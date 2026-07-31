---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

朴素地做模型服务，会在两个很具体的地方出问题。第一是内存：一个序列每生成一个 token，都会留下 key 和 value 向量，而且必须在这个序列存活期间一直待在显存里；最直接的实现是给每个请求预留一段按最坏情况估算的连续 buffer。绝大多数请求只用得上其中一小部分，而这份浪费会在每一个并发序列上重复一遍 —— 朴素服务代码往往撑不了几个并发，原因通常就在这里。第二是调度：如果请求被打包成 batch、步调一致地推进，那整个 batch 的速度就取决于最慢的那个成员，而中途到达的请求得等整个 batch 排空才能进来。

vLLM 的两个核心思想，正好一一对应这两个问题。**PagedAttention** 把 KV cache 存成从共享池里取出的、固定大小的小 block，借用的是操作系统做虚拟内存的那套办法，于是不需要连续，也不需要提前预留。**Continuous batching** 让调度器在每一步都能改变 batch 的成员组成，新请求立刻就能开始干活，而不用排队等位置。第三个机制 **prefix caching**，则在共享前缀的请求之间复用已经算好的 block。

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用大约 1200 行 Python 把这三个都实现了，而且在项目自己的 benchmark 上跑到 1434 tokens/s，对比真实 vLLM 的 1362（Qwen3-0.6B，256 并发序列，单张 RTX 4070 Laptop）。这个体量小到可以完整读完，但实现的是真机制而不是玩具。这篇文章跟着一个请求走一遍，走到 PagedAttention 时停下来把它彻底拆开。

## 一、仓库结构，以及穿过它的那条路径

`nanovllm` 这个 package 一共 19 个 Python 文件（仓库根目录另有 `bench.py` 和 `example.py`）。其中七个构成了请求流经的那条流水线，顺序固定 —— 这篇文章跟的就是这七个。剩下的代码一个请求同样会执行，但那些要么是标准的 transformer 组件（RMSNorm、RoPE、SwiGLU、张量并行的线性层），要么是管道工作（权重加载、配置），不属于服务逻辑。

![interactive:nanovllm-arch](#)

开始走之前，这棵树里有两处值得先指出来。`utils/context.py` 是一个模块级全局变量，负责把 `slot_mapping` 和 `block_tables` 从 `ModelRunner` 直接送到 `Attention` —— 它存在的意义就是让分页元数据不必作为参数层层穿过中间每一层。另外 `layers/` 大部分是普通模型代码，但有两个例外会去读这个全局量并因此改变行为：`attention.py`，分页真正发生的地方；以及 `embed_head.py`，它在 prefill 时会跳过输出投影的绝大部分（见第六节）。

## 二、一个 prompt 变成 `Sequence`

`LLMEngine.add_request()` 把 prompt 分词、包成 `Sequence`，塞进调度器的 `waiting` 队列。在下一轮调度之前，不会再发生任何事。

一个 `Sequence` 就是某个请求的完整状态：它的 token id、其中有多少已经被算进 KV cache 了（`num_cached_tokens`）、这一轮正在算多少个（`num_scheduled_tokens`），以及它拥有哪些物理缓存 block（`block_table`）。前两个计数器的区分，正是"分块 prefill"和"前缀缓存"这两件事能被表达出来的前提 —— 一个序列可以只算了一半，而引擎必须精确知道分界线在哪。

其中有一个方法比其余都重要：`block(i)` 把序列的 token 按 block 大小切片，`num_blocks` 则报告它一共跨多少块。这个切片给了每个 block 一份定义明确的内容，而这恰恰就是它可以被哈希的原因 —— prefix caching 全靠可哈希这件事运转。

## 三、调度器，以及 continuous batching 到底是什么

引擎的每一步都从"问调度器该跑什么"开始。答案永远是"prefill 工作"或者"给所有 running 序列各解一个 token"二选一 —— 绝不会同时。`schedule()` 先试前者，只有在什么都没排上时，才落到后者。

prefill 那一趟在两个预算下排空 `waiting` 队列：`max_num_seqs` 个序列和 `max_num_batched_tokens` 个 token。如果单个 prompt 比剩余 token 预算还大，它会被**分块**（chunked）—— 分几轮 prefill 完 —— 但一轮里最多只会劈开一个 prompt，而这条约束是靠一个连接词实现的：

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

`scheduled_seqs` 只有在这一轮的第一个候选者面前才是空的，所以第一个序列被允许超预算、被切开，而之后任何一个装不下的序列就干脆推迟到下一轮。被分块的序列还会留在 `waiting` 里，因为晋升到 `running` 的条件是 `num_cached_tokens + num_scheduled_tokens == num_tokens` —— 它就是这样在下一轮被重新捡起来的。

decode 那一趟是内存压力浮出水面的地方，它依赖了 Python 最少人知道的语法：

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```

`while ... else` 的 `else` 只在循环条件自然变假、而*没有*撞上 `break` 时才执行。所以只有 `can_append` 最终成功了，这个序列才会被调度。如果始终不成功，`preempt()` 就会驱逐一个牺牲者 —— 最近才加入 running 的那个，或者在实在没有别的可选时驱逐它自己 —— 释放它的 block，并把它塞回 `waiting` 的*队首*，这样一旦腾出空间它会被优先重新 prefill。

抢占不是错误处理路径。它是针对 KV cache 被填满这件事设计好的应对方式，也是引擎在高负载下会优雅降级而不是直接失败的原因。

prefill 优先这个顺序带来的结果是：batch 的成员组成是流动的。此刻到达的请求，下一次 `schedule()` 调用就能开始 prefill，完全不需要任何正在跑的序列先结束。一步步走一遍，和静态 batching 的差别就很具体了：

![interactive:nanovllm-scheduler](#)

## 四、搭出 GPU 张量

被调度的序列会变成扁平张量。对 prefill，所有序列还没算的 token 被拼接成一条一维张量，用累积偏移量标记每个序列从哪开始、到哪结束。任何地方都不会被 padding 到统一长度 —— 这就是"变长打包"（varlen）的含义，也是为什么一批长度差异极大的 prompt，代价恰好等于这些长度之和。

这里维护了两套偏移量数组，而它们之间的差值本身携带了信息：

```python
seqlen_q = seq.num_scheduled_tokens
seqlen_k = end
cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```

query 长度只算*新*的 token；key 长度则要覆盖整个序列，包括那段新 token 仍然必须注意到的缓存前缀。所以 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 恰好就是"这一轮发生了前缀缓存命中"这个条件，而 nano-vllm 正是拿它来判断到底需不需要把 block table 交给 attention kernel。

另一个产物是 `slot_mapping`：为每个新 token 算出它该被写进的绝对物理槽位，做法是遍历这个序列的 `block_table`，把每个 block id 换算成基址偏移。decode 是这件事的退化情况 —— 每个序列只有一个 token，槽位就是最后一个 block 的末尾。这个数组就是交给 PagedAttention 的接力棒，而且它是 attention 层唯一需要的东西 —— 有了它，attention 就能往一块自己对物理布局一无所知的缓存里写数据。

## 五、PagedAttention

走到这里先停下来，因为整个引擎的其余部分都是围着这个机制搭起来的。

把问题重新说一遍：一个序列的 KV cache 会不可预测地增长，而你在它开始时并不知道最终会有多大。按最坏情况预留，几乎全是浪费。解法就是操作系统用的那个 —— 不再要求内存连续。把缓存切成固定大小的 block（nano-vllm 默认配置里是 256 个 token），维护一个共享的池子，再给每个序列一张 **block table**，把它的逻辑 block 编号映射到当时恰好空闲的那个物理 block。

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

序列自己看到的视图依然是顺序的 —— block 0、block 1、block 2 —— 而物理 block 爱在哪在哪。剩下的唯一浪费，是序列最后一个 block 里没填满的那部分，无论序列长到多少，这份浪费都被 block 大小卡住了上限。

### 分配，以及基于内容寻址的复用

`can_allocate()` 在真正落实任何事之前先探测缓存。它遍历序列的满 block，对每一个做哈希、而且*串上前一个 block 的哈希*，再拿每个摘要去 `hash_to_block_id` 字典里查。这个循环里有三处细节撑起了整个设计。

链式哈希意味着一个 block 的身份不是"这 256 个 token"，而是"这 256 个 token，在这个位置上，跟在这段确切历史之后"—— 于是两个序列只有一路一致回到第 0 个 token 才可能共享一个 block，这让复用是安全的，而不只是看起来合理。循环在第一次未命中时就 break，因为前缀匹配按定义就是一段开头的连续区间。而每次疑似命中时，它都会拿存着的 `token_ids` 和序列真实的 token 再核对一遍，所以哈希碰撞的结果是一次未命中，而不是悄悄把别的请求的 KV 数据端上来。

返回值是被复用的，读调度器之前值得先知道：`-1` 表示"空闲 block 不够，先别调度这个"，其他任何值都是命中的开头 block *数量*。调度器正是拿这个数量去减，算出真正需要计算的 token 有多少。

`allocate()` 随后落实 —— 命中的 block 增加 `ref_count`，未命中的从空闲队列里取新的。而 decode 期间的增长被刻意做得几乎免费：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

`len(seq) % block_size == 1` 恰好在"刚刚溢出一个 block 边界"的那个 token 上为真，所以每 256 个 token 才申请一次新 block，另外 255 个一分钱不花。`can_append` 里那个比较悄悄利用了 Python 中 `bool` 就是 `int` 这一点：在边界 token 上它读作"空闲 block ≥ 1"，其他时候读作"空闲 block ≥ 0" —— 恒真。

释放则是遍历 block table 逐个减引用计数，只放掉归零的那些。因为按引用计数而不是按归属释放，一段共享前缀能比"碰巧创造了它"的那个序列活得更久。看着池子一步步变化，比读代码更容易记住这件事：

![interactive:nanovllm-blocks](#)

![Prefix caching: identical leading blocks share one physical block](blogs/images/nanovllm-prefix-cache.svg?v=1)

还有一个图里画不出来的细节：block 是在被*算完之后*才哈希的，不是在分配时。`hash_blocks()` 跑在 `postprocess()` 里，只覆盖这一轮里刚刚变满的 block —— 这就是为什么一个序列末尾那个没填满的 block 永远不会进缓存：它的内容还没定下来，也就没有一个稳定的身份可以当键。

### 把 K/V 写进去、读回来

写入是一个 scatter，由一个 Triton kernel 完成，每个 token 一个 program：各自加载自己 token 的 key 和 value，从 `slot_mapping` 里读出目的地，然后存进缓存对应槽位。其中有一个守卫值得一提 —— `if slot == -1: return`。这个哨兵值正是 CUDA graph 重放能安全工作的前提，因为被捕获的 graph 跑在固定 batch size 上，用不到的 padding 条目会被标成 `-1` 并跳过。

读取才是这层间接寻址真正兑现价值的地方，而结论是：`Attention.forward()` 自己从不去收集任何东西。它把 `block_table` 直接传给 `flash_attn_varlen_func`（prefill）或 `flash_attn_with_kvcache`（decode），由 kernel 在内部完成那个散落的 gather。唯一真正的分支是这个：

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
```

前缀缓存命中时，刚算出来的 `k` 和 `v` 只覆盖新 token，但 attention 必须也看到缓存里的前缀 —— 于是它们被整个*替换*成完整的 cache，之后 kernel 通过 block table 把所有东西读回来。这就是"分页存储"和"普通 attention"之间的全部接缝。

如果想把上面这些片段完整读一遍，而不只是看摘录：

![interactive:nanovllm-code-blockmgr](#)

上面这一切完全没有改变 attention 的数学 —— 依然是缩放点积再 softmax，和以前一模一样。分页改变的只是 K 和 V 住在哪里。（flash-attention kernel 内部的 tiling 与带宽推理是 [GPU field guide](#/blog?id=gpu-guide-for-dl) 那篇的主题；这篇文章把那个 kernel 当作既定条件。）

## 六、采样

整个采样器只有九行：按温度缩放 logits、softmax，然后由一行看起来很奇怪的代码完成采样。

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

把每个概率除以一个独立的 `Exponential(1)` 采样值再取 `argmax`，恰好等价于从分类分布里采样 —— 这就是换了个形式的 Gumbel-max trick，因为除以一个指数分布变量，在对数空间里就是减去一个 Gumbel 变量。不用 `torch.multinomial` 而这样写的理由是：它全部由逐元素算术加一次 argmax 组成，没有依赖数据的控制流，于是 `torch.compile` 能把整段融合成一个 kernel。

上面一层还有个相关的小技巧。prefill 期间模型会为每个 prompt token 算出隐状态，但只有每个序列的最后一个 token 才能产出下一个 token，所以 `ParallelLMHead` 在投影之前先切片：

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

对一批长 prompt 来说，这就把"在几千个位置上做词表规模的投影"变成了"只在寥寥几个位置上做"。

## 七、闭合循环

`postprocess()` 收尾这一轮：把刚填满的 block 哈希掉、推进已缓存 token 计数、追加新 token，并让撞到 EOS 或 `max_tokens` 的序列退场 —— 它们的 block 立刻回到空闲池，而不是等某次延后的清扫。

它同时也包含了分块 prefill 的另一半：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

prompt 还没走完的序列会把这次采样出的 token 直接丢掉，因为从半截 prompt 预测出来的 token 没有意义。只有等 prompt 被完整消费完，生成才真正开始。这一条和调度器"决定把 prompt 切开"是一体两面，两边必须对得上，否则序列会在 prefill 中途吐出垃圾。

`generate()` 把这一切裹进一个循环，一直跑到两个队列都空为止，然后把每个序列累积的 id 解码回文本。

![interactive:nanovllm-code-sched](#)

## 八、和真实 vLLM 比，少了什么

nano-vllm 确实还包含两样这次没细讲的东西，都属于工程而非算法。张量并行为每个 rank 起一个 `ModelRunner` 进程，由 rank 0 通过一块 `SharedMemory` 广播方法调用 —— 大约四十行手写的 RPC。CUDA graph 捕获则为一组固定的 batch size（`[1, 2, 4, 8] + range(16, max_bs+1, 16)`）预先录好 decode 步骤，之后重放"不小于当前 batch size 的最小那份" graph（`next(x for x in self.graph_bs if x >= bs)`），而不是每步都从 Python 重新派发每个算子；这也解释了为什么 `slot_mapping` 需要 `-1` 这个哨兵值，以及为什么只有形状可预测的 decode 才用 graph。

真正缺席的，是让生产级 vLLM 变得庞大的那些东西：投机解码、量化、多种调度策略、把 prefill 和 decode 拆到不同机器上的分离式服务，以及一个通用到能支持 Qwen3 之外各种架构的模型加载层。这些都不会改变上面这些机制 —— 它们是核心正确之后，才一层层堆上去的东西。
