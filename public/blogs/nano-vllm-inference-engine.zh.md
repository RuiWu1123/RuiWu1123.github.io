---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

假设你刚训完一个模型，想把它服务起来。第一版闭着眼都能写：接一个请求，跑一次前向，采一个 token，接上去，再跑一次，遇到 EOS 就停。它能跑。你拿去演示，有人问这东西能扛多少用户，你一测，答案大概是四个。

值得把"为什么只有四个"说精确，因为这两个原因，正是整个推理引擎领域存在的理由。

第一个原因是 KV cache 又大又不可预知。一个序列每生成一个 token，都会留下 key 和 value 向量，而且只要这个序列还活着，它们就得一直待在显存里。所以你得给每个请求分一块 buffer。可你事先并不知道用户是要 20 个 token 还是 2000 个，只能按最大值分。结果是每个请求都只用掉自己预留的一丁点，剩下的你在每一个并发序列上同时付一遍钱。你的显存看起来是满的，装的大部分是空气。

第二个原因要等你开始 batching 才浮出来。batching 显然是对的，GPU 就喜欢大矩阵，于是你把请求攒成一组一起推进。可这一组要等最慢的那个成员结束才算完；一个在这组启动后一步才到达的请求，得站在门外等整组彻底散场；而早早生成完的序列还占着位置，除了 padding 什么也不贡献。

vLLM 最出名的两项贡献，瞄的正是这两处。PagedAttention 不再要求 KV cache 是每个序列一整块连续 buffer，改成从共享池里发放固定大小的 block——和当年操作系统决定不再给进程连续物理内存，是同一个动作。Continuous batching 则让这一组在每一步都能换人，新请求下一轮就能进来。还有第三个思路叫 prefix caching，它注意到请求之间常常共享一长段前缀，于是让它们把算好的 block 也共享掉。

想靠读真实 vLLM 来学这些，麻烦在于它们被埋在多年优化的下面。[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用大约 1200 行 Python 重写了同一个内核，而且它不是玩具：在它自己的 benchmark 上是 1434 tok/s，对比真实 vLLM 的 1362（Qwen3-0.6B，256 并发序列，单张 RTX 4070 Laptop）。小到一个下午读得完，真到值得一读。

下面的顺序是：先看仓库长什么样、一个请求从头到尾经历了什么，然后把上面那两个机制各拿出一节来细看。

## 一、地图

`nanovllm` 这个 package 一共 19 个 Python 文件。其中七个负责扛一个请求，剩下的是普通的 transformer 部件（RMSNorm、RoPE、SwiGLU、张量并行的线性层）和一些管道工作。点任意文件可以读它真实的源码：

![interactive:nanovllm-arch](#)

有一个文件得先提一句，因为它后面出现时会显得很怪。`utils/context.py` 是一个模块级全局变量：`ModelRunner` 把分页元数据写进去，`Attention` 在二十几层之下再把它读出来。这么做是为了让 block table 不必作为参数穿过中间每一个 `forward()`。它属于那种看起来明显不对、直到你试着用另一种方式写一遍才明白的设计。

## 二、一个请求经历了什么

`add_request()` 把 prompt 分词，包成一个 `Sequence`，也就是一个请求的全部状态：它的 token、有多少已经算进缓存、当下正在算多少、以及它持有哪些物理 block。包好之后它进入 `waiting` 队列，然后在下一轮之前什么都不会发生。

接着 `step()` 一遍遍地跑，每一轮做同样五件事。调度器决定这轮跑什么，要么是处理 prompt 的活，也就是 prefill，要么是每个序列各出一个 token 的活，也就是 decode，同一轮里绝不混。block manager 给选中的序列发放 KV-cache block，前缀对得上的就直接复用。model runner 把这些序列压平成 GPU 张量，其中包括一个 `slot_mapping`，说明每个新 token 的 K/V 该落在缓存的哪个位置。模型跑起来，这是一次完全普通的 Qwen3 前向，只不过它的 attention 层是隔着分页间接层去读写的。最后 sampler 把 logits 变成 token id，`postprocess()` 把它们接上去、给刚填满的 block 算哈希、并让撞到 EOS 或 token 上限的序列退场，把它们的 block 还给池子。

`generate()` 就这样循环到两个队列都空，然后解码成文本。整个引擎就这些。这五步里有两步藏着真正有意思的决策，它们是这篇文章剩下的全部内容。

## 三、调度器：一个 batch 如何不再是一个 batch

调度器每一轮要回答的问题很简单：手上有一些生成到一半的请求，还有一些刚到的，这一轮该让 GPU 干什么？

先看最朴素的答案。攒够一个 batch 的请求，一起发给模型；整批一起做 prefill，再一起一轮轮 decode；等最后一条也生成完，一起把结果返回。这种"一组凑齐了才动、动完了才走"的做法，就是静态 batching。它的毛病开头已经见过：整组的速度由最慢的那个成员决定，而中途到达的请求，只能站在门外等这一组彻底散场。

nano-vllm 的规则恰好反过来，而且只有一句：永远优先 prefill，只有实在没有 prefill 可做时，才跑 decode。

这句话看起来只是个优先级设定，实际上它就是 continuous batching 的全部。因为 prefill 排在前面，一个三十毫秒前才到的请求，下一轮就能开始处理 prompt，它不必等任何人结束；又因为 decode 每次只把每个在跑的序列推进一个 token，每一轮结束都成了一个天然的切换点，序列可以随时进、随时出。到这里，"batch"这个词其实已经不成立了——不存在一组固定成员，只有"这一轮碰巧在跑的那些"。

把这条策略放到三个请求上跑一遍，再拿同样三个请求跑一遍静态 batching，差别是这样的：

![Same three requests under continuous vs. static batching](blogs/images/nanovllm-batching-timeline.svg?v=1)

两幅图都是真的把 nano-vllm 的调度逻辑跑一遍生成的，不是手画的。同样的活，11 轮对 15 轮；而且请求到达越分散，差距越大，因为那正是静态 batching 花更多时间干等的场景。

图里有两处需要解释，它们恰好都是实现变得有意思的地方。

先看 C：第 3 轮被放进来，第 4 轮就被抢占了。看下面那排 block 占用条，池子在那一轮撞到了 FULL。某个正在跑的序列需要再要一个 block 才能继续，而一个都没有了。这时 `preempt()` 挑一个牺牲者，也就是最近才被放进来的那个正在跑的序列，释放它全部的 block，把它推回等待队列的最前面。C 丢掉了已有的工作，到第 8 轮才被重新 prefill。

这看起来像出了故障，实际上它是设计的一部分。调度器是故意放进来比它能保证内存的更多的序列的，因为大多数序列会提前结束，而按最坏情况预留恰恰是我们一开始想逃离的东西。抢占就是让这份乐观变得安全的泄压阀：引擎在压力下会变慢，而不是倒掉。

代码用一个大多数 Python 程序员没用过的语法表达了这件事：

```python
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

`while...else` 的 `else` 只在循环因为条件变假而退出时才执行，撞上 `break` 就绝不执行。所以这段的意思是：一直驱逐别人，直到这个序列能拿到 block，然后才调度它；如果已经没有别人可驱逐了，它就抢占自己，并且什么也不调度。换成别的写法就得额外引进一个标志变量，确实更难看。

另一处是分块 prefill。单个 prompt 可能比整轮的 token 预算还长。与其拒绝它，或者干脆撑爆预算，调度器把它切到几轮里去，但一轮里最多只切一个。实现这条规则的，是一个连接词：

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

`scheduled_seqs` 只有面对这一轮第一个候选者时才是空的。所以第一个序列被允许超预算、被切开；之后任何一个装不下的，就老实等下一轮。这也解释了为什么 `Sequence` 要把 `num_cached_tokens` 和 `num_scheduled_tokens` 分开记：一个序列可以真的只算了一半，而调度器和缓存必须对"分界线在哪"达成一致。

## 四、PagedAttention：KV cache 不再连续

回到第一个问题。一个序列的缓存会不可预测地增长，而你必须在知道它多大之前就先定下大小。

解法是从操作系统那里原样搬过来的，而且这个类比值得好好说，因为它不是修辞。一个进程以为自己拥有一段平坦连续的地址空间，物理上它的内存却是散落的页，中间靠页表做翻译。PagedAttention 做的就是这件事：一个序列以为自己拥有一段连续的 KV cache，物理上那是散落在共享池里的固定大小 block，每个序列有一张 block table，把逻辑块号映射到物理块号。

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

好处是，分配可以随着序列增长一次只给一个 block。一个序列任何时刻只持有它真正用掉的量，外加最多一个没填满的块。浪费被 block 大小——256 个 token——卡住了上限，无论这个序列最终跑多久。

但还有第二个好处，从操作系统那个类比里看不出来，而且可以说更重要。一旦 block 只是表里的条目，两个序列就可以指向同一个 block。如果两个请求共享一段 system prompt，它们就能共享这段的 KV cache：算一次，存一份。

实现机制是内容寻址。每个满 block 都会被哈希，而关键在于，它串上了前一个 block 的哈希：

```python
h = self.compute_hash(token_ids, h)
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

链式哈希是让复用正确、而不只是"大概率对"的原因。一个 block 的身份不是"这 256 个 token"，而是"这 256 个 token，在这个位置上，跟在恰好这段历史之后"。所以匹配成功就意味着两个序列一路一致回到第 0 个 token，而这恰恰就是它们的 K/V 真正相同的条件。循环在第一次未命中就 break，因为前缀匹配按定义就是开头的一段连续区间。而即便疑似命中，它还会再比一遍存着的 `token_ids`，于是哈希碰撞的结果是退化成未命中，而不是悄悄把别的请求的缓存端上来。

block 是按引用计数释放的，不是按归属。完整的生命周期是这样：

![One block pool, four moments](blogs/images/nanovllm-block-lifecycle.svg?v=1)

注意最后一帧：A 结束了，但 block 0 和 1 哪也没去，因为 B 还在用。共享前缀比创造它的那个序列活得更久。这也意味着缓存能挺过抢占——一个被抢占的序列重新 prefill 时，只要它的 block 还在，就会直接命中，这就是为什么抢占的代价比看起来小。

有个细节容易绊人：block 是在被算完之后才哈希的，不是在分配时。`hash_blocks()` 跑在 `postprocess()` 里，只覆盖这一轮里刚填满的 block。一个序列末尾那个没填满的 block 永远不会进缓存，因为它的内容还没定型，还没有一个稳定的身份可以当键。

decode 期间的增长几乎不要钱，而这段代码算个小玩笑：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`len(seq) % block_size == 1` 恰好在"溢出到新 block"的那个 token 上为真。在 256 个里的那一个 token 上，它读作"空闲 block 不少于 1"；在另外 255 个上，它读作"空闲 block 不少于 0"，恒真。它利用了 Python 里 `bool` 就是 `int` 这一点，而且它在干正经活：决定一个序列会不会被抢占的，就是这个判断。

### 怎么把 K/V 写进散落的 block、又读出来

还剩两个问题，而它们才是让分页真的能在 GPU 上跑起来的那两个。

写入是一次 scatter。每个新 token 的 K/V 必须落到它的 block table 所指向的那个物理槽位，而这些槽位并不连续。`ModelRunner` 预先把目的地算进一个 `slot_mapping` 张量，然后一个 Triton kernel 每个 token 起一个 program：加载 K 和 V，读出目的地，写进去。它只有一个守卫，`if slot == -1: return`，存在的意义是让 CUDA graph 重放能工作，因为被捕获的 graph 跑在固定 batch size 上，用不到的槽位会被标成 `-1`。

读取是更难的那个，而 nano-vllm 的答案是：它根本不自己读。它把 `block_table` 交给 attention kernel，让 kernel 去 gather。`flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 都原生接受 block table，所以那个散落的布局从头到尾不需要被拼成一个连续张量。整个 attention 层唯一真正的分支就这一个：

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
```

前缀命中时，刚算出来的 `k`/`v` 只覆盖新的那些 token，但 attention 必须也看到缓存里的前缀，于是它们被整个换成完整的 cache，之后 kernel 通过 block table 把所有东西读回来。这两行分支，就是"分页存储"和"普通 attention"之间的全部接缝。

而这正是该带走的结论：上面这一切都没有改变数学。attention 依然是缩放点积再 softmax。分页改变的只是 K 和 V 物理上住在哪，并且把这层间接寻址推进了一个本来就要按块遍历的 kernel 里。至于那个 kernel 内部在干什么——tiling，以及为什么不把完整 attention 矩阵摆出来很重要——是 [GPU field guide](#/blog?id=gpu-guide-for-dl) 那篇的主题。

上面这些片段的完整源码：

![interactive:nanovllm-code-blockmgr](#)

## 五、剩下的，简单说

采样只有九行，其中一行是个漂亮的小技巧：

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

把每个概率除以一个独立的 `Exponential(1)` 采样值再取 argmax，恰好就是在从分类分布里采样。这是 Gumbel-max trick，因为除以一个指数分布变量，在对数空间里就是减去一个 Gumbel 变量。那为什么不直接用 `torch.multinomial`？因为这段全是逐元素算术加一次 argmax，没有依赖数据的控制流，`torch.compile` 能把它整个融成一个 kernel。

上面一层还有个相关的省法。prefill 期间模型会为每个 prompt token 产出隐状态，但每个序列只有最后一个 token 能预测出东西，所以 `ParallelLMHead` 在投影之前先切片：

```python
last_indices = context.cu_seqlens_q[1:] - 1
x = x[last_indices].contiguous()
```

对一批长 prompt，这就把"在几千个位置上做词表规模的投影"变成了"只在寥寥几个位置上做"。

还有一件贯穿全程的事：任何地方都不 padding。prefill 把所有序列的 token 拼成一条扁平的一维张量，用累积偏移量标记边界，于是一批长短极不均匀的 prompt，代价恰好等于它们长度之和。而那两套偏移量数组本身还携带信息：`cu_seqlens_q` 只数新 token，`cu_seqlens_k` 覆盖包含缓存前缀的整个序列，所以 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 就是"这一轮发生了前缀命中"的判据。

最后是两个值得知道的优化。decode 步很小，时间被 Python 的启动开销主导，所以 nano-vllm 为一组固定的 batch size（`[1, 2, 4, 8] + range(16, max_bs+1, 16)`）预先录好整个 decode 步，之后重放不小于当前 batch 的最小那份 graph；只有 decode 这么干，prefill 的形状没法预测。张量并行则为每个 rank 起一个 `ModelRunner`，由 rank 0 通过一块 `SharedMemory` 广播方法调用，大约四十行手写的 RPC。

## 六、nano-vllm 没有的东西

投机解码、量化、多种调度策略、把 prefill 和 decode 拆到不同机器的分离式服务，以及一个通用到能吃下 Qwen3 之外任何模型的加载层。这份清单基本就是生产级 vLLM 之所以庞大的原因，而其中没有任何一项会改变上面那两个机制。

这也正是先读 nano-vllm 的理由：让推理引擎成立的那些想法，恰好就是能塞进 1200 行里的那些，其余的都是这些想法正确之后才堆积起来的东西。
