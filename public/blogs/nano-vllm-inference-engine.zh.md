---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

假设你刚训完一个模型，准备把它部署成一个线上服务。最直接的做法几乎不用思考就能写出来：来一个请求就跑一次前向，采样出一个 token，拼回序列末尾，再跑下一次，直到模型吐出 EOS 为止。这段代码是能正常工作的。但等你把它拿去演示，有人问起它同时能撑住多少个用户，你实测一下会发现，这个数字大概是四。

为什么只有四，这件事值得拆开来仔细看，因为导致它的那两个原因，恰恰就是推理引擎这一整个方向存在的意义。

第一个原因是 KV cache 既占地方，大小又没法提前知道。序列每生成一个 token，都会留下一份 key 和 value 向量，而且只要这个序列还没结束，它们就得一直占着显存。所以你必须为每个请求预留一块 buffer。可问题在于，你事先无从判断用户这次是要 20 个 token 还是 2000 个，保险起见只能按最大长度来预留。于是每个请求实际用掉的都只是其中很小一部分，而剩下那些没用上的空间，你在每一个并发请求上都要重复浪费一次。显存看上去是满的，装的其实大半是空气。

第二个原因要等你开始做 batching 才会暴露出来。把请求攒成一批一起算，这个方向本身没错，毕竟 GPU 天生就擅长大矩阵。但这一批要等最慢的那个成员算完才算结束；在这批已经开跑之后才到达的请求，只能在外面干等，直到整批彻底散场；而那些早早生成完的序列，位置还占着，除了贡献 padding 之外什么也做不了。

vLLM 最为人熟知的两项设计，针对的正是这两点。PagedAttention 不再要求 KV cache 是每个序列独占的一整块连续空间，而是改成从一个共享的池子里按固定大小分配 block，这和当年操作系统决定不再给进程分配连续物理内存，是同一个思路。Continuous batching 则让这一批的成员在每一步都可以变动，新来的请求下一轮就能加入。除此之外还有一个叫 prefix caching 的机制，它注意到不同请求之间往往共享一大段相同的前缀，于是让它们把这段前缀算出来的 block 也共用掉。

想通过阅读 vLLM 本体来理解这些设计，难点在于它们被埋在多年积累的工程优化底下。[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用一千两百行左右的 Python 重写了同一套内核，而且它并不是个演示玩具：按项目自己给出的 benchmark，它跑到 1434 tok/s，而同样条件下的 vLLM 是 1362（Qwen3-0.6B，256 路并发，单张 RTX 4070 Laptop）。规模小到一个下午能读完，实现又足够真实，值得认真读一遍。

下面的安排是这样：先看清楚它的代码结构，以及一个请求从进来到返回会经过什么，然后把上面那两个核心机制各拿出一节来展开。

## 一、代码地图

`nanovllm` 这个 package 一共十九个 Python 文件，其中真正参与处理一个请求的是七个，剩下的要么是常规的 transformer 组件，比如 RMSNorm、RoPE、SwiGLU 和张量并行的线性层，要么是加载权重、读配置这类杂活。点开任意一个文件都能看到它真实的源码：

![interactive:nanovllm-arch](#)

这里有个文件需要先打个招呼，否则它后面出现的时候会让人费解。`utils/context.py` 里放的是一个模块级的全局变量：`ModelRunner` 把分页相关的元数据写进去，`Attention` 在二十多层之后再从里面把它读出来。这样做的目的，是让 block table 不必作为参数一路穿过中间每一层的 `forward()`。这种设计乍看很不讲究，但真要换成规规矩矩传参数的写法，你会发现那样更难受。

## 二、一个请求会经历什么

请求进来之后，`add_request()` 先把 prompt 分词，然后包成一个 `Sequence`。这个对象承载了一个请求的全部状态：它的 token 序列、其中已经算进缓存的有多少、当前这一轮正在算的有多少，以及它手上持有哪些物理 block。包好之后它被放进 `waiting` 队列，在下一轮调度到来之前不会有任何事情发生。

真正干活的是 `step()`，它会被反复调用，每一轮都按同样的顺序做五件事。首先由调度器决定这一轮跑什么，要么是处理 prompt 的 prefill，要么是让每个序列各吐一个 token 的 decode，两者在同一轮里不会混着来。接着 block manager 为选中的这些序列分配 KV-cache block，凡是前缀能对上的就直接复用现成的。然后 model runner 把这批序列压平成 GPU 张量，其中包含一个 `slot_mapping`，用来指明每个新 token 的 K/V 应该写到缓存的哪个位置。再往下模型开始前向，这一步就是一次普普通通的 Qwen3 计算，唯一的区别在于它的 attention 层是隔着分页这层间接寻址去读写缓存的。最后 sampler 把 logits 变成 token id，`postprocess()` 负责收尾：把新 token 接到序列上，给这一轮刚填满的 block 计算哈希，并让已经遇到 EOS 或者达到长度上限的序列退场，把它们占用的 block 归还给池子。

`generate()` 就是把上面这个循环一直转下去，直到两个队列都空掉，最后把 token 解码回文本。整个引擎的骨架就是这些。这五步里有两步藏着真正值得研究的取舍，也就是这篇文章接下来要讲的全部内容。

## 三、调度器：batch 是怎么变得不再固定的

调度器每一轮需要回答的问题其实很朴素：现在手上既有一些生成到一半的请求，又有一些刚刚到达的，这一轮到底该让 GPU 去算什么？

先看最容易想到的那种答案。攒够一批请求之后一起发给模型，整批一起做完 prefill，再一起一轮一轮地 decode，等最后一条也生成结束，把结果统一返回。这种"人不齐就不开始、没走完就不散场"的组织方式，就是所谓的静态 batching。它的毛病在开头已经见过了：整批的推进速度取决于最慢的那个成员，而中途到达的请求只能在门外等着，等这一批彻底结束才轮得到自己。

nano-vllm 的规则正好反过来，而且一句话就能说完：只要还有 prefill 可做就优先做 prefill，实在没有 prefill 了，才去跑 decode。

这句话表面上只是定了个优先级，实际上 continuous batching 的全部内容就在这里。由于 prefill 排在前面，一个三十毫秒前才到达的请求，下一轮就能开始处理它的 prompt，完全不需要等谁结束；又由于 decode 每次只把每个在跑的序列往前推进一个 token，每一轮的结束都自然成了一个可以换人的节点，序列因此可以随时加入、随时离开。走到这一步，"batch"这个词其实已经名存实亡了，因为根本不存在一组固定的成员，只有"这一轮恰好在跑的那些序列"。

把这条规则放到三个请求上实际跑一遍，再让同样三个请求走一遍静态 batching，两者的差别是这样的（A 和 B 在第 1 轮到达，C 在第 3 轮到达；为了让内存压力看得出来，block 大小缩到 4 个 token，整个池子只有 6 个 block）：

![Same three requests under continuous vs. static batching](blogs/images/nanovllm-batching-timeline.svg?v=2)

两幅图都是把 nano-vllm 的调度逻辑真跑一遍之后生成的，不是照着感觉画的。同样的工作量，一个用了 11 轮，另一个用了 15 轮；而且请求到达的时间越分散，这个差距会拉得越开，因为那正是静态 batching 最容易空等的情况。

图里有两处需要解释，它们恰好也是实现层面最有意思的两处。

先看 C 这一行：它在第 3 轮被放进来，第 4 轮就被踢出去了。原因是那一轮池子里的 block 已经分光，某个正在跑的序列想再要一个 block 才能继续，却一个都拿不到。这时 `preempt()` 会挑一个牺牲者，也就是最近才被放进来的那个正在跑的序列，把它占用的 block 全部释放，然后把它塞回等待队列的最前面。C 之前算过的东西就此作废，一直到第 8 轮才被重新 prefill。

这一幕看上去像是出了故障，实际上它是被设计出来的。调度器是故意放进来比它能保证显存的更多的序列的，因为大部分序列都会提前结束，而按最坏情况预留恰恰是我们一开始想摆脱的做法。抢占就是让这份乐观不至于翻车的泄压阀：压力大的时候引擎会变慢，但不会崩。

这件事在代码里是用一个大多数 Python 使用者没写过的语法表达的：

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

`while...else` 里的 `else` 只有在循环因为条件不再成立而正常退出时才会执行，一旦撞上 `break` 就会被跳过。所以这段的含义是：不停地驱逐别人，直到当前这个序列能拿到 block 为止，拿到了才调度它；如果已经没有别人可以驱逐，它就只好抢占自己，这一轮什么也不调度。换成不用 `while...else` 的写法，就得额外引进一个标志变量，读起来反而更绕。

另一处是分块 prefill。单个 prompt 有可能比一整轮的 token 预算还要长。与其把它拒之门外，或者干脆让它撑爆预算，调度器选择把它拆到好几轮里去处理，但每一轮最多只拆一个。而实现这条约束的，只是一个连接词：

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

`scheduled_seqs` 只有在面对这一轮第一个候选者的时候才是空的。所以第一个序列可以超出预算、被拆开来算，而之后任何一个装不下的，都老老实实等下一轮。这也顺带解释了 `Sequence` 为什么要把 `num_cached_tokens` 和 `num_scheduled_tokens` 分开记：一个序列确实可能只算了一半，而调度器和缓存两边必须对"算到哪儿了"有完全一致的认识。

## 四、PagedAttention：让 KV cache 不必连续

回到最开始那个问题。一个序列的缓存会不断增长，而且长到多少事先不知道，你却必须在它开始之前就把空间定下来。

解决办法是从操作系统那里整个搬过来的，这个类比值得认真讲，因为它并不只是个比方。一个进程以为自己拥有一段平坦而连续的地址空间，但物理上它的内存是散落在各处的页，中间靠页表来完成翻译。PagedAttention 做的完全是同一件事：一个序列以为自己拥有一段连续的 KV cache，物理上那是散落在共享池里的一个个固定大小的 block，而每个序列各自持有一张 block table，负责把逻辑块号翻译成物理块号。

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

这样做的直接好处是，分配可以跟着序列的增长一块一块地给。任何时刻，一个序列手上只有它真正用掉的那些 block，外加最多一个没填满的。被浪费掉的空间因此被 block 大小卡住了上限，也就是 256 个 token，无论这个序列最后生成了多长，这个上限都不会变。

但还有第二个好处，从操作系统那个类比里是看不出来的，而且它可能更重要。既然 block 只是表里的一个条目，那么两个序列完全可以指向同一个 block。如果两个请求共享同一段 system prompt，它们就能共用这段前缀的 KV cache，算一次，存一份。

具体的实现方式是内容寻址。每个填满的 block 都会被计算一个哈希，而关键之处在于，这个哈希串上了它前一个 block 的哈希：

```python
h = self.compute_hash(token_ids, h)
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

链式哈希是让这种复用真正正确、而不只是"多半没错"的原因。一个 block 的身份并不是"这 256 个 token"，而是"这 256 个 token，出现在这个位置上，并且前面跟着的恰好是这一段历史"。所以两个 block 的哈希对得上，就意味着这两个序列从第 0 个 token 起就完全一致，而这正是它们的 K/V 确实相同的充分条件。循环在第一次没对上的时候就 break，因为前缀匹配按定义就是从头开始的一段连续区间。而即便哈希看起来命中了，代码还会再比对一次存下来的 `token_ids`，这样哈希碰撞的后果只是退化成一次未命中，而不会把别人的缓存悄悄端上来。

block 的释放依据是引用计数，而不是谁最早申请了它。完整的生命周期是这样：

![One block pool, four moments](blogs/images/nanovllm-block-lifecycle.svg?v=2)

留意最后一帧：A 已经结束了，但 block 0 和 1 并没有被回收，因为 B 还在用它们。这段共享的前缀比当初创造它的那个序列活得更久。这同时也意味着缓存能够挺过抢占，一个被踢出去的序列在重新 prefill 时，只要它原来的 block 还留在池子里，就能直接命中，这就是抢占的代价其实没有看上去那么大的原因。

有个容易让人绊一下的细节：block 是在算完之后才被哈希的，而不是在分配的时候。`hash_blocks()` 跑在 `postprocess()` 里，而且只处理这一轮里刚刚填满的那些 block。所以一个序列末尾那个没填满的 block 永远进不了缓存，因为它的内容还没定下来，也就还谈不上有一个稳定的身份可以拿来当键。

decode 期间的扩容几乎不花什么代价，而这段代码本身算个小玩笑：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`len(seq) % block_size == 1` 只有在某个 token 恰好溢出到新 block 的那一刻才成立。在 256 个 token 里的那一个上，这行代码读作"空闲 block 不少于 1"；在其余 255 个上，它读作"空闲 block 不少于 0"，恒真。它利用了 Python 里 `bool` 本身就是 `int` 这一点，而且它干的是正经活，一个序列会不会被抢占，判断依据就是这一行。

到这里还剩两个问题没解决，而它们才是让分页在 GPU 上真正跑得起来的关键。

先说写入，它是一次 scatter。每个新 token 的 K/V 必须落到它的 block table 所指定的那个物理槽位上，而这些槽位彼此并不相邻。`ModelRunner` 会提前把这些目的地算好放进 `slot_mapping` 张量，然后由一个 Triton kernel 来完成搬运，每个 token 起一个 program，做的事情就是加载 K 和 V、读出目的地、写进去。整个 kernel 只有一处守卫，即 `if slot == -1: return`，它存在的意义是让 CUDA graph 的重放能够工作，因为被捕获的 graph 跑在固定的 batch size 上，那些用不到的槽位会被标成 `-1` 跳过。

读取是更棘手的那个，而 nano-vllm 的应对方式是根本不自己读。它把 `block_table` 直接交给 attention kernel，让 kernel 自己去把散落的 block 收集起来。`flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 都原生支持传入 block table，所以那个散落的物理布局从头到尾都不需要被拼回成一个连续张量。整个 attention 层里唯一称得上分支的地方只有这一处：

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
```

前缀命中的时候，刚算出来的 `k` 和 `v` 只覆盖了新增的那部分 token，可 attention 必须连缓存里的前缀一起看到，于是它们被整个替换成完整的 cache，之后由 kernel 通过 block table 把所有内容读回来。这两行分支，就是分页存储和普通 attention 之间的全部接缝。

而这也正是最该带走的一点：上面这一整套机制，没有任何一处改动了数学本身。attention 依然是缩放点积之后做 softmax。分页真正改变的只是 K 和 V 物理上存放在哪里，并且把这层间接寻址下推给了一个本来就要按块遍历的 kernel。至于那个 kernel 内部究竟在做什么，比如 tiling，以及为什么不把完整的 attention 矩阵摆出来这件事很关键，是 [GPU field guide](#/blog?id=gpu-guide-for-dl) 那篇文章的主题。

## 五、其余几处值得一提的设计

采样部分总共九行，其中一行是个挺漂亮的技巧：

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

把每个概率除以一个独立的 `Exponential(1)` 采样值，再取 argmax，这个操作恰好严格等价于从分类分布里采样。它其实就是 Gumbel-max trick，因为除以一个指数分布的随机变量，在对数空间里就相当于减去一个 Gumbel 变量。那为什么不直接调 `torch.multinomial`？因为这样写全程只有逐元素运算和一次 argmax，不存在依赖数据取值的控制流，`torch.compile` 可以把它整段融合成一个 kernel。

再往上一层还有一个性质相近的省法。prefill 期间模型会为每一个 prompt token 都算出隐状态，但每个序列里真正能用来预测下一个 token 的只有最后那一个，所以 `ParallelLMHead` 在做投影之前先切了一刀：

```python
last_indices = context.cu_seqlens_q[1:] - 1
x = x[last_indices].contiguous()
```

面对一批长 prompt，这一刀把"在好几千个位置上做词表规模的投影"变成了"只在几个位置上做"。

还有一件事贯穿始终，那就是全程不做任何 padding。prefill 会把所有序列的 token 拼成一条扁平的一维张量，用累积偏移量来标记每条序列的边界，所以一批长短极不均匀的 prompt，代价恰好等于它们长度之和。而这两套偏移量数组本身也携带着信息：`cu_seqlens_q` 只统计新 token，`cu_seqlens_k` 覆盖的却是包含缓存前缀在内的整条序列，因此 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 这个条件本身，就是"这一轮发生了前缀命中"的判据。

最后是两个值得知道的优化。decode 这一步计算量很小，时间反而被 Python 侧的算子启动开销吃掉，所以 nano-vllm 针对一组固定的 batch size（`[1, 2, 4, 8] + range(16, max_bs+1, 16)`）把整个 decode 步骤预先录制成 CUDA graph，运行时直接重放不小于当前 batch 的那份最小的 graph。这个手段只用在 decode 上，因为 prefill 的形状没法提前预判。张量并行那边则是给每个 rank 起一个 `ModelRunner` 进程，由 rank 0 通过一块 `SharedMemory` 把方法调用广播出去，加起来大约四十行，相当于手写了一个简易 RPC。

## 六、nano-vllm 没有实现的部分

投机解码、量化、多种可切换的调度策略、把 prefill 和 decode 拆到不同机器上的分离式部署，以及一个通用到能吃下 Qwen3 之外各种架构的模型加载层。这份清单基本上就是生产级 vLLM 之所以体量庞大的原因，而其中没有任何一项会改变前面讲的那两个机制。

这恰恰就是先读 nano-vllm 的理由：真正让一个推理引擎成立的想法，正好就是能塞进一千两百行里的那些，其余的都是在这些想法已经正确之后，才一层层堆上去的东西。
