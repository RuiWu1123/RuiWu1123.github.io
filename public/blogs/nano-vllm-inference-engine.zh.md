---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

把一个语言模型部署成线上服务，最先卡住你的通常不是算力，而是显存，而且往往是白白浪费掉的那一部分。

根源是 KV cache。序列每生成一个 token，都会留下一份 key 和 value 向量，只要这个序列还没结束，它们就得一直占着显存。麻烦在于你事先并不知道这个序列会生成多长：这次请求可能只要 20 个 token，也可能要 2000 个。于是最直接的实现只能按上限预留，给每个请求都分一块连续空间，大到足以装下最长的输出。真正用掉的通常只是其中很小一部分，而这种浪费会在每个并发请求上重复一遍。

第二个约束来自批处理。GPU 天生擅长大矩阵，所以你自然会想把请求攒成一批一起算。可一旦攒成了批，这批就要等最慢的成员结束才算完；在这批开跑之后才到达的请求，只能等整批彻底散场；而那些早早生成完的序列还占着位置，实际上只是在陪跑。

vLLM 最为人熟知的两项设计，正好一项对着一个问题。PagedAttention 不再要求 KV cache 是每个序列独占的一整块连续空间，而是改成从一个共享的池子里按固定大小分配 block，这和操作系统当年不再给进程分配连续物理内存，是同一个思路。Continuous batching 则让这一批的成员在每一步都可以变动，新来的请求下一轮就能加入。除此之外还有一个叫 prefix caching 的机制，它盯的是另一件事：不同请求之间往往共享一大段相同的前缀，那就让它们共用同一份前缀 block。

想通过阅读 vLLM 本身来理解这些设计，难点在于它们埋在多年积累的工程优化底下。[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用一千两百行左右的 Python 重写了同一套核心逻辑，而且它并不是个演示玩具：按项目自己给出的 benchmark，它跑到 1434 tok/s，而同样条件下的 vLLM 是 1362（Qwen3-0.6B，256 路并发，单张 RTX 4070 Laptop）。规模小到一个下午能读完，实现又足够真实，值得认真读一遍。

## 一、代码地图

`nanovllm` 这个 package 一共十九个 Python 文件，其中真正参与处理请求的只有七个，剩下的要么是常规的 transformer 组件，比如 RMSNorm、RoPE、SwiGLU 和张量并行的线性层，要么是加载权重、读配置这类杂活。点开任意一个文件都能看到真实的源码：

![interactive:nanovllm-arch](#)

这里有个文件得先提一句，否则它后面出现的时候会让人摸不着头脑。`utils/context.py` 里放的是一个模块级的全局变量：`ModelRunner` 把分页相关的元数据写进去，`Attention` 在二十多层之后再从里面把它读出来。这样做的目的，是让 block table 不必作为参数一路穿过中间每一层的 `forward()`。这种设计乍看很不讲究，但真要换成规规矩矩传参数的写法，你会发现那样反而更别扭。

## 二、一个请求会经历什么

请求进来之后，`add_request()` 先把 prompt 分词，然后包成一个 `Sequence`。一个请求的全部状态都在这个对象里：token 序列、已经算进缓存的 token 数、这一轮正在算的 token 数，以及手上持有的物理 block。包好之后放进 `waiting` 队列，然后一直干等到下一轮调度。

真正干活的是 `step()`，它会被反复调用，每一轮都按同样的顺序做五件事。首先由调度器决定这一轮跑什么，要么是 prefill，把 prompt 一口气算掉；要么是 decode，让每个在跑的序列各吐一个 token。两者在同一轮里不会混着来。接着 block manager 为选中的这些序列分配 KV-cache block，凡是前缀能对上的就直接复用现成的。然后 model runner 把这批序列压平成 GPU 张量，其中包含一个 `slot_mapping`，用来指明每个新 token 的 K/V 应该写到缓存的哪个位置。再往下就是模型前向，这一步不过是一次普普通通的 Qwen3 计算，唯一的区别在于它的 attention 层是隔着分页这层间接寻址去读写缓存的。最后 sampler 把 logits 变成 token id，`postprocess()` 负责收尾：把新 token 接到序列上，给这一轮刚填满的 block 计算哈希，并让已经遇到 EOS 或者达到长度上限的序列退场，把它们占用的 block 归还给池子。

`generate()` 就是把上面这个循环一直转下去，直到两个队列都空掉，最后把 token 解码回文本。整个引擎的骨架就是这些。这五步里有两步藏着真正值得研究的取舍，也就是这篇文章接下来要讲的全部内容。

## 三、调度器：batch 是怎么变得不再固定的

调度器每一轮需要回答的问题其实很朴素：现在手上既有一些生成到一半的请求，又有一些刚刚到达的，这一轮到底该让 GPU 去算什么？

先看最容易想到的答案。把当前排队的请求凑成一批，整批一起做完 prefill，然后整批一起一轮一轮地 decode，等这批里最后一个也生成结束，才放下一批进来。这种"人不齐就不开始、没走完就不散场"的组织方式，就是所谓的静态 batching。它的毛病开头已经说过：整批的推进速度取决于最慢的那个成员，而在这批开跑之后才到达的请求，只能在门外等着。

nano-vllm 的规则正好反过来，而且一句话就能说完：只要还有 prefill 可做就优先做 prefill，实在没有 prefill 了，才去跑 decode。

这句话表面上只是定了个优先级，实际上 continuous batching 的全部内容就在这里。prefill 排在前面，所以刚到的请求下一轮就轮得上，它的 prompt 立刻能开始算，不用等谁结束。decode 每次只把在跑的序列各推进一个 token，于是每一轮的结束都天然是一个可以换人的节点，序列想进就进，想走就走。走到这一步，"batch"这个词其实已经名存实亡了，因为根本不存在一组固定的成员，只有"这一轮恰好在跑的那些序列"。

把这条规则放到三个请求上实际跑一遍，再让同样三个请求走一遍静态 batching，差别是这样的：

![Same three requests under continuous vs. static batching](blogs/images/nanovllm-batching-timeline.svg?v=3)

两幅图都是把 nano-vllm 的调度逻辑真跑一遍之后生成的，不是照着感觉画的。为了让内存压力在这么小的例子里能显出来，这里把参数缩小了：一个 block 装 4 个 token，整个池子只有 6 个 block，也就是全部 KV cache 加起来只装得下 24 个 token；每一轮最多处理 8 个 token。真实默认值是一个 block 装 256 个 token，池子有多大则由启动时剩余显存决定。

图里有几处得对照着看。prefill 格子里的数字是这一轮处理掉的 prompt token 数，所以它可以是 6 或 7；decode 格子里恒定是 1，因为一轮就只生成一个 token。灰色带横杠的格子表示这条序列虽然在跑，但这一轮什么也没做。

灰格子这件事值得单独拿出来说，因为它是"prefill 优先"要付的直接代价。prefill 轮和 decode 轮是互斥的，一轮要么是 prefill，要么是 decode。所以每当有新请求需要 prefill，那一整轮里所有正在解码的序列都得停一拍：看 continuous 那张图的第 2、3、6 轮，A 全都在原地等着。这等于拿正在跑的序列的一点延迟，换新请求少等一会儿。这笔买卖很划算，因为 prefill 一次能吞掉整段 prompt，decode 一轮却只产出一个 token。

两张图最扎眼的差别在 C 这一行。静态那张里，C 第 3 轮就到了，却一直排到第 10 轮才轮上，整整八轮什么也没干。这期间 B 早在第 4 轮就结束，显存也空了出来，可 C 依然进不来，因为它不属于当前这一批。continuous 那张里，C 第 3 轮就被放进来，之后它的 decode 和 A 的 decode 挤在同一轮里并行推进。同样的工作量，continuous 跑完用了 12 轮，静态用了 15 轮。

C 那一行还有第二处值得看：它第 3 轮刚被 prefill，第 4 轮就被踢了出去。那一轮池子里的 6 个 block 已经分光，而 B 恰好写满了最后一个 block，得再拿一块才能继续。这时 `preempt()` 会挑一个牺牲者，也就是正在跑的序列里最晚进来的那个，把它占用的 block 全部释放，再塞回等待队列的最前面。C 刚算完的 prompt 就此作废，直到第 6 轮 B 结束腾出空间，它才被重新 prefill。

这一幕看上去像是出了故障，其实是有意为之。调度器故意多放序列进来，放的比它真正保得住显存的数量还多，赌的是大部分序列都会提前结束，而按最坏情况预留恰恰是一开始就想摆脱的做法。抢占就是让这种乐观不至于翻车的泄压阀：压力大的时候引擎会变慢，但不会崩。

代码里用了一个大多数人没写过的 Python 语法来表达这件事：

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

`while...else` 里的 `else` 只有在循环因为条件不再成立而正常退出时才会执行，一旦撞上 `break` 就会被跳过。所以这段的含义是：不停地驱逐别人，直到当前这个序列能拿到 block 为止，拿到了才调度它；如果已经没有别人可以驱逐，就只好把这个序列本身抢占掉，这一轮什么都不调度。换成不用 `while...else` 的写法，就得额外引入一个标志变量，读起来反而更绕。

另一处是分块 prefill。单个 prompt 有可能比一整轮的 token 预算还要长。调度器既没有把它拒之门外，也没有让它撑爆预算，而是把它拆到好几轮里去，但每一轮最多只拆一个。实现这条约束的，只是一个 `and`：

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

`scheduled_seqs` 只有在面对这一轮第一个候选者的时候才是空的。所以第一个序列可以超出预算、被拆开来算，而之后任何一个装不下的，都老老实实等下一轮。这也顺带解释了 `Sequence` 为什么要把 `num_cached_tokens` 和 `num_scheduled_tokens` 分开记：一个序列确实可能只算了一半，而调度器和缓存两边对"算到哪儿了"必须完全一致。

## 四、PagedAttention：让 KV cache 不必连续

回到最开始那个问题。一个序列的缓存会不断增长，而且长到多少事先不知道，你却必须在它开始之前就把空间定下来。

解决办法是从操作系统那里整个搬过来的，这个类比值得认真讲，因为它并不只是个比方。一个进程以为自己拥有一段平坦而连续的地址空间，但物理上它的内存是散落在各处的页，中间靠页表来完成翻译。PagedAttention 做的完全是同一件事：一个序列以为自己拥有一段连续的 KV cache，物理上那是散落在共享池里的一个个固定大小的 block，而每个序列各自持有一张 block table，负责把逻辑块号翻译成物理块号。

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

这样做的直接好处是，空间可以跟着序列的增长，一块一块地给。任何时刻，一个序列手上只有它真正用掉的那些 block，外加最多一个没填满的。于是浪费的空间有了上限，就是一个 block 的大小，也就是 256 个 token。无论这个序列最后生成了多长，这个上限都不会变。

还有第二个好处，操作系统那个类比里看不出来，而且可能更重要。既然 block 只是表里的一个条目，那么两个序列完全可以指向同一个 block。如果两个请求共享同一段 system prompt，它们就能共用这段前缀的 KV cache，算一次，存一份。

具体的实现方式是内容寻址。每个填满的 block 都会算一个哈希，关键在于：这个哈希串上了它前一个 block 的哈希：

```python
h = self.compute_hash(token_ids, h)
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

有了链式哈希，这种复用才是真正正确的，而不只是"多半没错"。一个 block 的身份并不是"这 256 个 token"，而是"这 256 个 token，出现在这个位置上，并且它前面正好接着这一段历史"。所以两个 block 的哈希对得上，就意味着这两个序列从第 0 个 token 起就完全一致，而这正是它们的 K/V 确实相同的充分条件。循环在第一次没对上的时候就 break，因为前缀匹配本来就是从头开始的一段连续区间。而即便哈希看起来命中了，代码还会再比对一次存下来的 `token_ids`，这样哈希碰撞的后果只是退化成一次未命中，而不会把别人的缓存悄悄端上来。

block 什么时候释放，看的是引用计数，而不是谁最早申请了它。完整的生命周期是这样：

![One block pool, four moments](blogs/images/nanovllm-block-lifecycle.svg?v=2)

留意最后一帧：A 已经结束了，但 block 0 和 1 并没有被回收，因为 B 还在用它们。这段共享的前缀比最早算出它的那个序列活得更久。这同时也意味着缓存能够挺过抢占，一个被踢出去的序列在重新 prefill 时，只要它原来的 block 还留在池子里，就能直接命中。抢占的代价之所以没有看上去那么大，原因就在这儿。

有个容易让人绊一下的细节：block 是在算完之后才被哈希的，而不是在分配的时候。`hash_blocks()` 跑在 `postprocess()` 里，而且只处理这一轮里刚刚填满的那些 block。所以一个序列末尾那个没填满的 block 永远进不了缓存，因为它的内容还没定下来，也就还谈不上有一个稳定的身份可以拿来当键。

decode 期间的扩容几乎不花什么代价，这行判断也写得相当取巧：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`len(seq) % block_size == 1` 只有在某个 token 恰好溢出到新 block 的那一刻才成立。256 个 token 里只有那一个会撞上这种情况，这时它读作"空闲 block 不少于 1"；剩下 255 次，它读作"空闲 block 不少于 0"，恒真。它利用了 Python 里 `bool` 本身就是 `int` 这一点，而且它干的是正经活，一个序列会不会被抢占，判断依据就是这一行。

到这里还剩两个问题没解决，而它们才是让分页在 GPU 上真正跑得起来的关键。

先说写入，它是一次 scatter。每个新 token 的 K/V 必须落到 block table 指定的物理槽位上，而这些槽位彼此并不相邻。`ModelRunner` 会提前把这些目的地算好放进 `slot_mapping` 张量，然后由一个 Triton kernel 来完成搬运，每个 token 起一个 program，做的事情就是加载 K 和 V、读出目的地、写进去。整个 kernel 只有一处防护判断，也就是 `if slot == -1: return`，这一句是专为 CUDA graph 重放留的：捕获下来的 graph 跑在固定 batch size 上，用不到的槽位会标成 `-1` 跳过。

读取更麻烦一些，而 nano-vllm 的应对方式是干脆不自己读。它把 `block_table` 直接交给 attention kernel，让 kernel 自己去把散落的 block 收集起来。`flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 都原生支持传入 block table，所以散落的物理布局从头到尾都不用拼回成一个连续张量。整个 attention 层里称得上分支的地方就这一处：

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
```

前缀命中的时候，刚算出来的 `k` 和 `v` 只覆盖了新增的那部分 token，可 attention 必须连缓存里的前缀一起看到，于是干脆把它们整个换成完整的 cache，之后由 kernel 通过 block table 把所有内容读回来。这两行分支，就是分页存储和普通 attention 之间的全部接缝。

这也正是最值得记住的一点：上面这一整套机制，没有任何一处改动了数学本身。attention 依然是缩放点积之后做 softmax。分页真正改变的只是 K 和 V 物理上存放在哪里，并且把这层间接寻址交给了本来就按块遍历的 kernel。至于 kernel 内部到底在做什么，tiling 是怎么回事，为什么绝不把完整的 attention 矩阵摆出来，那是 [GPU field guide](#/blog?id=gpu-guide-for-dl) 那篇的主题。

## 五、其余几处值得一提的设计

采样部分总共九行，其中一行是个挺漂亮的技巧：

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

把每个概率除以一个独立的 `Exponential(1)` 采样值，再取 argmax，这个操作严格等价于从分类分布里采样。它其实就是 Gumbel-max trick，因为除以一个指数分布的随机变量，在对数空间里就相当于减去一个 Gumbel 变量。那为什么不直接调 `torch.multinomial`？因为这样写全程只有逐元素运算和一次 argmax，没有依赖数据的控制流，`torch.compile` 可以把它整段融合成一个 kernel。

再往上一层还有一处类似的省事办法。prefill 期间模型会为每一个 prompt token 都算出隐状态，但每个序列里真正能用来预测下一个 token 的只有最后那一个，所以 `ParallelLMHead` 在做投影之前先切了一刀：

```python
last_indices = context.cu_seqlens_q[1:] - 1
x = x[last_indices].contiguous()
```

面对一批长 prompt，这一刀把"在好几千个位置上做词表规模的投影"变成了"只在几个位置上做"。

还有一件事贯穿始终，那就是全程不做任何 padding。prefill 会把所有序列的 token 拼成一条扁平的一维张量，用累积偏移量来标记每条序列的边界，所以一批长短极不均匀的 prompt，代价恰好等于它们长度之和。而 `cu_seqlens_q` 和 `cu_seqlens_k` 这两套数组本身也说明了问题：`cu_seqlens_q` 只统计新 token，`cu_seqlens_k` 覆盖的却是包含缓存前缀在内的整条序列，因此 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 这个条件本身，就是"这一轮发生了前缀命中"的判据。

最后是两个值得知道的优化。decode 这一步计算量很小，时间反而被 Python 侧的算子启动开销吃掉，所以 nano-vllm 针对一组固定的 batch size（`[1, 2, 4, 8] + range(16, max_bs+1, 16)`）把整个 decode 步骤预先录制成 CUDA graph，运行时直接重放大小刚好够用的那份 graph。这个手段只用在 decode 上，因为 prefill 的形状没法提前预判。张量并行那边则是给每个 rank 起一个 `ModelRunner` 进程，由 rank 0 通过一块 `SharedMemory` 把方法调用广播出去，加起来大约四十行，相当于手写了一个简易 RPC。

## 六、nano-vllm 没有实现的部分

投机解码、量化、多种可切换的调度策略、把 prefill 和 decode 拆到不同机器上的分离式部署，以及一个通用到能吃下 Qwen3 之外各种架构的模型加载层。生产级 vLLM 之所以体量庞大，基本就是因为这些；而其中没有任何一项会动到前面讲的那两个机制。

这恰恰就是先读 nano-vllm 的理由：真正让一个推理引擎成立的想法，正好就是能塞进一千两百行里的那些，剩下的都是等这些想法立住之后，才一层层堆上去的。
