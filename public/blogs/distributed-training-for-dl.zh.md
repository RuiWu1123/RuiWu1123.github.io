---
title: "Distributed Training from First Principles: A Field Guide to Parallelism Strategies"
date: "2026/7/19"
---

[这个系列的第一篇文章](#/blog?id=gpu-field-guide-for-dl) 建立了单张 GPU 的模型：warp、SM、内存层级、roofline 模型。[第二篇](#/blog?id=triton-for-dl) 讲的是怎么为这一张 GPU 写自己的 kernel。而这一篇要讲的问题，是从"一张 GPU 就够了"这个假设不再成立的那一刻开始的：你的模型、或者你的 batch、或者你的序列长度，已经装不进一张卡了，你必须决定怎么把工作拆到多张卡上去。这个决定有个名字——**并行策略**——而在 DP、DDP、ZeRO、TP、PP、SP、CP 这一堆缩写背后，真正不同的想法其实出奇地少。这篇文章会按照这个领域实际发现它们的顺序，从每一个策略具体要解决的问题出发，把它们一个个搭建起来，最后讲清楚现代训练是怎么把四五个这样的想法同时叠在一起，却不会互相打架的。

这篇文章大量参考了猛猿的《图解大模型训练》系列，那依然是我见过对这部分内容讲得最清楚的中文资料之一；这里的图和讲解方式是我自己在读这个系列以及它背后的原始论文（GPipe、PipeDream、ZeRO、Megatron-LM、DeepSpeed Ulysses、Ring Attention）时重新组织出来的。

## 1. 装不下的两种东西

在给任何策略起名字之前，有必要先精确地说清楚"装不下"这三个字到底指的是什么，因为这一个说法背后其实藏着两个结构完全不同的问题，需要用不同的办法解决。

第一种情况是：**你的数据集，或者你想要的 batch size，太大了，串行跑一遍要等太久**。模型本身在一张 GPU 上跑得很舒服，你只是样本数量比你愿意等一张 GPU 跑完的量要多。自然的解法是：给每张 GPU 一份完整的模型副本，和一份不同的数据切片，然后想办法让所有副本保持一致——这就是 **data parallelism（数据并行）**，也是这篇文章里最古老、最简单的想法。

第二种情况是：**模型本身——它的参数、它的优化器状态，或者它在前向传播中产生的激活值——根本没法塞进一张 GPU 的 HBM**，不管你的 batch 有多小都一样。这时候"每张 GPU 一份完整副本"已经不再是一个选项了：根本没有多余的空间去"复制"。解法必须是把模型本身拆到多张 GPU 上，而这个拆分有好几种结构上完全不同的做法，每一种在通信开销、显存节省和实现复杂度上的取舍都不一样：**pipeline parallelism（流水线并行，按层拆）**、**tensor parallelism（张量并行，拆一层内部的运算）**，以及 **sequence/context parallelism（序列/上下文并行，沿序列维度拆——用在哪怕一层的激活值，对一条很长的序列来说都装不下的时候）**。

这篇文章里的每一个策略，都是在回答这两个问题里的一个，而最后一节要讲的，是真实的训练任务往往需要**同时**回答这两个问题，并且按照第一篇文章第 7 节讲过的物理拓扑结构，把这些策略一层套一层地嵌在一起：通信频繁、体量大的部分留在节点内的 NVLink 域里；通信少、允许慢一点的部分才被允许跨到网络上去。

## 2. 数据并行：从一个瓶颈到一个环

最朴素版本的数据并行简单到几乎不需要专门起个名字：每张 GPU 持有一份完整的模型副本，在自己的 batch 切片上跑前向和反向，然后所有算出来的梯度需要被平均，才能做优化器更新（否则各个副本会越跑越不一致）。最早的实现是通过 **parameter server（参数服务器）** 来做这件事：一个专门指定的节点（或进程），所有 worker 把自己的梯度发给它，它做平均之后再把结果发回去。

![Naive DP vs Ring-AllReduce](blogs/images/dp-ddp-ring-allreduce.svg?v=1)
^Parameter server 的入站带宽是所有 worker 共享的，GPU 越多情况越糟。Ring-AllReduce 把同样的计算重新组织了一遍，使得任何一个节点都不会成为瓶颈。

Parameter server 这个设计有一个明显的缺陷：server 那根网线是所有 worker 共享的，所以它的流量会随着 GPU 数量线性增长，最终会变成整个集群都要等的瓶颈。**DDP**（PyTorch 的 `DistributedDataParallel`，也是今天几乎所有人默认使用的方案）用 **Ring-AllReduce** 取代了这套机制，这个算法有一个相当优雅的性质：每个参与者永远只和逻辑环上的两个邻居通信，而且每张 GPU 需要搬运的数据总量，**和环里有多少张 GPU 无关**。

这个算法分两个阶段跑。在 **scatter-reduce** 阶段，每张 GPU 的梯度缓冲区会被切成 N 份（N = GPU 数量），经过 N−1 步，每张 GPU 一边把一份数据传给邻居，一边从另一个邻居那里接收并累加进来的数据——N−1 步之后，每张 GPU 手里都有一份"在所有 GPU 上完全求和过"的数据，但只有这一份。在 **all-gather** 阶段，同样的环结构被用来把这 N 份"已经求和完毕"的数据传播开，让每张 GPU 最终都拿到完整的、已经求和好的梯度。每个阶段每张 GPU 搬运的数据量是总梯度大小的 (N−1)/N，两个阶段加起来一共是 **2(N−1)/N × 梯度大小**——随着 N 增长，这个值会收敛到梯度大小的 2 倍，然后就不再增长了。这正是 DDP 能够扩展到很多 GPU、而 parameter server 在结构上做不到这一点的数学原因。

![interactive:ring-allreduce](#)

## 3. ZeRO：DDP 里还剩下的冗余

Ring-AllReduce 解决了朴素 DP 的**通信**瓶颈，但前面描述的 DDP 依然有一个显存问题：N 张 GPU 里的每一张，都持有更新模型所需要的**全部**东西的一份完整冗余副本——不只是参数本身，还有优化器自己的那一堆记账数据。对于用 mixed-precision 训练的 Adam 来说，这些记账数据比模型本身还要大。以 Φ 表示参数量，每个参数需要：一份 fp16 的参数副本（2 字节）用于快速的前向/反向运算、一份 fp16 梯度（2 字节）——以及这部分经常被低估——一份 fp32 的参数主副本，加上 fp32 的 Adam 动量和方差项（4+4+4=12 字节），用来保证优化器更新在数值上是稳定的。这样算下来，plain DDP 会在**每一张 GPU 上**完整复制 **2Φ + 2Φ + 12Φ = 16Φ 字节** 的"model state"——一个 75 亿参数的模型光是这一项就需要 120GB，比单张 GPU 的 HBM 还大，而这时候连一个激活值都还没存呢。

**ZeRO**（Zero Redundancy Optimizer，来自微软 DeepSpeed）问了一个很自然的问题：既然每张 GPU 反正都要持有这 16Φ 字节的一份一模一样的副本，那为什么不让每张 GPU 只持有其中 1/N 的**分片**，然后在真正需要的时候，把某一步实际需要的那一小块重新拼出来呢？ZeRO 把这个想法拆成了三个逐渐激进的 stage：

**ZeRO-1** 只把优化器状态（也是最大的一块，12Φ）分片到 N 张 GPU 上，参数和梯度依然完整复制。每张 GPU 的显存变成 **2Φ + 2Φ + 12Φ/N**——在合理的 N 下已经接近 4 倍的显存缩减。

**ZeRO-2** 进一步把梯度也分片，因为一旦优化器状态已经被分片了，每张 GPU 其实只需要它实际负责更新的那部分参数所对应的那一小块梯度。每张 GPU 的显存变成 **2Φ + (2Φ + 12Φ)/N = 2Φ + 14Φ/N**。

**ZeRO-3** 更进一步，把参数本身也分片了。这是结构上变化最大的一步：GPU 不再持有完整的参数张量，而是要在每一层前向或反向计算**之前**，用 **all-gather** 把自己需要的那个具体分片拼回来，用完之后再释放掉。每张 GPU 的显存降到 **16Φ/N**——几乎是和 GPU 数量成正比的缩减——代价是每一层、每一步都要多付出 ZeRO-1/2 不需要的 all-gather 通信。

![ZeRO memory breakdown](blogs/images/zero-memory-breakdown.svg?v=1)
^在 N=64 时，ZeRO-1 已经能把显存降到 DDP 的大约四分之一；ZeRO-3 能降到 16Φ/N——真正和你的 GPU 数量成正比，而不只是一个固定的倍数。

这一切都不是免费的：ZeRO-3 多出来的通信意味着它特别适合"显存不够、但带宽有富余（比如一整个快速的 NVLink 域）"这种情况；而当瓶颈主要是优化器状态本身时，ZeRO-1 往往是更务实的默认选择。下面这个面板可以让你选一个模型大小、一个 GPU 数量和一个 stage，直接看到每张 GPU 上真实的显存数字——包括它什么时候会超出单张 GPU 的 HBM。

![interactive:zero-memory](#)

## 4. 流水线并行：按层拆模型

ZeRO 分片的是"同一个模型的冗余副本"；如果一个模型的各层，哪怕不算任何冗余，单独排在一起就已经装不进一张 GPU，ZeRO 是帮不上忙的。直接的解法是 **pipeline parallelism（流水线并行）**：把前几层放在 GPU 0 上，接下来几层放在 GPU 1 上，以此类推，这样一次前向传播就变成了 GPU 之间的一场接力赛，而不是一张 GPU 独自扛下所有事情。

如果朴素地实现，这会很糟糕：GPU 1 什么都做不了，必须等 GPU 0 把整个 batch 的前向传播全部跑完并交接过来才行；反向传播的情况反过来也是一样——在任意一个时刻，你的 P 张 GPU 里只有一张在真正干活，剩下 P−1 张都在空闲。这段空闲时间叫做 **pipeline bubble**，朴素版本几乎浪费掉了你的大部分硬件。

**GPipe**（Google，2019）用 **microbatching（微批次）** 解决了这个问题：把一个 batch 切成 M 个更小的 microbatch，一个接一个地喂进流水线。当 GPU 0 在跑 microbatch 2 的前向时，GPU 1 已经可以在跑 microbatch 1 的前向了——流水线被填满了，一旦填满，每个 stage 就能同时保持繁忙。GPipe 的调度是先把**所有** microbatch 的前向跑完，再跑**所有**的反向，这样推理起来很简单，但意味着每个 microbatch 的激活值都必须一直留在显存里，直到它对应的反向传播真正发生为止。

**PipeDream**，以及 Megatron-LM 在实践中使用的 **1F1B**（"one-forward-one-backward"）调度，会更激进地交错执行：一旦某个 microbatch 的激活值不再是维持流水线运转所必需的，它对应的反向传播就会立刻被调度执行，而不是等所有其他 microbatch 的前向都跑完再说。这**不会**减少总的 bubble 时间——出于后面会讲到的一个微妙但重要的原因，它其实是完全相同的比例——但它会大幅减少任意一个 stage 在同一时刻需要在显存里保留多少个 microbatch 的激活值，而这在实践中往往才是更紧的约束。

![Three pipeline schedules](blogs/images/pipeline-bubble-schedules.svg?v=1)
^朴素的模型并行在任意时刻只有 P 张 GPU 里的 1 张在忙。GPipe 和 1F1B 最终达到的总 bubble 比例完全一样——两者的区别在于峰值激活值显存，而不是墙钟时间。

Bubble 比例有一个很干净的封闭形式：P 个 pipeline stage、M 个 microbatch 时，每张 GPU 空闲的时间比例是 **(P−1) / (P−1+M)**。这一个公式几乎解释了你会读到的所有关于流水线并行的实践建议：microbatch 越多，bubble 永远越小（当 M→∞ 时，bubble 比例趋向于 0）；而更深的 pipeline（更大的 P，通常是模型更大时才需要）需要成比例更多的 microbatch 才能把 bubble 压小——这正是为什么流水线并行通常要搭配一个足够大的 global batch size，而在 P 很深、batch 又很小的时候会很吃亏。

![interactive:pipeline-bubble](#)

## 5. 张量并行：拆开一层内部的运算

流水线并行拆的是"哪些层放在哪里"；它完全没有回答，当单独一层的权重矩阵本身就太大装不下，或者你想要比"整层"更细粒度的并行度来减少 pipeline bubble 的影响时，该怎么办。**Tensor parallelism（张量并行）**（因 **Megatron-LM** 而普及）把一层**内部**的矩阵乘法拆到多张 GPU 上。

最干净的例子是 transformer 的 MLP block，它是两个线性层中间夹一个非线性：`Y = B(GeLU(A(X)))`。Megatron 把矩阵 A **按列**切分到各张 GPU 上——GPU 0 拿到 A 的第 0..k 列，GPU 1 拿剩下的——这样每张 GPU 都能独立算出自己那一份 `GeLU(A(X))`，**完全不需要通信**（GeLU 是逐元素的，它不在乎每张 GPU 只拿到中间张量的一部分）。矩阵 B 则**按行**切分，这个切法是特意选的，使得每张 GPU 算出来的部分结果，在各 GPU 间求和之后，能重新拼出正确的最终输出——而这恰好只需要在 block 的最末尾做**一次 all-reduce**。

![Megatron tensor parallelism](blogs/images/megatron-tensor-parallel.svg?v=1)
^先列并行、后行并行：中间的 GeLU 完全不需要通信，整个 block 只需要付出一次 all-reduce 的代价。

这种"先列后行"的模式（Megatron 把"前向是恒等、反向是 all-reduce"的算子叫 `f`，把"前向是 all-reduce、反向是恒等"的算子叫 `g`）用同样的方式应用在 attention 上，把整个 head 拆到不同 GPU 上，而不是在一个 head 内部再拆。一个完整的 transformer 层——一个 attention block、一个 MLP block——恰好需要 **前向 2 次 all-reduce、反向 2 次 all-reduce**，不管这个 tensor-parallel 组里有多少张 GPU 都是如此。

问题在于，这些通信发生在**每一层、每一次前向和反向**——比 DP 偶尔同步一次梯度、或者 PP 偶尔交接一次激活值，频繁了一个数量级。这正是为什么第一篇文章第 7 节的内容在这里格外重要：张量并行需要一整个节点内部快速、全带宽的 NVLink 域，一旦把它跑到更慢的跨节点网络上，就是"线性扩展"曲线一旦跨出单节点就不再线性的、最常见也最具体的原因之一。

## 6. 序列并行：填补张量并行留下的空隙

前面描述的张量并行有一个不太起眼的低效之处：LayerNorm、dropout、残差相加这些操作，并不像矩阵乘法那样能沿着 hidden 维度干净地切分——它们需要每个 token **完整**的 hidden 向量才能算对。Megatron 最早的 TP 实现处理这个问题的办法很简单：直接在 tensor-parallel 组里的每张 GPU 上**完整复制**这些操作的激活值，这意味着这部分区域的激活值显存完全没有从张量并行中得到任何好处。

**Sequence parallelism（Megatron SP）** 用一个观察填补了这个空隙：LayerNorm、dropout、残差相加都是**逐 token** 独立运算的，所以与其沿着 hidden 维度切分（这是 TP 在做的，而这些操作又用不上），你完全可以改成沿着**序列**维度切分——每张 GPU 只拥有一部分 token 对应的这些激活值，完全没有冗余。

![Megatron sequence parallelism](blogs/images/megatron-sequence-parallel.svg?v=1)
^SP 区域（LayerNorm、dropout、残差）按序列位置切分；TP 区域（attention、MLP）按 hidden 维度切分。每个接缝处用一次 all-gather 和一次 reduce-scatter，在两种布局之间切换。

SP 区域和 TP 区域之间的接缝，用的是 **all-gather**（在需要完整序列的 TP block 之前，把序列重新拼起来）和 **reduce-scatter**（在之后把输出重新按序列分片）——而不是 TP 的 all-reduce——这里有必要说清楚一点：这是一次**显存**优化，而不是**带宽**优化：一次 all-gather 加一次 reduce-scatter，搬运的总字节数和一次 all-reduce 完全一样。你消除了一个真实存在的显存冗余（被复制的 LayerNorm/dropout 激活值），而为此付出的通信账单，本质上和你原本为张量并行付的那笔账是一样的。

## 7. TP 和 SP 都解决不了的问题：单独一条序列就装不下

把 DP、ZeRO、PP、TP、SP 组合在一起，通常被称为 **3D parallelism**（算上 SP 的话是 4D），在很长一段时间里，这被当作"如何训练一个单张 GPU 装不下的模型"的完整答案。但这里有一种失效模式，是上面这些都解决不了的：当序列长度足够长时，**单次 attention 运算的激活值，哪怕只是一张 GPU 负责的那部分 hidden 维度，对一条序列来说**，也会装不下——这和模型参数量完全无关。Attention 的激活值显存大致按序列长度的**平方**增长（也就是 attention 矩阵本身，在任何 FlashAttention 式的技巧介入之前），而即便用上了 FlashAttention 那种线性显存的技巧，对于足够长的序列，K/V cache 和 running statistics 依然最终会超出单张 GPU 的 HBM。

张量并行在这里帮不上忙，因为它是沿着 hidden 维度切分的，不是沿着序列维度——TP 组里的每张 GPU 依然需要看到整条序列才能算自己那一份。真正需要的是一种专门针对 attention 计算、把序列维度本身切到多张 GPU 上的办法——这是和前面讲的完全不同的一个维度，有时候也被称为第二种意义上的"序列并行"（这里称为 context parallelism，以便和第 6 节 Megatron SP 区分开），也是这篇文章接下来要讲的内容。

## 8. DeepSpeed Ulysses：用 All-to-All 把切分方式转置一下

最直接的想法是：如果序列被切到了多张 GPU 上（每张 GPU 持有全部 attention head，但只有序列的一部分），你没法在本地算 attention，因为每个 query 都需要看到**整条**序列上的所有 key 和 value，而不只是本地这一小段。**DeepSpeed Ulysses** 的技巧，是用一次 **All-to-All** 集合通信，把这个切分方式转置一下：All-to-All 之后，每张 GPU 拿到的是**整条**序列，但只有一部分 attention **head**。由于不同的 head 是完全独立的计算，每张 GPU 现在可以为自己负责的那部分 head 算出完整、正确的 attention，**完全不需要再通信**——attention 算完之后，第二次 All-to-All 会把布局换回去，因为这一层剩下的部分还是期待序列并行的布局。

![DeepSpeed Ulysses All-to-All](blogs/images/deepspeed-ulysses-alltoall.svg?v=1)
^一次 All-to-All 把"全部 head、部分序列"转置成"部分 head、完整序列"——这正是本地、无需通信的 attention 计算所需要的。

这是一个优雅、通信开销很低的方案，但有一个结构性的限制值得明说：并行度的上限是 attention head 的数量（用了 grouped-query attention 的话，是 KV head 组的数量）——你没法把它切到比你手头 head 数量还多的 GPU 上，这给固定模型架构下 Ulysses 单独能扩展的程度设了一个天花板。

## 9. Ring Attention：分布式的 FlashAttention

**Ring Attention** 走了一条完全不同的路，而且没有任何 head 数量上的天花板。回忆一下 [Triton 那篇文章里讲的 FlashAttention 走读](#/blog?id=triton-for-dl)：单张 GPU 计算 attention 的方式，是让一个 query tile 常驻，把 key/value tile 一块一块地流式送进来，同时维护一个 running max 和 running sum（也就是"online softmax"），这样完整的 attention 矩阵就永远不需要被物化出来。Ring Attention 拿的正是这同一个循环，**把它分布到多张 GPU 上，而不是在一张 GPU 内部循环**：每张 GPU 让自己那一份 Q 保持固定，所有 GPU 排成一个逻辑环，K/V 分片沿着这个环传递。每一步，一张 GPU 对当前刚到达的那个 K/V 分片计算本地 attention，更新自己的 online-softmax 统计量——关键的是，这个计算是在**下一个分片已经在传输路上**的同时发生的，所以只要每一步的算力足够掩盖传输时间，这个环在墙钟时间上根本不需要额外付出任何通信代价。

![Ring Attention](blogs/images/ring-attention.svg?v=1)
^每张 GPU 上 Q 保持固定；K/V 沿着环旋转。和单张 GPU 的 FlashAttention 循环里同样的 online-softmax 累积方式，只是从"跨越一个 for 循环"变成了"跨越多个设备"。

Ring Attention 没有 Ulysses 那种 head 数量上限——你可以往环里加任意多张 GPU 来切分序列——但它依赖计算和通信的重叠在实践中真的能生效，这需要每个环步骤里有足够多的 FLOPs 相对于互联带宽来说，这一点直接呼应了这个系列第一篇文章里的 roofline 问题。

这里还有一个更具体的问题值得单独说一下：在**因果掩码（causal mask）**下（自回归语言模型的标准情形），序列早期位置的 query 只需要关注很少几个 key，而靠后位置的 query 几乎要关注整条序列。如果朴素地把序列的连续片段分给各张 GPU，持有最早那段的 GPU 每个环步骤要做的工作量，会比持有最后那段的 GPU 少得多——这是一个真实存在的负载不均衡问题，不只是理论上的。

## 10. Megatron Context Parallel：把环上的负载压平

**Megatron 的 Context Parallelism** 拿 Ring Attention 的机制，用一种 **zigzag（之字形）** 的分片分配方式，恰好解决了这个负载不均衡问题：不再是 GPU 0 拿到第一个连续片段、GPU 3 拿到最后一个，而是给每张 GPU 分配早期和后期片段的一个**混合**——多到足以让每张 GPU 在每个环步骤里，无论位置如何，都承担大致相同的因果掩码工作量。

![Ring Attention and load balancing](blogs/images/ring-attention.svg?v=1)
^朴素的连续分片方式，会在因果掩码下让持有早期位置的 GPU 工作量不足；zigzag 分配方式让每张 GPU 都拿到早期和后期位置的均衡组合。

这是一次真正实用的工程改进，而不是一个全新的算法想法——底层的通信模式依然是 Ring Attention 那种旋转的 K/V——但它是"一个 context-parallel 实现能不能干净地扩展到生产环境的因果语言模型训练"和"悄悄浪费掉环两端一大批 GPU 算力"之间的区别。

## 11. 合在一起：3D 变成 4D、5D

上面这些策略互不排斥，任何足够大的模型在生产环境的训练，都会把好几个策略嵌套在一起用——而嵌套的顺序，直接由第一篇文章第 7 节讲过的互联拓扑决定。一条经验法则是：**哪个维度通信最频繁、体量最大，就放在最快、最紧密的链路上；哪个维度通信最少，就可以放在最慢、最远的链路上。**

在实践中，这意味着：**tensor parallelism（以及和它放在一起的 context/sequence parallelism）放在最内层**，局限在单个节点内部的 NVLink/NVSwitch 域里，因为它在每一层、每一次前向和反向都要通信。**Pipeline parallelism 放在中间一层**，跨越数量不多的几个节点，因为它只需要在少数几个 stage 边界上交接激活值。**Data parallelism 放在最外层**，可以跨越任意多的节点（甚至跨数据中心），因为对整个模型梯度的 all-reduce 每一步只发生一次，可以容忍最慢的那种链路。

![3D/4D/5D parallelism combined](blogs/images/dist-training-3d-parallelism.svg?v=1)
^最内层 = 最快、最频繁的链路（TP、CP/SP，NVLink）。中间层 = 中等频率（PP，少数几个节点）。最外层 = 最不频繁，可以容忍任何链路（DP）。

有一个很具体的方式可以看出这个顺序为什么不是随便定的：如果你不小心把张量并行跑到了**跨节点**而不是节点内部，你就是在让整个集群里最慢的那条链路，去承担整个训练任务里频率最高、体量最大的通信——这正是一个"理应线性扩展"的训练任务，一旦跨出单节点吞吐量就断崖式下跌的最常见、也最具体的原因之一。

## 12. 选择并行策略的一份简明指南

**你的问题是数据太多，还是模型太大？** 第 1 节的问题，也是分岔路口的第一步。如果模型本身在一张 GPU 上跑得很舒服，你只是想要更高的吞吐，plain DDP（Ring-AllReduce）通常就够了——只有在它不够用的时候，才需要动用这篇文章后面的内容。

**你的瓶颈是冗余的 model state，而不是模型本身的规模吗？** 如果 DDP 复制的那 16Φ 字节的 model state，就是把你推过单张 GPU HBM 上限的原因，ZeRO-1 或 ZeRO-2 通常是第一个该尝试的东西——它几乎是一个即插即用的改动（DeepSpeed、FSDP），不需要碰你的模型代码，这一点和 TP、PP 都不一样。

**哪怕用了 ZeRO-3，模型的一个分片依然装不下吗？** 这时候你需要真正意义上的模型并行——如果你能接受用一点 bubble 换来实现上的简单，用 pipeline parallelism；如果你有一整个快速的 NVLink 域，想要更细粒度、bubble 代价更低的并行，用 tensor parallelism；而在实践中，对非常大的模型，往往是两者一起用。

**你的 pipeline 深度，配的 microbatch 数量够吗？** 第 4 节的公式，(P−1)/(P−1+M)——一个"扩展不上去"的 pipeline，很多时候只是相对于它的 stage 数，microbatch 数量太少了，而不是有什么更深层的问题。

**你的 tensor-parallel（或 context-parallel）组，是不是跨到了一条慢链路上？** 第 5 节和第 11 节的问题，也是多节点扩展性最常见的隐形杀手。如果吞吐量恰好在节点边界处断崖式下跌，这是第一个该排查的地方。

**你的序列长度本身就是瓶颈吗，和模型大小无关？** 如果卡住你的是 attention 本身的激活值显存装不下，你需要第 7–10 节里的工具：如果你的 head 数量能舒服地支撑你想要的并行度，用 Ulysses；如果你需要扩展到 head 数量允许范围之外，或者需要对因果掩码做负载均衡感知，用 Ring Attention（配上 Megatron 的 zigzag 均衡）。

**你是真的测量过时间花在哪里了，还是在猜？** 这个系列前两篇文章里同样的原则，在这里依然完全适用，不需要任何修改：先 profile，再重构。一个"慢"的分布式任务，出问题的地方是 dataloader、是某个 NCCL 环境变量设置错了、还是不小心让某个 collective 跨了节点，和"从根本上选错了并行策略"这几种可能性其实同样常见——而分辨它们的工具（Nsight Systems、PyTorch 自带的分布式 profiler，以及这篇文章交互面板里那些朴素的算术）通常都比推倒重写来得快。

*这篇文章里的公式和图（16Φ、bubble 比例、Ring-AllReduce 的 2(N−1)/N）都是 ZeRO、GPipe、PipeDream、Megatron-LM 这些论文里标准、经得住时间的结论，在它们各自的设定（mixed-precision Adam / dense transformer）下是准确的；真实系统还会加上一堆工程细节（梯度累积、activation checkpointing、通信与计算重叠、各并行维度混用不同的并行度），这些会让具体数字发生变化，但不会改变每种策略真正撬动的是哪个杠杆。和前两篇一样，这里真正重要的概念——冗余到底藏在哪里、"频繁、大体量"的通信到底需要网络提供什么、以及嵌套顺序为什么不是随便定的——远比任何一个框架用来表达它们的具体 API 更经得住时间。*
