---
title: "Triton from First Principles: Writing Fast GPU Kernels Without Writing CUDA"
date: "2026/7/9"
---

[上一篇文章](#/blog?id=gpu-field-guide-for-dl) 建立了关于 GPU 的心智模型：warp 和 SM、内存层级、roofline 模型，以及矩阵乘法是如何被 tiling 到 shared memory 上的。这些内容完全不需要写一行 CUDA。而这一篇要讲的，是你 **真的** 想自己写点什么的时刻——一个 fused 的 elementwise 操作、一个自定义的 attention 变体、一个 PyTorch 没有提供的 normalization 层——但又不想为此去学 CUDA C++、手动 thread indexing，以及手工规避 shared memory 的 bank conflict。

这正是 Triton 填补的空白。它是一门嵌入 Python 的语言，最早来自 OpenAI，现在作为 PyTorch 生态的一部分被维护，让你可以在"一块块数据"而不是"一个个线程"的层面上写 GPU kernel，并把它们编译成和手写 CUDA 目标完全相同的 PTX。而且它其实已经悄悄运行在你用过的大量代码之下：`torch.compile` 的 Inductor 后端会为它生成的大多数 fused 算子自动生成 Triton kernel，而目前发布的大多数 FlashAttention 系列实现都存在一个 Triton 版本。这篇文章假设你已经具备 GPU 那篇里的心智模型——warp、SM、shared memory、arithmetic intensity、tiling——并直接在此基础上，解释 Triton 到底对这些硬件做了什么、以及它在哪里会力所不及。

## 1. Triton 到底改变了什么

写一个 CUDA kernel，意味着写的代码是 **每个线程各跑一遍** 的，并且要显式地在 GPU 那篇提到的每一层硬件上进行推理：你是哪个线程（`threadIdx.x`、`blockIdx.x`）、你这个线程具体该碰哪块数据、什么时候要手动把数据搬到 shared memory 里暂存、以及如何手工避免 bank conflict 和 warp divergence。这是一个表达能力极强的模型——硬件几乎所有的能力都能从这个模型里触达——而这种表达能力，恰恰也是它写起来慢、又容易在细节上悄悄写错的原因。

Triton 的核心想法，是把编程的基本单位往上挪一层：你不再写"线程 `i` 该做什么"，而是写"这个 program instance 该对 **一整块数据** 做什么"。一个 Triton kernel 操作的是向量（Triton 把这门语言里类似张量的值直接称为"tensor"，说实话这个命名有点容易混淆），而不是标量。你把整块 tile 加载进来，对整块 tile 做运算，再把整块 tile 存回去——而把这个 tile 级别的描述，落实到具体的线程、warp，以及（在编译器能做到的地方）shared memory 上，完全是编译器的责任，不需要你手写任何一部分。

![CUDA vs Triton programming model](blogs/images/cuda-vs-triton-model.svg?v=1)
^思考的基本单位从"一个线程、一个标量"变成了"一个 program、一整块 tile"。这条线以下的东西——线程索引、warp 分配、相当一部分 shared memory 的暂存——都变成了编译器的问题，而不再是你的问题。

这是一次实打实的"用通用性换生产力"的交易，值得诚实地说清楚这笔交易的方向。你放弃的是 CUDA 的一部分完整表达能力——任意的 per-thread 控制流、手工摆放的 shared memory 布局，以及最新一代架构刚发布时那些还没被普遍支持的硬件特性。换回来的是：kernel 代码量大幅缩短、编译器会替你在 block size 和 warp 数量上自动调优、并且你可以把整个 kernel 当作 Python 来写、读、调试。对于一个深度学习研究者真正需要写的那类"自定义 kernel"——fused 的 elementwise 链条、自定义的 normalization 或 activation 函数、attention 变体、量化 / 反量化 kernel——这笔交易绝大多数时候都非常划算。第 7 节会回过头来讲清楚，这笔交易在哪里会变得不划算。

## 2. 编程模型：program、offsets 与 mask

每一个 Triton kernel 都是一个被 `@triton.jit` 装饰的 Python 函数，它不是只被启动一次，而是被并行启动很多次——每次称为一个 **program instance**——这和 CUDA kernel 启动一整个 thread block 组成的 grid 是同一回事。block size 是一个编译期常量（`tl.constexpr`），而 program instance 的总数则由你在启动时指定，通常写作 `triton.cdiv(n_elements, BLOCK_SIZE)`——也就是"向上取整除法"，因为最后一个 program 几乎从来凑不满一整块。

下面是最小的完整例子，两个向量相加：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
```

你以后会读到的每一个 Triton kernel，基本都是这同一个五步套路：先确定自己是哪个 program，把这个 program 该处理的那部分问题算成一个 **向量** 形式的 offsets，为任何可能越界的位置构造一个布尔 mask，透过这个 mask 去加载数据，做一些看起来很普通的向量运算，再透过同一个 mask 存回去。

![Anatomy of a Triton kernel](blogs/images/triton-kernel-anatomy.svg?v=1)
^定位 → 计算 offsets → mask → load / 计算 → store。每一个 Triton kernel 都是这同一个五步流水线的变体。

这里的 mask 值得多停留一下，因为它做的事情在 CUDA 里并没有一个干净的对应物：它让 `BLOCK_SIZE` 可以固定取一个 2 的幂（编译器和硬件都更喜欢这样），而 `n_elements` 可以是任意的运行时数值。`tl.load(..., mask=mask)` 对被 mask 掉的 lane 根本不会发出任何内存访问——不是"加载了垃圾值再丢弃"，而是"压根不加载"——所以一个偏大的 tail block，代价只是浪费几个 lane 的算力空转，而不是额外的内存带宽开销。下面这个面板可以让你选一个总元素数和一个 `BLOCK_SIZE`，直接看到到底是哪个 program 里的哪些 lane 被 mask 掉了。

![interactive:triton-grid](#)

## 3. 从 Python 函数到 SM，中间发生了什么

有必要明确地对回 GPU 那篇第 2 节的内容，因为 Triton 的全部价值就活在这个"中间地带"里。当你写下 `add_kernel[grid](..., BLOCK_SIZE=1024)` 时，Triton 的编译器会决定用多少个真实的 CUDA 线程和 warp 去执行一个 program instance——这由另一个参数 `num_warps` 控制（默认是 4，也就是 128 个线程合作处理一个 program 对应的 1024 个元素的 tile）。你在上面那段 kernel 里根本没有写过 `threadIdx.x`；决定这 128 个协作线程里谁读 1024 个元素中的哪一个，是编译器代码生成器的工作，而对于这种规则、规整的访问模式，它做得不比一个仔细的人差多少。

这也是为什么 Triton 里的"grid"和 GPU 那篇里的"grid"是完全同一件事：一次 Triton launch，在底层真的会生成一个货真价实的 CUDA thread block 组成的 grid，每个 program instance 对应一个 block，而这个 grid 仍然会按照第 2 节描述的方式被调度到芯片的各个 SM 上——包括 occupancy、常驻 warp 数的上限，以及如果你发射的 program 数超过芯片一次能跑的数量，依然会出现多个"wave"。Triton 并没有改变底层任何一条调度规则；它改变的是写 kernel 的时候，谁需要去操心这些规则。

## 4. 内存访问模式：比 mask 更重要的那个邻居

GPU 那篇第 3 节讲过，搬运数据的代价远大于对数据做运算，这个道理原封不动地搬到 Triton 里依然成立——只是 Triton 的 tile 级抽象，更容易让你在没留意的时候，不小心写出一个访问模式很差的 kernel，因为底层到底在发生什么，可能被这层抽象悄悄藏起来了。

真正重要的区分是 **连续（contiguous）** 访问和 **跨步（strided）** 访问。像 vector-add kernel 里那样 `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`，产生的是一段连续的地址——GPU 可以用少量的宽内存事务，一次性服务完整块 tile。但只要你的下标看起来像 `offs = row * stride + col`，而 `col` 又不是变化最快的那一维，或者你是在从一个 row-major 矩阵里取出一列，你得到的就是一个 **跨步** 模式：同样多的有效元素，却散落在内存各处，每一个都要单独发起一次内存事务，事务里剩下的带宽全都浪费在你根本不需要的字节上。

![Coalesced vs strided memory access](blogs/images/triton-memory-access-patterns.svg?v=1)
^元素数量相同，事务数量却天差地别。这是一个"正确"的 Triton kernel 慢了 3–10 倍最常见的原因——往往不是 bug，只是访问模式从"流式读取"变成了"到处散射"。

一个真实会踩坑的例子：按行做 softmax，每个 program 处理矩阵的一行。

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = x_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denom = tl.sum(numerator, axis=0)
    result = numerator / denom

    out_start_ptr = out_ptr + row_idx * row_stride
    tl.store(out_start_ptr + col_offsets, result, mask=mask)
```

只要矩阵是 row-major，并且每个 program 沿着一行读取，这就是连续访问——每行只需一次宽而便宜的 load。但如果把输入转置了（或者更糟，不小心让每个 program 去处理 row-major 矩阵的一 **列**），完全相同的逻辑运算就会变成访问模式很差的跨步读取，FLOPs 数量却一点没变——这正是 GPU 那篇里 roofline 那一课的翻版，只不过这次是以一行下标写错的方式出现，而不是一个抽象概念。

## 5. Autotuning：BLOCK_SIZE、num_warps 与 num_stages

第 2 节里那个 `BLOCK_SIZE=1024` 是没有给出理由就选定的，这是故意的：通常并没有一种靠谱的方法能凭经验手选它。更大的 `BLOCK_SIZE` 意味着每次内存事务能有更多复用、需要调度的 program instance 也更少，但每个 program 消耗的寄存器和 shared memory 也更多——直接照搬 GPU 那篇第 2 节的结论，这会降低 occupancy。`num_warps` 牵涉的是同一种权衡：每个 program 的 warp 数越多，延迟隐藏能力越强（一个 warp 在等 load 的时候，另一个可以去计算），但同时也有更多常驻线程在争抢同一个有限的寄存器堆。

Triton 给出的答案不是去猜，而是去 **实测**：`@triton.autotune` 接受一份候选配置列表，在第一次调用时，真的用每一种配置在真实输入形状上跑一遍 kernel、计时，然后把跑得最快的那个缓存下来。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

`num_stages` 值得单独说一下：它控制的是 **软件流水线（software pipelining）**——编译器会提前多少次迭代，去预取下一块 tile 的 `tl.load`，同时还在计算当前这块 tile。这其实是编译器自动帮你做了和第 2 节 occupancy 一样的"用算力掩盖内存延迟"的把戏，只不过这次是往上挪了一层，发生在同一个 program 的循环迭代之间，而不是 warp 调度层面。`num_stages` 调得越高，需要用来保存那些"正在飞行中"的 tile 的寄存器和 shared memory 也越多，所以它同样要面对本节前面讲的那种 occupancy 权衡。

![The autotuning search space](blogs/images/triton-autotune-space.svg?v=1)
^最优点既不在任何一个极端，它的具体位置也会随着 kernel、输入形状和 GPU 型号而移动——这正是为什么这是一次"实测搜索"，而不是一条公式。

可以玩玩下面这个面板："每个线程处理的元素数"这个统计量，才是真正能预测你落在哪种失败模式里的数字——太低，就会有线程闲着什么都不做；太高，就会变成一个线程串行处理很多元素，而调度器手头能用来掩盖下一次 load 的独立 warp 也变少了。

![interactive:autotune](#)

## 6. 案例研究：把 FlashAttention 写成一个 Triton tiling 循环

GPU 那篇第 5 节把 attention 当作一个 memory-bound 操作的例子：在小 batch size 下，它的瓶颈不是矩阵乘法本身，而是要把完整的 attention 矩阵物化并反复读写——一个 O(序列长度²) 的张量，朴素实现要为算分数写一次到 HBM，softmax 之后又写一次，再读回来去和 V 做加权求和。FlashAttention 的全部技巧，就是永远不去物化这个矩阵，而它几乎完美地展示了 GPU 那篇第 6 节的 tiling 思路，只是多了一份额外的记账工作。

这个循环让 query tile `Q_i` 在整个生命周期里都常驻 shared memory，然后把 key / value tile `K_j`、`V_j` 一块一块地流式送进来，同时维护一个running 的最大值和一个 running（经过重新缩放的）求和——也就是"online softmax"——这样每一份部分结果都能被安全地合并进最终输出，而不需要一次性看到整行的分数。

```python
# schematic — illustrates the loop structure, not a drop-in kernel
m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)   # running max
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running sum
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)        # running output

for start_n in range(0, seq_len, BLOCK_N):
    k = tl.load(k_ptrs)                      # one small K tile
    v = tl.load(v_ptrs)                      # one small V tile

    scores = tl.dot(q, tl.trans(k)) * sm_scale
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    p = tl.exp(scores - m_new[:, None])
    correction = tl.exp(m_i - m_new)

    l_i = l_i * correction + tl.sum(p, axis=1)
    acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new

    k_ptrs += BLOCK_N * stride_kn
    v_ptrs += BLOCK_N * stride_vn

acc = acc / l_i[:, None]
```

![FlashAttention as a Triton tiling loop](blogs/images/triton-flashattention-tiling.svg?v=1)
^和 matmul tiling 同样的"复用 tile"思路，再加上一个 running softmax，让每一块 K / V tile 都能被直接揉进答案里，而不需要同时看到别的 tile 的分数。

这里每一次 `tl.dot` 调用，都是在直接调用 GPU 那篇第 4 节里的 Tensor Core——Triton 的编译器会把 `tl.dot` 识别成一个矩阵乘法形状的操作，直接把它编译成 Tensor Core 指令，这也是为什么手写的 Triton attention kernel 能够跑得很接近手写的 CUDA / CUTLASS 版本：真正昂贵的运算是在完全相同的硅片单元上发生的，Triton 帮你省下来的，其实是围绕这个矩阵乘法本身的 tiling 和 shared memory 记账工作，而不是矩阵乘法这件事本身。

## 7. Triton 停在哪里，CUDA / CUTLASS 从哪里接手

如果把 Triton 说成 CUDA 的彻底替代品，那是不诚实的，所以有必要具体说清楚这层抽象在哪些地方会露出破绽。Triton 的编译器在 tiling、向量化和指令选择上做出的是通用、良好的决策——但"通用、良好"并不等于"针对这个具体形状、这块具体芯片的最优解"，目前确实有几个地方，这个差距是真实存在的：

**最新硬件特性总是先出现在 CUDA 里。** 比如 Hopper 的 Tensor Memory Accelerator（TMA，一个专门用于 shared memory 暂存的异步拷贝引擎）、thread block cluster，或者让 block 里不同 warp 承担刻意不同角色的 warp specialization 模式，这些都会先在 CUDA / CUTLASS 里被更早、更完整地暴露出来，而 Triton 的编译器需要时间，才能为每一代新架构引入的每一种新机制补上支持。

**非常不规则的控制流，或者按 lane 各异的数据依赖分支**，是 CUDA 的 per-thread 模型天然擅长处理的东西，而 Triton 的 per-tile 模型处理起来就没那么优雅——Triton 在整块 tile 都遵循同一种计算形状时最舒服，这覆盖了深度学习里的大多数情况，但不是全部。

**一个关键 kernel 最后那几个百分点的性能**——那种值得让一个工程师团队专门去手工调寄存器分配和 shared memory bank 布局的场景——目前依然经常是手写 CUTLASS 更快，而不是 Triton，这也是为什么目前发布的最快的 FlashAttention 实现，往往同时存在一个 Triton 版本（好用、可移植、容易改动）和一个手写 CUDA / CUTLASS 版本（还要更快一点，但也更难动）。

这些都不会让 Triton 变成一个"次一等"的工具——它只是恰好在一个大得多的问题范围里，成为了更合适的那个工具，而不是原始 CUDA 更合适的那部分。一个深度学习研究者这辈子真正需要手写的自定义 kernel，绝大多数都恰恰是 Triton 生来就是为之设计的那类场景："规则、tile 形状、不需要在硬件刚发布第一天就用上它最新的特性"。而且相当一部分收益其实是白得的：`torch.compile` 的 Inductor 后端已经会为它能识别出来的大多数 fused elementwise 链条，自动生成并自动调优 Triton kernel——所以很多研究者早就在跑 Triton 生成的代码，只是自己从没写过一个 `@triton.jit` 装饰器。

## 8. 写自己的 kernel 时的一份简短清单

**先检查正确性，再检查速度。** 先在小的、容易检查的形状上，和一份参考的 PyTorch 实现做对比（用适合你所用精度的容差调用 `torch.allclose`）。一个跑得快但结果错的 kernel 毫无意义，而 Triton 的 mask 很容易让你在最后一块 tile 的边界条件上，不知不觉写错。

**先用 interpreter 调试逻辑，再上真实 GPU。** `TRITON_INTERPRET=1` 会用一个 Python 层面的解释器来运行你的 kernel，而不是真的编译它，这能把那些晦涩的 device 端报错，变成正常的 Python 报错堆栈——在把一个 bug 归咎于什么"深奥的 warp 行为"之前，值得先试试这个。

**你的访问模式是连续的吗？** 第 4 节的问题。如果一个 kernel 比它的 FLOPs 数量所暗示的速度慢很多，先检查一下你写的下标，到底是在真的连续读写，还是不小心变成了跨步散射。

**你是让 `@triton.autotune` 去搜索，还是自己拍脑袋定了一组配置？** 第 5 节的问题——在调试正确性阶段，手选一个 `BLOCK_SIZE` / `num_warps` 是合理的起点，但如果对一个你真正在意速度的 kernel 也一直这么做，通常会白白损失掉本可以拿到的性能，而代价只是加一个 `@triton.autotune` 装饰器。

**这段逻辑 `torch.compile` 本来会不会已经帮你生成了？** 在动手写 kernel 之前，值得先检查一下 Inductor 的自动 fusion 是不是已经生成了很接近的东西——对于一串普通的 elementwise / reduction 操作，它经常已经做到了，所以更诚实的第一步动作，往往是先试试 `torch.compile(model)`，而不是直接手写一个 kernel。

**如果还是不够快，瓶颈到底是内存还是算力？** 还是 GPU 那篇里那个 roofline 问题，现在可以直接用 `triton.testing.do_bench` 测墙钟时间、用 Nsight Compute 测实际带宽和 FLOPs 利用率来回答——Triton 生成的 kernel 在这些 profiler 里的表现，和手写 CUDA kernel 一模一样，因为等它真正跑到 GPU 上的时候，它本来就是同一种东西。

*这篇文章里的代码是示意性的，而不是可以直接生产使用的——真实的 kernel 要处理更多边界情况，而且 Triton 的 API 表面在一代代发布之间变化得很快。真正经得住时间的想法，是那些和 GPU 那篇里完全一致、原封不动继承下来的想法：用 tile 取代 thread、用实测搜索取代手工调参，以及为你真正需要做的那部分运算，尽量少搬一点数据。*
