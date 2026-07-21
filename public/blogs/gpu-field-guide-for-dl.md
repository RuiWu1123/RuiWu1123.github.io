---
title: "Inside the GPU: A Field Guide for Deep Learning Researchers"
date: "2026/7/2"
---

Most of us who train models never really have to think about the GPU. You write `model.to("cuda")`, you write `loss.backward()`, and a few weeks later a number goes down. The hardware underneath is a black box that occasionally throws `CUDA out of memory` at you and otherwise stays quietly out of the way.

This post is for the moment that black box stops staying out of the way — when you're trying to figure out why your training step takes 400ms instead of 40ms, why doubling your batch size didn't slow anything down, why your beautifully parallel 8-GPU job scales like a 3-GPU job, or why mixed precision training exists at all. None of these questions require you to know how to write CUDA kernels. They do require a working mental model of what a GPU actually *is*, which is what this post tries to build — from zero, but built specifically for the questions a deep learning researcher runs into, not for a general audience. If you already know what a warp is, most of the first section will be review; stick around for the roofline model and the multi-GPU section, which is where the actually load-bearing intuitions live.

## 1. The one idea everything else follows from

A CPU and a GPU are both built out of transistors, and both do arithmetic. The difference is almost entirely about *what they optimize for*.

A CPU is optimized for **latency**: get one instruction stream through as fast as possible, including all its branches, dependencies, and unpredictability. To do that, a CPU core spends most of its silicon on things that have nothing to do with arithmetic — branch prediction, out-of-order execution, large caches, speculative execution — so that a single thread of control can run about as fast as physics allows. A CPU typically has somewhere between 8 and 128 of these cores.

A GPU is optimized for **throughput**: it assumes you have thousands of independent, near-identical pieces of work to do (the pixels of a frame, or the elements of a tensor), and it doesn't care how long any *one* of them takes, as long as it can push huge numbers of them through per second. So a GPU spends comparatively little silicon on control logic and branch prediction, and devotes almost all of it to raw arithmetic units. Instead of a few complex cores, you get thousands of simple ones.

![CPU vs GPU chip layout](blogs/images/cpu-vs-gpu.svg?v=3)
^A CPU devotes most of its area to control logic and cache so a single instruction stream runs fast. A GPU devotes most of its area to arithmetic units (ALUs), running the same instruction across thousands of threads at once.

This is why "a GPU has 16,896 cores and a CPU has 16" is true but misleading if you take it at face value — a GPU core (usually called a CUDA core) is a tiny, simple ALU that can only do a scalar multiply-add. It is not comparable to a CPU core, which is a whole independent processor. The right comparison isn't core-for-core; it's "how much useful arithmetic can this chip do per second, given a workload with enough parallelism," and by that measure GPUs win by 1–2 orders of magnitude on exactly the kind of workload deep learning is: the same handful of operations (matmul, add, multiply, a nonlinearity) applied identically across millions of independent elements.

That's also the constraint to keep in mind going forward: a GPU is fast *because* it assumes your work is regular and parallel. Every time your code introduces data-dependent branching, small sequential dependencies, or irregular memory access, you're fighting the thing that makes a GPU a GPU.

## 2. Grids, blocks, threads, warps: how work gets scheduled

To use that arithmetic throughput, you need a programming model that expresses "do the same thing, on independent data, a huge number of times." That model, in CUDA terms, is a hierarchy:

A **kernel launch** starts a **grid**. A grid is divided into **thread blocks** (or just "blocks"). Each block is divided into **threads**. When you write a PyTorch op, you don't see any of this — the framework and its underlying kernels (from cuDNN, cuBLAS, or a custom Triton/CUDA kernel) choose the grid and block dimensions for you — but the shape of that hierarchy still fully determines your performance.

The hardware-level detail that actually matters is this: threads don't run one at a time, and they don't run fully independently either. They run in groups of **32, called a warp**, and all 32 threads in a warp execute the *same instruction* at the *same time*, just on different data (this execution model is called SIMT — single instruction, multiple threads). If you have an `if` statement and 16 threads in a warp take the `true` branch while 16 take the `false` branch, the warp doesn't run both halves in parallel — it runs the `true` branch with the other 16 threads masked off, then the `false` branch with the first 16 masked off. This is called **warp divergence**, and it's one of the more common ways to silently lose a large fraction of your theoretical throughput inside a custom kernel.

A GPU's chip is, physically, an array of **Streaming Multiprocessors (SMs)** — an H100 has 132 of them. Each SM has its own warp schedulers, its own CUDA cores, its own Tensor Cores, its own register file, and a pool of on-chip shared memory. Crucially: **a whole thread block is scheduled onto exactly one SM**, and stays there for its entire lifetime — it never migrates mid-execution. Multiple blocks can be resident on the same SM at once (this is what "occupancy" refers to), but a single block never spans two SMs.

![Inside one Streaming Multiprocessor](blogs/images/gpu-sm-anatomy.svg?v=3)
^Each SM contains several warp schedulers, banks of CUDA cores (scalar/vector arithmetic) and Tensor Cores (matrix arithmetic), a register file, and shared memory / L1 cache. A block lives entirely on one SM for its whole lifetime.

How many blocks can be simultaneously resident on one SM depends on how much of that SM's limited resources — registers, shared memory, and the maximum number of resident warps (typically 64 warps = 2048 threads on recent architectures) — each block needs. A kernel that uses fewer registers per thread, or less shared memory per block, can pack more blocks onto an SM at once, which usually means better latency hiding (while one warp waits on a slow memory load, the scheduler can switch to another warp that's ready to compute) and, up to a point, better throughput. This is the essence of "occupancy," a term you'll see in every GPU profiler.

The interactive panel below lets you pick a block size and a total number of blocks, and see how that maps onto 132 SMs — including how many sequential "waves" of blocks it takes to get through the whole grid, since only so many blocks fit on the chip at once.

![interactive:grid-block](#)

The takeaway that matters in practice: **launching too few blocks under-uses the chip** (some SMs sit idle), and **very small blocks with heavy per-thread resource usage under-use each SM** (occupancy is low even though every SM has work). Both failure modes show up as "my GPU utilization is low" in `nvidia-smi`, for entirely different underlying reasons — which is exactly why that single utilization number is a bad diagnostic tool on its own.

## 3. The memory hierarchy: the thing that actually limits you

Here is the fact that reframes almost everything else in this post: **on nearly every modern accelerator, moving data costs far more, in relative terms, than doing arithmetic on it.** A single fused-multiply-add is almost free. Fetching the two numbers it operates on from the wrong place in memory can cost 100–1000× more in cycles than the arithmetic itself. Deep learning performance work is, disproportionately, about respecting this fact.

GPUs (like CPUs) have a memory hierarchy: several tiers of storage that trade off capacity against speed.

![GPU memory hierarchy](blogs/images/memory-hierarchy.svg?v=3)
^Every step down this pyramid is roughly 5–20× more capacity and 5–20× less bandwidth. Fast kernels reuse data as high up this pyramid as possible instead of re-fetching it from below.

A few of these levels are worth calling out specifically, because they correspond directly to concepts you'll run into constantly:

**Registers** are private to a single thread and are, functionally, free to access — but there are very few of them per thread, and using too many per thread is one of the main things that reduces occupancy (fewer blocks fit on an SM if each thread hogs more registers).

**Shared memory / L1** is on-chip, shared by all threads in a block, and roughly 20–30× faster than going to device memory. It is the main tool a kernel author has for turning a memory-bound operation into a compute-bound one: load a chunk of data from device memory into shared memory once, then have many threads reuse it from there instead of each one re-fetching it. This is exactly what makes matrix multiplication fast on a GPU, and we'll come back to it in Section 5.

**HBM (High Bandwidth Memory)** is what everyone means when they say "GPU memory," what `nvidia-smi` reports, and what an out-of-memory error is complaining about. It's fast by DRAM standards — an H100 has roughly 3.35 TB/s of HBM3 bandwidth — but it's still one to two orders of magnitude slower than on-chip memory, and every tensor, activation, and weight your model touches has to live here between kernel calls.

**Host (CPU) DRAM, over PCIe**, is where your dataloader's output sits before it's copied to the GPU, and where CPU-side tensors live. PCIe Gen5 gives you on the order of 60–64 GB/s in each direction — two orders of magnitude below HBM bandwidth. This is precisely why a slow dataloader can silently cap your GPU utilization at 40%: the GPU finishes a step and then sits idle, waiting for the next batch to cross this comparatively narrow bridge from the CPU side.

**Remote GPU memory, over NVLink or over the network**, is what a *different* GPU's activations, gradients, or KV-cache look like from your GPU's perspective — another one to two orders of magnitude slower again, and the subject of Section 6.

A useful mental habit: whenever you look at a slow operation, ask "how many bytes did this have to move, and from how far down this pyramid?" before asking "how many FLOPs did this do?" For a huge share of deep learning workloads, the honest answer to the second question is "not that many, actually" — and the slowness is entirely explained by the first.

## 4. Tensor Cores, and why precision is a systems decision, not just a numerics one

CUDA cores do scalar or small-vector fused-multiply-adds: one multiply, one add, per instruction, per thread. Starting with the Volta architecture (2017), NVIDIA added a second kind of arithmetic unit to every SM: the **Tensor Core**, which computes a small *matrix* multiply-accumulate in a single operation (conceptually, something like a 4×4 times 4×4 matrix multiply, accumulated into a 4×4 result, all in one instruction). Because deep learning is, at its core, an enormous pile of matrix multiplications, Tensor Cores can deliver roughly an order of magnitude more throughput than CUDA cores for exactly the operations that dominate training and inference — provided your data is in a format the Tensor Cores support.

That last clause is where precision stops being a purely numerical concern and becomes a systems one. Tensor Cores are typically fastest at lower-precision formats, because narrower numbers mean more of them fit through the same silicon and memory bandwidth per cycle. On an H100, dense (non-sparse) peak throughput roughly doubles as you drop from one format to the next:

| Format | Peak dense throughput (H100 SXM5, approx.) | Typical role |
|---|---|---|
| FP32 (CUDA core) | ~67 TFLOPs | Reference / non-tensor-core math |
| TF32 (Tensor Core) | ~495 TFLOPs | "Free" speedup for FP32-written training code |
| BF16 / FP16 (Tensor Core) | ~990 TFLOPs | Standard mixed-precision training |
| FP8 (Tensor Core) | ~1979 TFLOPs | Newer training/inference, careful scaling required |

(Figures are NVIDIA's published dense, non-sparse numbers for H100 SXM5; the newer Blackwell B200 pushes this further still, with roughly 8 TB/s of HBM3e bandwidth, ~5 PFLOPs dense BF16, and a native FP4 format for inference.)

This is why mixed-precision training exists at all: it isn't a numerics trick invented for its own sake, it's a direct consequence of the hardware rewarding narrower formats so heavily. The numerics complexity — loss scaling, keeping a master copy of weights in FP32, choosing BF16 over FP16 for its wider dynamic range — exists entirely in service of safely capturing that 2–4× hardware speedup (and a proportional reduction in memory traffic, which, per Section 3, often matters even more than the raw FLOPs). FP8 training pushes the same trade further and needs correspondingly more careful scaling; it is generally not "free" the way TF32 is, and is still an active area of methodological work rather than a drop-in switch.

The practical implication: if you're not sure why a paper bothers to specify its precision so carefully, or why a training recipe insists on keeping certain layers (like the final loss computation, or normalization statistics) in higher precision, it's usually about exactly this tension — narrower formats are a large, real, hardware-given speedup, purchased at the cost of dynamic range and precision that your numerics have to be robust to.

## 5. The roofline model: is your kernel compute-bound, or memory-bound?

Sections 3 and 4 both point at the same underlying question, and there's a standard tool for answering it precisely: the **roofline model**.

Define **arithmetic intensity** as the ratio of floating-point operations performed to bytes of memory moved (FLOPs / byte) for a given operation. Every operation has some peak achievable throughput given by:

**achievable TFLOPs/s = min( peak compute, arithmetic intensity × peak memory bandwidth )**

At low arithmetic intensity, you're **memory-bound**: the GPU's arithmetic units spend most of their time idle, waiting on data, and your achievable throughput scales linearly with how much bandwidth you can use. At high arithmetic intensity, you're **compute-bound**: memory can keep the arithmetic units fed easily, and your achievable throughput is capped at the chip's peak FLOPs, no matter how much bandwidth you have to spare. The crossover point — where compute and bandwidth become equally limiting — is called the **ridge point**, and it depends only on the chip (peak compute ÷ peak bandwidth), not on your code.

Most of the individual operations inside a transformer sit in very different places on this curve. Elementwise operations (adding a bias, a ReLU, a residual add) do one or two FLOPs per element and have to read and write that element from HBM — arithmetic intensity close to 0.1–0.5, deep in memory-bound territory. Normalization and softmax are only slightly better. A large, well-shaped matrix multiplication, by contrast, reuses each loaded value many times (as we'll see in the next section) and can have an arithmetic intensity in the hundreds — solidly compute-bound, and exactly where Tensor Cores earn their keep.

Play with the panel below: pick a Tensor Core precision (which sets the ridge point) and either an operation profile or a manual arithmetic-intensity slider, and watch which regime you land in.

![interactive:roofline](#)

This single idea explains a lot of otherwise-confusing empirical facts: why fusing several small elementwise ops into one kernel (what `torch.compile` and hand-written fused kernels both try to do) can produce a large speedup with *zero* change in FLOPs — you're not doing less arithmetic, you're doing fewer round-trips to HBM, which is what was actually limiting you; why attention at long sequence lengths and small batch sizes can be disappointingly slow relative to its FLOP count — it's memory-bound, dominated by moving the attention matrix and KV-cache around, not by the matmuls themselves (this is a large part of what motivated FlashAttention: it restructures the computation specifically to avoid materializing memory-bound intermediate tensors); and why a bigger batch size so often "comes for free" in terms of wall-clock time per sample — it increases arithmetic intensity by amortizing each weight load over more examples, sliding you further into compute-bound territory where you were leaving throughput on the table before.

## 6. Matrix multiplication: mapping the operation that matters most onto the chip

Since matmul is both the dominant FLOP consumer in deep learning and the clearest illustration of everything above, it's worth looking at how it's actually executed, at a level of detail one step below "cuBLAS handles it."

A naive matmul implementation would have each output element read an entire row of A and an entire column of B directly from HBM, compute a dot product, and write one output value. This is disastrously memory-bound: every element of A gets re-read from HBM once for every column of B, and vice versa — a huge amount of redundant traffic to the slowest tier of memory.

The actual implementation instead **tiles** the problem. The output matrix C is divided into tiles, and each tile is assigned to one thread block. That block cooperatively loads the small strips of A and B it needs — once — into on-chip shared memory. From there, every thread in the block computes its piece of the output tile by reading repeatedly from *shared memory* rather than HBM, reusing each loaded value many times before it's evicted.

![Matmul tiling diagram](blogs/images/matmul-tiling.svg?v=3)
^Loading a tile of A and a tile of B into shared memory once, then reusing them for every element of the output tile, is what turns matmul from a memory-bound operation into a compute-bound one.

This is the concrete mechanism behind the abstract idea of "arithmetic intensity" from the previous section: tiling turns one HBM load into many arithmetic operations, and the bigger the tile (which usually means the bigger your batch size or hidden dimension), the more reuse you get per byte loaded, and the closer you get to the chip's peak FLOPs. It is also, very directly, why very small matrices — a hidden dimension of 32, or a batch size of 1 — chronically under-use a GPU: there simply isn't enough reuse available per tile to amortize the cost of loading it, no matter how well-tuned the kernel is.

## 7. Multiple GPUs: interconnects set the ceiling on parallelism strategy

Everything above described a single GPU. Training anything at contemporary scale means coordinating many of them, and the physical topology connecting them is not a footnote — it directly determines which parallelization strategy will actually scale.

Within a single node, modern systems connect GPUs via **NVLink** (roughly 900 GB/s–1.8 TB/s per GPU, depending on generation), often through an **NVSwitch** that gives every GPU in the node a full-bandwidth path to every other GPU. Between nodes, GPUs typically talk over a network fabric like **InfiniBand** or RoCE, at roughly 400 Gb/s (about 50 GB/s) per GPU — again, an order of magnitude drop, this time from intra-node NVLink to inter-node networking.

![Multi-GPU topology](blogs/images/multi-gpu-topology.svg?v=3)
^Bandwidth drops by roughly an order of magnitude at every hop outward: on-chip → HBM → NVLink (within a node) → network fabric (across nodes).

This asymmetry is the whole reason different parallelism strategies exist and get combined the way they do:

**Data parallelism** — every GPU holds a full copy of the model, processes a different slice of the batch, and the resulting gradients are averaged (all-reduced) across GPUs once per step — communicates relatively infrequently and can tolerate the slower, inter-node link reasonably well. This is why it's the strategy that scales most easily across many nodes.

**Tensor parallelism** — splitting individual weight matrices across GPUs, so that a single layer's computation is itself distributed — requires communication *inside* every layer, potentially many times per forward and backward pass. This volume of traffic needs the fast, low-latency NVLink domain; running tensor parallelism across the slower inter-node link is a common, very concrete reason a "linear scaling" plot stops looking linear once a job spans more than one node.

**Pipeline parallelism** — splitting the model's layers across GPUs, with different GPUs handling different stages of the same forward/backward pass — communicates less frequently than tensor parallelism (just the activations between stage boundaries) and so tolerates inter-node links better, at the cost of the well-known "pipeline bubble" idle time.

In practice, large training runs use a mix — tensor parallelism inside a node (staying on NVLink), pipeline and/or data parallelism across nodes (tolerating the network) — precisely because the topology diagram above dictates which trade-offs are survivable at which distance. When someone says a training job doesn't scale efficiently past a certain number of nodes, the first thing worth checking is whether a parallelism strategy that assumes NVLink-class bandwidth is accidentally being asked to run across the network fabric instead.

## 8. Putting it together: a short field guide for when training feels slow

None of the sections above are purely theoretical — they map onto a fairly short checklist that covers most "why is my training slow" investigations:

**Is the GPU actually busy?** `nvidia-smi` utilization tells you whether *some* kernel is running on the GPU at any given instant, not whether that kernel is doing useful work efficiently. A slow dataloader, or unnecessary CPU–GPU synchronization points in your training loop (a stray `.item()`, `.cpu()`, or `print(tensor)` inside the hot loop), will often show up as GPU utilization that looks fine on average, while the GPU is actually idling between short bursts.

**Is the kernel memory-bound or compute-bound?** Section 5's roofline question. Profilers (Nsight Compute, or the built-in PyTorch profiler) will tell you achieved FLOPs and achieved bandwidth utilization directly; if bandwidth utilization is high and FLOP utilization is low, no amount of "faster GPU" will help nearly as much as reducing memory traffic (operator fusion, `torch.compile`, avoiding unnecessary intermediate tensors).

**Is occupancy reasonable?** Section 2's question — are you launching enough parallel work, with a block size that doesn't waste registers or shared memory, to actually fill the chip? This mostly matters for hand-written or custom kernels rather than standard PyTorch ops, which are usually already reasonably tuned.

**Are you using the precision the hardware rewards?** Section 4's question — mixed precision (BF16/FP16, or FP8 where your recipe supports it) is very often the single highest-leverage change available, precisely because the hardware is built to reward it so heavily.

**If multi-GPU, does your parallelism strategy match your topology?** Section 6's question — tensor-parallel traffic crossing a slow inter-node link, rather than staying inside the fast NVLink domain, is one of the most common reasons multi-node scaling disappoints.

**Are you memory-limited rather than compute-limited, and is that showing up as OOM rather than slowness?** The same memory hierarchy that determines speed also determines capacity — activation memory, optimizer states, and the trade-off of gradient checkpointing (recomputing activations instead of storing them, trading compute for HBM capacity) are all instances of Section 3's pyramid, just viewed through the capacity axis instead of the bandwidth axis.

None of this requires writing a CUDA kernel yourself. But every one of these questions becomes answerable — instead of a guess — once "compute vs. bandwidth," "warps vs. occupancy," and "NVLink vs. network" stop being unfamiliar terms and start being the mental model you reach for by default when a training job isn't behaving the way the FLOP count on paper says it should.

*Numbers in this post (H100/B200 specs, bandwidth, TFLOPs) are drawn from NVIDIA's published datasheets and are meant to convey the right order of magnitude and the right relationships between quantities, not to serve as a citation-grade spec sheet — exact numbers vary by SKU, cooling, and driver/firmware version, and shift with every new hardware generation. The concepts — SIMT execution, the memory hierarchy, the roofline model, and interconnect topology — are far more durable than any specific number attached to them.*
