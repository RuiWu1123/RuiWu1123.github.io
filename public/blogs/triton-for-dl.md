---
title: "Writing Fast GPU Kernels Without Touching CUDA"
date: "2026/7/9"
---

The [previous post](#/blog?id=gpu-field-guide-for-dl) built a mental model of the GPU: warps and SMs, the memory hierarchy, the roofline model, and how matrix multiplication gets tiled onto shared memory. None of that required writing a single line of CUDA. This post is about the moment you *do* want to write something yourself — a fused elementwise op, a custom attention variant, a normalization layer that PyTorch doesn't ship — and would rather not learn CUDA C++, thread indexing, and manual shared-memory bank-conflict avoidance to do it.

That's the gap Triton fills. It's a Python-embedded language, originally from OpenAI and now maintained as part of the PyTorch ecosystem, that lets you write GPU kernels at the level of "blocks of data" instead of "individual threads," and compiles them down to the same PTX that hand-written CUDA targets. It is also, quietly, already running under a huge amount of code you've used: `torch.compile`'s Inductor backend generates Triton kernels automatically for most of the fused ops it produces, and most published FlashAttention-family implementations exist in a Triton version. This post assumes the mental model from the GPU post — warp, SM, shared memory, arithmetic intensity, tiling — and builds directly on top of it to explain what Triton actually does with that hardware, and where it stops being able to help.

## 1. What Triton actually changes

Writing a CUDA kernel means writing code that runs *once per thread*, and reasoning explicitly about every layer of the hierarchy from the GPU post: which thread you are (`threadIdx.x`, `blockIdx.x`), which data your specific thread should touch, when to stage data through shared memory yourself, and how to avoid bank conflicts and warp divergence by hand. It's an extremely capable model — essentially everything the hardware can do is reachable from it — and that capability is exactly why it's slow to write and easy to get subtly wrong.

Triton's core idea is to move the unit of programming up one level: instead of writing "what does thread `i` do," you write "what does this program instance do to *a block of data*." A Triton kernel operates on vectors (Triton calls the language's tensor-like values simply "tensors," confusingly enough), not scalars. You load a whole tile, do arithmetic on the whole tile, and store the whole tile back — and the compiler is responsible for lowering that tile-level description down onto actual threads, warps, and (where it can) shared memory, entirely without you writing any of that by hand.

![CUDA vs Triton programming model](blogs/images/cuda-vs-triton-model.svg?v=1)
^The unit of thought moves from "one thread, one scalar" to "one program, one tile." Everything below that line — thread indices, warp assignment, a good deal of shared-memory staging — becomes the compiler's problem instead of yours.

This is a genuine trade of generality for productivity, and it's worth being honest about which direction that trade goes. You give up some of CUDA's full expressiveness — arbitrary per-thread control flow, hand-placed shared-memory layouts, the newest hardware-specific instructions on the bleeding edge of a new architecture. In exchange you get kernels that are dramatically shorter, that a compiler auto-tunes for you across block sizes and warp counts, and that you can write, read, and debug as Python. For the large majority of the "custom kernel" work a deep learning researcher actually needs — fused elementwise chains, custom normalization or activation functions, attention variants, quantization/dequantization kernels — that trade is very favorable. Section 7 comes back to exactly where it stops being favorable.

## 2. The programming model: programs, offsets, and masks

Every Triton kernel is a Python function decorated with `@triton.jit`, launched not once but many times in parallel — once per **program instance** — exactly the way a CUDA kernel launches a grid of thread blocks. The block size is a compile-time constant (`tl.constexpr`), and the total number of program instances is whatever you specify at launch time, typically `triton.cdiv(n_elements, BLOCK_SIZE)` — "ceiling division," because the last program almost never gets a perfectly full block.

Here's the smallest complete example, adding two vectors:

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

Every Triton kernel you'll ever read has this same five-step shape: figure out which program you are, compute this program's slice of the problem as a *vector* of offsets, build a boolean mask for anything that would run off the end of the array, load through that mask, do ordinary-looking vector arithmetic, store back through the same mask.

![Anatomy of a Triton kernel](blogs/images/triton-kernel-anatomy.svg?v=1)
^Locate → compute offsets → mask → load/compute → store. Every Triton kernel is a variation on this same five-step pipeline.

The mask deserves a moment of attention because it's doing something that has no clean CUDA equivalent: it's what lets `BLOCK_SIZE` be a fixed power of two (which the compiler and hardware both prefer) while `n_elements` is an arbitrary runtime value. `tl.load(..., mask=mask)` simply never issues memory traffic for masked-off lanes — it's not "load garbage and discard it," it's "don't load at all" — so an over-sized tail block costs you a few idle lanes of compute, not extra memory bandwidth. The panel below lets you pick a total element count and a `BLOCK_SIZE` and watch exactly which lanes in which program end up masked.

![interactive:triton-grid](#)

## 3. What happens between the Python function and the SM

It's worth being explicit about the mapping back to Section 2 of the GPU post, because Triton's whole value proposition lives in this gap. When you write `add_kernel[grid](..., BLOCK_SIZE=1024)`, Triton's compiler decides how many actual CUDA threads and warps to use to execute one program instance — controlled by a separate parameter, `num_warps` (default 4, meaning 128 threads cooperate on one program's `BLOCK_SIZE`-sized tile). You never write `threadIdx.x` anywhere in the kernel above; the compiler's code generator is the thing that decides which of the 128 cooperating threads reads which element of the 1024-element tile, and it does that job about as well as a careful human would for simple, regular access patterns like this one.

This is also why "the grid" in Triton means the exact same thing it meant in the GPU post: a Triton launch produces a genuine CUDA grid of thread blocks under the hood, one block per program instance, and that grid still gets scheduled onto the chip's SMs exactly the way Section 2 described — including occupancy, resident-warp limits, and multiple "waves" if you launch more programs than the chip can run at once. Triton doesn't change any of the underlying scheduling rules; it changes who has to think about them while writing the kernel.

## 4. Memory access patterns: the mask's more important sibling

Section 3 of the GPU post established that moving data costs far more than computing on it, and that lesson transfers to Triton completely unchanged — Triton just makes it easier to accidentally write a kernel with a bad access pattern, because the tile-level abstraction can hide what's happening underneath if you're not paying attention.

The distinction that matters is **contiguous** versus **strided** access. `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`, as in the vector-add kernel, produces a contiguous run of addresses — the GPU can service that whole tile with a small number of wide memory transactions. But the moment your indexing looks like `offs = row * stride + col` and `col` doesn't vary fastest, or you're gathering a column out of a row-major matrix, you get a **strided** pattern: the same number of useful elements, but scattered across memory, each one costing its own transaction with the rest of that transaction's bandwidth wasted on bytes you didn't ask for.

![Coalesced vs strided memory access](blogs/images/triton-memory-access-patterns.svg?v=1)
^Same element count, very different transaction count. This is the single most common reason a "correct" Triton kernel is 3–10× slower than it should be — not a bug, just an access pattern that scatters instead of streams.

A representative example where this actually bites: a row-wise softmax, where each program handles one row of a matrix.

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

As long as the matrix is row-major and each program reads along a row, this is contiguous — one wide, cheap load per row. Transpose the input (or, worse, launch one program per *column* of a row-major matrix by mistake) and the exact same logical operation becomes badly strided, for no change in FLOPs at all — precisely the roofline lesson from the GPU post, now showing up as a one-line indexing mistake rather than an abstract concept.

## 5. Autotuning: BLOCK_SIZE, num_warps, and num_stages

Section 2's `BLOCK_SIZE=1024` was picked without justification, and that's deliberate: there usually isn't a principled way to pick it by hand. A bigger `BLOCK_SIZE` means more reuse per memory transaction and fewer program instances to schedule, but also more registers and more shared memory consumed per program — which, straight out of Section 2 of the GPU post, lowers occupancy. `num_warps` interacts with the same trade: more warps per program means more latency hiding (while one warp waits on a load, another can compute) but also more resident threads competing for the same limited register file.

Triton's answer is not to guess but to **measure**: `@triton.autotune` takes a list of candidate configurations, actually runs the kernel with each one on real input shapes the first time it's called, times them, and caches the fastest.

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

`num_stages` is worth calling out on its own: it controls **software pipelining** — how many iterations ahead the compiler prefetches the next tile's `tl.load` while the current tile is still being computed on. This is the compiler doing, automatically, the same "hide memory latency behind arithmetic" trick that occupancy does at the warp-scheduling level in Section 2, but one level up, across loop iterations of a single program. Higher `num_stages` needs more registers and shared memory to hold the in-flight tiles, so it's subject to the same occupancy trade-off as everything else in this section.

![The autotuning search space](blogs/images/triton-autotune-space.svg?v=1)
^The optimum isn't at either extreme, and its exact location shifts with the kernel, the input shape, and the GPU — which is exactly why this is a measured search, not a formula.

Play with the panel below: the "elements per thread" statistic is the number that actually predicts which failure mode you're in — too low, and you have idle threads doing nothing; too high, and one thread is serially grinding through many elements while the scheduler has fewer independent warps available to hide the next load behind.

![interactive:autotune](#)

## 6. Case study: FlashAttention as a Triton tiling loop

Section 5 of the GPU post used attention as a motivating example of a memory-bound operation: at small batch sizes, its bottleneck isn't the matmuls, it's materializing and re-reading the full attention matrix — an O(sequence length²) tensor that a naive implementation writes to HBM once for the scores, once again after the softmax, and reads back for the weighted sum over V. FlashAttention's entire trick is to never materialize that matrix, and it's a nearly perfect illustration of the tiling idea from Section 6 of the GPU post applied with one extra piece of bookkeeping.

The loop keeps a query tile `Q_i` resident in shared memory for its whole lifetime, and streams key/value tiles `K_j`, `V_j` through it one at a time, maintaining a running maximum and a running (rescaled) sum as it goes — the "online softmax" — so that each partial result can be safely combined into the final output without ever having seen the whole row of scores at once.

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
^The same tile-reuse idea as matmul tiling, plus a running softmax so that each K/V tile can be folded into the answer without needing the other tiles' scores at the same time.

Every `tl.dot` here is a call down into the Tensor Cores from Section 4 of the GPU post — Triton's compiler recognizes `tl.dot` as a matmul-shaped operation and lowers it to Tensor Core instructions directly, which is a large part of why hand-written Triton attention kernels can get close to hand-written CUDA/CUTLASS ones: the expensive arithmetic is happening on exactly the same silicon, and the thing Triton is actually saving you from writing by hand is the tiling and shared-memory bookkeeping around it, not the matmul itself.

## 7. Where Triton stops, and CUDA/CUTLASS picks back up

It would be dishonest to present Triton as a strict replacement for CUDA, so it's worth being specific about where the abstraction leaks. Triton's compiler makes good, general-purpose decisions about tiling, vectorization, and instruction selection — but "good, general-purpose" is not the same as "the best possible for this exact shape on this exact chip," and there are a few concrete places where that gap is currently real:

**Newest-hardware features arrive in CUDA first.** Things like Hopper's Tensor Memory Accelerator (TMA, a dedicated asynchronous copy engine for shared memory staging) and thread block clusters, or warp specialization patterns where different warps in a block are given deliberately different roles, are exposed earlier and more completely in CUDA/CUTLASS than in Triton, whose compiler needs time to grow support for each new mechanism a new architecture introduces.

**Very irregular control flow or data-dependent branching per lane** is something CUDA's per-thread model handles natively and Triton's per-tile model handles less gracefully — Triton is happiest when the whole tile follows the same computational shape, which is most of deep learning, but not all of it.

**The very last few percent of a critical kernel** — the ones that justify a team of engineers hand-tuning register allocation and shared-memory bank layout — are still frequently faster in hand-written CUTLASS than in Triton, which is why the fastest published FlashAttention implementations exist in both a Triton version (good, portable, easy to modify) and a hand-written CUDA/CUTLASS version (a bit faster still, harder to touch).

None of this makes Triton a lesser tool — it makes it the right tool for a different, much larger fraction of the problem than raw CUDA is. Most custom kernels a deep learning researcher will ever need to write are exactly the "regular, tile-shaped, doesn't need the newest hardware feature on day one" case Triton is built for. And a large share of the benefit arrives for free either way: `torch.compile`'s Inductor backend already generates and autotunes Triton kernels automatically for most fused elementwise chains it can identify, so plenty of researchers are already running Triton-generated code without writing a `@triton.jit` decorator themselves.

## 8. A short field guide for writing your own kernel

**Check correctness before you check speed.** Compare against a reference PyTorch implementation on small, easy-to-inspect shapes first (`torch.allclose` with a sensible tolerance for the precision you're using). A fast, wrong kernel is worthless, and Triton's masks make it easy to get boundary conditions subtly wrong on the very last tile.

**Debug logic with the interpreter before debugging on a real GPU.** `TRITON_INTERPRET=1` runs your kernel through a Python-level interpreter instead of compiling it, which turns cryptic device-side errors into normal Python stack traces — worth reaching for before assuming a bug is something exotic about warp behavior.

**Is your access pattern contiguous?** Section 4's question. If a kernel is much slower than its FLOP count suggests it should be, check whether the indexing you wrote is actually reading and writing contiguous runs, or accidentally scattering across a stride.

**Are you letting `@triton.autotune` search, rather than guessing one config?** Section 5's question — hand-picking a single `BLOCK_SIZE`/`num_warps` is a reasonable start while debugging correctness, but leaving it there on a kernel you actually care about the speed of usually leaves real performance on the table, for the cost of one `@triton.autotune` decorator.

**Would `torch.compile` already generate this for you?** Before hand-writing a kernel, it's worth checking whether Inductor's automatic fusion already produces something close — for chains of ordinary elementwise/reduction ops, it frequently does, and the honest first move is often `torch.compile(model)` rather than a custom kernel.

**If it's still not fast enough, is the bottleneck memory or compute?** The same roofline question from the GPU post, now answerable directly with `triton.testing.do_bench` for wall-clock time and Nsight Compute for achieved bandwidth and FLOP utilization — Triton-generated kernels show up in these profilers exactly like hand-written CUDA ones, because by the time they reach the GPU, that's exactly what they are.

*The code in this post is illustrative rather than production-ready — real kernels carry more edge-case handling, and Triton's API surface has moved quickly release to release. The durable ideas are the ones that carry over unchanged from the GPU post: tiles over threads, measured search over hand-tuning, and moving as little data as possible for the arithmetic you actually need to do.*
