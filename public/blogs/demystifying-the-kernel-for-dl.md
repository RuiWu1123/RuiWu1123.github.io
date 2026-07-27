---
title: "Demystifying the Kernel: What It Is, How It's Chosen, and Why Some Are Faster"
date: "2026/7/27"
---

"Dispatch it to a kernel" is one of those phrases that's easy to nod along to without actually picturing anything concrete. This post opens that phrase up: what a kernel actually looks like as a piece of code, how one specific kernel gets chosen out of several candidates for a given call, what happens once it's chosen, and — the part that matters most in practice — why two kernels that compute the exact same correct result can run at very different speeds. One small example with real numbers runs through the whole thing:

```python
A = [[1,  2,  3],
     [4,  5,  6],
     [7,  8,  9],
     [10, 11, 12]]

B = [[1, 0],
     [0, 1],
     [1, 1]]

C = A @ B
# C = [[4, 5], [10, 11], [16, 17], [22, 23]]
```

`A` is 4×3, `B` is 3×2, `C` is 4×2, and the output element `C[2][1] = 17` gets traced all the way through.

## 1. What a kernel actually is

A kernel is a function, but it's written from an unusual point of view: not "loop over every output element," but "here's what one thread does, given its own position." Here's a kernel that computes our matrix multiply, in generic pseudo-syntax for a C-like GPU kernel language:

```c
// every thread executes this same function
kernel void matmul_dot(float* A, float* B, float* C, int M, int N, int K) {
    int i = this_thread.row;      // which output row this thread owns
    int j = this_thread.col;      // which output column this thread owns

    float acc = 0;
    for (int k = 0; k < K; k++) {
        acc += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = acc;
}
```

Two things about this are worth stopping on. First, there's no loop over `i` and `j` anywhere in the function body — the "loop" over all 8 output elements doesn't exist in the code at all. It exists as 8 separate launches of this same function, each with a different `(i, j)` baked into `this_thread`, running concurrently. Second, `A`, `B`, and `C` aren't the nice nested-list objects from the Python layer anymore; they're flat blocks of memory, and turning a 2D position into a memory address is the kernel's own job, done by hand: `A[i * K + k]` is how you address "row `i`, column `k`" of a matrix stored as one long strip of `M × K` numbers.

![Which numbers feed C[2][1]](blogs/images/matmul-thread-parallel.svg?v=2)

Run this function with `i=2, j=1`, and it reads row 2 of `A`, `[7, 8, 9]`, and column 1 of `B`, `[0, 1, 1]`, out of memory, then works through exactly this trace:

```c
// i=2, j=1, K=3:
// acc = 0
// acc += A[6] * B[1]  =  7 * 0  =  0    (A[6] is A[2][0], B[1] is B[0][1])   -> acc = 0
// acc += A[7] * B[3]  =  8 * 1  =  8    (A[7] is A[2][1], B[3] is B[1][1])   -> acc = 8
// acc += A[8] * B[5]  =  9 * 1  =  9    (A[8] is A[2][2], B[5] is B[2][1])   -> acc = 17
// C[5] = 17   (C[5] is C[2][1], since C is 4x2 and i*N+j = 2*2+1 = 5)
```

![Inside one thread: the arithmetic trace](blogs/images/matmul-dot-product-trace.svg?v=1)

This code is written in, or compiled down into, a compiled, C-like language — never the interpreted language the outer training script uses. The GPU's compute units execute their own native instruction set; nothing about a kernel runs by being read and interpreted line by line while it's executing.

## 2. How this specific kernel gets chosen

Before this section even starts, the framework already knows it wants *some* GPU implementation of matrix multiply for float32 inputs — that's a separate routing decision, made by matching the operands' device and numeric type against a lookup table. What this section is about is narrower and more interesting: which specific implementation of "GPU float32 matmul," out of several that exist.

The function from section 1 is a real, correct kernel — but it's not the only one that computes a matrix multiply, and it's not necessarily the one that gets picked. Behind a single operation like matmul sits an entire library of kernels, each tuned for a different range of shapes, and picking the right one for this specific call is not, in general, a runtime bake-off where every candidate actually gets run and timed — for a call this small, that comparison would cost more than the computation itself.

Real systems use some mix of three strategies instead. Most commonly, a fast heuristic — a shallow lookup keyed on shape ranges, tuned ahead of time by the library's own developers through offline profiling — picks a kernel directly. Some systems add a caching layer on top: the first time a particular shape is seen, a handful of candidates actually get benchmarked, and the winner is remembered so later calls with the same shape skip straight to it. And some systems compile a new kernel specialized to a given shape on the spot instead of choosing among a pre-shipped set — a one-time compilation cost the first time, cached afterward the same as the previous strategy.

```python
kernel = kernel_library.lookup(A.shape, B.shape)
if kernel is None:
    kernel = kernel_library.generic_fallback   # always correct, not shape-tuned
```

What happens if none of that produces a specialized match? Every serious kernel library ships a generic fallback: correct for any valid shape, tuned for none of them. Our tiny 4×3-by-3×2 example is realistically far below the size most tuning effort targets — it's completely ordinary for a call this small to land on the generic fallback rather than a shape-specialized kernel. Hold onto that fact; section 4 comes back to exactly what "not shape-tuned" costs.

## 3. Once chosen, how it actually takes effect

It's tempting to picture "using" a kernel as shipping its code to the GPU on every call, but that's not what happens. The kernel's compiled code is already resident on the GPU — loaded once, typically the first time it's needed, and kept there. What actually travels per call is a small launch descriptor: which already-loaded kernel to run, plus this call's specific arguments — where `A`, `B`, and `C` live in GPU memory, and the shape parameters the kernel needs, like `M=4, N=2, K=3` from section 1's function signature.

![The launch descriptor, not the kernel itself](blogs/images/matmul-queue-doorbell.svg?v=1)

That descriptor gets written into a queue shared between CPU and GPU — typically a ring buffer, a fixed block of memory both sides can see, reused in a loop. Writing it into the next open slot isn't enough on its own; something also has to tell the GPU a new entry exists, commonly done by writing to a specific hardware register acting as a doorbell. The GPU's own front-end hardware notices that write, pulls the descriptor off the queue, and hands the work out to its many on-board compute units based on which are currently free. From there, what "taking effect" means is exactly section 1's function, launched many times over — once per `(i, j)` pair, 8 times for our example — all running concurrently, each with its own thread reading its own slice of `A` and `B` and writing its own single number into `C`.

## 4. Why some kernels are faster than others

Section 1's kernel is correct. It is also, deliberately, a bad kernel — worth using precisely because it makes the next point easy to see with real numbers.

Look again at the two threads computing `C[2][0]` and `C[2][1]`. Both of them need row 2 of `A`, `[7, 8, 9]` — the same three numbers, from the same three memory locations. Section 1's kernel doesn't know that. Each thread runs its own independent copy of the function, each one reads `A[6]`, `A[7]`, `A[8]` on its own, straight from slow memory, with no awareness that a neighboring thread just fetched — or is about to fetch — the exact same values.

![Why reuse beats redundant reads](blogs/images/matmul-kernel-reuse.svg?v=1)

At this tiny scale that's one wasted read out of two — not much. At real matrix sizes it stops being a rounding error: in this same naive scheme, row 2 of `A` gets independently re-read once for every column of `B`, and column 1 of `B` gets independently re-read once for every row of `A`. The total memory traffic this generates grows much faster than the actual amount of arithmetic does, and moving data is, on most hardware most of the time, considerably more expensive than the arithmetic performed on it once it arrives.

A faster kernel is written around exactly this observation. Instead of every thread fending for itself, threads are grouped so that the group cooperatively loads a shared chunk of `A` and `B` into fast, on-chip memory once, and every thread in that group computes its own output element by reading repeatedly from that fast, shared copy — not from slow memory again. The total arithmetic is identical to section 1's kernel; what changes is how much data has to move to support it. This is the entire content of "some kernels are faster than others" for an operation like matrix multiply: not a different algorithm, the same one, executed so that data gets reused instead of re-fetched.

This is also the real answer to the question section 2 left hanging. A generic, not-shape-tuned fallback kernel is usually exactly this: correct, but written the section-1 way, with no attempt at reuse, because a kernel that has to handle any shape at all can't assume much about how to group threads for reuse ahead of time. A shape-specialized kernel earns its spot in the library specifically because, for that range of shapes, it was written to reuse data the section-1 way's naive version throws away. (This is a specific case of a general framework — arithmetic intensity, and whether an operation is memory-bound or compute-bound — that the [GPU field guide](#/blog?id=gpu-field-guide-for-dl) covers in full in its section on the roofline model; this post has been one worked example of exactly what that framework is measuring.)

## What this adds up to

A kernel is a function written from one thread's perspective, operating on raw memory addresses it computes by hand, compiled down to the GPU's native instructions. Which kernel runs for a given call is decided by a library, a heuristic or a cache or a just-in-time compiler, and a generic fallback for anything none of those match. Once chosen, "using" it means sending a small descriptor to code that's already resident on the GPU, not the code itself. And whether it's fast comes down to one thing more than any other: whether it was written to reuse the data it loads, or to fetch the same numbers from slow memory again every time a different thread happens to need them too.
