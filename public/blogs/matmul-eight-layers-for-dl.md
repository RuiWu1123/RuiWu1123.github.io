---
title: "Eight Layers Down: A Matrix Multiply from Python to the GPU and Back"
date: "2026/7/27"
---

Write a line like `C = A @ B` in ordinary training code, and it looks like a single, instantaneous thing: one statement, one operation, one result. It isn't. Between that line being interpreted and a real number landing in memory as part of `C`, the request passes through a stack of distinct layers, four of them on the CPU, one at the boundary, three on the GPU, each with a specific, narrow job. This post walks down that stack, one layer at a time, using one small, concrete example the whole way through: `A` is a 4×3 matrix, `B` is a 3×2 matrix, and `C = A @ B` is the 4×2 result.

![Eight layers, one matrix multiply](blogs/images/matmul-eight-layers.svg?v=1)

## 1. Python: the line itself

At this layer, `C = A @ B` isn't doing arithmetic yet. `A` and `B` are objects that carry a shape, a numeric type, and a note about which device they live on; the line itself is a request phrased in those terms — "multiply these two things" — handed to whatever machinery sits underneath the object. Nothing has been read, multiplied, or added. This layer's entire job is to describe the computation, not perform it.

## 2. Framework dispatch: which implementation family applies

The framework looks at what it was just handed — the device the operands live on, their numeric type, their shapes — and decides which family of implementation is even applicable. A matrix multiply meant to run on this kind of hardware, with this kind of number, is a different piece of code entirely from one meant for different hardware or a different numeric type, even though both are conceptually "matrix multiply." This layer's job is routing: send the request toward the right family of implementation, nothing more.

## 3. Kernel selection: which specific recipe fits this shape

Within the family this dispatch settled on, there usually isn't just one implementation of matrix multiply — there are several, each tuned for a different range of shapes. One version of the recipe might be efficient for something the size of our 4×3-by-3×2 example; a different version of the same recipe, built for matrices thousands of times larger, would produce the same correct answer if used here, just less efficiently. This layer's job is picking the specific recipe suited to these exact shapes.

## 4. Launch configuration: how the work gets divided up

This is where the shape of the problem starts mattering directly, not just as metadata. `C` has 4×2 = 8 elements, and — this is the layer's central insight — each of those 8 elements can, in principle, be computed completely independently of the other 7. This layer's job is to decide how to carve those 8 independent pieces of work into groups sized to match how the target GPU likes to receive work, and package that plan up alongside the chosen kernel from layer 3.

## 5. Driver / queue: the CPU hands off

The launch plan and kernel from the layers above get translated into a request the driver places into a queue belonging to the GPU. This is the boundary this whole stack has been building toward: everything above this layer is software running on the CPU, describing a computation; everything below it is the computation actually happening. And critically, the CPU's involvement ends here for now — it doesn't wait for the queue to be processed. It returns control immediately, free to go dispatch whatever comes next.

## 6. GPU scheduling: work meets compute units

The GPU has its own scheduling logic, independent of the CPU, that pulls this request off its queue and hands the pieces of work packaged in layer 4 out to its many on-board compute units, based on which of them are currently free. For our tiny 8-element example this happens almost instantly; for a matrix multiply with millions of independent elements, this is the layer doing the real work of spreading that flood of independent pieces across all the hardware available to chew on it.

## 7. Compute unit execution: threads doing dot products

This is where the actual arithmetic happens, and it's also where the concrete example earns its keep. Each of `C`'s 8 elements is exactly one row of `A` dotted with one column of `B` — 8 completely independent dot products, sharing no data dependency with each other. A thread assigned to one of them reads that row of `A` and that column of `B` out of memory, walks through the multiply-and-add sequence a dot product requires, and writes the single resulting number into `C`'s corresponding memory location. Every other thread is doing exactly the same kind of work, on a different row/column pair, at the same time.

![Why a matrix multiply parallelizes so well](blogs/images/matmul-thread-parallel.svg?v=1)

This is the reason matrix multiply is such a natural fit for this kind of hardware in the first place: the underlying math doesn't need to be forced into independent pieces through some clever trick. It's already, structurally, a large pile of identical, independent work — one dot product per output element — and that's exactly the shape of workload this hardware exists to run.

## 8. Completion, and the way back

Once every thread has finished writing its element, `C` exists, complete, sitting in GPU memory. But that completion, by itself, doesn't do anything — nothing forces the CPU to notice or wait for it. If the next line of training code needs `C` for another GPU computation, nothing has to happen at this layer at all: that next operation just gets dispatched the same way, chained onto this one, with the GPU working through both in order from its queue. The CPU only actually stops and waits here if something concrete requires it — printing a value, converting a result to an ordinary host-side number, or any other point where the program can't proceed without the real answer in hand. And because the result lives in GPU memory, satisfying that requirement means one more transfer, copying the relevant piece of `C` back to CPU memory, before Python can treat it as an ordinary number again.

## What the eight layers add up to

Four layers of CPU-side software, one boundary layer, three layers of GPU-side execution, and a return path that only gets walked if something downstream actually asks for the answer. All eight of them are involved in a single `C = A @ B`, even for an example this small, where the real arithmetic is eight dot products of length three — next to nothing. The layers above the boundary cost roughly the same regardless of whether the matrices involved are tiny or enormous, which is a large part of why real training code tries to make each individual dispatched operation cover as much actual computation as possible: the eight-layer trip is worth taking once for a computation that keeps a GPU busy for a long time, and much less worth taking, over and over, for one that doesn't. (The [GPU field guide](#/blog?id=gpu-field-guide-for-dl) goes further into what layers 6 and 7 look like once the workload is not eight elements but many millions.)
