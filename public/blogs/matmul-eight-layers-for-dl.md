---
title: "Eight Layers Down: A Matrix Multiply from Python to the GPU and Back"
date: "2026/7/27"
---

Write a line like `C = A @ B` in ordinary training code, and it looks like a single, instantaneous thing: one statement, one operation, one result. It isn't. Between that line being interpreted and a real number landing in memory as part of `C`, the request passes through a stack of distinct layers, four of them on the CPU, one at the boundary, three on the GPU, each with a specific, narrow job — and, running through all of them, one throughline worth tracking on its own: which of these layers are written in an interpreted, high-level language, and which are written in a compiled one, and why that split exists where it does. This post walks down the stack one layer at a time, using one small example with real numbers the whole way through:

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

`A` is 4×3, `B` is 3×2, `C` is 4×2. Every layer below reuses these exact numbers, and section 7 traces one specific output element, `C[2][1] = 17`, all the way down to individual multiply-and-add steps.

![Eight layers, one matrix multiply](blogs/images/matmul-eight-layers.svg?v=1)

## 1. Python: the line itself

At this layer, `C = A @ B` hasn't done any arithmetic. `A` and `B` are objects carrying a shape, a numeric type, and a note about which device they live on. The line itself is written in an interpreted, high-level language — the kind of language training code is written in specifically because it's fast to write and easy to read, not because it's fast to run. That tradeoff is fine for this one line, but it stops being fine the moment something needs to happen a huge number of times over the course of a training run, which is exactly the situation starting at the next layer. Every layer from here on is written in a compiled systems language instead, for one concrete reason: the decisions those layers make happen on the order of hundreds of thousands of times over a training run, and if each of those decisions had to be made by an interpreter reading code line by line, that overhead alone would start to compete with the actual computation for time.

## 2. Framework dispatch: a lookup table, not a chain of if-else

The framework has to decide which concrete implementation applies, based on the device `A` and `B` live on and their numeric type. It is tempting to picture this as a long chain of `if device == "gpu" and dtype == "float32": ...`, but that's not how it's actually built, for a good reason: real frameworks support dozens of operations across several device types and several numeric types, and a hand-written if-else chain covering every combination would be both enormous and slow to walk through on every single call.

What real dispatch mechanisms use instead is closer to a lookup table, built once, ahead of time:

```python
# built once, at framework build/init time — not written by hand per call
dispatch_table = {
    ("matmul", "gpu", "float32"): gpu_float32_matmul,
    ("matmul", "cpu", "float32"): cpu_float32_matmul,
    ("matmul", "gpu", "float16"): gpu_float16_matmul,
    # ... one entry generated for every operation × device × dtype combination
}
backend = dispatch_table[(request.op, request.device, request.dtype)]
```

A table lookup like this costs roughly the same regardless of how many combinations exist, which is the entire reason it beats a sequential if-else chain at this scale. And this layer, like every layer below it, lives in compiled code: the same per-call-frequency argument from section 1 applies here directly — this lookup happens on every dispatched operation, and an interpreter re-reading dispatch logic that often would be paying real, avoidable overhead.

## 3. Kernel selection: a library, a fallback, and sometimes a compiler

`backend` doesn't point to one single implementation of matrix multiply — it points to an entire kernel library: several precompiled routines, each tuned for a different range of shapes. Picking the right one for this specific call is not, in general, a runtime bake-off where every candidate kernel actually gets run and timed — for a call this small, that would cost more than the computation itself. What real systems do instead is some mix of three strategies. Most commonly, a fast heuristic — a shallow lookup keyed on shape ranges, tuned ahead of time by the library's own developers through offline profiling — picks a kernel directly. Some systems add a caching layer on top: the first time a particular shape is seen, a handful of candidates actually get benchmarked, and the winner is remembered so later calls with the same shape skip straight to it. And some systems skip the pre-shipped library altogether for a given shape and instead compile a new kernel specialized to it on the spot — paying a one-time compilation cost the first time, then caching the result exactly like the previous strategy would.

What happens if none of that produces a specialized match? Every serious kernel library ships a generic fallback: an implementation correct for any valid shape, tuned for none of them. If nothing more specific is found, execution falls back to it — correctness is never in question, only how efficiently the specific shape gets handled. Our tiny 4×3-by-3×2 example is realistically far below the size most tuning effort targets; it's a completely ordinary thing for a call this small to land on the generic fallback rather than a shape-specialized kernel, and that isn't a failure case, it's this layer working as designed.

```python
kernel = kernel_library.lookup(A.shape, B.shape)
if kernel is None:
    kernel = kernel_library.generic_fallback   # always correct, not shape-tuned
```

And the kernel itself — whichever one gets picked — is never written in the interpreted language the training script uses. A GPU's compute units execute their own native instruction set, and kernel code is written in, or compiled down into, a C-like language built specifically for describing what one thread among many should do. Some newer tools let a kernel be written in syntax that reads like the high-level scripting language, but that's a surface convenience: before any of it runs, it's compiled into the GPU's native instructions ahead of time, the same as kernel code written any other way. Nothing about a kernel executes by being interpreted line by line while it runs.

## 4. Launch configuration: how the work gets divided up

This is where the shape of the problem starts mattering directly, not just as metadata. `C` has 4×2 = 8 elements, and — this is the layer's central insight — each of those 8 elements can, in principle, be computed completely independently of the other 7:

```python
work_items = [(i, j) for i in range(4) for j in range(2)]
# (0,0) (0,1) (1,0) (1,1) (2,0) (2,1) (3,0) (3,1)
launch_plan = group_for_hardware(work_items)
# e.g. grouped into a couple of small batches, sized to how the GPU likes to receive work
```

This layer's job is deciding how to carve those 8 independent pieces of work into groups, and packaging that plan up alongside the kernel chosen in layer 3.

## 5. Driver / queue: the launch descriptor, not the kernel

It's easy to picture this step as shipping the whole kernel over to the GPU on every call, but that's not what happens, and the distinction matters. The kernel's compiled code is already resident on the GPU — loaded once, typically the first time it's needed, and kept there — so what actually travels per call is a small launch descriptor: which already-loaded kernel to run, plus this call's specific arguments (where `A`, `B`, and `C` live in GPU memory, and any shape parameters the kernel needs, like the loop bound `3` in our dot-product example).

![The launch descriptor, not the kernel itself](blogs/images/matmul-queue-doorbell.svg?v=1)

How that descriptor physically gets there: driver code — compiled, for the same reason every layer below section 1 is compiled, plus a second reason specific to this layer, that it has to manipulate raw memory addresses and hardware registers directly, which an interpreted language generally isn't built to do safely — writes the descriptor into a queue shared between CPU and GPU, typically implemented as a ring buffer: a fixed block of memory, reused in a loop, that both sides can see. Writing the descriptor into the next open slot isn't enough by itself; the driver also has to tell the GPU a new entry exists, commonly by writing to a specific hardware register that acts as a doorbell — the GPU's own front-end hardware notices that write and knows to go check the queue. Once that happens, the CPU is done. It doesn't wait for the entry to be processed; it returns control immediately.

```python
launch = {
    "kernel_id": kernel.id,                       # which already-loaded kernel to run
    "args": [A.gpu_ptr, B.gpu_ptr, C.gpu_ptr, shape_params],
}
command_queue.write(launch)   # into a shared ring buffer
gpu_doorbell.ring()           # tell the GPU: new work is waiting
return                        # the CPU does not wait for this to be processed
```

This is the boundary the whole stack has been building toward. Everything above this layer is compiled software running on the CPU, describing a computation; everything below it is the computation actually happening.

## 6. GPU scheduling: work meets compute units

The GPU's own front-end hardware, having noticed the doorbell, pulls the descriptor off the queue and hands the work items packaged in layer 4 out to its many on-board compute units, based on which of them are currently free. For our tiny 8-element example this happens almost instantly. For a matrix multiply with millions of independent elements, this is the layer doing the real work of spreading that flood of independent pieces across all the hardware available to chew on it.

## 7. Compute unit execution: threads doing dot products

This is where the actual arithmetic happens. Each of `C`'s 8 elements is exactly one row of `A` dotted with one column of `B` — 8 completely independent dot products, sharing no data dependency with each other.

![Which numbers feed which output cell](blogs/images/matmul-thread-parallel.svg?v=2)

Take the thread responsible for `C[2][1]`. It reads row 2 of `A`, `[7, 8, 9]`, and column 1 of `B`, `[0, 1, 1]`, out of memory, and walks through the multiply-and-add sequence a dot product requires:

```python
# thread responsible for output element C[i][j]
acc = 0
for k in range(3):
    acc += A[i][k] * B[k][j]
C[i][j] = acc

# concretely, for i=2, j=1:
# acc = 0
# acc += A[2][0] * B[0][1]  =  7 * 0  =  0   -> acc = 0
# acc += A[2][1] * B[1][1]  =  8 * 1  =  8   -> acc = 8
# acc += A[2][2] * B[2][1]  =  9 * 1  =  9   -> acc = 17
# C[2][1] = 17
```

![Inside one thread: the arithmetic trace](blogs/images/matmul-dot-product-trace.svg?v=1)

Every other thread is doing exactly this kind of multiply-and-add sequence, on a different row/column pair, at the same time — one thread for `C[0][0]`, reading row 0 and column 0; one for `C[3][1]`, reading row 3 and column 1; and so on, all 8 running concurrently. This is the reason matrix multiply is such a natural fit for this kind of hardware in the first place: the underlying math doesn't need to be forced into independent pieces through some clever trick. It's already, structurally, a large pile of identical, independent work — one dot product per output element.

## 8. Completion, and the way back

Once every thread has finished writing its element, `C` exists, complete, sitting in GPU memory:

```python
# GPU memory now holds:
# C = [[4, 5], [10, 11], [16, 17], [22, 23]]
```

But that completion, by itself, doesn't do anything — nothing forces the CPU to notice or wait for it. If the next line of training code needs `C` for another GPU computation, nothing has to happen at this layer at all: that next operation just gets dispatched the same way, chained onto this one, with the GPU working through both in order from its queue. The CPU only actually stops and waits here if something concrete requires it:

```python
print(C[2][1])
# this is where the CPU actually blocks — waits for the kernel to finish,
# then copies 17 back from GPU memory before Python can treat it as a plain number
```

Because the result lives in GPU memory, satisfying that `print` means one more transfer — copying the relevant piece of `C` back to CPU memory — before Python can use it as an ordinary number again.

## What the eight layers add up to

Line them up by language and the split is exactly one boundary wide: layer 1 is interpreted, layers 2 through 7 are compiled — either compiled systems code on the CPU side, or code compiled down to the GPU's own native instructions on the GPU side — and layer 8 only crosses back into interpreted territory if, and exactly when, something downstream actually asks for a real number. That split exists for a single, consistent reason repeated at every layer: interpretation is fine for something that happens once, and increasingly not fine for something that happens hundreds of thousands of times over a training run, which is what everything past layer 1 does. The layers above the boundary in section 5 cost roughly the same regardless of whether the matrices involved are tiny or enormous, which is a large part of why real training code tries to make each individual dispatched operation cover as much actual computation as possible: the eight-layer trip is worth taking once for a computation that keeps a GPU busy for a long time, and much less worth taking, over and over, for one that doesn't. (The [GPU field guide](#/blog?id=gpu-field-guide-for-dl) goes further into what layers 6 and 7 look like once the workload is not eight elements but many millions.)
