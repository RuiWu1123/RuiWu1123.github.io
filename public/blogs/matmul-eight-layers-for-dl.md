---
title: "Eight Layers Down: A Matrix Multiply from Python to the GPU and Back"
date: "2026/7/27"
---

Write a line like `C = A @ B` in ordinary training code, and it looks like a single, instantaneous thing: one statement, one operation, one result. It isn't. Between that line being interpreted and a real number landing in memory as part of `C`, the request passes through a stack of distinct layers, four of them on the CPU, one at the boundary, three on the GPU, each with a specific, narrow job. This post walks down that stack one layer at a time, using one small example with real numbers the whole way through:

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

At this layer, `C = A @ B` isn't doing arithmetic yet. `A` and `B` are objects carrying a shape, a numeric type, and a note about which device they live on; the line is a request phrased in those terms, handed to whatever machinery sits underneath the object:

```python
# what this layer actually does — describe, don't compute
request = describe_operation("matmul", operands=(A, B))
# nothing has been read, multiplied, or added yet
```

This layer's entire job is to describe the computation, not perform it.

## 2. Framework dispatch: which implementation family applies

The framework looks at what it was just handed — the device the operands live on, their numeric type, their shapes — and decides which family of implementation is even applicable:

```python
if request.device == "gpu" and request.dtype == "float32":
    backend = gpu_float32_matmul
```

A matrix multiply meant to run on this kind of hardware, with this kind of number, is a different piece of code entirely from one meant for different hardware or a different numeric type, even though both are conceptually "matrix multiply." This layer's job is routing, nothing more: send the request toward the right family of implementation.

## 3. Kernel selection: which specific recipe fits this shape

Within `gpu_float32_matmul`, there usually isn't just one implementation of matrix multiply — there are several, each tuned for a different range of shapes:

```python
kernel = pick_recipe(A.shape, B.shape)
# recipe_small:  tuned for shapes around 4x3 @ 3x2  <- picked here
# recipe_large:  tuned for shapes thousands of times bigger
```

`recipe_large` would produce the exact same correct answer if it were used on our tiny example instead — it just carries setup cost that only pays off on much bigger matrices. This layer's job is picking the specific recipe suited to these exact shapes.

## 4. Launch configuration: how the work gets divided up

This is where the shape of the problem starts mattering directly, not just as metadata. `C` has 4×2 = 8 elements, and — this is the layer's central insight — each of those 8 elements can, in principle, be computed completely independently of the other 7:

```python
work_items = [(i, j) for i in range(4) for j in range(2)]
# (0,0) (0,1) (1,0) (1,1) (2,0) (2,1) (3,0) (3,1)
launch_plan = group_for_hardware(work_items)
# e.g. grouped into a couple of small batches, sized to how the GPU likes to receive work
```

This layer's job is deciding how to carve those 8 independent pieces of work into groups, and packaging that plan up alongside the kernel chosen in layer 3.

## 5. Driver / queue: the CPU hands off

The launch plan and kernel from the layers above get translated into a request the driver places into a queue belonging to the GPU:

```python
instruction = build_instruction(kernel, launch_plan)
gpu_queue.enqueue(instruction)
return   # the CPU does not wait for this to be processed
```

This is the boundary the whole stack has been building toward: everything above this layer is software running on the CPU, describing a computation; everything below it is the computation actually happening. The CPU's involvement ends here for now — it returns control immediately, free to dispatch whatever comes next.

## 6. GPU scheduling: work meets compute units

The GPU has its own scheduling logic, independent of the CPU, that pulls this request off its queue and hands the work items packaged in layer 4 out to its many on-board compute units, based on which of them are currently free:

```python
# GPU side, running independently of the CPU
instruction = gpu_queue.pop()
free_units = available_compute_units()
assign(instruction.work_items, free_units)   # all 8 (i, j) pairs handed out
```

For our tiny 8-element example this happens almost instantly. For a matrix multiply with millions of independent elements, this is the layer doing the real work of spreading that flood of independent pieces across all the hardware available to chew on it.

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

Every other thread is doing exactly this kind of multiply-and-add sequence, on a different row/column pair, at the same time — one thread for `C[0][0]`, reading row 0 and column 0; one for `C[3][1]`, reading row 3 and column 1; and so on, all 8 running concurrently. This is the reason matrix multiply is such a natural fit for this kind of hardware in the first place: the underlying math doesn't need to be forced into independent pieces through some clever trick. It's already, structurally, a large pile of identical, independent work — one dot product per output element — and that's exactly the shape of workload this hardware exists to run.

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

Four layers of CPU-side software, one boundary layer, three layers of GPU-side execution, and a return path that only gets walked if something downstream actually asks for the answer. All eight of them are involved in a single `C = A @ B`, even for an example this small, where the real arithmetic is eight dot products of length three — next to nothing. The layers above the boundary cost roughly the same regardless of whether the matrices involved are tiny or enormous, which is a large part of why real training code tries to make each individual dispatched operation cover as much actual computation as possible: the eight-layer trip is worth taking once for a computation that keeps a GPU busy for a long time, and much less worth taking, over and over, for one that doesn't. (The [GPU field guide](#/blog?id=gpu-field-guide-for-dl) goes further into what layers 6 and 7 look like once the workload is not eight elements but many millions.)
