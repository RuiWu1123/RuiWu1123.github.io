---
title: "Disk to Disk: The Full Lifecycle of an LLM Pretraining Run"
date: "2026/7/27"
---

Most explanations of large-model training live entirely on one side of a line. Material about the training algorithm itself stays inside the model: forward pass, loss, gradients, an update rule, none of which says anything about where the underlying computation actually happens. Material about the hardware and systems side stays below the model: memory, interconnects, kernel scheduling, none of which says anything about how a batch of text got there in the first place. This post is about the line in between, and about connecting both ends of it: what actually happens, in order, starting from a raw text file sitting on disk and ending with trained parameters written back to a different file on disk. It deliberately leaves out specific frameworks, specific hardware, and specific numbers — all of that changes every year. What doesn't change is the shape of the pipeline itself, and that's what this post is about.

![The full loop: disk to disk](blogs/images/pretraining-lifecycle-loop.svg?v=1)

## 1. From raw text on disk to a batch of token ids

Training data starts as files sitting in storage — plain text, at whatever scale the run calls for. Before any of it reaches a GPU, a CPU-side process reads chunks of that raw text and runs it through a tokenizer: a fixed mapping from text into a sequence of integer ids drawn from a fixed vocabulary. Those id sequences get trimmed or joined to a fixed length, and a fixed number of them get stacked together into a single batch — a two-dimensional grid of integers, some number of sequences by some sequence length.

None of this touches a GPU. It's disk I/O and CPU-side string processing, and it's slow relative to what comes next, which is exactly why it doesn't happen just-in-time. In practice this step runs continuously in the background, staying one or more batches ahead of what training actually needs at any given moment, so a freshly prepared batch is already sitting in CPU memory by the time the previous one has been consumed. The structural point that matters for the rest of this post: batch preparation is a CPU-and-storage-bound process that runs *concurrently with*, not sequentially before, everything described below.

## 2. Getting a batch onto the GPU, and where the model already lives

A batch sitting in CPU (host) memory can't be computed on directly by a GPU; it has to be physically copied into the GPU's own (device) memory first, over whatever connects the two. That copy happens once per batch, every single step, for the entire duration of training — it's a small, repeated tax paid every time new data needs to reach the GPU.

Model parameters follow a completely different pattern. Before training starts, they're initialized (via some randomization scheme) and copied onto the GPU's device memory exactly once. From that point on, they stay resident there for the whole run — modified in place, step after step, never re-transferred from the host the way batch data is. (If the model is too large for one GPU's memory to hold at all, that's a real constraint, and section 5 comes back to it.) The two data flows moving through GPU memory during training look nothing alike: batch data arrives fresh every step, gets consumed, and is discarded; parameters arrive once and sit there, mutated in place, for the entire run.

## 3. The bridge: what a line of training code actually does

Here's the puzzle this section resolves. A line like `output = compute(a, b)`, sitting inside ordinary training code, executes on the CPU — the Python interpreter runs that line, on the CPU, like any other line of code. And yet the actual computation is supposed to happen on the GPU. Those two facts only fit together once you stop thinking of that line as *performing* a computation and start thinking of it as *dispatching* one.

![CPU dispatches, GPU executes asynchronously](blogs/images/cpu-dispatch-gpu-execute.svg?v=1)

Three things happen, in order, entirely on the CPU side, before any actual computation occurs anywhere: the framework looks at that high-level operation and picks the specific low-level implementation appropriate for it; a driver-layer call packages that low-level operation into an instruction; and that instruction gets placed into a queue that belongs to the GPU. Critically, the CPU does not sit and wait for that instruction to finish. It returns control immediately and moves on to dispatching whatever comes next, unless something later in the program actually needs the result right away.

```python
def training_step(batch, model):
    output = model.forward(batch)   # dispatched — this call returns before the GPU is done
    loss = loss_fn(output, batch.labels)
    loss.backward()                 # also dispatched, not executed here
    optimizer.step()                # also dispatched
    # by the time this function returns, the CPU has issued a long list of
    # instructions into the GPU's queue; the GPU may still be working through it
```

The GPU side runs independently and asynchronously: it pulls instructions off its own queue in order, and for each one, expands the underlying computation into an enormous number of parallel threads, spread across the many independent compute units the GPU has on board, and runs them concurrently. (The [GPU field guide](#/blog?id=gpu-field-guide-for-dl) covers what happens once an instruction actually reaches the hardware, in far more depth than this post needs.)

The mental model worth keeping is two separate timelines running at once. The CPU timeline is a fast sequence of dispatch actions. The GPU timeline is a comparatively slower backlog of real computation, worked through from a queue. As long as the CPU keeps dispatching faster than the GPU empties that backlog, the two stay decoupled, and the GPU is never left idle waiting on the CPU to hand it the next thing to do. This is the entire mechanism that lets "the code running" and "the computation happening" be two different events, at two different times, on two different pieces of hardware.

## 4. One training step: forward, backward, update

Every compute-heavy line inside a training step is exactly the dispatch-then-async-execute pattern from section 3, chained together a very large number of times. What ties the individual dispatches together into "one training step" is a strict order, enforced by a hard data dependency: forward has to produce its output before backward can compute anything, and backward has to produce gradients before the update step can use them.

Forward pushes the batch through the model's sequence of operations and produces a prediction, and from that prediction, a loss value. Mechanically, this is layer after layer of the same handful of operation types, repeated at scale — mostly large matrix multiplications, plus a few other structured operations — each one dispatched and executed exactly as described in section 3.

Backward computes how much each parameter contributed to that loss: the gradients. As forward runs, the framework keeps track of which computation depended on which; backward replays that dependency structure in reverse, dispatching essentially the same kind of large matrix computation as forward, just propagating in the opposite direction through the model.

The update step then uses those gradients to nudge every parameter by some small amount. This is a much smaller amount of total computation than forward or backward, but it goes through the identical dispatch mechanism — just applied to every parameter directly, rather than to activations flowing through the network.

None of this requires walking through the underlying math to understand what's structurally going on: a training step isn't three different kinds of thing happening. It's the same CPU-dispatch/GPU-execute mechanism from section 3, applied an enormous number of times, to computation that happens to fall into three phases with a hard order between them.

## 5. When one GPU isn't enough

Model parameters, the batch of data for a single step, and the intermediate activations produced while forward runs all have to fit in GPU memory simultaneously, and all three grow as model size or data scale grows. Past some point they simply stop fitting on one GPU, and there's no way around splitting something — parameters, data, or activations — across multiple GPUs, and eventually across multiple machines. (The [distributed training post](#/blog?id=distributed-training-for-dl) goes through the specific ways this splitting happens in detail; real training runs typically combine more than one of them at once.)

What matters here is what splitting does to the picture built up in sections 3 and 4: each GPU still runs its own copy of the exact same dispatch-then-execute loop, on whatever slice of parameters, data, or activations it was assigned. Most of the time, one GPU's CPU-side process is dispatching instructions and that GPU is executing them, completely independently of what every other GPU is doing.

But not always. At specific points, determined by exactly what got split and how, one GPU's next computation depends on a result that currently lives on a different GPU. At that point, the GPUs involved have to communicate — send their own partial results to each other, combine them — before any of them can proceed. That communication might be between GPUs sharing the same machine, or between GPUs on entirely separate machines connected over a network; the specifics differ, but the structural role is identical either way.

The resulting rhythm of a multi-GPU training step is independent computation, then a synchronization point, then independent computation again, repeating as many times within a single step as the particular splitting scheme requires. Nothing about sections 3 and 4 changes underneath this — it's the same mechanism, just running in parallel across many GPUs, with mandatory handshakes wherever the computation itself requires one.

## 6. The loop

Everything in sections 1 through 5 describes a single step: one batch, prepared on the CPU, moved onto the GPU, run through forward, backward, and update via CPU-dispatched, GPU-executed instructions, with synchronization across GPUs wherever the work was split. Training a model is exactly this sequence, repeated an enormous number of times — tens of thousands of steps is typical.

This is why section 1's pipelining detail matters structurally, not just as an efficiency footnote. The loop only keeps moving if a freshly prepared batch is sitting ready the moment the previous step's computation finishes, which means the CPU-side reading, tokenizing, and batching process has to run continuously — in step with, but decoupled from, the GPU-side computation loop — for the entire duration of training.

## 7. Closing the loop: back to disk

GPU memory isn't persistent. If whatever process is managing the GPU ever stops, for any reason, whatever is currently sitting in that memory is gone. So periodically, the current state of the parameters — along with whatever additional bookkeeping is needed to resume training correctly — gets copied back: GPU memory to CPU memory, and from CPU memory out to persistent storage. This is a checkpoint, and structurally it's the exact reverse of the transfer described in section 2, moving in the opposite direction, at a much lower frequency than every step.

Training keeps looping through sections 3 to 6 until some predetermined stopping condition is reached — a fixed number of steps, a fixed amount of data consumed, or some other target. At that point, the same GPU-to-CPU-to-disk movement that produces every intermediate checkpoint happens one final time, and what gets written out is the actual product of the entire process: the trained model's parameters, sitting in a file on disk.

That closes the loop this post opened with. Training starts with raw text sitting in a file on disk and ends with parameters sitting in a different file on disk. Everything in between — tokenizing, batching, moving data onto a GPU, dispatching instructions from the CPU, executing them asynchronously as massively parallel computation on the GPU, keeping many GPUs in step with each other whenever the work is split — is one loop, run tens of thousands of times, until it produces the file the entire process existed to create.
