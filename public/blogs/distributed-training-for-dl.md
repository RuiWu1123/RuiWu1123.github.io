---
title: "Distributed Training from First Principles: A Field Guide to Parallelism Strategies"
date: "2026/7/19"
---

The [first post in this series](#/blog?id=gpu-field-guide-for-dl) built a model of one GPU: warps, SMs, the memory hierarchy, the roofline model. The [second](#/blog?id=triton-for-dl) was about writing your own kernel for that one GPU. This post is about the question that shows up the moment "one GPU" stops being the unit you're working with: your model, or your batch, or your sequence length, no longer fits on a single device, and you have to decide how to split the work across many. That decision has a name — a *parallelism strategy* — and there turn out to be surprisingly few genuinely different ideas underneath the alphabet soup of DP, DDP, ZeRO, TP, PP, SP, and CP. This post builds each of them up from the specific problem it solves, in the order the field actually discovered them, and ends with how modern training runs combine four or five of these ideas at once without contradicting each other.

This post draws heavily on 猛猿's excellent *图解大模型训练* series, which remains one of the clearest treatments of this material anywhere; the diagrams and framing here are my own, built while working through that series and the source papers it's based on (GPipe, PipeDream, ZeRO, Megatron-LM, DeepSpeed Ulysses, Ring Attention).

## 1. Two things that don't fit

Before naming any strategy, it's worth being precise about what "doesn't fit" actually means, because there are two structurally different problems hiding under that one phrase, and they call for different solutions.

The first is that your **dataset, or your desired batch size, is too large to process serially in a reasonable amount of time**. The model itself fits comfortably on one GPU; you just have more examples than you want to wait for one GPU to churn through. The natural fix is to give every GPU a full copy of the model and a different slice of the data, and somehow keep all the copies in agreement — this is **data parallelism**, and it's the oldest, simplest idea in this post.

The second is that **the model itself — its parameters, its optimizer state, or the activations it produces during a forward pass — doesn't fit in one GPU's HBM at all**, no matter how small your batch is. Giving every GPU a full copy is no longer an option; there's nothing left over to make a copy *of*. The fix has to be to split the model itself across GPUs, and there are several structurally different ways to do that split, each of which trades off differently against communication cost, memory savings, and implementation complexity: **pipeline parallelism** (split by layer), **tensor parallelism** (split inside a layer's math), and **sequence/context parallelism** (split along the sequence axis, for when even a single layer's activations for one long sequence don't fit).

Every strategy in this post is an answer to one of these two problems, and the last section is about the fact that real training runs need answers to *both* at once, nested inside each other according to the physical topology from the first post's Section 7: fast, frequent communication stays within a node's NVLink domain; slow, infrequent communication is allowed to cross the network.

## 2. Data parallelism: from a bottleneck to a ring

The naive version of data parallelism looks almost too simple to need a name: every GPU holds a full copy of the model, runs forward and backward on its own slice of the batch, and then all the resulting gradients need to be averaged before anyone takes an optimizer step (otherwise the copies drift apart). The earliest implementations did this averaging through a **parameter server**: a designated node (or process) that every worker sends its gradients to, which averages them and sends the result back.

![Naive DP vs Ring-AllReduce](blogs/images/dp-ddp-ring-allreduce.svg?v=1)
^The parameter server's incoming bandwidth is shared across every worker and gets worse as you add more GPUs. Ring-AllReduce restructures the same computation so that no single node is ever a bottleneck.

The parameter-server design has an obvious flaw: the server's network link is shared across every single worker, so its traffic grows linearly with the number of GPUs, and it eventually becomes the bottleneck the entire cluster waits on. **DDP** (PyTorch's `DistributedDataParallel`, and the standard approach almost everyone uses today) replaces this with **Ring-AllReduce**, an algorithm with a genuinely elegant property: every participant only ever talks to its two neighbors in a logical ring, and the total data moved per GPU is *independent of how many GPUs are in the ring*.

The algorithm runs in two phases. In the **scatter-reduce** phase, each GPU's gradient buffer is split into N chunks (N = number of GPUs), and over N−1 steps, each GPU passes a chunk to its neighbor while simultaneously accumulating an incoming chunk from its other neighbor — after N−1 steps, every GPU holds one chunk that is the *fully reduced sum* across all GPUs, but only for that one chunk. In the **all-gather** phase, the same ring structure is used to circulate those N fully-reduced chunks around so that every GPU ends up with the complete, fully-reduced gradient. Each phase moves (N−1)/N of the total gradient size per GPU, for a total of **2(N−1)/N × gradient size** — which, as N grows, converges to a flat 2× the gradient size and never grows further. This is the mathematical reason DDP scales to large GPU counts in a way a parameter server structurally cannot.

![interactive:ring-allreduce](#)

## 3. ZeRO: DDP still has redundancy left to remove

Ring-AllReduce fixes the *communication* bottleneck of naive DP, but DDP as described still has a memory problem: every one of the N GPUs holds a full, redundant copy of everything needed to update the model — not just the parameters, but the optimizer's own bookkeeping. For Adam with mixed-precision training, that bookkeeping is larger than the model itself. Per parameter, mixed-precision Adam needs: an fp16 copy of the parameter (2 bytes) for fast forward/backward math, an fp16 gradient (2 bytes), and — this is the part people underestimate — an fp32 master copy of the parameter plus fp32 Adam momentum and variance terms (4+4+4 = 12 bytes) kept for numerically stable optimizer updates. Using Φ for the parameter count, that's a well-known **2Φ + 2Φ + 12Φ = 16Φ bytes** of "model state" that plain DDP replicates, in full, on *every single GPU* — a 7.5B-parameter model needs 120GB of this alone, more than a single GPU's HBM, before a single activation has been stored.

**ZeRO** (Zero Redundancy Optimizer, from Microsoft's DeepSpeed) asks the obvious question: if every GPU is going to hold an identical copy of this 16Φ bytes anyway, why not have each GPU own only a 1/N *shard* of it, and reconstruct whichever piece any given step actually needs, on demand? ZeRO ships this idea in three increasingly aggressive stages:

**ZeRO-1** partitions only the optimizer states (the largest piece, 12Φ) across the N GPUs, while still replicating parameters and gradients in full. Per-GPU memory becomes **2Φ + 2Φ + 12Φ/N** — already close to a 4× reduction at reasonable N.

**ZeRO-2** additionally partitions the gradients, since once the optimizer states are sharded, each GPU only ever needs the slice of the gradient that corresponds to the parameter slice it's actually responsible for updating. Per-GPU memory becomes **2Φ + (2Φ + 12Φ)/N = 2Φ + 14Φ/N**.

**ZeRO-3** goes all the way and partitions the parameters themselves. This is the biggest structural change: a GPU no longer holds the full parameter tensor at all, and has to **all-gather** the specific shard it needs, layer by layer, immediately before that layer's forward or backward computation, then release it again afterward. Per-GPU memory drops to **16Φ/N**, an almost-N-fold reduction — at the cost of extra all-gather communication on every layer, every step, that ZeRO-1/2 didn't need.

![ZeRO memory breakdown](blogs/images/zero-memory-breakdown.svg?v=1)
^At N=64, ZeRO-1 already gets you to roughly a quarter of DDP's memory; ZeRO-3 gets you to 16Φ/N — genuinely proportional to your GPU count, not just a fixed multiple.

None of this is free: ZeRO-3's extra communication means it's the right choice specifically when you're memory-bound and have bandwidth to spare (a fast NVLink domain), and ZeRO-1 is often the pragmatic default when your optimizer states alone are the problem. The panel below lets you pick a model size, a GPU count, and a stage, and see the actual per-GPU number — including the point where it stops fitting on a single GPU's HBM at all.

![interactive:zero-memory](#)

## 4. Pipeline parallelism: splitting the model by layer

ZeRO shards *redundant copies* of the same model; it doesn't help if the model's layers, laid out for a single forward pass, simply don't fit on one GPU even without any redundancy. The direct fix is **pipeline parallelism**: put the first several layers on GPU 0, the next several on GPU 1, and so on, so that a forward pass is a relay race across GPUs instead of a single GPU doing everything.

Done naively, this is disastrous: GPU 1 can't do anything until GPU 0 finishes the entire forward pass for the entire batch and hands it off, and the same is true, in reverse, for the backward pass — at any instant, only one of your P GPUs is doing any work at all, and the other P−1 sit idle. This idle time is called the **pipeline bubble**, and the naive version wastes almost all of your hardware.

**GPipe** (Google, 2019) fixes this with **microbatching**: split each batch into M smaller microbatches, and feed them into the pipeline one after another. While GPU 0 is running microbatch 2's forward pass, GPU 1 can already be running microbatch 1's forward pass — the pipeline fills up, and once it's full, every stage is busy simultaneously. GPipe's schedule runs *all* microbatches' forward passes through the whole pipeline first, then *all* the backward passes, which is simple to reason about but means every microbatch's activations have to be kept in memory until the corresponding backward pass finally happens.

**PipeDream**, and the **1F1B** ("one-forward-one-backward") schedule used by Megatron-LM in practice, interleave more aggressively: as soon as a microbatch's activation is no longer needed to keep the pipeline fed, its backward pass is scheduled immediately, rather than waiting for every other microbatch's forward pass to finish first. This does **not** reduce the total bubble time — it turns out to be exactly the same fraction, for a subtle but important reason covered below — but it dramatically reduces how many microbatches' activations any one stage has to hold in memory at once, which is usually the more binding constraint in practice.

![Three pipeline schedules](blogs/images/pipeline-bubble-schedules.svg?v=1)
^Naive model parallelism keeps only 1 of P GPUs busy at any moment. GPipe and 1F1B both reach the same total bubble fraction — the difference between them is peak activation memory, not wall-clock time.

The bubble fraction has a clean closed form: with P pipeline stages and M microbatches, the fraction of every GPU's time spent idle is **(P−1) / (P−1+M)**. This single formula explains most of the practical advice you'll ever read about pipeline parallelism: more microbatches always shrinks the bubble (as M→∞, the bubble fraction →0), and a deeper pipeline (larger P, needed when the model is bigger) needs proportionally more microbatches to keep the bubble small — which is exactly why pipeline parallelism is usually paired with a healthy global batch size, and struggles at very deep P with a small batch.

![interactive:pipeline-bubble](#)

## 5. Tensor parallelism: splitting the math inside one layer

Pipeline parallelism splits *which layers* live where; it says nothing about what happens when a single layer's weight matrix is itself too large to fit, or when you want parallelism at a finer grain than "whole layers" to reduce the pipeline bubble's impact. **Tensor parallelism** (popularized by **Megatron-LM**) splits the matrix multiplication *inside* a layer across GPUs.

The cleanest illustration is a transformer's MLP block, which is two linear layers with a nonlinearity between them: `Y = B(GeLU(A(X)))`. Megatron splits matrix A **by columns** across GPUs — GPU 0 gets columns 0..k of A, GPU 1 gets the rest — so each GPU can independently compute its slice of `GeLU(A(X))` with **no communication at all** (GeLU is elementwise, so it doesn't care that each GPU only has part of the intermediate tensor). Matrix B is then split **by rows**, chosen specifically so that each GPU's partial result, summed across GPUs, reconstructs the correct final output — which needs exactly **one all-reduce**, at the very end of the block.

![Megatron tensor parallelism](blogs/images/megatron-tensor-parallel.svg?v=1)
^Column-parallel first, row-parallel second: the intermediate GeLU never needs communication, and the whole block costs exactly one all-reduce.

This column-then-row pattern (Megatron calls the identity-in-forward/all-reduce-in-backward operator `f`, and the all-reduce-in-forward/identity-in-backward operator `g`) is applied the same way to attention, splitting whole heads across GPUs rather than splitting within a head. A full transformer layer — one attention block, one MLP block — needs exactly **2 all-reduces in the forward pass and 2 in the backward pass**, regardless of how many GPUs the tensor-parallel group spans.

The catch is that this communication happens on *every single layer, every single forward and backward pass* — an order of magnitude more frequently than DP's occasional gradient sync, or PP's occasional activation handoff. This is precisely why the first post's Section 7 matters here specifically: tensor parallelism needs the fast, full-bandwidth NVLink domain within a single node, and running it across the slower inter-node network is one of the most common, very concrete reasons a "linear scaling" plot stops looking linear the moment a job spans more than one node.

## 6. Sequence parallelism: filling in tensor parallelism's gaps

Tensor parallelism, as described, has a quiet inefficiency: operations like LayerNorm, dropout, and the residual add don't split cleanly along the hidden dimension the way a matrix multiply does — they need the *entire* hidden vector for each token to compute correctly. Megatron's original TP implementation handles this by simply **replicating** these operations' activations in full on every GPU in the tensor-parallel group, which means the activation memory for these regions gets *zero* benefit from tensor parallelism at all.

**Sequence parallelism** (Megatron SP) closes this gap with an observation: LayerNorm, dropout, and the residual add all operate independently *per token*, so instead of splitting along the hidden dimension (which TP does, and which these ops can't use), you can split along the **sequence** dimension instead — each GPU owns a different subset of tokens' worth of these activations, with zero redundancy at all.

![Megatron sequence parallelism](blogs/images/megatron-sequence-parallel.svg?v=1)
^SP regions (LayerNorm, dropout, residual) split by sequence position; TP regions (attention, MLP) split by hidden dimension. An all-gather and a reduce-scatter at each seam swap between the two layouts.

The seams between an SP region and a TP region use an **all-gather** (to reassemble the full sequence before a TP block that needs it) and a **reduce-scatter** (to re-shard the output back into per-sequence pieces afterward) instead of TP's all-reduce — and it's worth being precise that this is a *memory* optimization, not a *bandwidth* one: an all-gather plus a reduce-scatter moves exactly as many total bytes as one all-reduce would have. You get rid of a real memory redundancy (the replicated LayerNorm/dropout activations) for essentially the same communication bill you were already paying for tensor parallelism.

## 7. The problem TP and SP don't solve: sequences that don't fit by themselves

Combining DP, ZeRO, PP, TP, and SP is often called **3D parallelism** (or 4D, counting SP), and it was, for a long time, treated as a complete answer to "how do you train a model too big for one GPU." But there's a failure mode none of the above addresses: at long enough sequence lengths, a **single attention operation's activations, for a single GPU's share of the hidden dimension, on a single sequence** stop fitting in memory — independent of how many parameters the model has. Attention's activation memory scales roughly with the *square* of sequence length (the attention matrix itself, before any FlashAttention-style tricks) and even with FlashAttention's linear-memory trick, the K/V cache and running statistics for a long enough sequence still eventually exceed one GPU's HBM.

Tensor parallelism doesn't help here, because it splits along the hidden dimension, not the sequence dimension — every GPU in a TP group still needs to see the entire sequence to do its slice of the math. What's needed is a way to split the sequence dimension itself across GPUs specifically for the attention computation — a genuinely different axis from anything covered so far, sometimes called **sequence parallelism** in a second, distinct sense (context parallelism, to disambiguate it from Section 6's Megatron SP), and it's the subject of the rest of this post.

## 8. DeepSpeed Ulysses: transpose the split with All-to-All

The most direct idea: if the sequence is split across GPUs (each GPU holding all attention heads, but only a fraction of the sequence positions), you can't compute attention locally, because every query needs to see every key and value across the *entire* sequence, not just the local shard. **DeepSpeed Ulysses**'s trick is to use a single **All-to-All** collective to transpose the split: after the All-to-All, every GPU holds the *entire* sequence, but only a fraction of the attention *heads*. Since different heads are completely independent computations, each GPU can now compute full, correct attention for its own subset of heads with **zero further communication** — and a second All-to-All after attention swaps the layout back for the rest of the layer, which still expects a sequence-parallel layout.

![DeepSpeed Ulysses All-to-All](blogs/images/deepspeed-ulysses-alltoall.svg?v=1)
^One All-to-All transposes "all heads, partial sequence" into "partial heads, full sequence" — exactly what local, communication-free attention needs.

This is an elegant, low-communication-overhead solution with one structural limitation worth being explicit about: the parallelism degree is capped by the number of attention heads (or, with grouped-query attention, the number of KV head groups) — you cannot usefully split across more GPUs than you have heads to give them, which puts a ceiling on how far Ulysses alone can scale for a fixed model architecture.

## 9. Ring Attention: distributed FlashAttention

**Ring Attention** takes a different approach that has no head-count ceiling at all. Recall the [Triton post's FlashAttention walkthrough](#/blog?id=triton-for-dl): a single GPU computes attention by keeping a query tile resident and streaming key/value tiles through it one at a time, maintaining a running max and running sum (the "online softmax") so the full attention matrix is never materialized. Ring Attention takes exactly this loop and **distributes it across GPUs instead of looping within one**: each GPU keeps its own shard of Q fixed, and the GPUs are arranged in a logical ring, passing K/V shards around it. At each step, a GPU computes local attention against whichever K/V shard has currently arrived, updates its running online-softmax statistics, and — critically — this computation happens *while the next shard is already being transferred*, so as long as there's enough arithmetic work per step to hide the transfer time, the ring's communication costs nothing extra in wall-clock time at all.

![Ring Attention](blogs/images/ring-attention.svg?v=1)
^Q stays fixed per GPU; K/V rotate around the ring. The same online-softmax accumulation from a single GPU's FlashAttention loop, spread across devices instead of across a for-loop.

Ring Attention has no head-count limitation the way Ulysses does — you can add as many GPUs to the ring as you have sequence to split — but it depends on the compute/communication overlap actually working in practice, which needs enough FLOPs per ring step relative to the interconnect's bandwidth, tying this directly back to the roofline questions from the first post in this series.

There's a second, more specific problem worth naming: under a **causal mask** (the standard case for autoregressive language models), a query at an early sequence position only attends to a few keys, while a query at a late position attends to nearly the whole sequence. If you assign contiguous chunks of the sequence to GPUs naively, the GPU holding the earliest chunk does dramatically less work per ring step than the GPU holding the latest chunk — a real load-imbalance problem, not just a theoretical one.

## 10. Megatron Context Parallel: balancing the ring

**Megatron's Context Parallelism** takes Ring Attention's mechanism and fixes exactly this load-imbalance problem with a **zigzag** (round-robin) chunk assignment: instead of GPU 0 getting the first contiguous chunk and GPU 3 getting the last, each GPU is given a *mix* of early and late chunks — enough that every GPU ends up doing roughly the same amount of causal-masked work per ring step, regardless of position.

![Ring Attention and load balancing](blogs/images/ring-attention.svg?v=1)
^Naive contiguous chunking leaves early-position GPUs underused under a causal mask; a zigzag assignment gives every GPU a balanced mix of early and late positions.

This is a genuinely practical engineering refinement rather than a new algorithmic idea — the underlying communication pattern is still Ring Attention's rotating K/V — but it's the difference between a context-parallel implementation that scales cleanly to production causal-LM training and one that quietly wastes a large fraction of its GPUs on the early end of every ring.

## 11. Putting it together: 3D becomes 4D and 5D

None of the strategies above are mutually exclusive, and production training of any sufficiently large model uses several of them nested inside each other — with the nesting order dictated directly by the interconnect topology from the first post's Section 7. The rule of thumb: **whichever axis communicates most frequently and in the largest volume goes on the fastest, tightest link; whichever axis communicates least tolerates the slowest, most distant link.**

In practice, that means: **tensor parallelism (and context/sequence parallelism alongside it) goes innermost**, confined to the GPUs within a single node's NVLink/NVSwitch domain, because it communicates on every layer of every forward and backward pass. **Pipeline parallelism goes next**, spanning a modest number of nodes, since it only needs to hand off activations at a small number of stage boundaries. **Data parallelism goes outermost**, spanning as many nodes (or even data centers) as you like, since an all-reduce over the full model's gradients happens only once per step, and can tolerate the slowest link available.

![3D/4D/5D parallelism combined](blogs/images/dist-training-3d-parallelism.svg?v=1)
^Innermost = fastest, most frequent link (TP, CP/SP, NVLink). Middle = moderate frequency (PP, a handful of nodes). Outermost = least frequent, tolerates any link at all (DP).

A concrete way to see why this ordering isn't arbitrary: if you accidentally ran tensor parallelism *across* nodes instead of within one, you'd be asking the slowest link in your entire cluster to carry the highest-frequency, highest-volume traffic in the whole training run — and this single misconfiguration is one of the most common, most concrete reasons a "should scale linearly" training job's throughput falls off a cliff the moment it spans more than one node.

## 12. A field guide for choosing a strategy

**Is your problem too much data, or too big a model?** Section 1's question, and the first fork in the road. If the model comfortably fits on one GPU and you just want more throughput, plain DDP (Ring-AllReduce) is usually all you need — reach for the rest of this post only once it doesn't.

**Is your bottleneck the redundant model state, not the model's actual size?** If DDP's 16Φ bytes of replicated model state is what's pushing you over a GPU's HBM, ZeRO-1 or ZeRO-2 is usually the first thing to reach for — it's a nearly drop-in change (DeepSpeed, FSDP) that doesn't touch your model code, unlike TP or PP.

**Does even one shard of the model, at ZeRO-3, still not fit?** That's when you need genuine model parallelism — pipeline parallelism if you can tolerate a bubble in exchange for simplicity, tensor parallelism if you have a fast NVLink domain and want finer-grained parallelism with a lower bubble cost, and in practice, both at once for very large models.

**Are you using enough microbatches for your pipeline depth?** Section 4's formula, (P−1)/(P−1+M) — a pipeline that's "not scaling" is very often just running with too few microbatches for how many stages it has, rather than any deeper problem.

**Is your tensor-parallel (or context-parallel) group crossing a slow link?** Section 5 and Section 11's question, and the single most common silent killer of multi-node scaling. If throughput falls off a cliff exactly at the node boundary, this is the first thing to check.

**Is your sequence length the actual bottleneck, independent of model size?** If activation memory for attention itself is what doesn't fit, you need Section 7–10's tools: Ulysses if your head count comfortably supports the parallelism degree you want, Ring Attention (with Megatron's zigzag balancing) if you need to scale past what head count allows or need causal-mask-aware load balancing.

**Have you actually measured where the time goes, or are you guessing?** The same discipline from the first two posts in this series applies here without modification: profile before you restructure. A slow "distributed" job is exactly as often a dataloader problem, a mis-set NCCL environment variable, or an accidental cross-node collective as it is a fundamentally wrong parallelism strategy — and the tools to tell these apart (Nsight Systems, PyTorch's distributed profiler, and the plain arithmetic in this post's interactive panels) are usually faster to reach for than a rewrite.

*The formulas and figures in this post (16Φ, the bubble fraction, Ring-AllReduce's 2(N−1)/N) are the standard, durable results from the ZeRO, GPipe, PipeDream, and Megatron-LM papers, and are accurate to the mixed-precision-Adam / dense-transformer setting they were derived for — real systems add engineering details (gradient accumulation, activation checkpointing, communication overlap, mixed degrees of each parallelism axis) that shift the exact numbers without changing which lever each strategy actually pulls. As in the first two posts, the concepts here — where the redundancy lives, what "frequent, high-volume" communication needs from the network, and why nesting order isn't arbitrary — are far more durable than any single framework's API for expressing them.*
