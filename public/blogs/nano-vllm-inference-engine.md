---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

Training a model and serving it are different engineering problems. Training cares about throughput on huge, uniform batches known ahead of time. Serving cares about latency on a stream of requests that arrive at unpredictable moments, ask for unpredictable output lengths, and need a response back as fast as possible. A framework built for the first problem is a poor fit for the second, which is why an entire category of software, the inference engine, exists specifically to run a trained model efficiently against live traffic. vLLM is the best-known open-source example of one.

This post uses [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a from-scratch reimplementation of vLLM's core ideas in about 1,200 lines of readable Python, as the vehicle for actually understanding what an inference engine does and why. Real vLLM is tens of thousands of lines, spread across schedulers, custom kernels, and years of incremental optimization; nano-vllm keeps only the essential mechanisms, one clean implementation of continuous batching, one of paged KV-cache memory, one of prefix caching, and gets remarkably close to vLLM's own throughput doing it (1434 tokens/s vs. vLLM's 1362 tokens/s, Qwen3-0.6B, 256 concurrent sequences, on a single RTX 4070 Laptop GPU, per the project's own benchmark). That combination, small enough to read end to end, faithful enough to teach the real thing, is exactly what makes it worth walking through.

## 1. The problem an inference engine actually solves

Picture the simplest possible way to serve a language model: a request arrives, you run it through the model one token at a time until it's done, then you move to the next request. Two things go wrong immediately at any real scale.

The first is memory. Every token a sequence generates needs its key and value vectors kept around for every future step, the KV cache, and that cache has to live somewhere in GPU memory for as long as the sequence is active. The naive fix is to reserve a contiguous buffer sized for the longest sequence you'll ever allow, per request, up front. Most requests don't come close to using it. That gap between reserved and actually-used memory is wasted on every single request running concurrently, and it's the single biggest reason naive serving code can only fit a handful of concurrent sequences before running out of GPU memory.

The second is scheduling. If requests are batched together and run in lockstep, a batch can only move forward as fast as its slowest member, and a new request that arrives mid-batch has to wait for the whole batch to finish before it can join. Static batching leaves the GPU alternating between "not enough work queued" and "blocked on stragglers," neither of which is good for throughput or for the latency any individual request experiences.

vLLM's two headline contributions are a direct answer to each problem: **PagedAttention** manages the KV cache in small, fixed-size, non-contiguous blocks instead of one large reserved buffer per sequence, borrowing the idea directly from how operating systems page virtual memory; and **continuous batching** lets the scheduler add and remove individual sequences from the running batch at every step, rather than waiting for a batch to fully drain. Nano-vllm implements both, plus a third mechanism, prefix caching, that reuses already-computed KV blocks across requests that happen to share a prefix (a repeated system prompt, for instance). The rest of this post follows one prompt through the code that implements all three.

## 2. The map: one request's path through the repo

Before tracing the lifecycle line by line, here's the shape of the whole system: seven files, each with a small number of functions that matter, called in the order a request actually flows through them.

![Nano-vLLM: one request's path through the repo](blogs/images/nanovllm-architecture-map.svg?v=1)

![interactive:nanovllm-arch](#)

`LLMEngine` is the entry point and the outer loop. `Scheduler` decides which sequences run on a given step. `BlockManager` owns the physical KV-cache memory and hands out blocks. `ModelRunner` turns scheduling decisions into actual GPU tensors. `Qwen3ForCausalLM` is the transformer itself. `Attention`, highlighted in the diagram, is where PagedAttention actually happens, and gets its own dedicated section below rather than a quick pass-through. `Sampler` turns the model's output logits into the next token id. The dashed arrow at the bottom is the loop: `generate()` calls `step()` repeatedly until every sequence in the batch has finished.

## 3. Stage 1-2: a prompt becomes a `Sequence`, and gets scheduled

A request enters through `LLMEngine.add_request()`, which tokenizes the prompt and wraps it in a `Sequence` object, a small piece of state tracking the token ids generated so far, which physical KV blocks belong to it, and how many of its prompt tokens are already covered by a cached prefix. That `Sequence` gets appended to a `waiting` queue and nothing else happens until the next scheduling step.

Scheduling is where continuous batching actually lives:

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # 1. try to schedule prefill work first
    scheduled_seqs, num_batched_tokens = [], 0
    while self.waiting and num_batched_tokens < self.max_num_batched_tokens:
        seq = self.waiting[0]
        if not self.block_manager.can_allocate(seq):
            break
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        self.block_manager.allocate(seq)
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True

    # 2. otherwise, advance every running sequence by one decode step
    while self.running:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    self.running.extend(scheduled_seqs)
    return scheduled_seqs, False
```

Every call to `schedule()` prefers **prefill** work, processing the prompt tokens of new or partially-processed sequences, over **decode** work, generating one more token for sequences already running. Only when there is no prefill work left to pack into this step does the scheduler fall through to advancing the running sequences by a single decode step. This ordering is what makes the batch composition fluid step to step: a brand-new request can start prefilling on the very next call to `schedule()`, without waiting for any currently-running sequence to finish. That is continuous batching, in nine lines of Python.

The `while ... else` block handles the case where a running sequence can't get the next KV block it needs (`can_append` fails, more on why in section 6): the scheduler frees up room by evicting the most recently added running sequence back to `waiting` via `preempt()`, which deallocates its blocks so it can be re-prefilled later. That's the pressure valve for when concurrent demand exceeds available cache memory.

## 4. Stage 3-4: reserving cache space, then building GPU tensors

Before a sequence can run, `BlockManager.can_allocate()` / `allocate()` (called inside `schedule()` above) have to reserve however many fixed-size KV-cache blocks its prompt needs, possibly reusing already-cached blocks from an identical prefix. That mechanism is the subject of the next section; for now, treat it as a black box that returns a `block_table`, a list of physical block ids, for each sequence.

`ModelRunner` takes the scheduled sequences and their block tables and turns them into the actual tensors the model will consume. For prefill, every scheduled sequence's uncached prompt tokens get flattened into one 1-D tensor, no padding at all, with cumulative offsets (`cu_seqlens_q`, `cu_seqlens_k`) marking where each sequence's tokens start and end:

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids, positions, cu_seqlens_q, cu_seqlens_k = [], [], [0], [0]
    slot_mapping = []
    for seq in seqs:
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(range(seq.num_cached_tokens, len(seq)))
        cu_seqlens_q.append(cu_seqlens_q[-1] + len(seq) - seq.num_cached_tokens)
        cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            end = start + (self.block_size if i != seq.num_blocks - 1
                            else seq.last_block_num_tokens)
            slot_mapping.extend(range(start, end))
    # ... packed into GPU tensors, one flat forward pass, zero padding waste
```

This "varlen" (variable-length) packing is why nano-vllm never pads a batch to its longest member: five sequences of very different lengths just become one long tensor with markers for where each one begins. `prepare_decode()` is the equivalent function for a decode step, one new token per running sequence instead of a whole prompt, along with each sequence's current length (`context_lens`) and its `block_table`, needed so the attention step below knows exactly which physical cache blocks to read from. The `slot_mapping` tensor built here, mapping every new token to the exact physical cache slot it should be written into, is the thread that connects this section to the PagedAttention deep-dive next.

## 5. Stage 5: running the model

With tensors prepared, `ModelRunner` calls the actual transformer, `Qwen3ForCausalLM`. Nothing about this stage is inference-engine-specific, it's a standard decoder-only forward pass, embedding the input ids, running them through N decoder layers, and normalizing the result:

```python
class Qwen3Model(nn.Module):
    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

Each decoder layer contains one `Attention` block, and that block is where the engine-specific work resumes. Rather than fold it into this pass-through, it gets its own section, because this is the one place where "how do you serve a model efficiently" and "how does a transformer work" genuinely intersect.

## 6. Breaking the chain: PagedAttention, from block table to kernel

Go back to the memory problem from section 1: reserving one contiguous, worst-case-sized buffer per sequence wastes most of what it reserves. PagedAttention's fix is the same one operating systems settled on decades ago for physical memory: stop requiring contiguity. Divide the KV cache into small fixed-size blocks (256 tokens each, in nano-vllm's default config), draw them from one shared pool across every sequence in the system, and give each sequence a small **block table**, a list mapping its logical block index (0, 1, 2, ...) to whichever physical block happens to be free.

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

A sequence's own view of its KV cache still looks perfectly sequential, block 0, then block 1, then block 2. What differs from the naive approach is that those logical blocks can land anywhere in the physical pool, blocks 7, 2, and 15 in the diagram above, scattered rather than contiguous. Nothing is wasted except the partial fill inside the very last block of a sequence, at most 255 tokens' worth regardless of how long the sequence eventually grows, instead of an entire reserved-but-unused worst-case buffer.

`BlockManager` is the code that owns this pool and hands out blocks, and it does one more thing worth pausing on: content-addressed reuse.

```python
def allocate(self, seq: Sequence):
    h = -1
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1) if h != -1 else -1
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            block_id = self.free_block_ids[0]     # cache miss: take a fresh block
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size
            block = self.blocks[block_id]
            block.ref_count += 1                  # cache hit: reuse, bump refcount
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

Every full block's token ids get hashed, chained with the hash of the block before it, so the hash captures not just "these 256 tokens" but "these 256 tokens, occurring right after this specific history." If a new sequence's leading blocks hash-match blocks that already exist in the pool, `BlockManager` reuses the same physical block instead of recomputing and rewriting it, incrementing a reference count instead. This is exactly why two requests sharing a long system prompt only pay for that prompt's KV cache once:

![Prefix caching: identical leading blocks share one physical block](blogs/images/nanovllm-prefix-cache.svg?v=1)

Sequence A and Sequence B diverge after their shared system prompt, but their block tables point at the same physical blocks 4 and 9 for as long as their prefixes match, only splitting apart, block 7 versus block 12, once their actual content differs.

Two things remain: getting new K/V vectors into these scattered physical blocks, and reading them back out during attention. Writing is a scatter operation, `store_kvcache_kernel`, a Triton kernel: every newly computed key/value vector for a token gets written directly to the physical cache slot given by that token's entry in `slot_mapping`, the same tensor built back in section 4's `prepare_prefill`. Reading happens inside `Attention.forward()`, and it's here that the block-table indirection actually pays off, `flash_attn_varlen_func` and `flash_attn_with_kvcache` both accept a `block_table` argument directly, so the attention kernel itself gathers the right scattered physical blocks per sequence without the surrounding Python code ever needing to materialize a contiguous cache:

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            o = flash_attn_varlen_func(q, k, v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q, k_cache, v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables, causal=True)
        return o
```

That flash-attention kernel underneath is itself the subject of the tiling and memory-bandwidth discussion in the [GPU field guide post](#/blog?id=gpu-guide-for-dl), if the "why does avoiding a materialized attention matrix matter" question is interesting on its own, that post's roofline section is the deeper dive. What matters here is narrower: PagedAttention doesn't change the attention math at all, softmax over scaled dot products is still softmax over scaled dot products. It changes only where the K and V vectors physically live, and the block table plus this kernel-level gather is the entire mechanism that makes non-contiguous storage invisible to the math.

## 7. Back to the chain: sampling

The model's forward pass ends in logits, one score per vocabulary token, and `Sampler` turns those into an actual next token id. Nano-vllm's sampler applies temperature scaling and then samples using the Gumbel-max trick, an equivalent but `torch.compile`-friendly alternative to `torch.multinomial`:

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

Dividing each probability by an independent `Exponential(1)` draw and taking the `argmax` is mathematically equivalent to sampling directly from the categorical distribution `probs`, but it's built entirely out of elementwise ops and an argmax, no data-dependent control flow, which is exactly the shape `torch.compile` can turn into one efficient fused kernel.

## 8. Closing the loop: postprocess, preemption, and `generate()`

Back in `LLMEngine.step()`, once the model has produced a batch of next-token ids, `Scheduler.postprocess()` appends each new token to its sequence, checks whether that sequence hit an end-of-sequence token or its maximum length, and if so marks it finished and returns its blocks to the free pool. Sequences that aren't finished simply stay in `running` and get picked up again on the scheduler's next call.

`generate()` is the outer loop tying all of this together: submit every prompt via `add_request()`, then call `step()` repeatedly, one call per scheduling round, prefill or decode, until every sequence has finished. That loop-back is the dashed arrow at the bottom of the architecture diagram in section 2. Preemption, from section 3, is the mechanism that keeps this loop correct even under memory pressure: if the block pool can't grow fast enough to keep every running sequence supplied, the least-recently-added running sequence gets evicted back to `waiting` rather than the whole system stalling or erroring out, and it simply gets re-prefilled, with whatever blocks are still valid reused via the same content-hash mechanism from section 6, once room frees up. Once a sequence is finished, its accumulated token ids get detokenized back into text, and that's the response the caller receives.

## 9. What nano-vllm leaves out

Nano-vllm's honesty about its own scope is part of what makes it a good teaching example: it implements the three mechanisms above thoroughly and cuts almost everything else. It does include, briefly, a couple of features not covered in the walkthrough above because they're closer to engineering than to core algorithm: tensor parallelism, splitting the model itself across multiple GPUs, coordinated through `torch.multiprocessing` and a small shared-memory RPC mechanism rather than a heavier distributed framework; and CUDA graph capture, pre-recording the sequence of GPU operations for a handful of fixed batch sizes so that decode steps can replay a graph instead of re-dispatching each op from Python, cutting per-step launch overhead.

What it leaves out entirely is most of what makes production vLLM larger: speculative decoding, multiple scheduling policies tuned for different SLA targets, quantization support, disaggregated prefill/decode serving across separate machines, and a much more general model-loading and kernel-selection layer that has to work across architectures nano-vllm never has to consider, since it targets exactly one model family. None of that changes the core mechanics this post walked through; it's what gets layered on top once "get the core algorithm right in 1,200 lines" turns into "run this in production at scale."
