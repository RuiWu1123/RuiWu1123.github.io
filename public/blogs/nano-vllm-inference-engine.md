---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

Say you've just finished training a model and you want to serve it. The first version is easy, and you can write it from memory: take a request, run the forward pass, sample a token, append it, run again, stop at EOS. It works. You demo it, someone asks how many users it can handle, and you find out the answer is roughly four.

It's worth being precise about why, because the two reasons are what the entire field of inference engines exists to fix.

The first is that the KV cache is enormous and you don't know how big it will be. Every token a sequence generates leaves behind key and value vectors that have to stay resident for as long as the sequence lives. So you allocate a buffer per request — and since you can't know in advance whether the user wants 20 tokens or 2,000, you allocate for the maximum. Almost every request then uses a sliver of what it reserved, and you're paying for the rest on every concurrent sequence at once. Your GPU looks full. It's mostly holding air.

The second only shows up once you start batching. Batching is obviously right — the GPU wants big matrices — so you group requests and step them together. But now the batch finishes when its *slowest* member finishes, and a request that shows up one step after the batch starts has to wait for the whole thing to drain before it can even begin. Sequences that finished early sit in the batch contributing nothing but padding.

vLLM's two well-known contributions are aimed exactly at these. **PagedAttention** stops requiring the KV cache to be one contiguous per-sequence buffer, and instead hands out fixed-size blocks from a shared pool — the same move operating systems made when they stopped giving processes contiguous physical memory. **Continuous batching** lets the batch change membership at *every* step, so a new request joins on the next iteration instead of waiting its turn. A third idea, **prefix caching**, notices that requests often share a long prefix and lets them share the computed blocks too.

The trouble with reading real vLLM to learn these is that they're buried under years of optimization. [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) is a reimplementation of the same core in ~1,200 lines of Python, and it isn't a toy — on its own benchmark it does 1434 tok/s against real vLLM's 1362 (Qwen3-0.6B, 256 concurrent sequences, one RTX 4070 Laptop). Small enough to read in an afternoon, real enough to be worth reading.

So: first a look at the repo and what happens to one request end to end, then the two mechanisms above, one section each, in detail.

## 1. The map

The `nanovllm` package is 19 Python files. Seven carry a request; the rest are ordinary transformer parts (RMSNorm, RoPE, SwiGLU, tensor-parallel linears) and plumbing. Click any file to read its actual source:

![interactive:nanovllm-arch](#)

One file deserves a note because it will look strange later. `utils/context.py` is a module-level global. `ModelRunner` writes the paging metadata into it, and `Attention` reads it back out, twenty-odd layers deeper. That's there so the block tables don't have to be passed as arguments through every `forward()` in between — a pragmatic hack, and the kind of thing that reads as obviously wrong until you try to write it the other way.

## 2. What happens to one request

`add_request()` tokenizes the prompt and wraps it in a `Sequence` — the state of one request: its tokens, how many are already computed into the cache, how many are being computed right now, and which physical blocks it owns. It goes on a `waiting` queue, and nothing happens until the next round.

Then `step()` runs, over and over, and each round does the same five things. The **scheduler** picks what runs — either prompt-processing work (*prefill*) or one-token-per-sequence work (*decode*), never both in the same round. The **block manager** hands out KV-cache blocks for whatever got picked, reusing cached ones where the prefix matches. The **model runner** flattens those sequences into GPU tensors, including a `slot_mapping` that says where each new token's K/V belongs in the cache. The **model** runs — a completely ordinary Qwen3 forward pass, except that its attention layers read and write through the paging indirection. The **sampler** turns logits into token ids, and `postprocess()` appends them, hashes any blocks that just filled up, and retires sequences that hit EOS or their token limit, returning their blocks to the pool.

`generate()` loops that until both queues are empty, then detokenizes. That's the whole engine. Two of those five steps are where the interesting decisions live, and they're the rest of this post.

## 3. The scheduler: how a batch stops being a batch

Start with the question the scheduler actually has to answer every round: given some requests mid-generation and some that just arrived, what should the GPU do next?

The naive answer is "finish what you started." That's static batching, and it's the thing that made your first server bad. nano-vllm's answer is the opposite, and it fits in one sentence: **always prefer prefill; run decode only when there's no prefill left to do.**

That sounds like a scheduling detail. It's the whole trick. Because prefill is checked first, a request that arrived thirty milliseconds ago gets its prompt processed on the very next round — it does not wait for anyone to finish. And because decode advances every running sequence by exactly one token, sequences enter and leave the group constantly without disturbing it. There is no "batch" in the sense of a fixed cohort; there's just whatever happens to be running this round.

Here is that policy playing out on three requests, against static batching on the same three:

![Same three requests under continuous vs. static batching](blogs/images/nanovllm-batching-timeline.svg?v=1)

Both panels are generated by actually running nano-vllm's scheduling logic, not drawn by hand. Same work, 11 rounds versus 15 — and the gap widens as arrivals get more spread out, because that's exactly when static batching spends more time waiting.

Two things in that diagram need explaining, and both are where the implementation gets interesting.

**C gets admitted at round 3 and preempted at round 4.** Look at the block-usage bars: the pool hits FULL. A running sequence needed one more block to continue and there were none. When that happens, `preempt()` picks a victim — the most recently admitted running sequence — frees all its blocks, and pushes it back to the *front* of the waiting queue. C loses its work and is re-prefilled at round 8.

That sounds like a failure, and it's worth being clear that it isn't. It's the designed answer to overcommitment. The scheduler deliberately admits more sequences than it can guarantee memory for, because most sequences finish early and reserving for the worst case is what we were trying to escape in the first place. Preemption is the release valve that makes the optimism safe. The engine slows down under pressure instead of falling over.

The code expresses this with a construct most Python programmers have never used:

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break
else:
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    self.block_manager.may_append(seq)
    scheduled_seqs.append(seq)
```

A `while...else` runs its `else` only if the loop exited because the condition went false — never after a `break`. So: keep evicting others until this sequence can get its block, and only then schedule it. If we run out of others to evict, the sequence preempts *itself* and nothing is scheduled. The alternative spelling needs a flag variable and is genuinely worse.

**The other thing is chunked prefill.** A single prompt can be longer than the whole per-round token budget. Rather than reject it or blow the budget, the scheduler splits it across rounds — but only ever one prompt per round, and the rule that enforces it is a single conjunct:

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

`scheduled_seqs` is empty only for the round's first candidate. So the first sequence is allowed to overrun and be split; any later one that doesn't fit just waits for the next round. This is why `Sequence` tracks `num_cached_tokens` and `num_scheduled_tokens` separately — a sequence can be genuinely half-computed, and both the scheduler and the cache have to agree on exactly where the boundary is.

## 4. PagedAttention: the KV cache stops being contiguous

Back to the first problem. A sequence's cache grows unpredictably, and you must commit to a size before you know it.

The fix is a straight lift from operating systems, and it's worth stating the analogy properly because it's not decorative. A process thinks it has a flat contiguous address space; physically its memory is scattered pages, and a page table does the translation. PagedAttention does exactly this: a sequence thinks it has a contiguous KV cache; physically it's fixed-size blocks scattered across a shared pool, and a per-sequence **block table** maps logical block index to physical block id.

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

The payoff: allocation happens one block at a time as the sequence grows, so a sequence only ever holds what it has actually used, plus at most one partially-filled block. The waste is bounded by the block size — 256 tokens — no matter how long the sequence eventually runs.

But there's a second payoff that isn't obvious from the OS analogy, and it's arguably the better one. Once blocks are just entries in a table, **two sequences can point at the same block**. If two requests share a system prompt, they can share its KV cache — computed once, stored once.

The mechanism is content addressing. Each full block is hashed, and critically, chained with the hash of the block before it:

```python
h = self.compute_hash(token_ids, h)
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

The chaining is what makes reuse *correct* rather than merely likely. A block's identity isn't "these 256 tokens" but "these 256 tokens, at this position, following exactly this history" — so a match implies the two sequences agree all the way back to token zero, which is precisely the condition under which their K/V are actually identical. The loop breaks at the first miss because a prefix match is by definition a leading run. And on an apparent hit it still compares the stored `token_ids`, so a hash collision degrades to a miss rather than silently serving another request's cache.

Blocks are freed by reference count, not by owner. Here's the whole lifecycle:

![One block pool, four moments](blogs/images/nanovllm-block-lifecycle.svg?v=1)

Note the last frame: A finishes, but blocks 0 and 1 don't go anywhere, because B is still using them. The shared prefix outlives the sequence that created it. This also means the cache survives preemption — a preempted sequence that gets re-prefilled will hit on any of its blocks still resident, which is why preemption costs less than it looks like it should.

One detail that trips people up: blocks are hashed *after* being computed, not when allocated. `hash_blocks()` runs in `postprocess()` and only covers blocks that filled up that round. A sequence's trailing partial block never enters the cache, because its contents aren't final — it has no stable identity to key on yet.

Growth during decode is nearly free, and the code is a small joke:

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`len(seq) % block_size == 1` is true exactly on the token that spills into a new block. On that one token in 256 it reads "free blocks ≥ 1"; on the other 255 it reads "free blocks ≥ 0", which is trivially true. It relies on `bool` being an `int` in Python and it is doing real work — this is the check that decides whether a sequence gets preempted.

### Getting K/V into and out of scattered blocks

Two problems remain, and they're the ones that make paging actually run on a GPU.

Writing is a scatter. Each new token's K/V has to land in whatever physical slot its block table implies, and those slots aren't contiguous. `ModelRunner` precomputes the destinations into a `slot_mapping` tensor, and a Triton kernel does one token per program: load K and V, read the destination, store. It has one guard — `if slot == -1: return` — which exists so CUDA graph replay works, since captured graphs run at fixed batch sizes and the unused slots get marked `-1`.

Reading is the harder one, and the answer is that nano-vllm doesn't do it. It hands `block_table` to the attention kernel and lets the kernel gather. `flash_attn_varlen_func` and `flash_attn_with_kvcache` both accept block tables natively, so the scattered layout never has to be materialized into a contiguous tensor. The only real branch in the whole attention layer is this:

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
```

On a prefix hit, the freshly computed `k`/`v` only cover the *new* tokens, but attention has to see the cached prefix too — so they're swapped out for the full caches, and the kernel reads everything through the block table. That two-line branch is the entire seam between paged storage and ordinary attention.

Which is the thing to take away: none of this changes the math. Attention is still softmax over scaled dot products. Paging changes only where K and V physically live, and pushes the indirection down into a kernel that was going to iterate over blocks anyway. (What that kernel does internally — tiling, and why avoiding a materialized attention matrix matters — is the subject of the [GPU field guide](#/blog?id=gpu-guide-for-dl).)

Full source for the pieces above:

![interactive:nanovllm-code-blockmgr](#)

## 5. The rest, briefly

**Sampling** is nine lines, and one of them is a nice trick:

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

Dividing each probability by an independent `Exponential(1)` draw and taking the argmax samples from the categorical distribution exactly — it's the Gumbel-max trick, since dividing by an exponential is subtracting a Gumbel in log space. Why not `torch.multinomial`? Because this is pure elementwise arithmetic plus an argmax, with no data-dependent control flow, so `torch.compile` fuses the whole thing into one kernel.

There's a related saving one layer up. During prefill the model produces hidden states for every prompt token, but only the last token of each sequence can predict anything, so `ParallelLMHead` slices before projecting:

```python
last_indices = context.cu_seqlens_q[1:] - 1
x = x[last_indices].contiguous()
```

For a batch of long prompts that turns a vocab-sized projection over thousands of positions into one over a handful.

**No padding, anywhere.** Prefill concatenates every sequence's tokens into one flat 1-D tensor with cumulative offsets marking the boundaries. A batch of wildly uneven prompts costs exactly the sum of their lengths. And the two offset arrays carry information: `cu_seqlens_q` counts only new tokens while `cu_seqlens_k` spans the whole sequence including cached prefix, so `cu_seqlens_k[-1] > cu_seqlens_q[-1]` *is* the test for "a prefix hit happened this round."

**CUDA graphs and tensor parallelism** are the two optimizations worth knowing about. Decode steps are tiny and dominated by Python launch overhead, so nano-vllm pre-records the whole decode step for a fixed set of batch sizes (`[1, 2, 4, 8] + range(16, max_bs+1, 16)`) and replays the smallest captured graph at or above the current batch. Only decode — prefill shapes are unpredictable. Tensor parallelism spawns one `ModelRunner` per rank and has rank 0 broadcast method calls through a `SharedMemory` buffer: a hand-rolled RPC in about forty lines.

## 6. What nano-vllm leaves out

Speculative decoding, quantization, multiple scheduling policies, disaggregated prefill/decode across machines, and a model loader general enough for anything beyond Qwen3. That list is most of what makes production vLLM large — and none of it changes the two mechanisms above. Which is the argument for reading nano-vllm first: the ideas that make inference engines work are the ones that fit in 1,200 lines, and everything else is what accumulates once those ideas are correct.
