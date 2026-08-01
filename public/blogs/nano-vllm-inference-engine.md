---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

The first thing that stops you when you put a language model into production is usually not compute. It's memory, and specifically the memory you waste.

The culprit is the KV cache. Every token a sequence generates leaves behind key and value vectors, and they have to stay in GPU memory for as long as that sequence is alive. The awkward part is that you don't know in advance how long it will generate: this request might want 20 tokens, or 2,000. So the straightforward implementation reserves for the ceiling, giving every request a contiguous buffer big enough for its longest possible output. What actually gets used is usually a small fraction of that, and the waste repeats on every concurrent request.

The second constraint arrives once you start batching. GPUs are built for big matrices, so grouping requests together is the obvious move. But once they're a group, the group only finishes when its slowest member does; a request that arrives after the group starts has to wait for the whole thing to disperse; and sequences that finished early keep their seats without contributing anything.

vLLM's two best-known designs take one constraint each. PagedAttention stops requiring the KV cache to be one contiguous per-sequence buffer and hands out fixed-size blocks from a shared pool instead. That's the same move operating systems made when they stopped giving processes contiguous physical memory. Continuous batching lets the group change membership at every step, so a new request joins on the next round. A third idea, prefix caching, notices that requests often share a long prefix and lets them share the computed blocks too.

The trouble with learning these from real vLLM is that they're buried under years of optimization. [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) reimplements the same core in about 1,200 lines of Python, and it isn't a toy: on its own benchmark it does 1434 tok/s against real vLLM's 1362 (Qwen3-0.6B, 256 concurrent sequences, one RTX 4070 Laptop). Small enough to read in an afternoon, real enough to be worth reading.

## 1. The map

The `nanovllm` package is 19 Python files. Seven of them carry a request; the rest are ordinary transformer parts (RMSNorm, RoPE, SwiGLU, tensor-parallel linears) and plumbing. Click any file to read its real source:

![interactive:nanovllm-arch](#)

One file is worth flagging now, because it looks strange when it shows up later. `utils/context.py` is a module-level global: `ModelRunner` writes the paging metadata into it, and `Attention` reads it back out twenty-odd layers deeper. That's there so the block tables don't have to be passed as an argument through every `forward()` in between. It's the kind of design that reads as obviously wrong until you try writing it the other way.

## 2. What happens to one request

`add_request()` tokenizes the prompt and wraps it in a `Sequence`, which holds the whole state of one request: its tokens, how many are already computed into the cache, how many are being computed right now, and which physical blocks it holds. Then it goes onto the `waiting` queue, and nothing happens until the next round.

After that `step()` runs over and over, doing the same five things each round. The scheduler decides what runs, either prompt-processing work (prefill) or one-token-per-sequence work (decode), never both in the same round. The block manager hands out KV-cache blocks for whatever got picked, reusing cached ones wherever the prefix matches. The model runner flattens those sequences into GPU tensors, including a `slot_mapping` that says where each new token's K/V belongs in the cache. Then the model runs, and this is a completely ordinary Qwen3 forward pass, except that its attention layers read and write through the paging indirection. Finally the sampler turns logits into token ids, and `postprocess()` appends them, hashes any blocks that just filled up, and retires sequences that hit EOS or their token limit, returning their blocks to the pool.

`generate()` loops that until both queues are empty, then detokenizes. That's the whole engine. Two of those five steps hold the decisions worth studying, and they're the rest of this post.

## 3. The scheduler: how a batch stops being a batch

The question the scheduler has to answer every round is simple: some requests are halfway through generating, some just arrived, and it has to decide what the GPU does next.

Start with the naive answer. Collect the requests currently queued into one batch, prefill the whole batch, then decode it together round after round, and only admit the next batch once the last member of this one has finished. That "nobody starts until the group is assembled, nobody leaves until the group is done" arrangement is static batching. Its problem is the one from the opening: the group moves at the speed of its slowest member, and anything that arrives after it starts waits outside.

nano-vllm's rule is the opposite, and it's one sentence: always prefer prefill; run decode only when there's no prefill left to do.

That looks like a scheduling detail. It's actually the whole of continuous batching. Because prefill is checked first, a request that just arrived gets its prompt processed on the very next round, without waiting for anyone to finish. And because decode advances each running sequence by exactly one token, every round boundary becomes a natural switching point where sequences can join and leave freely. At which point the word "batch" has stopped meaning anything, because there is no fixed cohort, only whatever happens to be running this round.

Here's that rule on three requests, next to static batching on the same three:

![Same three requests under continuous vs. static batching](blogs/images/nanovllm-batching-timeline.svg?v=5)

Both panels come from actually running nano-vllm's scheduling logic, not from drawing by hand. The parameters are shrunk so memory pressure shows up in an example this small: a block holds 4 tokens, the pool has 6 blocks, and each round processes at most 8 tokens. The real defaults are 256 tokens per block, with the pool sized from whatever GPU memory is free at startup.

One rule has to be stated before the figure makes arithmetic sense: **a block belongs to exactly one sequence.** The tail of A and the head of B are never packed into the same block. So a sequence needs ceil(its length / 4) blocks, and its last block is usually partly empty, with those leftover slots unusable by anyone else.

Work through the state after round 3. A's prompt is 5 tokens, so ceil(5/4) = 2 blocks; B's is also 2; C's 7-token prompt is 2 as well. That's 6 blocks, and the pool is fully handed out. Only 17 tokens are actually stored, and the pool nominally holds 24, but the 7 free slots are scattered across three different sequences' unfilled tail blocks and cannot be lent to each other. That "looks free, can't be used" gap is the internal fragmentation the block size buys you, and capping it at one block per sequence is exactly what the next section is about.

Here's how to read the cells. The number in a prefill cell is how many prompt tokens that round consumed, so 5 or 7; a decode cell is always 1, because a round generates exactly one token. One thing the figure can't show but is worth knowing: a prefill round also emits the first generated token at the end, which is why A, labelled gen 8, only needs seven more decode rounds. A grey cell with a dash means the sequence is running but did nothing that round.

That last one deserves its own paragraph, because it's the direct cost of preferring prefill. A round is either a prefill round or a decode round, never both. So whenever a new request needs prefilling, every sequence already decoding stalls for that entire round: look at rounds 2, 3 and 6 in the continuous panel, where A sits still. It trades a little latency for the sequences already running against waiting time for the new arrival, and the trade is worth it, because one prefill round swallows an entire prompt while one decode round produces a single token.

The sharpest difference between the panels is C's row. Under static batching, C arrives at round 3 and then queues until round 10, doing nothing for seven rounds. B had finished back at round 4 and its memory was free the whole time, but C still couldn't start, because it wasn't part of the current cohort. Under continuous batching C is admitted at round 3, and from then on its decode steps ride along in the same rounds as A's. Same work, 11 rounds against 15.

C's row has a second thing worth looking at: it gets preempted at round 5. By then C has accumulated 9 tokens, and `9 % 4 == 1` means it has just stepped into a third block and needs one more to keep writing. But the pool was fully handed out back at round 3, and A and B had already been scheduled this round and taken off the running queue, so when C's turn came there was nobody left to evict. So it preempts itself: all its blocks are released, it goes back to the front of the waiting queue, and the prompt it just computed is discarded. By round 6 the space B released on finishing, plus what C gave back, is enough to re-prefill it.

That is precisely the least conspicuous branch in the code below, `else: self.preempt(seq)`, the "nobody else to evict, so evict myself" path.

That looks like a failure. It's actually the design. The scheduler deliberately admits more sequences than it can guarantee memory for, because most sequences finish early, and reserving for the worst case is precisely what we were trying to escape. Preemption is the release valve that makes the optimism safe: the engine slows down under pressure instead of falling over.

The code says this with a construct most Python programmers have never used:

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

A `while...else` runs its `else` only when the loop exits because the condition went false, never after a `break`. So this reads: keep evicting others until this sequence can get its block, and only then schedule it; if there's nobody left to evict, it preempts itself and nothing gets scheduled. The alternative spelling needs an extra flag variable and is genuinely uglier.

The other thing is chunked prefill. A single prompt can be longer than the whole per-round token budget. Rather than reject it or blow through the budget, the scheduler splits it across rounds, but only ever one prompt per round, and the rule that enforces this is a single conjunct:

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

`scheduled_seqs` is empty only for the round's first candidate. So the first sequence is allowed to overrun and get split; any later one that doesn't fit simply waits for the next round. This is also why `Sequence` tracks `num_cached_tokens` and `num_scheduled_tokens` separately: a sequence really can be half-computed, and the scheduler and the cache have to agree on exactly where the boundary sits.

## 4. PagedAttention: the KV cache stops being contiguous

Back to the first problem. A sequence's cache grows unpredictably, and you have to commit to a size before you know what it is.

The fix is lifted straight from operating systems, and the analogy is worth stating properly because it isn't decorative. A process believes it owns a flat contiguous address space; physically its memory is scattered pages, with a page table translating between the two. PagedAttention does exactly that: a sequence believes it owns a contiguous KV cache; physically that's fixed-size blocks scattered across a shared pool, and each sequence carries a block table mapping logical block index to physical block id.

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

The payoff is that allocation can happen one block at a time as the sequence grows. A sequence holds only what it has actually used, plus at most one partially-filled block. The waste is capped by the block size, 256 tokens, no matter how long it eventually runs.

But there's a second payoff, one the OS analogy doesn't suggest, and arguably the better one. Once blocks are just entries in a table, two sequences can point at the same block. If two requests share a system prompt, they can share its KV cache: computed once, stored once.

The mechanism is content addressing. Every full block gets hashed, and crucially, chained with the hash of the block before it:

```python
h = self.compute_hash(token_ids, h)
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

The chaining is what makes reuse correct rather than merely likely. A block's identity isn't "these 256 tokens" but "these 256 tokens, at this position, following exactly this history". So a match implies the two sequences agree all the way back to token zero, which is precisely the condition under which their K/V really are identical. The loop breaks at the first miss, because a prefix match is by definition a leading run. And even on an apparent hit it re-compares the stored `token_ids`, so a hash collision degrades into a miss instead of quietly serving another request's cache.

Blocks are freed by reference count, not by owner. Here's the whole lifecycle:

![One block pool, four moments](blogs/images/nanovllm-block-lifecycle.svg?v=2)

Look at the last frame: A finishes, but blocks 0 and 1 don't go anywhere, because B is still using them. The shared prefix outlives the sequence that created it. It also means the cache survives preemption: a preempted sequence being re-prefilled will hit on whichever of its blocks are still resident, which is why preemption costs less than it looks like it should.

One detail trips people up: blocks are hashed after being computed, not when allocated. `hash_blocks()` runs inside `postprocess()` and covers only blocks that filled up during that round. A sequence's trailing partial block never enters the cache, because its contents aren't final yet, so it has no stable identity to key on.

Growth during decode is nearly free, and the code is a small joke:

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`len(seq) % block_size == 1` is true exactly on the token that spills into a new block. On that one token in 256 it reads "free blocks ≥ 1"; on the other 255 it reads "free blocks ≥ 0", which is trivially true. It leans on `bool` being an `int` in Python, and it's doing real work: this is the check that decides whether a sequence gets preempted.

Two problems are left, and they're the ones that make paging actually run on a GPU.

Writing is a scatter. Each new token's K/V has to land in whatever physical slot its block table implies, and those slots aren't contiguous. `ModelRunner` precomputes the destinations into a `slot_mapping` tensor, and a Triton kernel runs one program per token: load K and V, read the destination, store. It has exactly one guard, `if slot == -1: return`, which exists so CUDA graph replay works, since captured graphs run at fixed batch sizes and the unused slots get marked `-1`.

Reading is the harder one, and nano-vllm's answer is that it doesn't do it. It hands `block_table` to the attention kernel and lets the kernel gather. Both `flash_attn_varlen_func` and `flash_attn_with_kvcache` accept block tables natively, so that scattered layout never has to be materialized into a contiguous tensor. The only real branch in the entire attention layer is this:

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
```

On a prefix hit, the freshly computed `k`/`v` only cover the new tokens, but attention has to see the cached prefix too, so they're swapped out for the full caches, and the kernel reads everything back through the block table. Those two lines are the entire seam between paged storage and ordinary attention.

Which is the thing worth taking away: none of this changes the math. Attention is still softmax over scaled dot products. Paging changes only where K and V physically live, and pushes the indirection down into a kernel that was going to iterate over blocks anyway. What that kernel does internally, meaning tiling and why not materializing the full attention matrix matters, is the subject of the [GPU field guide](#/blog?id=gpu-guide-for-dl).

## 5. The rest, briefly

Sampling is nine lines, and one of them is a nice trick:

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

Dividing each probability by an independent `Exponential(1)` draw and taking the argmax samples from the categorical distribution exactly. It's the Gumbel-max trick, since dividing by an exponential is subtracting a Gumbel in log space. So why not just call `torch.multinomial`? Because this is pure elementwise arithmetic plus an argmax, with no data-dependent control flow, so `torch.compile` fuses the whole thing into one kernel.

There's a related saving one layer up. During prefill the model produces hidden states for every prompt token, but only the last token of each sequence can predict anything, so `ParallelLMHead` slices before projecting:

```python
last_indices = context.cu_seqlens_q[1:] - 1
x = x[last_indices].contiguous()
```

For a batch of long prompts that turns a vocab-sized projection over thousands of positions into one over a handful.

One more thing runs through all of it: nothing is padded, anywhere. Prefill concatenates every sequence's tokens into one flat 1-D tensor with cumulative offsets marking the boundaries, so a batch of wildly uneven prompts costs exactly the sum of their lengths. And those two offset arrays carry information of their own: `cu_seqlens_q` counts only new tokens while `cu_seqlens_k` spans the whole sequence including cached prefix, so `cu_seqlens_k[-1] > cu_seqlens_q[-1]` is the test for "a prefix hit happened this round".

Finally, two optimizations worth knowing about. Decode steps are tiny and dominated by Python launch overhead, so nano-vllm pre-records the whole decode step for a fixed set of batch sizes (`[1, 2, 4, 8] + range(16, max_bs+1, 16)`) and replays the smallest captured graph at or above the current batch; only decode, since prefill shapes are unpredictable. Tensor parallelism spawns one `ModelRunner` per rank and has rank 0 broadcast method calls through a `SharedMemory` buffer, a hand-rolled RPC in about forty lines.

## 6. What nano-vllm leaves out

Speculative decoding, quantization, multiple scheduling policies, disaggregated prefill/decode across machines, and a model loader general enough for anything beyond Qwen3. That list is most of what makes production vLLM large, and not one item on it changes either of the two mechanisms above.

Which is the argument for reading nano-vllm first: the ideas that make an inference engine work are the ones that fit in 1,200 lines, and everything else is what accumulates once those ideas are correct.
