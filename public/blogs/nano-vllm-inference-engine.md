---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

Serving a language model naively goes wrong in two specific ways. First, memory: every token a sequence generates leaves behind key and value vectors that must stay in GPU memory for as long as the sequence is alive, and the obvious implementation reserves one contiguous worst-case-sized buffer per request. Most requests use a fraction of it, and that waste multiplies across every concurrent sequence — it is usually what caps naive serving code at a handful of them. Second, scheduling: if requests are batched and stepped in lockstep, the batch advances at the speed of its slowest member, and a request arriving mid-batch waits for the whole batch to drain.

vLLM's two central ideas answer these one for one. **PagedAttention** stores the KV cache in small fixed-size blocks drawn from a shared pool, borrowing the trick operating systems use for virtual memory, so nothing has to be contiguous and nothing has to be reserved up front. **Continuous batching** lets the scheduler change the batch's membership at every single step, so a new request starts work immediately instead of waiting for a slot. A third mechanism, **prefix caching**, reuses already-computed blocks across requests that share a prefix.

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) implements all three in about 1,200 lines of Python, and reaches 1434 tokens/s against real vLLM's 1362 on the project's own benchmark (Qwen3-0.6B, 256 concurrent sequences, one RTX 4070 Laptop GPU). That makes it small enough to read completely while still being the real mechanism rather than a toy. This post follows one request through it, stopping to open up PagedAttention properly when we reach it.

## 1. The repo, and the path through it

The `nanovllm` package is 19 Python files (plus `bench.py` and `example.py` at the repo root). Seven of them form the pipeline a request moves through, in a fixed order — those are the ones this post follows. A request executes plenty of the rest too, but that code is either standard transformer machinery (RMSNorm, RoPE, SwiGLU, tensor-parallel linear layers) or plumbing (weight loading, config), not serving logic.

![interactive:nanovllm-arch](#)

Two details in that tree are worth flagging before we start walking it. `utils/context.py` is a module-level global that carries `slot_mapping` and `block_tables` from `ModelRunner` directly down to `Attention` — it exists so the paging metadata doesn't have to be threaded as an argument through every intervening layer. And most of `layers/` is ordinary model code, with two exceptions that read that global and change behaviour because of it: `attention.py`, which is where paging actually happens, and `embed_head.py`, which skips most of the output projection during prefill (section 6).

## 2. A prompt becomes a `Sequence`

`LLMEngine.add_request()` tokenizes the prompt, wraps it in a `Sequence`, and pushes it onto the scheduler's `waiting` queue. Nothing else happens until the next scheduling round.

A `Sequence` is the complete state of one request: its token ids, how many of them have already been computed into the KV cache (`num_cached_tokens`), how many are being computed this round (`num_scheduled_tokens`), and which physical cache blocks it owns (`block_table`). The distinction between those first two counters is what makes chunked prefill and prefix caching expressible at all — a sequence can be half-computed, and the engine needs to know exactly where the boundary is.

One method matters more than the rest: `block(i)` slices the sequence's tokens into block-sized chunks, and `num_blocks` reports how many such chunks it spans. That slicing gives every block a well-defined content, which is precisely what makes it hashable — and hashability is what prefix caching runs on.

## 3. The scheduler, and what continuous batching actually is

Each engine step begins by asking the scheduler what to run. The answer is always either "prefill work" or "one decode token for everything running" — never both. `schedule()` tries the first, and only falls through to the second if it scheduled nothing.

The prefill pass drains the `waiting` queue under two budgets: `max_num_seqs` sequences and `max_num_batched_tokens` tokens. If a single prompt is larger than the remaining token budget it gets **chunked** — prefilled across several rounds — but only ever one prompt per round, enforced by a single conjunct:

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

`scheduled_seqs` is empty only for the round's first candidate, so the first sequence is allowed to overrun the budget and be split, while any later one that doesn't fit simply defers to the next round. A chunked sequence also stays in `waiting`, because the promotion to `running` is gated on `num_cached_tokens + num_scheduled_tokens == num_tokens` — that's how it gets picked up again.

The decode pass is where memory pressure surfaces, and it leans on Python's least-known construct:

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
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

A `while ... else` runs its `else` only when the loop condition went false *without* hitting `break`. So the sequence gets scheduled only if `can_append` eventually succeeded. If it never does, `preempt()` evicts a victim — the most recently added running sequence, or the sequence itself when nothing else is left — deallocating its blocks and pushing it back onto the *front* of `waiting` so it is re-prefilled first when room appears.

Preemption is not an error path. It is the designed response to the KV cache filling up, and the reason the engine degrades gracefully under load instead of failing.

The consequence of prefill-first ordering is that batch membership is fluid: a request that arrives now can start prefilling on the very next `schedule()` call, without any currently-running sequence finishing. Stepping through it makes the difference from static batching concrete:

![interactive:nanovllm-scheduler](#)

## 4. Building the GPU tensors

Scheduled sequences become flat tensors. For prefill, every sequence's uncomputed tokens are concatenated into one 1-D tensor, with cumulative offsets marking where each sequence starts and ends. Nothing is padded to a common length — this is what "varlen" packing means, and it is why a batch of wildly different prompt lengths costs exactly the sum of those lengths.

Two separate offset arrays are maintained, and the gap between them carries real information:

```python
seqlen_q = seq.num_scheduled_tokens
seqlen_k = end
cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```

The query length counts only the *new* tokens; the key length spans the whole sequence, including any cached prefix the new tokens must still attend to. So `cu_seqlens_k[-1] > cu_seqlens_q[-1]` is exactly the condition "a prefix cache hit happened this round," and nano-vllm uses precisely that test to decide whether block tables need to be handed to the attention kernel at all.

The other output is `slot_mapping`: for every new token, the absolute physical slot it should be written into, obtained by walking the sequence's `block_table` and converting each block id into a base offset. Decode is the degenerate case — one token per sequence, so the slot is just the tail of the last block. This array is the handoff to PagedAttention, and it is the only thing the attention layer needs in order to write into a cache whose physical layout it knows nothing about.

## 5. PagedAttention

Here the walkthrough stops, because this is the mechanism the rest of the engine is built around.

The problem, restated: a sequence's KV cache grows unpredictably, and you don't know its final size when it starts. Reserving for the worst case wastes almost everything. The fix is the one operating systems use — stop requiring the memory to be contiguous. Carve the cache into fixed-size blocks (256 tokens in nano-vllm's default config), keep one shared pool of them, and give each sequence a **block table** mapping its logical block index to whatever physical block was free.

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

The sequence's own view stays sequential — block 0, block 1, block 2 — while the physical blocks sit wherever. The only waste left is the unfilled remainder of a sequence's last block, bounded by the block size no matter how long the sequence grows.

### Allocation, and content-addressed reuse

`can_allocate()` probes the cache before anything is committed. It walks the sequence's full blocks, hashing each one *chained with the previous block's hash*, and looks each digest up in a `hash_to_block_id` dictionary. Three details in that loop carry the whole design.

The chaining means a block's identity is not "these 256 tokens" but "these 256 tokens, at this position, following this exact history" — so two sequences can only share a block if they agree all the way back to token zero, which is what makes the reuse safe rather than merely plausible. The loop breaks at the first miss, since a prefix match is by definition a leading run. And on every apparent hit it re-checks the stored `token_ids` against the sequence's actual tokens, so a hash collision produces a miss rather than silently serving another request's KV data.

The return value is overloaded, which is worth knowing before reading the scheduler: `-1` means "not enough free blocks, don't schedule this yet," and any other value is the *count* of leading blocks that hit. That count is what the scheduler subtracts to work out how many tokens genuinely need computing.

`allocate()` then commits — hits get their `ref_count` incremented, misses draw fresh blocks off the free deque. Growth during decode is deliberately almost free:

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

`len(seq) % block_size == 1` is true exactly on the token that spills past a block boundary, so a new block is requested on one token in every 256 and the other 255 cost nothing. The comparison in `can_append` quietly relies on Python's `bool` being an `int`: it reads as "free blocks ≥ 1" on a boundary token and "free blocks ≥ 0" — trivially true — otherwise.

Freeing walks the block table decrementing reference counts, releasing only blocks that reach zero. Because it's refcounted rather than owner-based, a shared prefix outlives whichever sequence happened to create it. Watching the pool evolve carries this better than the code does:

![interactive:nanovllm-blocks](#)

![Prefix caching: identical leading blocks share one physical block](blogs/images/nanovllm-prefix-cache.svg?v=1)

One subtlety the diagram can't show: blocks are hashed *after* being computed, not when allocated. `hash_blocks()` runs during `postprocess()` and covers only blocks that filled up during that round, which is why a sequence's trailing partial block never enters the cache — its contents aren't settled yet, so it has no stable identity to key on.

### Getting K/V in and out

Writing is a scatter, done by a Triton kernel with one program per token: each loads its token's key and value, reads its destination from `slot_mapping`, and stores into the cache at that slot. There's one guard worth noting — `if slot == -1: return`. That sentinel is what makes CUDA graph replay safe, since captured graphs run at a fixed batch size and the unused padding entries are marked `-1` and skipped.

Reading is where the indirection pays off, and the punchline is that `Attention.forward()` never gathers anything itself. It passes `block_table` straight to `flash_attn_varlen_func` (prefill) or `flash_attn_with_kvcache` (decode), and the kernel does the scattered gather internally. The only real branch is this one:

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
```

On a prefix cache hit the freshly computed `k` and `v` cover only the new tokens, but attention has to see the cached prefix too — so they are *replaced* wholesale by the full caches, and the kernel reads everything back through the block table. That is the entire seam between "paged storage" and "ordinary attention."

Full source for the pieces above, if you want to read them end to end rather than in excerpt:

![interactive:nanovllm-code-blockmgr](#)

The attention math is completely unchanged by any of this — softmax over scaled dot products, exactly as always. Paging changes only where K and V live. (The tiling and bandwidth reasoning inside the flash-attention kernel itself is the subject of the [GPU field guide](#/blog?id=gpu-guide-for-dl); this post treats that kernel as given.)

## 6. Sampling

The whole sampler is nine lines: scale the logits by temperature, softmax, and then one strange-looking line does the sampling.

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

Dividing each probability by an independent `Exponential(1)` draw and taking the `argmax` samples from the categorical distribution exactly — it's the Gumbel-max trick in disguise, since dividing by an exponential is the same as subtracting a Gumbel in log space. The reason to write it this way rather than call `torch.multinomial` is that it is pure elementwise arithmetic plus an argmax, with no data-dependent control flow, so `torch.compile` can fuse the whole thing into one kernel.

There is a related trick one layer up. During prefill the model computes hidden states for every prompt token, but only the last token of each sequence can produce a next token, so `ParallelLMHead` slices before projecting:

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

For a batch of long prompts this turns a vocab-sized projection over thousands of positions into one over a handful.

## 7. Closing the loop

`postprocess()` finishes the round: hash any blocks that filled up, advance the cached-token count, append the new token, and retire sequences that hit EOS or `max_tokens` — returning their blocks to the free pool immediately rather than at some later sweep.

It also contains the other half of chunked prefill:

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

A sequence still partway through its prompt throws the sampled token away, because a token predicted from half a prompt is meaningless. Generation only starts once the prompt is fully consumed. This is the counterpart to the scheduler's decision to split the prompt in the first place, and the two have to agree or the sequence would emit garbage mid-prefill.

`generate()` wraps everything in a loop that runs until both queues are empty, then detokenizes each sequence's accumulated ids back into text.

![interactive:nanovllm-code-sched](#)

## 8. What's missing compared to real vLLM

nano-vllm does include two things this walkthrough skipped, both engineering rather than algorithm. Tensor parallelism spawns one `ModelRunner` process per rank and has rank 0 broadcast method calls over a `SharedMemory` buffer — a hand-rolled RPC in about forty lines. And CUDA graph capture pre-records the decode step for a fixed set of batch sizes (`[1, 2, 4, 8] + range(16, max_bs+1, 16)`), replaying the smallest captured graph at or above the current batch size (`next(x for x in self.graph_bs if x >= bs)`) instead of re-dispatching every op from Python; this is why `slot_mapping` needs its `-1` sentinel and why graphs are only used for decode, where shapes are predictable.

What's genuinely absent is most of what makes production vLLM large: speculative decoding, quantization, multiple scheduling policies, disaggregated prefill/decode across machines, and a model-loading layer general enough for architectures beyond the single Qwen3 family this targets. None of that changes the mechanics above — it's what accumulates on top once the core is correct.
