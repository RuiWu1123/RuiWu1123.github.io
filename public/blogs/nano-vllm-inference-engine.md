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

`LLMEngine.add_request()` tokenizes the prompt and wraps it in a `Sequence`, then hands it to the scheduler:

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

A `Sequence` is the complete state of one request: its token ids, how many of them have already been computed into the KV cache (`num_cached_tokens`), how many are being computed this round (`num_scheduled_tokens`), and which physical cache blocks it owns (`block_table`). Two derived properties matter later:

```python
@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size

def block(self, i):
    assert 0 <= i < self.num_blocks
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

`block(i)` slices the sequence's tokens into block-sized chunks. That slicing is what makes prefix caching possible, because it gives every block a well-defined content to hash.

## 3. The scheduler, and what continuous batching actually is

Each engine step begins by asking the scheduler what to run. The answer is always either "prefill work" or "one decode token for everything running" — never both:

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_batched_tokens = 0

    # prefill
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]
        remaining = self.max_num_batched_tokens - num_batched_tokens
        if remaining == 0:
            break
        if not seq.block_table:
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
        else:
            num_tokens = seq.num_tokens - seq.num_cached_tokens
        if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
            break
        if not seq.block_table:
            self.block_manager.allocate(seq, num_cached_blocks)
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        num_batched_tokens += seq.num_scheduled_tokens
        if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True

    # decode
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
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

The prefill loop drains the `waiting` queue under two budgets: `max_num_seqs` sequences and `max_num_batched_tokens` tokens. If a single prompt is larger than the remaining token budget, it gets **chunked** — prefilled across several rounds — but the `and scheduled_seqs` guard means only the first sequence in a round may be chunked, so nano-vllm never splits two prompts in the same round. A chunked sequence stays in `waiting` (the status change only fires once `num_cached_tokens + num_scheduled_tokens == num_tokens`), which is how it gets picked up again next round.

The `while ... else` in the decode loop is Python's least-known construct: the `else` runs only if the `while` condition went false without hitting `break`. So a sequence is scheduled only if `can_append` eventually succeeded. If it never does, `preempt()` evicts a victim — the most recently added running sequence, or the sequence itself if nothing else is running — freeing its blocks and pushing it back to the front of `waiting`:

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

Preemption is not an error path. It is the designed response to the KV cache filling up, and the reason the engine degrades gracefully under load instead of failing.

The consequence of prefill-first ordering is that batch membership is fluid: a request that arrives now can start prefilling on the very next `schedule()` call, without any currently-running sequence finishing. Stepping through it makes the difference from static batching concrete:

![interactive:nanovllm-scheduler](#)

## 4. Building the GPU tensors

Scheduled sequences become flat tensors. For prefill, every sequence's uncomputed tokens are concatenated into one 1-D tensor with cumulative offsets marking the boundaries — no padding anywhere:

```python
for seq in seqs:
    start = seq.num_cached_tokens
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q
    seqlen_k = end
    input_ids.extend(seq[start:end])
    positions.extend(range(start, end))
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```

Note `seqlen_q` and `seqlen_k` differ whenever a prefix was cached: the query length is only the *new* tokens, while the key length spans the whole sequence including the cached prefix it must attend to. `cu_seqlens_k[-1] > cu_seqlens_q[-1]` is precisely the test nano-vllm uses to detect that a prefix cache hit occurred and that block tables therefore need to be passed to the attention kernel.

The other output is `slot_mapping`: for every new token, the exact physical slot in the cache it should be written to, computed by walking the sequence's `block_table`.

```python
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
    slot_mapping.extend(range(slot_start, slot_end))
```

Decode is the same idea with one token per sequence, so the slot is just the tail of the last block:

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

## 5. PagedAttention

Here the walkthrough stops, because this is the mechanism the rest of the engine is built around.

The problem, restated: a sequence's KV cache grows unpredictably, and you don't know its final size when it starts. Reserving for the worst case wastes almost everything. The fix is the one operating systems use — stop requiring the memory to be contiguous. Carve the cache into fixed-size blocks (256 tokens in nano-vllm's default config), keep one shared pool of them, and give each sequence a **block table** mapping its logical block index to whatever physical block was free.

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

The sequence's own view stays sequential — block 0, block 1, block 2 — while the physical blocks sit wherever. The only waste left is the unfilled remainder of a sequence's last block, bounded by the block size no matter how long the sequence grows.

### Allocation, and content-addressed reuse

`can_allocate()` does the cache probing, walking the sequence's full blocks and chain-hashing them:

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()

def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

Three things are load-bearing. Each hash is chained with the previous block's hash, so a block's identity is "these tokens, in this position, after this exact history" — two sequences can only match if they agree from token zero. The loop `break`s at the first miss, because a prefix match is by definition a leading run. And the explicit `self.blocks[block_id].token_ids != token_ids` re-check guards against hash collisions rather than trusting the digest.

The return value is overloaded: `-1` means "not enough free blocks, don't schedule this yet," any other value is the count of blocks that came back as cache hits. That's the value the scheduler subtracts to work out how many tokens actually need computing.

`allocate()` then commits it — reusing cached blocks by bumping their reference count, and taking fresh blocks for the rest:

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    assert not seq.block_table
    h = -1
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

Growth during decode is deliberately cheap:

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

`len(seq) % block_size == 1` is true exactly when the sequence has just spilled past a block boundary, so a new block is requested on exactly one token in every 256. Note the comparison in `can_append` relies on Python's `bool` being an `int`: it reads as "free blocks ≥ 1" on a boundary token and "free blocks ≥ 0" otherwise.

Freeing is by reference count, not by owner, which is what lets a shared prefix outlive the sequence that created it:

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

Watching the pool evolve makes the sharing and the reference counting easier to hold onto than the code does:

![interactive:nanovllm-blocks](#)

![Prefix caching: identical leading blocks share one physical block](blogs/images/nanovllm-prefix-cache.svg?v=1)

One subtlety: blocks are hashed *after* they are computed, not when allocated. `hash_blocks()` runs in `postprocess()` and only covers blocks that became full during the round, which is why a sequence's final partial block never enters the cache — its contents aren't settled yet.

### Getting K/V in and out

Writing is a scatter, done by a Triton kernel — one program per token, each copying that token's key and value into the slot `slot_mapping` assigns it:

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

The `slot == -1` early return is what makes CUDA graph replay safe: captured graphs run at a fixed batch size, so padding entries are marked `-1` and skipped.

Reading is where the indirection pays off. `Attention.forward()` never gathers blocks itself — it hands `block_table` to the attention kernel, which does the gather internally:

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(q, k, v,
                                   max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                   max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                   softmax_scale=self.scale, causal=True, block_table=context.block_tables)
    else:    # decode
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                                    softmax_scale=self.scale, causal=True)
    return o
```

Note the branch: on a prefix cache hit, `k` and `v` are *replaced* by the full caches, because the freshly computed `k`/`v` only cover the new tokens while attention must see the cached prefix too. The kernel then reads everything through `block_table`.

The attention math is completely unchanged by any of this — softmax over scaled dot products, exactly as always. Paging changes only where K and V live. (The tiling and bandwidth reasoning inside the flash-attention kernel itself is the subject of the [GPU field guide](#/blog?id=gpu-guide-for-dl); this post treats that kernel as given.)

## 6. Sampling

Logits become a token id in nine lines:

```python
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
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

`postprocess()` finishes the round: hash any blocks that filled up, advance the cached-token count, append the new token, and retire finished sequences.

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

The `continue` is the chunked-prefill case: a sequence still mid-prompt discards the sampled token, because a token predicted from a partial prompt is meaningless. Only once the prompt is fully consumed does generation actually begin.

`generate()` wraps it all in a loop that runs until both queues are empty, then detokenizes each sequence's accumulated ids back into text.

## 8. What's missing compared to real vLLM

nano-vllm does include two things this walkthrough skipped, both engineering rather than algorithm. Tensor parallelism spawns one `ModelRunner` process per rank and has rank 0 broadcast method calls over a `SharedMemory` buffer — a hand-rolled RPC in about forty lines. And CUDA graph capture pre-records the decode step for a fixed set of batch sizes (`[1, 2, 4, 8] + range(16, max_bs+1, 16)`), replaying the smallest captured graph at or above the current batch size (`next(x for x in self.graph_bs if x >= bs)`) instead of re-dispatching every op from Python; this is why `slot_mapping` needs its `-1` sentinel and why graphs are only used for decode, where shapes are predictable.

What's genuinely absent is most of what makes production vLLM large: speculative decoding, quantization, multiple scheduling policies, disaggregated prefill/decode across machines, and a model-loading layer general enough for architectures beyond the single Qwen3 family this targets. None of that changes the mechanics above — it's what accumulates on top once the core is correct.
