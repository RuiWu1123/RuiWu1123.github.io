---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

朴素地做模型服务，会在两个很具体的地方出问题。第一是内存：一个序列每生成一个 token，都会留下 key 和 value 向量，而且必须在这个序列存活期间一直待在显存里；最直接的实现是给每个请求预留一段按最坏情况估算的连续 buffer。绝大多数请求只用得上其中一小部分，而这份浪费会在每一个并发序列上重复一遍 —— 朴素服务代码往往撑不了几个并发，原因通常就在这里。第二是调度：如果请求被打包成 batch、步调一致地推进，那整个 batch 的速度就取决于最慢的那个成员，而中途到达的请求得等整个 batch 排空才能进来。

vLLM 的两个核心思想，正好一一对应这两个问题。**PagedAttention** 把 KV cache 存成从共享池里取出的、固定大小的小 block，借用的是操作系统做虚拟内存的那套办法，于是不需要连续，也不需要提前预留。**Continuous batching** 让调度器在每一步都能改变 batch 的成员组成，新请求立刻就能开始干活，而不用排队等位置。第三个机制 **prefix caching**，则在共享前缀的请求之间复用已经算好的 block。

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用大约 1200 行 Python 把这三个都实现了，而且在项目自己的 benchmark 上跑到 1434 tokens/s，对比真实 vLLM 的 1362（Qwen3-0.6B，256 并发序列，单张 RTX 4070 Laptop）。这个体量小到可以完整读完，但实现的是真机制而不是玩具。这篇文章跟着一个请求走一遍，走到 PagedAttention 时停下来把它彻底拆开。

## 一、仓库结构，以及穿过它的那条路径

`nanovllm` 这个 package 一共 19 个 Python 文件（仓库根目录另有 `bench.py` 和 `example.py`）。其中七个构成了请求流经的那条流水线，顺序固定 —— 这篇文章跟的就是这七个。剩下的代码一个请求同样会执行，但那些要么是标准的 transformer 组件（RMSNorm、RoPE、SwiGLU、张量并行的线性层），要么是管道工作（权重加载、配置），不属于服务逻辑。

![interactive:nanovllm-arch](#)

开始走之前，这棵树里有两处值得先指出来。`utils/context.py` 是一个模块级全局变量，负责把 `slot_mapping` 和 `block_tables` 从 `ModelRunner` 直接送到 `Attention` —— 它存在的意义就是让分页元数据不必作为参数层层穿过中间每一层。另外 `layers/` 大部分是普通模型代码，但有两个例外会去读这个全局量并因此改变行为：`attention.py`，分页真正发生的地方；以及 `embed_head.py`，它在 prefill 时会跳过输出投影的绝大部分（见第六节）。

## 二、一个 prompt 变成 `Sequence`

`LLMEngine.add_request()` 把 prompt 分词、包成 `Sequence`，然后交给调度器：

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

一个 `Sequence` 就是某个请求的完整状态：它的 token id、其中有多少已经被算进 KV cache 了（`num_cached_tokens`）、这一轮正在算多少个（`num_scheduled_tokens`），以及它拥有哪些物理缓存 block（`block_table`）。有两个派生属性后面会用到：

```python
@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size

def block(self, i):
    assert 0 <= i < self.num_blocks
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

`block(i)` 把序列的 token 按 block 大小切片。正是这个切片让 prefix caching 成为可能，因为它给每个 block 一份定义明确、可以拿去哈希的内容。

## 三、调度器，以及 continuous batching 到底是什么

引擎的每一步都从"问调度器该跑什么"开始。答案永远是"prefill 工作"或者"给所有 running 序列各解一个 token"二选一 —— 绝不会同时：

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

prefill 循环在两个预算下排空 `waiting` 队列：`max_num_seqs` 个序列和 `max_num_batched_tokens` 个 token。如果单个 prompt 比剩余 token 预算还大，它会被**分块**（chunked）—— 分几轮 prefill 完 —— 但 `and scheduled_seqs` 这个守卫意味着一轮里只有第一个序列可以被分块，所以 nano-vllm 绝不会在同一轮里劈开两个 prompt。被分块的序列会留在 `waiting` 里（状态变更只在 `num_cached_tokens + num_scheduled_tokens == num_tokens` 时才触发），下一轮就是这样被再次捡起来的。

decode 循环里的 `while ... else` 是 Python 最少人知道的语法：`else` 只在 `while` 条件自然变假、而没有撞上 `break` 时才执行。所以只有 `can_append` 最终成功了，这个序列才会被调度。如果始终不成功，`preempt()` 就会驱逐一个牺牲者 —— 最近才加入 running 的那个，或者在没有别的序列可选时驱逐它自己 —— 释放其 block 并把它塞回 `waiting` 队首：

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

抢占不是错误处理路径。它是针对 KV cache 被填满这件事设计好的应对方式，也是引擎在高负载下会优雅降级而不是直接失败的原因。

prefill 优先这个顺序带来的结果是：batch 的成员组成是流动的。此刻到达的请求，下一次 `schedule()` 调用就能开始 prefill，完全不需要任何正在跑的序列先结束。一步步走一遍，和静态 batching 的差别就很具体了：

![interactive:nanovllm-scheduler](#)

## 四、搭出 GPU 张量

被调度的序列会变成扁平张量。对 prefill，所有序列还没算的 token 被拼接成一条一维张量，用累积偏移量标记边界 —— 任何地方都不 padding：

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

注意 `seqlen_q` 和 `seqlen_k` 在有前缀命中缓存时是不相等的：query 长度只算*新*的 token，而 key 长度要覆盖整个序列、包括那段必须被注意到的缓存前缀。`cu_seqlens_k[-1] > cu_seqlens_q[-1]` 正是 nano-vllm 用来判断"发生了前缀缓存命中、因而需要把 block table 传给 attention kernel"的条件。

另一个产物是 `slot_mapping`：为每个新 token 算出它该被写进缓存里的哪个物理槽位，做法是遍历这个序列的 `block_table`。

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

decode 是同一个思路，只是每个序列只有一个 token，所以槽位就是最后一个 block 的末尾：

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

## 五、PagedAttention

走到这里先停下来，因为整个引擎的其余部分都是围着这个机制搭起来的。

把问题重新说一遍：一个序列的 KV cache 会不可预测地增长，而你在它开始时并不知道最终会有多大。按最坏情况预留，几乎全是浪费。解法就是操作系统用的那个 —— 不再要求内存连续。把缓存切成固定大小的 block（nano-vllm 默认配置里是 256 个 token），维护一个共享的池子，再给每个序列一张 **block table**，把它的逻辑 block 编号映射到当时恰好空闲的那个物理 block。

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

序列自己看到的视图依然是顺序的 —— block 0、block 1、block 2 —— 而物理 block 爱在哪在哪。剩下的唯一浪费，是序列最后一个 block 里没填满的那部分，无论序列长到多少，这份浪费都被 block 大小卡住了上限。

### 分配，以及基于内容寻址的复用

`can_allocate()` 负责探测缓存，遍历序列的满 block 并做链式哈希：

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

这里有三处是承重的。每个哈希都串上了前一个 block 的哈希，所以一个 block 的身份是"这些 token，在这个位置上，跟在这段确切历史之后" —— 两个序列只有从第 0 个 token 就一致才可能匹配上。循环在第一次未命中时就 `break`，因为前缀匹配按定义就是一段开头的连续区间。而那个显式的 `self.blocks[block_id].token_ids != token_ids` 复查，是在防哈希碰撞，而不是无条件信任摘要值。

返回值是被复用的：`-1` 表示"空闲 block 不够，先别调度这个"，其他任何值都是命中缓存的 block 数量。调度器正是拿这个值去减，算出真正需要计算的 token 有多少。

`allocate()` 随后落实这件事 —— 命中的 block 通过增加引用计数来复用，剩下的取新 block：

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

decode 期间的增长被刻意做得很轻：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

`len(seq) % block_size == 1` 恰好在序列刚刚越过一个 block 边界时为真，所以平均每 256 个 token 才会申请一次新 block。注意 `can_append` 里的比较利用了 Python 中 `bool` 就是 `int` 这一点：在边界 token 上它读作"空闲 block ≥ 1"，其他时候读作"空闲 block ≥ 0"。

释放是按引用计数、而不是按归属进行的，这正是共享前缀能比创造它的那个序列活得更久的原因：

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

看着池子一步步变化，比读代码更容易把共享和引用计数这两件事记住：

![interactive:nanovllm-blocks](#)

![Prefix caching: identical leading blocks share one physical block](blogs/images/nanovllm-prefix-cache.svg?v=1)

一个细节：block 是在被*算完之后*才哈希的，不是在分配时。`hash_blocks()` 跑在 `postprocess()` 里，而且只覆盖这一轮里刚刚变满的 block —— 这就是为什么一个序列最后那个没填满的 block 永远不会进缓存：它的内容还没定下来。

### 把 K/V 写进去、读回来

写入是一个 scatter，由一个 Triton kernel 完成 —— 每个 token 一个 program，各自把自己的 key 和 value 拷进 `slot_mapping` 指派的槽位：

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

`slot == -1` 这个提前返回，正是 CUDA graph 重放能安全工作的前提：被捕获的 graph 跑在固定 batch size 上，于是 padding 出来的条目被标成 `-1` 并跳过。

读取才是这层间接寻址真正兑现价值的地方。`Attention.forward()` 自己从不去收集 block —— 它把 `block_table` 交给 attention kernel，由 kernel 在内部完成收集：

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

注意那个分支：在前缀缓存命中时，`k` 和 `v` 会被*替换*成完整的 cache，因为刚算出来的 `k`/`v` 只覆盖新 token，而 attention 必须也看到缓存里的前缀。之后 kernel 就通过 `block_table` 把这些全部读出来。

上面这一切完全没有改变 attention 的数学 —— 依然是缩放点积再 softmax，和以前一模一样。分页改变的只是 K 和 V 住在哪里。（flash-attention kernel 内部的 tiling 与带宽推理是 [GPU field guide](#/blog?id=gpu-guide-for-dl) 那篇的主题；这篇文章把那个 kernel 当作既定条件。）

## 六、采样

logits 变成 token id，只用九行：

```python
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

把每个概率除以一个独立的 `Exponential(1)` 采样值再取 `argmax`，恰好等价于从分类分布里采样 —— 这就是换了个形式的 Gumbel-max trick，因为除以一个指数分布变量，在对数空间里就是减去一个 Gumbel 变量。不用 `torch.multinomial` 而这样写的理由是：它全部由逐元素算术加一次 argmax 组成，没有依赖数据的控制流，于是 `torch.compile` 能把整段融合成一个 kernel。

上面一层还有个相关的小技巧。prefill 期间模型会为每个 prompt token 算出隐状态，但只有每个序列的最后一个 token 才能产出下一个 token，所以 `ParallelLMHead` 在投影之前先切片：

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

对一批长 prompt 来说，这就把"在几千个位置上做词表规模的投影"变成了"只在寥寥几个位置上做"。

## 七、闭合循环

`postprocess()` 收尾这一轮：把刚填满的 block 哈希掉、推进已缓存 token 计数、追加新 token、并让结束的序列退场。

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

那个 `continue` 对应的是分块 prefill 的情况：prompt 还没处理完的序列会丢弃这次采样出的 token，因为从半截 prompt 预测出来的 token 没有意义。只有等 prompt 被完整消费完，生成才真正开始。

`generate()` 把这一切裹进一个循环，一直跑到两个队列都空为止，然后把每个序列累积的 id 解码回文本。

## 八、和真实 vLLM 比，少了什么

nano-vllm 确实还包含两样这次没细讲的东西，都属于工程而非算法。张量并行为每个 rank 起一个 `ModelRunner` 进程，由 rank 0 通过一块 `SharedMemory` 广播方法调用 —— 大约四十行手写的 RPC。CUDA graph 捕获则为一组固定的 batch size（`[1, 2, 4, 8] + range(16, max_bs+1, 16)`）预先录好 decode 步骤，之后重放"不小于当前 batch size 的最小那份" graph（`next(x for x in self.graph_bs if x >= bs)`），而不是每步都从 Python 重新派发每个算子；这也解释了为什么 `slot_mapping` 需要 `-1` 这个哨兵值，以及为什么只有形状可预测的 decode 才用 graph。

真正缺席的，是让生产级 vLLM 变得庞大的那些东西：投机解码、量化、多种调度策略、把 prefill 和 decode 拆到不同机器上的分离式服务，以及一个通用到能支持 Qwen3 之外各种架构的模型加载层。这些都不会改变上面这些机制 —— 它们是核心正确之后，才一层层堆上去的东西。
