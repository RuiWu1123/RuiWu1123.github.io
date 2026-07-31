---
title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM"
date: "2026/7/31"
---

训练一个模型和把它服务上线，是两个完全不同的工程问题。训练关心的是在提前知道、规模巨大又相对均匀的 batch 上跑出多高的吞吐；服务关心的则是一串到达时间不可预测、生成长度不可预测的请求，怎么样尽快给出响应。为第一个问题设计的框架，拿来解决第二个问题会很吃力，这也是为什么会存在"推理引擎"这样一整类专门软件，专门负责把一个训练好的模型高效地跑在真实流量上。vLLM 就是开源世界里最知名的一个。

这篇文章用 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 作为理解推理引擎到底在做什么、为什么这么做的载体——这是一个用约 1200 行、可读性很高的 Python 代码，从零重写了 vLLM 核心思想的项目。真正的 vLLM 有几万行代码，涉及各种调度器、自定义 kernel，以及多年积累下来的渐进式优化；nano-vllm 只保留了最核心的机制：一份干净的 continuous batching 实现，一份干净的 paged KV-cache 内存实现，一份干净的 prefix caching 实现，而且吞吐还相当接近 vLLM 本体（按项目自己的 benchmark，在单张 RTX 4070 Laptop 上跑 Qwen3-0.6B、256 并发序列，1434 tokens/s 对 vLLM 的 1362 tokens/s）。这种"小到可以通读、又忠实到能教会你真东西"的组合，正是它值得被这样一篇文章通读一遍的原因。

## 一、推理引擎到底在解决什么问题

想象一种最朴素的模型服务方式：请求到达，逐个 token 跑完模型直到生成结束，再处理下一个请求。在任何真实规模下，这样做马上会出两个问题。

第一个是内存。一个序列每生成一个 token，它的 key、value 向量（也就是 KV cache）就必须一直留着，供之后每一步使用，这份 cache 得在这个序列存活期间一直占着 GPU 内存。最朴素的做法是，每个请求提前按"能允许的最长长度"预留一段连续内存。但绝大多数请求根本用不到这么多。预留量和实际用量之间的这个差距，在每一个并发运行的请求上都在被浪费，这也是朴素服务代码往往撑不了几个并发序列就把 GPU 内存耗尽的头号原因。

第二个是调度。如果请求被打包成 batch、步调一致地一起跑，那这个 batch 的前进速度就取决于跑得最慢的那个成员；而一个在 batch 跑到一半时到达的新请求，得等整个 batch 跑完才能加入。静态 batching 会让 GPU 在"排队的活不够多"和"卡在拖后腿的成员上"这两种状态之间来回切换，无论对吞吐还是对单个请求体验到的延迟都不友好。

vLLM 最出名的两项贡献，正好分别对应这两个问题的解法：**PagedAttention** 用小的、固定大小、非连续的 block 来管理 KV cache，而不是给每个序列预留一整块连续大内存——这个思路直接借用了操作系统几十年前就想明白的物理内存分页方案；**continuous batching** 让调度器可以在每一步都往正在运行的 batch 里加入或移出单个序列，而不是等一整个 batch 完全跑完。nano-vllm 把这两个机制都实现了，还额外实现了第三个机制——prefix caching，把已经算过的 KV block 在共享同一个前缀（比如重复出现的 system prompt）的不同请求之间复用。接下来这篇文章会沿着一个 prompt 的生命周期，把这三者背后的真实代码走一遍。

## 二、大地图：一个请求在仓库里的完整路径

在逐行追踪生命周期之前，先看一眼整个系统的形状：七个文件，每个文件里只有几个真正重要的函数，按请求实际流经它们的顺序排列。

![Nano-vLLM: one request's path through the repo](blogs/images/nanovllm-architecture-map.svg?v=1)

![interactive:nanovllm-arch](#)

`LLMEngine` 是入口，也是最外层的循环。`Scheduler` 决定某一步到底该跑哪些序列。`BlockManager` 掌管物理 KV-cache 内存，负责发放 block。`ModelRunner` 把调度决策转换成真正的 GPU 张量。`Qwen3ForCausalLM` 是 transformer 本体。`Attention`，也就是图中被高亮的那一块，是 PagedAttention 真正发生的地方，后面会单独开一节深入讲，而不是简单带过。`Sampler` 把模型输出的 logits 转换成下一个 token id。图下方的虚线箭头就是那个循环：`generate()` 会反复调用 `step()`，直到 batch 里的每一个序列都结束。

## 三、第一、二站：一个 prompt 变成 `Sequence`，然后被调度

一个请求通过 `LLMEngine.add_request()` 进入系统，这个函数会把 prompt 分词，包成一个 `Sequence` 对象——一小份状态，记录着目前已经生成的 token id、这个序列拥有哪些物理 KV block，以及它的 prompt 里有多少 token 已经被某个缓存前缀覆盖了。这个 `Sequence` 会被塞进 `waiting` 队列，在下一次调度之前不会发生任何事。

调度正是 continuous batching 真正落地的地方：

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

每次调用 `schedule()`，都会优先处理 **prefill** 工作——也就是新请求或还没处理完 prompt 的序列——而不是 **decode** 工作——给已经在跑的序列多生成一个 token。只有当这一步已经没有 prefill 工作可以打包了，调度器才会退而求其次，让正在运行的序列各推进一个 decode 步。正是这种优先级顺序，让每一步的 batch 组成都是流动的：一个全新到达的请求可以在下一次 `schedule()` 调用里立刻开始 prefill，完全不用等任何一个正在跑的序列结束。这就是 continuous batching，用九行 Python 就写完了。

`while ... else` 那段处理的是一种情况：某个正在运行的序列拿不到它下一步需要的 KV block（`can_append` 失败了，具体原因第六节会讲）。这时调度器会把最近才加入 running 的那个序列通过 `preempt()` 驱逐回 `waiting`，从而腾出空间；`preempt()` 会释放它占用的 block，让它之后可以重新 prefill。这就是当并发需求超过可用缓存内存时的泄压阀。

## 四、第三、四站：先占内存，再搭 GPU 张量

一个序列开始运行之前，`BlockManager.can_allocate()` / `allocate()`（在上面 `schedule()` 内部被调用）得先按它 prompt 需要的量，预留若干个固定大小的 KV-cache block，如果碰巧和某个已有前缀完全一致，还可能直接复用已经缓存好的 block。这个机制是下一节的主题；这里先把它当成一个黑盒，它对每个序列返回一个 `block_table`——一份物理 block id 的列表。

`ModelRunner` 拿到被调度的序列和它们的 block table，把它们变成模型真正会消费的张量。对于 prefill，所有被调度序列里还没缓存的 prompt token 会被拍平成一个一维张量，完全不做 padding，用累积偏移量（`cu_seqlens_q`、`cu_seqlens_k`）标记每个序列的 token 从哪里开始、到哪里结束：

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

这种"变长打包"（varlen）正是 nano-vllm 从不需要把 batch 填充到最长成员长度的原因：五个长度差异很大的序列，直接变成一条长张量，外加几个标记每个序列起止位置的数字。`prepare_decode()` 是 decode 步对应的函数：每个正在运行的序列只贡献一个新 token，而不是整段 prompt，同时还会带上每个序列当前的长度（`context_lens`）和它的 `block_table`——后面的 attention 步骤需要靠它才知道该去读哪些物理缓存 block。这里构造出来的 `slot_mapping` 张量，把每个新 token 精确映射到它该写入的物理缓存槽位，正是连接这一节和接下来 PagedAttention 深入章节的那根线。

## 五、第五站：跑模型

张量准备好之后，`ModelRunner` 调用真正的 transformer，也就是 `Qwen3ForCausalLM`。这一步本身没有任何推理引擎特有的东西，就是标准的 decoder-only 前向传播：把输入 id 变成 embedding，跑过 N 层 decoder layer，最后做一次归一化：

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

每个 decoder layer 内部都有一个 `Attention` block，引擎相关的工作正是从这里重新开始。与其把它简单塞进这段流水账，不如单独开一节讲——因为这是"怎么高效服务一个模型"和"transformer 本身怎么运作"这两个话题真正交汇的地方。

## 六、跳出链条：PagedAttention，从 block table 到 kernel

回到第一节提到的内存问题：给每个序列预留一整块连续、按最坏情况估计大小的内存，绝大部分预留出来的空间都被浪费了。PagedAttention 的解法，和操作系统几十年前对物理内存做的事情一模一样：不再要求连续性。把 KV cache 切成小的、固定大小的 block（nano-vllm 默认配置里是每块 256 个 token），从一个所有序列共享的池子里取用，再给每个序列一张小小的 **block table**——一份把逻辑 block 编号（0、1、2……）映射到某个空闲物理 block 的对照表。

![The block table: logical blocks map to scattered physical slots](blogs/images/nanovllm-block-table.svg?v=1)

一个序列自己看到的 KV cache 依然是完全顺序的：block 0，然后 block 1，然后 block 2。和朴素方案不同的地方在于，这些逻辑 block 可以落在物理池子里的任何位置——就像上图里的物理 block 7、2、15，是散落的而不是连续的。除了每个序列最后一个 block 里没填满的那部分（最多浪费 255 个 token 的空间，无论这个序列最终会长到多长），几乎不存在任何浪费，不再是整块预留却大半用不上的最坏情况缓冲区。

`BlockManager` 就是掌管这个池子、负责发放 block 的代码，它还顺手做了一件很值得停下来看看的事：基于内容做寻址复用。

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

每个填满的 block，它的 token id 都会被哈希，而且这个哈希是链式的，串上了前一个 block 的哈希值，所以哈希捕捉到的不只是"这 256 个 token 是什么"，而是"这 256 个 token，紧跟在这一整段特定的历史后面"。如果一个新序列开头的几个 block，哈希值和池子里已有的 block 对上了，`BlockManager` 就会直接复用同一个物理 block，而不是重新计算、重新写入，只是把引用计数加一。这正是为什么两个共享同一段很长 system prompt 的请求，那段 system prompt 的 KV cache 只需要被真正算过、存过一次：

![Prefix caching: identical leading blocks share one physical block](blogs/images/nanovllm-prefix-cache.svg?v=1)

序列 A 和序列 B 在共享的 system prompt 之后开始分叉，但只要前缀还一致，它们的 block table 就都指向同一批物理 block 4 和 9；只有等实际内容真正不一样了（block 7 对 block 12），才会各自分开。

还剩两件事：怎么把新算出来的 K/V 向量写进这些散落的物理 block 里，以及 attention 计算时怎么把它们读回来。写入是一个 scatter 操作，`store_kvcache_kernel`，一个 Triton kernel：每个 token 新算出来的 key/value 向量，都会被直接写到 `slot_mapping` 里给出的那个物理缓存槽位——就是第四节 `prepare_prefill` 里构造的那个张量。读取发生在 `Attention.forward()` 内部，也正是在这里，block table 这层间接寻址真正开始发挥作用：`flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 都直接接受一个 `block_table` 参数，所以真正去把每个序列散落的物理 block 收集起来的工作，是 attention kernel 自己做的，外层的 Python 代码完全不需要先把 cache 拼成一整块连续内存：

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

底层用到的这个 flash-attention kernel 本身，正是 [GPU field guide 那篇文章](#/blog?id=gpu-guide-for-dl)里 tiling 和内存带宽那部分讨论的对象——如果"为什么不把完整的 attention 矩阵摆出来"这个问题本身让你感兴趣，那篇文章里 roofline 那一节会讲得更深。这里想强调的是更窄的一点：PagedAttention 完全没有改变 attention 的数学本身，缩放点积再 softmax，还是缩放点积再 softmax。它改变的只是 K、V 向量物理上存在哪里，而 block table 加上这层 kernel 级别的收集操作，正是让"非连续存储"这件事对数学计算完全透明的全部机制。

## 七、回到链条：采样

模型的前向传播最终产出 logits——词表里每个 token 一个分数——`Sampler` 把它们变成真正的下一个 token id。nano-vllm 的采样器先做温度缩放，再用 Gumbel-max trick 采样，这是一种和直接调用 `torch.multinomial` 数学上等价、但对 `torch.compile` 更友好的写法：

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

把每个概率除以一个独立的 `Exponential(1)` 采样值，再取 `argmax`，在数学上和直接从分类分布 `probs` 里采样是等价的，但整个过程只由逐元素运算和一次 argmax 组成，没有任何依赖数据的控制流分支——这正是 `torch.compile` 能把它编译成一个高效融合 kernel 所需要的形状。

## 八、闭合循环：postprocess、抢占，以及 `generate()`

回到 `LLMEngine.step()`：一旦模型产出了这一批新 token id，`Scheduler.postprocess()` 会把每个新 token 追加到对应序列，检查这个序列是不是碰到了结束符或者达到了最大长度，如果是，就标记它结束，并把它占用的 block 归还给空闲池。还没结束的序列则原样留在 `running` 里，等调度器下一次调用时继续被处理。

`generate()` 是把这一切串起来的最外层循环：先通过 `add_request()` 提交每一个 prompt，然后反复调用 `step()`——每次调用对应一轮调度，可能是 prefill 也可能是 decode——直到所有序列都结束。这个"循环回去"正是第二节架构图底部那根虚线箭头。第三节提到的抢占机制，就是保证这个循环在内存压力下依然正确的关键：如果 block 池子的扩张速度跟不上所有正在运行序列的需求，最近才加入 running 的那个序列会被驱逐回 `waiting`，而不是让整个系统卡死或报错；等腾出空间之后，它会被重新 prefill，而且第六节提到的那套基于内容哈希的机制，还能让它复用那些依然有效的旧 block。一个序列真正结束之后，它积累下来的 token id 会被重新解码回文本，这就是调用方最终拿到的响应。

## 九、nano-vllm 省掉了什么

nano-vllm 对自己范围的诚实，也是它作为教学范例好用的原因之一：上面这三个机制，它都实现得很完整，其他几乎全部省掉了。它确实还顺带实现了两个没有出现在上面这条主线里的功能，因为它们更偏工程而不是核心算法：张量并行（tensor parallelism），把模型本身切分到多张 GPU 上，靠 `torch.multiprocessing` 加一个小型共享内存 RPC 机制来协调，而不是用更重的分布式框架；以及 CUDA graph 捕获，针对几个固定的 batch size 提前录制好 GPU 操作序列，让 decode 步可以直接重放一份 graph，而不用每一步都从 Python 重新派发每个算子，从而削减单步的启动开销。

它完全没有实现的，是生产级 vLLM 之所以庞大得多的大部分原因：投机解码（speculative decoding）、为不同 SLA 目标调优的多种调度策略、量化支持、把 prefill 和 decode 拆到不同机器上的分离式服务，以及一个通用得多、需要跨各种模型架构工作的模型加载与 kernel 选择层——而 nano-vllm 因为只针对一个模型家族，完全不用考虑这些。这些都不会改变这篇文章走过的核心机制；它们是"用 1200 行代码把核心算法做对"变成"把这套东西真正大规模跑在生产环境里"之后，才会被一层一层叠加上去的东西。
