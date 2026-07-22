---
title: "Sparse by Design: How Mixture-of-Experts Actually Works in 2026"
date: "2026/7/21"
---

Mixture-of-Experts (MoE) is the architecture underneath nearly every frontier language model shipping in 2026, and the term covers a much wider range of real designs than it sounds like it should. "MoE" by itself tells you almost nothing about how a model routes tokens, how it keeps experts balanced, how many experts exist versus how many actually fire, or whether the computation happening inside an "expert" even looks like a normal feed-forward block anymore. This post is about the mechanics underneath that one label: what a modern MoE layer actually does, where each of its moving parts came from historically, and which specific design levers today's frontier labs are actively pulling on to push the architecture further (routing strategy, gating function, load-balancing method, expert granularity, latent compression, adaptive per-token compute).

## 1. Anatomy of a modern MoE layer

Strip away everything vendor-specific and a 2026-era MoE layer looks like this: a token arrives, a small router network scores every expert's "affinity" for that token, the top-K highest-scoring experts actually run their feed-forward computation, and, in almost every current design, one additional shared expert runs on every token regardless of what the router decided. The outputs get combined into a single vector, weighted by the router's own scores, and that's the layer's output.

![Anatomy of a modern MoE layer](blogs/images/moe-layer-anatomy.svg?v=1)
^One token, one layer: every routed expert is a candidate, only a handful actually run. The shared expert is the one exception that always fires.

The entire appeal of this design is in the gap between two numbers. **Total parameters** scale with how many experts exist in the pool: this is the number vendors put in a headline ("2.8 trillion parameters"). **Active parameters** scale with how many experts actually run per token, top-K plus the shared expert, and this is the number that actually determines inference cost. A model can have an enormous total parameter count and still be cheap to run per token, provided K stays small relative to the pool. That gap between total and active is the entire reason MoE exists, and every design choice covered below is, one way or another, an attempt to widen that gap without hurting quality.

## 2. Where this actually comes from: nine years of lineage

Almost every mechanism discussed in this post is a variation on a much older idea, or a direct continuation of one specific 2024 paper. It's worth walking through that lineage before getting into what's changed recently. The idea is older than the deep-learning era it's usually associated with: **Jacobs et al., 1991** first proposed mixture-of-experts as a way to let different sub-networks specialize on different parts of a task, decades before anything resembling a modern transformer existed. What follows is the lineage that starts once the idea got adapted to large neural sequence models, the part of the history most people actually mean when they say "MoE."

![Nine years of MoE design](blogs/images/moe-lineage-timeline.svg?v=1)
^Most of 2026's frontier MoE designs trace back to one 2024 fork: DeepSeekMoE's fine-grained-experts-plus-shared-expert recipe.

**Shazeer et al., 2017** introduced the modern sparsely-gated MoE layer for deep learning: a noisy top-k softmax router with an auxiliary loss to keep expert usage balanced. This is the ancestor of essentially everything that follows.

**GShard (2020)** scaled the idea to a transformer trained with real expert parallelism across hardware: splitting experts across devices and using all-to-all communication to route tokens to whichever device holds their assigned expert. This is also where the standard load-balancing auxiliary loss formula comes from, and where a capacity factor (a hard cap on how many tokens each expert can accept before overflow tokens get dropped) entered the standard toolkit.

**Switch Transformer and GLaM (2021)** pushed in different directions from the same base: Switch Transformer simplified routing to top-1 (one expert per token) and showed MoE could train stably even at that extreme, while GLaM used top-2 routing to scale to 1.2 trillion parameters while activating only a small fraction of them per token, an early, clean demonstration of decoupling total parameters from compute cost.

**ST-MoE and expert-choice routing (2022)** addressed a training-stability problem (router logits could grow unbounded and destabilize training, addressed with a router z-loss) and explored an entirely different paradigm: instead of each token picking its top-K experts, expert-choice routing lets each expert pick its top-tokens, which guarantees perfect load balance by construction but complicates causal / autoregressive generation.

**Mixtral (2024)** didn't introduce new MoE mechanics so much as prove the whole approach could be trained and released as a strong open-weight model: top-2-of-8 routing with a fairly standard softmax router. It normalized MoE as a mainstream, non-exotic architecture choice for anyone building an open model.

**DeepSeekMoE (2024)** is the actual fork point for almost everything covered below. Its first contribution was fine-grained expert segmentation: instead of a handful of large experts, use many small ones, which gives the router far more combinatorial flexibility in how it allocates capacity to a token. Its second was isolating one or more shared experts that process every token unconditionally, so that common, general-purpose computation doesn't have to be redundantly re-learned across dozens of routed experts.

That fork is also where the shared-expert question starts, and different labs have answered it differently over the following two years, not only DeepSeek's own later models but also Alibaba's Qwen series, whose position on it reversed more than once. Section 7 below covers that in detail.

Two more recent, narrower ideas matter for what follows without being direct ancestors of the DeepSeekMoE line. **Hash Layers (Roller et al., 2021)** showed you could route tokens with a fixed, deterministic hash function instead of a learned gate at all, trading routing quality for guaranteed-even load and zero routing-network overhead (more on exactly how much quality that trade costs in section 4). And **Mixture-of-Depths (2024)**, which let transformers route tokens through or around entire compute blocks based on a learned per-token decision, is the clearest conceptual ancestor of "let some tokens skip real computation," alongside a separate 2024 technique for accelerating expert-parallel communication, ScMoE.

## 3. DeepSeek-V3: MoE's breakout moment

If DeepSeekMoE (2024) is the fork point, DeepSeek-V3 (late 2024) is the release that turned the fork into an industry default. It didn't change the fine-grained-experts-plus-shared-expert substrate at all; that part stayed unchanged. But it shipped three routing and training decisions on top of it that most of the field has since either adopted outright or explicitly defined itself against.

The headline change was **auxiliary-loss-free load balancing**. Instead of adding a separate loss term that competes with the language-modeling objective for gradient signal, DeepSeek-V3 gives each expert a learned bias term that gets added only to the routing decision (which experts get selected), not to the weight used to combine their outputs:

$$
\text{score}_i = g_i + b_i, \qquad \text{weight}_i = g_i
$$

$$
b_i \leftarrow \begin{cases} b_i - \gamma & i \text{ overloaded} \\[2pt] b_i + \gamma & i \text{ underloaded} \end{cases}
$$

Here $g_i$ is the router's own affinity score for expert $i$: the same kind of raw gating output covered in section 5, computed from the token's representation. $b_i$ is a separate, learned bias unique to expert $i$, adjusted by a small fixed step $\gamma$ once per training step, up if expert $i$ was underloaded on the last step and down if it was overloaded. The score, which includes the bias, decides the top-K selection: which experts actually run for a given token. The weight used afterward to combine those experts' outputs is $g_i$ alone, with the bias excluded, so the bias can only ever change which experts get picked, never how strongly their outputs count once picked. Balancing happens entirely through this per-step adjustment, with no competing loss coefficient to tune.

Layered on top, at a very small weight, is a complementary sequence-wise balance loss: a traditional GShard-style term, kept around specifically to catch imbalance within a single sequence that the bias term, updated at a coarser batch-level granularity, can miss on its own. The bias mechanism is the primary balancing signal; the sequence-wise loss is closer to a backstop. (Section 6 below places this alongside the other balancing mechanisms the rest of the field has tried.)

Two more decisions came bundled with it. **Node-limited routing** restricts each token's routing candidates to experts living on a bounded number of compute nodes rather than the entire cluster, capping how much all-to-all traffic a single token can generate. And a **multi-token prediction (MTP) module** trained alongside the main next-token objective provides a denser training signal and a natural building block for speculative decoding.

None of these four ideas individually reads as a dramatic departure from GShard-era MoE. What made DeepSeek-V3 MoE's "breakout moment" is that the combination shipped in a strong, openly documented model at a moment when the field needed exactly this reference point, and the field's reaction over the following eighteen months makes the point clearer than the mechanism does. GLM-5.2 keeps this MoE recipe almost completely unchanged. DeepSeek's own V4 keeps the bias mechanism as its baseline and only extends it. Even Qwen's independent lineage, which never adopted the bias-term mechanism itself, converged on the same underlying diagnosis: that learned routing needs some kind of insurance against router collapse. It reached for a shared expert as that insurance, for the same reason DeepSeek-V3 did. Three different labs, three different specific mechanisms, all downstream of the same finding.

## 4. Three ways to route: hash, learned, and Sinkhorn

Everything so far has assumed learned routing: a trainable gate scores experts, and gradient descent shapes those scores over time. That's the industry default, but it's a choice, not the only option, and comparing it against the alternatives explains why the field has spent so much engineering effort on load balancing rather than simply picking a routing mechanism that's balanced by construction.

![Three ways to assign tokens to experts](blogs/images/moe-routing-comparison.svg?v=1)
^Same 8 tokens, same 4 experts, three different assignment rules. Only one of the three both balances load and reads the tokens.

$$
\text{expert}(x) = h(\text{id}) \bmod N
$$

$$
g_i = \text{router}(x)\cdot e_i, \qquad \text{selected} = \operatorname{top\text{-}}k(g_1,\dots,g_N)
$$

$$
A \leftarrow \text{row\_normalize}(A), \qquad A \leftarrow \text{col\_normalize}(A)
$$

$$
\text{selected} = \operatorname{top\text{-}}k(A_i)
$$

Across all three, $x$ is the token's representation, $e_i$ is expert $i$'s learned key vector, and $N$ is the size of the expert pool. For hash routing, $h(\text{id})$ is a fixed hash of the token's position or identifier, with no learned parameters at all. For learned routing, $g_i$ is the router's trained affinity score for expert $i$, and top-$k$ selects the $k$ highest-scoring experts (the specific function computing $g_i$ from softmax, sigmoid, or a variant is covered in section 5). For Sinkhorn routing, $A$ is the affinity matrix for an entire batch, tokens by experts; the row and column normalization steps repeat for a handful of rounds before top-$k$ selection is applied to the normalized matrix $A_i$, though the combine weight afterward still uses the original, un-normalized $g_i$ rather than $A_i$, for reasons covered below.

Hash routing is the simplest of the three: it doesn't look at the token at all, just its position in the batch, so no two tokens ever fight over the same expert and no expert is ever starved. That guarantee is also its ceiling. Because assignment carries zero information about what the token actually is, the router has nothing to exploit, and the measured gain over a compute-matched dense model tops out around 1.5% at 16 experts and barely improves as the expert count grows, since adding more experts doesn't give a content-blind assignment rule any more signal to work with.

Learned routing is the opposite trade: because the gate is trained on the actual token representation, it can genuinely specialize (roughly 4% improvement at 16 experts, well over double hash routing's), but nothing in its training objective on its own prevents it from routing every token to the same handful of experts. That failure mode, routing collapse, tends to hit hardest in a network's first and last few layers, which is exactly the gap a shared expert (section 7) or an explicit balancing loss (section 6) is built to cover.

Sinkhorn routing tries to get both properties at once. Before top-k selection happens, the token-by-expert affinity matrix for an entire batch gets alternately normalized across its rows and its columns: row-normalize so each token's scores sum to one, column-normalize so each expert's incoming load sums to one, and repeat a handful of times. This converges toward a matrix that's simultaneously well-calibrated per token and evenly loaded per expert, achieving hash-level balance across every layer rather than only on average. The catch is that this near-perfect balance constrains how specialized any one expert can become relative to plain learned routing, and there's a specific implementation trap: because the Sinkhorn normalization is a fixed procedure with no gradient of its own, using its output directly as the combination weight silently detaches the router from the training signal entirely. The fix used in practice is to use the Sinkhorn-normalized matrix only to decide which experts get selected, while still computing the actual combination weights from the original, un-normalized logits, so the router keeps learning even though selection is balanced by a separate, non-differentiable process.

Learned routing's collapse risk is also where a much smaller, more obscure implementation detail turns out to matter more than it should. With top-1 routing specifically, the standard way of normalizing gate weights (dividing a token's gate value by the sum of gate values across its selected experts) collapses to dividing a number by itself, which always equals one. That silently zeroes out the gradient the router receives from the language-modeling loss, leaving only the load-balancing loss's gradient reaching the router at all: the router can learn to balance load perfectly while learning nothing about which expert actually suits which token. One documented fix is adding a "null expert": a phantom option that never actually computes anything, costs zero extra parameters and zero extra compute, but gives the router something real to contrast its top-1 choice against, restoring a working gradient signal from the main loss. It's a one-line fix for a bug that a router's usage statistics alone will never reveal, since perfectly balanced expert utilization looks identical whether or not the router is actually learning anything.

## 5. The gating function: from softmax to sigmoid and its variants

The router's core job is turning a token's representation into a score per expert, and which function does that scoring has quietly become one of the more consequential design choices in the whole architecture.

$$
\textbf{Softmax:}\quad g_i = \frac{\exp(x\cdot e_i)}{\sum_j \exp(x\cdot e_j)}
$$

$$
\textbf{Sigmoid:}\quad g_i = \frac{1}{1+\exp(-x\cdot e_i)}
$$

$$
\textbf{Sqrt(Softplus):}\quad g_i = \sqrt{\log(1+\exp(x\cdot e_i))}
$$

In every case, $x$ is the token's hidden representation and $e_i$ is expert $i$'s learned key vector, so $x\cdot e_i$ is their dot-product similarity, the raw logit. Softmax normalizes these logits across the whole pool of $N$ experts at once, so all $N$ scores are forced to sum to one: raising expert $i$'s logit necessarily lowers every other $g_j$. Sigmoid and Sqrt(Softplus) instead apply an independent function to each logit on its own, with no normalization across experts, so one expert's score moving has no effect on any other expert's score.

![Competing for a budget vs. scoring independently](blogs/images/moe-gating-comparison.svg?v=1)
^Same 5 experts, same starting logits, only expert 3's logit goes up. Under softmax, everyone else's score has to shrink to compensate; under sigmoid, nothing else moves at all.

Early MoE work (Shazeer, GShard, Switch Transformer, Mixtral) used softmax across all candidate experts, forcing every expert to compete for a shared probability budget: raising one expert's score necessarily lowers everyone else's, whether or not that's actually the right call for those other experts. DeepSeek-V3 switched to sigmoid, scoring each expert against a fixed threshold rather than against the rest of the pool. That distinction is largely cosmetic at single-digit expert counts, but it matters more as expert counts climb into the hundreds: with 256 or 896 experts, a single softmax adjustment gets divided across the entire pool, while sigmoid's independence stays constant no matter how large N gets.

That sigmoid choice has mostly held. [GLM-5.2](https://www.zhipuai.cn/zh/research/161) keeps it unchanged, and [DeepSeek-V4](https://arxiv.org/pdf/2606.19348) keeps the same independent-scoring philosophy while swapping the specific function to Sqrt(Softplus), described as only a minor adjustment from V3's sigmoid rather than a change in kind. [MAI-Thinking-1](https://microsoft.ai/pdf/mai-thinking-1.pdf) is the one clear departure, reverting to ordinary softmax computed on the token's original uncompressed representation, even though the actual expert computation happens in a compressed latent space (section 8). Notably, MAI-Thinking-1 is also the one architecture here that doesn't run MoE at every layer, interleaving dense feed-forward blocks with sparse MoE ones, which raises the possibility that the gating-function choice is somewhat coupled to how densely MoE is applied through the network rather than a universal improvement. [Kimi K3](https://www.kimi.com/blog/kimi-k3)'s gating function isn't disclosed in its release material. Qwen's lineage, notably, never left softmax at all through any of its generations, a reminder that sigmoid's advantages are real but not universal.

## 6. Load balancing: from auxiliary losses to bias terms to quantiles

Nothing in a router's training objective naturally prevents it from routing every token to the same handful of experts and ignoring the rest, a failure mode usually called routing collapse. How to prevent it without hurting quality is the part of MoE design that has visibly been reinvented the most in the last two years.

![DeepSeek-V3's bias-based balancing loop](blogs/images/moe-balancing-loop.svg?v=1)
^No competing loss term. The router's own bias adjusts itself, one training step at a time, based on last step's measured load.

$$
L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \, P_i
$$

$$
L_z = \frac{1}{B}\sum_{\text{tokens}} \Big(\log \sum_i \exp(\text{logit}_i)\Big)^{2}
$$

Here $f_i$ is the fraction of tokens actually sent to expert $i$ within a batch (a hard, non-differentiable count), and $P_i$ is the average routing probability the gate assigned to expert $i$ over that same batch (soft and differentiable). Multiplying the two together gives GShard's auxiliary loss $L_{\text{aux}}$ a gradient that pushes down specifically on experts that are both frequently selected and confidently scored, with $\alpha$ a tunable coefficient and $N$ the pool size. In the router z-loss $L_z$, $B$ is the batch size and the sum runs over every token's logit vector; squaring the log-sum-exp term penalizes large logits directly, keeping the softmax numerically stable regardless of whether load happens to be balanced. DeepSeek-V3 abandons both of these in favor of the per-expert bias mechanism covered in section 3 above.

GShard's original solution was that straightforward auxiliary loss, refined further by Switch Transformer and ST-MoE's router z-loss for logit stability. DeepSeek-V3 replaced the auxiliary loss entirely with its bias mechanism, balancing load without competing against the language-modeling loss for any gradient at all. DeepSeek-V4 keeps that bias mechanism but layers a small additional sequence-wise balance loss on top, specifically to catch imbalance within a single long sequence that looks fine only when averaged across an entire training batch, which matters given V4's million-token context target.

MAI-Thinking-1 goes a different direction, using a fairly traditional GShard-style auxiliary loss rather than a bias term. The paper reports that once $f_i$ and $P_i$ are aggregated globally across the whole training batch rather than computed per-microbatch, it performs about as well as the bias-based approach. Qwen3 reaches a strikingly similar conclusion independently, about a year earlier. The standard auxiliary loss is normally computed per micro-batch, and if a micro-batch happens to be domain-skewed (all code, say, because of how the data loader happened to batch it), that loss actively punishes the router for doing the sensible thing and sending code tokens to code-specialized experts, forcing artificial uniformity within every micro-batch. Qwen's fix, described in a dedicated write-up and formalized in a companion paper on load-balancing-loss implementation details, synchronizes each expert's selection frequency across all micro-batches before computing the loss, so the balance constraint applies to the corpus as a whole rather than to every individual slice of it. Individual micro-batches stay free to be domain-skewed while the aggregate stays balanced. Two labs, a year apart, converged on the same fix: the aggregation granularity of a balancing loss seems to matter more than the specific formula built on top of it.

Kimi K3 departs furthest from the DeepSeek-V3 default. Rather than a bias term at all, it uses what Moonshot calls Quantile Balancing, deriving expert allocation directly from the quantiles of the router's own score distribution, eliminating both the heuristic bias-update rule and a separate balancing hyperparameter that needed tuning. At Kimi K3's 896-of-16 sparsity, small routing imperfections have more room to compound than they do at DeepSeek-V3's 256-of-8, making one less hyperparameter to get wrong a reasonable thing for a frontier lab to want. Several different balancing mechanisms have shipped within about eighteen months of each other, and no released material makes a clean, controlled case that any one dominates the others, largely because each was tuned inside a specific architecture (different granularity, different gating function, different degree of dropless-ness) that makes an apples-to-apples comparison hard to extract from public reports alone.

## 7. Granularity and the shared-expert question

$$
\textbf{Coarse-grained:}\quad N \text{ experts, top-}K\text{ active}
$$

$$
\textbf{Fine-grained:}\quad N\!\cdot\! m \text{ experts, top-}(K\!\cdot\! m)\text{ active}
$$

Here $N$ is the number of experts in the coarse-grained baseline and $m$ is the segmentation factor DeepSeekMoE introduces. Each original expert's feed-forward block is split into $m$ equal-sized pieces, so the fine-grained pool has $N\cdot m$ experts instead of $N$, and top-$K$ becomes top-$(K\cdot m)$ to keep the same fraction of the pool active. The active compute per token is identical either way, since the same total width is doing the work, but the router now chooses from a far larger combinatorial space: with $N=8$ and $K=2$ there are only a handful of possible pairings, while $N\cdot m=64$ and $K\cdot m=16$ opens up millions.

![Same total capacity, sliced differently](blogs/images/moe-granularity-comparison.svg?v=1)
^8 experts, top-2 gives 4 possible pairs. Split the same total width into 64 pieces at top-16, and the number of ways to fill that budget explodes into the millions: same compute, far more ways to specialize.

DeepSeekMoE's fine-grained-experts idea is no longer under active debate the way the shared-expert half of its recipe is: it's the substrate nearly everything in this post starts from. Kimi K3 pushes this further than anyone else in this comparison, with 896 experts and top-16 selected.

Whether a shared expert belongs alongside those fine-grained experts is a genuinely open question, not a settled default. Alibaba's Qwen team has run the longest, most public experiment on exactly this question, reversing its own answer twice.

![Qwen's shared-expert reversal](blogs/images/moe-qwen-lineage.svg?v=1)
^Same lineage, opposite conclusion, then a partial walk-back: Qwen kept adding shared experts for a year, then removed them entirely, then a newer model reportedly brought one back.

Qwen1.5-MoE (March 2024) and Qwen2-MoE (mid-2024) both combined fine-grained routed experts with several always-on shared experts: first 4 shared plus 60 routed at top-4, then 8 shared plus 64 routed at top-8. Qwen2's own technical report states plainly that shared experts "facilitate the application of...experts across various tasks while reserving others for selective use in specific routing scenarios," citing the same router-collapse concern that motivated DeepSeekMoE and DeepSeek-V3's shared expert. Qwen3 (May 2025) then reversed course entirely: its technical report states directly that "unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts," relying instead on the global-batch load balancing loss covered in section 6 as its insurance against router collapse. Secondary reporting on Qwen3-Next (September 2025), not yet confirmed in a primary technical report, describes a shared expert reappearing alongside 10 routed-of-512 experts, which if accurate would make Qwen's own history a complete round trip inside about eighteen months: shared experts in, then out, then reportedly back in again.

MAI-Thinking-1 arrives at "no shared expert" from a different angle: the team tested one and found it only helped in architectures that run MoE at every layer. Because MAI-Thinking-1 interleaves dense and sparse layers instead, a shared expert added no measurable benefit in their own ablations, and the shipped model doesn't use one. [LongCat-2.0](https://longcat.chat/blog/longcat-2.0/) sidesteps the binary question entirely, splitting its routed experts into three task-specialized groups (Agent, Reasoning, Interaction) that are gated per task, rather than treating the whole pool as one undifferentiated set of interchangeable specialists plus a single shared one.

## 8. Compressing the computation itself: LatentMoE

$$
z = W_{\text{down}}\, x, \qquad y_i = \text{expert}_i(z), \qquad \text{out} = W_{\text{up}}\, y_i
$$

$$
\text{cost} \propto N\ell,\ K\ell
$$

Here $x$ is the token's full-dimension representation, with dimension $d$, and $z$ is its projection into a smaller latent space of dimension $\ell < d$ via a learned down-projection matrix $W_{\text{down}}$. The expert computation happens entirely in this compressed space, and a learned up-projection $W_{\text{up}}$ expands the result back to dimension $d$ before it's combined with the rest of the layer's output. Because both the per-expert compute and the all-to-all communication payload scale with $\ell$ rather than $d$, the cost of running $N$ experts at top-$K$ scales as shown above instead of the usual $Nd, Kd$. The gap between $d$ and $\ell$ is exactly the budget that a larger expert pool or a larger top-K can be paid for with, at no extra total cost.

![LatentMoE: compress, compute, expand](blogs/images/moe-latentmoe-flow.svg?v=1)
^Standard MoE pays the full token dimension d for every expert and every all-to-all hop. LatentMoE pays a compressed dimension l instead, and reinvests the difference into a bigger pool.

Every idea covered so far leaves the expert computation itself alone and only changes routing or balancing around it. LatentMoE, first proposed by NVIDIA in early 2026 and already shipping in NVIDIA's own Nemotron-3 Super and Ultra models, changes the computation directly.

MAI-Thinking-1 adopts this directly, growing from an earlier 192-expert, top-4 configuration to 512 experts at top-8, specifically once LatentMoE made the larger configuration affordable to communicate. It also runs the model fully dropless, with no token-dropping capacity limit at all, a choice the paper says was driven by capacity-related inconsistencies that showed up in balancing ablations once a hard cap was in the picture. Kimi K3's "Stable LatentMoE" appears to be the same latent-compression family applied at a more extreme granularity (896 experts, top-16), though Moonshot's own material doesn't disclose the compression ratio ($\ell$ relative to $d$) or expert-expansion ratio the way MAI-Thinking-1's report does.

## 9. Making compute itself adaptive

Everything above still assumes a fixed top-K: every token activates the same number of experts, whether it's a punctuation mark or a genuinely hard reasoning step. LongCat-2.0 breaks that assumption directly. Its expert pool includes what Meituan calls Zero-Computation Experts: experts that do no computation at all and simply return the input unchanged. The router decides, per token, how many real experts versus zero-computation experts to use. A simple token might route almost entirely to zero-computation experts and cost next to nothing; a difficult token engages more real expert capacity. This is a token-adaptive compute budget rather than a fixed top-K, conceptually closer to Mixture-of-Depths' 2024 idea of letting tokens skip computation blocks entirely than to anything in the DeepSeekMoE lineage: the same idea, applied inside the expert dimension instead of the depth dimension.

This is also the one area with a real, unanswered open question. Meituan hasn't published detailed load-balancing mechanics for this design, and the zero-computation-expert mechanism raises an obvious one: what stops the router from routing everything to zero-computation experts to minimize loss on easy tokens during training. The public material doesn't yet say.

## 10. The systems layer: what routing costs in communication

Every routed token has to physically travel, over an interconnect, to whichever accelerator holds its selected expert. This cost only grows more pressing as expert counts climb into the hundreds (the [distributed-training post](#/blog?id=distributed-training-for-dl) covers this all-to-all communication pattern in more depth). Several of the choices above are, from different angles, responses to that same cost. Hash routing's revival in DeepSeek-V4 (deterministic, no learned gate, applied specifically to its first three MoE layers in place of what used to be dense FFN blocks) trades routing quality for guaranteed-even load and zero routing-network communication overhead in those layers. LatentMoE's compressed latent space directly shrinks the payload that has to move in an all-to-all exchange, which is precisely the lever that let MAI-Thinking-1 and Kimi K3 afford larger expert pools in the first place. And DeepSeek-V4's other systems contribution, a pipelined expert-parallelism kernel open-sourced as MegaMoE, overlaps the dispatch, compute, and combine stages of expert-parallel MoE across "waves" of experts rather than running them sequentially, reporting meaningful speedups over a comparable overlap scheme called Comet. LongCat's ScMoE backbone is a similarly systems-driven choice, a shortcut-connected design built specifically to improve expert-parallel throughput. None of these are quality improvements in the way sigmoid gating or fine-grained experts are: they're the engineering that makes the quality improvements affordable to actually run.

Separately, DeepSeek-V4 introduces Anticipatory Routing, a training-stability technique rather than a communication one: when the model detects a loss spike, which the paper ties specifically to outlier activations inside MoE layers, it starts computing routing decisions from a slightly stale copy of the router's own weights rather than the current ones, decoupling how fast the router changes from how fast the rest of the model changes until the instability passes.

## 11. Today's configurations, side by side

GLM-5.2 and LongCat-2.0's own pages say little about MoE-specific numbers, so the figures below for those two are compiled from third-party technical coverage rather than each model's own release material; everything else is drawn directly from the source reports.

| Model | Total params | Active params | Routed experts | Shared experts | Top-K | Gating |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | 671B | 37B | 256 | 1 | 8 | Sigmoid |
| DeepSeek-V4-Flash | 284B | 13B | 256 | 1 | 6 | Sqrt(Softplus) |
| DeepSeek-V4-Pro | 1.6T | 49B | 384 | 1 | 6 | Sqrt(Softplus) |
| GLM-5.2 | 753B | 40B | 256 | 1 | 8 | Sigmoid |
| MAI-Thinking-1 | 962B | 34.7B | 512 | 0 | 8 | Softmax |
| Kimi K3 | 2.8T | not disclosed | 896 | yes, count undisclosed | 16 | not disclosed |
| LongCat-2.0 | 1.6T | ~48B | dynamic pool | 3 task groups instead | dynamic | router picks real vs. zero-compute |
| Qwen2-MoE | — | — | 64 | 8 | 8 | Softmax |
| Qwen3-MoE (235B) | 235B | 22B | 128 | 0 | 8 | Softmax |

![interactive:moe-sparsity](#)

![interactive:moe-lookup](#)

## 12. What's converged, and what's still an open argument

Fine-grained routed experts has stopped being a debated choice within about two years of DeepSeekMoE's publication: it's the substrate nearly everything in this post starts from. Sigmoid-family gating has displaced softmax at most, though notably not all, frontier labs, with MAI-Thinking-1's reversion to softmax being the one clear 2026 exception, and Qwen's entire lineage never having left softmax to begin with.

The shared-expert question is the opposite of converged. DeepSeek-V3, GLM-5.2, and Kimi K3 keep one; MAI-Thinking-1 tested one and dropped it; Qwen added shared experts for over a year, removed them entirely in Qwen3, and, per unconfirmed reporting, may have reintroduced one in Qwen3-Next. If there's a single design axis in this whole survey where the field visibly hasn't made up its mind, this is it.

Load balancing is still visibly being reinvented every few months: a learned per-expert bias term, that same bias term plus a sequence-level loss, a fully global-batch aggregated traditional loss (reached independently by both Qwen and MAI-Thinking-1), and quantile-derived allocation with no separate hyperparameter at all are different answers shipping within about eighteen months of each other, without a controlled comparison anywhere in the public record to say which wins. Underneath all of them sits the same three-way choice between hash, learned, and Sinkhorn routing covered in section 4, and every frontier model in this survey has made the same choice at that level (learned routing, with some insurance mechanism bolted on), even as they disagree sharply about which insurance mechanism to use.

The most consequential open fork above the balancing-mechanism layer is between two different ways of widening the gap between total and active parameters. LatentMoE keeps a static top-K but shrinks the effective cost of a larger K and larger N by moving computation into a compressed latent space: more experts, more of them active, but each one cheaper. LongCat-2.0's zero-computation experts instead keep the pool's nominal size fixed but make the number of "real" experts activated a dynamic, token-dependent quantity. These are not mutually exclusive, and it would not be surprising to see a future report combine both: latent-space compression to make a bigger pool affordable, and adaptive per-token activation to avoid spending that budget uniformly on tokens that don't need it. As with any snapshot of a fast-moving architecture, some of what reads as settled convergence here may look different in six months.
