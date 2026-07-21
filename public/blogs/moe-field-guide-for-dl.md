---
title: "Mixture-of-Experts from First Principles: A Field Guide to How MoE Actually Works in 2026"
date: "2026/7/21"
---

Mixture-of-Experts (MoE) is the architecture underneath nearly every frontier language model shipping in 2026, and the term covers a much wider range of real designs than it sounds like it should. "MoE" by itself tells you almost nothing about how a model routes tokens, how it keeps experts balanced, how many experts exist versus how many actually fire, or whether the computation happening inside an "expert" even looks like a normal feed-forward block anymore. This post is about the mechanics underneath that one label: what a modern MoE layer actually does, where each of its moving parts came from historically, and which specific design levers — gating function, load-balancing method, expert granularity, latent compression, adaptive per-token compute — today's frontier labs are actively pulling on to push the architecture further.

The concrete evidence throughout is drawn from a close read of five technical reports released within weeks of each other this summer — [Kimi K3](https://www.kimi.com/blog/kimi-k3) (Moonshot AI), [GLM-5.2](https://www.zhipuai.cn/zh/research/161) (Zhipu), [MAI-Thinking-1](https://microsoft.ai/pdf/mai-thinking-1.pdf) (Microsoft AI), [DeepSeek-V4](https://arxiv.org/pdf/2606.19348) (DeepSeek-AI), and [LongCat-2.0](https://longcat.chat/blog/longcat-2.0/) (Meituan) — plus the foundational MoE literature they all build on. A brief sourcing note: the Kimi K3, MAI-Thinking-1, and DeepSeek-V4 figures below come from a full read of each model's own release material. GLM-5.2's own announcement page says little about MoE-specific architecture (it focuses on coding benchmarks and infrastructure), so its expert-count and top-K figures here are compiled from third-party technical coverage rather than Zhipu's own text. LongCat-2.0's blog page didn't render its architecture details in a form this research could capture directly, so its numbers are likewise third-party. Anywhere a figure is secondary-sourced rather than confirmed from the model's own report, it's noted inline.

## 1. Anatomy of a modern MoE layer

Strip away everything vendor-specific and a 2026-era MoE layer looks like this: a token arrives, a small router network scores every expert's "affinity" for that token, the top-K highest-scoring experts actually run their feed-forward computation, and — in almost every current design — one additional shared expert runs on every token regardless of what the router decided. The outputs get combined into a single vector, weighted by the router's own scores, and that's the layer's output.

![Anatomy of a modern MoE layer](blogs/images/moe-layer-anatomy.svg?v=1)
^One token, one layer: every routed expert is a candidate, only a handful actually run — the shared expert is the one exception that always fires.

The entire appeal of this design is in the gap between two numbers. **Total parameters** scale with how many experts exist in the pool — this is the number vendors put in a headline ("2.8 trillion parameters"). **Active parameters** scale with how many experts actually run per token — top-K plus the shared expert — and this is the number that actually determines inference cost. A model can have an enormous total parameter count and still be cheap to run per token, provided K stays small relative to the pool. That gap between total and active is the entire reason MoE exists, and every design choice covered below is, one way or another, an attempt to widen that gap without hurting quality.

## 2. Where this actually comes from: nine years of lineage

Almost every mechanism discussed in this post is a variation on a much older idea, or a direct continuation of one specific 2024 paper — worth walking through before getting into what's changed recently.

![Nine years of MoE design](blogs/images/moe-lineage-timeline.svg?v=1)
^Most of 2026's frontier MoE designs trace back to one 2024 fork: DeepSeekMoE's fine-grained-experts-plus-shared-expert recipe.

**Shazeer et al., 2017** introduced the modern sparsely-gated MoE layer for deep learning: a noisy top-k softmax router with an auxiliary loss to keep expert usage balanced. This is the ancestor of essentially everything that follows.

**GShard (2020)** scaled the idea to a transformer trained with real expert parallelism across hardware — splitting experts across devices and using all-to-all communication to route tokens to whichever device holds their assigned expert. This is also where the standard load-balancing auxiliary loss formula comes from, and where a capacity factor (a hard cap on how many tokens each expert can accept before overflow tokens get dropped) entered the standard toolkit.

**Switch Transformer and GLaM (2021)** pushed in different directions from the same base: Switch Transformer simplified routing to top-1 (one expert per token) and showed MoE could train stably even at that extreme, while GLaM used top-2 routing to scale to 1.2 trillion parameters while activating only a small fraction of them per token — an early, clean demonstration of decoupling total parameters from compute cost.

**ST-MoE and expert-choice routing (2022)** addressed a training-stability problem (router logits could grow unbounded and destabilize training, addressed with a router z-loss) and explored an entirely different paradigm — instead of each token picking its top-K experts, expert-choice routing lets each expert pick its top-tokens, which guarantees perfect load balance by construction but complicates causal / autoregressive generation.

**Mixtral (2024)** didn't introduce new MoE mechanics so much as prove the whole approach could be trained and released as a strong open-weight model — top-2-of-8 routing with a fairly standard softmax router — and normalized MoE as a mainstream, non-exotic architecture choice for anyone building an open model.

**DeepSeekMoE (2024)** is the actual fork point for almost everything covered below. Its two contributions were, first, fine-grained expert segmentation — instead of a handful of large experts, use many small ones, which gives the router far more combinatorial flexibility in how it allocates capacity to a token — and second, isolating one or more shared experts that process every token unconditionally, so that common, general-purpose computation doesn't have to be redundantly re-learned across dozens of routed experts.

**DeepSeek-V3 (late 2024)** then made three changes that turned the DeepSeekMoE recipe into what's now closer to an industry default: sigmoid gating instead of softmax, an auxiliary-loss-free load-balancing method (a learned per-expert bias term added to routing scores and adjusted up or down based on recent load, rather than a separate loss term competing with the language-modeling objective), and a multi-token prediction (MTP) module trained alongside the main model.

Two more recent, narrower ideas matter for what follows without being direct ancestors of the DeepSeekMoE line: **Hash Layers (Roller et al., 2021)** showed you could route tokens with a fixed, deterministic hash function instead of a learned gate at all, trading routing quality for guaranteed-even load and zero routing-network overhead. And **Mixture-of-Depths (2024)**, which let transformers route tokens through or around entire compute blocks based on a learned per-token decision, is the clearest conceptual ancestor of "let some tokens skip real computation," alongside a separate 2024 technique for accelerating expert-parallel communication, ScMoE.

## 3. The gating function: from softmax to sigmoid and its variants

The router's core job is turning a token's representation into a score per expert, and which function does that scoring has quietly become one of the more consequential design choices in the whole architecture. Early MoE work — Shazeer, GShard, Switch Transformer, Mixtral — used softmax across all candidate experts, which forces every expert to compete for a shared probability budget: raising one expert's score necessarily lowers another's. DeepSeek-V3 switched to sigmoid, scoring each expert independently rather than in competition with the others, a change that turns out to matter more as expert counts climb into the hundreds rather than staying in the single digits.

That sigmoid choice has mostly held: GLM-5.2 keeps it unchanged, and DeepSeek-V4 keeps the same independent-scoring philosophy while swapping the specific function to Sqrt(Softplus) — described as only a minor adjustment from V3's sigmoid, not a change in kind. MAI-Thinking-1 is the one clear departure, reverting to ordinary softmax, computed on the token's original uncompressed representation even though the actual expert computation happens in a compressed latent space (more on that below) — notably, MAI-Thinking-1 is also the one architecture here that doesn't run MoE at every layer, interleaving dense feed-forward blocks with sparse MoE ones, which raises the possibility that the gating-function choice is somewhat coupled to how densely MoE is applied through the network rather than a universal improvement. Kimi K3's gating function isn't disclosed in its release material.

## 4. Load balancing: from auxiliary losses to bias terms to quantiles

Nothing in a router's training objective naturally prevents it from routing every token to the same handful of experts and ignoring the rest — a failure mode usually called routing collapse — and how to prevent it without hurting quality is the part of MoE design that has visibly been reinvented the most in the last two years.

GShard's original solution was a straightforward auxiliary loss added to the training objective, penalizing imbalanced expert usage, refined further by Switch Transformer and ST-MoE's router z-loss for logit stability. DeepSeek-V3 replaced the auxiliary loss with a different mechanism entirely: a learned per-expert bias term added to routing scores, nudged up when an expert is underused and down when it's overused, which balances load without competing against the language-modeling loss for gradient signal — hence "auxiliary-loss-free." DeepSeek-V4 keeps that bias mechanism but layers a small additional sequence-wise balance loss on top, specifically to catch imbalance within a single long sequence that looks fine only when averaged across an entire training batch — relevant given V4's million-token context target.

MAI-Thinking-1 goes a different direction, using a fairly traditional GShard-style auxiliary loss rather than a bias term — but the paper reports that once the loss is aggregated globally across the whole training batch rather than per-microbatch, it performs about as well as the bias-based approach, suggesting the aggregation strategy matters more than which specific balancing formula gets used. Kimi K3 departs furthest from the DeepSeek-V3 default: rather than a bias term at all, it uses what Moonshot calls Quantile Balancing, deriving expert allocation directly from the quantiles of the router's own score distribution — eliminating both the heuristic bias-update rule and a separate balancing hyperparameter that needed tuning. At Kimi K3's 16-of-896 sparsity, small routing imperfections have more room to compound than they do at DeepSeek-V3's 8-of-256, making one less hyperparameter to get wrong a reasonable thing for a frontier lab to want. Four different balancing mechanisms, shipping the same summer, with no released material making a clean, controlled case that any one dominates the others — largely because each was tuned inside a specific architecture (different granularity, different gating function, different degree of dropless-ness) that makes an apples-to-apples comparison hard to extract from public reports alone.

## 5. Granularity and the shared-expert question

DeepSeekMoE's fine-grained-experts-plus-shared-expert combination is no longer a design choice under active debate — it's the substrate nearly everything in this post starts from. The logic is straightforward: many small experts give the router more combinatorial flexibility in how it allocates capacity to a given token than a few large ones would, and isolating a shared expert that runs unconditionally means common, general-purpose computation doesn't have to be redundantly re-learned inside dozens of routed experts that are also trying to specialize.

Kimi K3 pushes the "fine-grained" half of that idea further than anyone else in this comparison — 896 experts with top-16 selected — while its release material confirms a shared-expert component exists (visible in its own architecture diagram) without disclosing how many. MAI-Thinking-1 is the interesting counter-example: the team tested shared experts and found they only helped in architectures that run MoE at every layer. Because MAI-Thinking-1 interleaves dense and sparse layers instead, a shared expert added no measurable benefit in their ablations, and the shipped model doesn't use one — a useful reminder that "the standard recipe" is a default, not a law, and specifically tied to running MoE at every layer. LongCat-2.0 takes a different structural approach entirely, splitting its routed experts into three task-specialized groups (Agent, Reasoning, Interaction) that are gated per task, rather than treating the whole pool as one undifferentiated set of interchangeable specialists.

## 6. Compressing the computation itself: LatentMoE

Every idea covered so far leaves the expert computation itself alone and only changes routing or balancing around it. LatentMoE, first proposed by NVIDIA in early 2026 and already shipping in NVIDIA's own Nemotron-3 Super and Ultra models, changes the computation directly: each token is projected into a smaller latent dimension before the expert computation and the all-to-all communication happen, then expanded back afterward. Because both the per-expert compute and the network traffic shrink by the same factor as the compression, the saved budget can be reinvested into more experts and a larger top-K at the same total cost.

MAI-Thinking-1 adopts this directly, growing from an earlier 192-expert / top-4 configuration to 512 experts at top-8 specifically once LatentMoE made the larger configuration affordable to communicate — and runs the model fully dropless (no token-dropping capacity limit at all), a choice the paper says was driven by capacity-related inconsistencies that showed up in balancing ablations once a hard cap was in the picture. Kimi K3's "Stable LatentMoE" appears to be the same latent-compression family applied at a more extreme granularity — 896 experts, top-16 — though Moonshot's own material doesn't disclose the compression factor or expert-expansion ratio the way MAI-Thinking-1's report does.

## 7. Making compute itself adaptive

Everything above still assumes a fixed top-K: every token activates the same number of experts, whether it's a punctuation mark or a genuinely hard reasoning step. LongCat-2.0 breaks that assumption directly. Its expert pool includes what Meituan calls Zero-Computation Experts — experts that do no computation at all and simply return the input unchanged — and the router decides, per token, how many real experts versus zero-computation experts to use. A simple token might route almost entirely to zero-computation experts and cost next to nothing; a difficult token engages more real expert capacity. This is a token-adaptive compute budget rather than a fixed top-K, conceptually closer to Mixture-of-Depths' 2024 idea of letting tokens skip computation blocks entirely than to anything in the DeepSeekMoE lineage — the same idea, applied inside the expert dimension instead of the depth dimension.

This is also the one area with a real, unanswered open question: Meituan hasn't published detailed load-balancing mechanics for this design, and the zero-computation-expert mechanism raises an obvious one — what stops the router from routing everything to zero-computation experts to minimize loss on easy tokens during training. The public material doesn't yet say.

## 8. The systems layer: what routing costs in communication

Every routed token has to physically travel, over an interconnect, to whichever accelerator holds its selected expert — a cost that only grows more pressing as expert counts climb into the hundreds (the [distributed-training post](#/blog?id=distributed-training-for-dl) covers this all-to-all communication pattern in more depth). Several of the choices above are, from different angles, responses to that same cost. Hash routing's revival in DeepSeek-V4 — deterministic, no learned gate, applied specifically to its first three MoE layers in place of what used to be dense FFN blocks — trades routing quality for guaranteed-even load and zero routing-network communication overhead in those layers. LatentMoE's compressed latent space directly shrinks the payload that has to move in an all-to-all exchange, which is precisely the lever that let MAI-Thinking-1 and Kimi K3 afford larger expert pools in the first place. And DeepSeek-V4's other systems contribution, a pipelined expert-parallelism kernel open-sourced as MegaMoE, overlaps the dispatch, compute, and combine stages of expert-parallel MoE across "waves" of experts rather than running them sequentially, reporting meaningful speedups over a comparable overlap scheme called Comet. LongCat's ScMoE backbone is a similarly systems-driven choice, a shortcut-connected design built specifically to improve expert-parallel throughput. None of these are quality improvements in the way sigmoid gating or fine-grained experts are — they're the engineering that makes the quality improvements affordable to actually run.

Separately, DeepSeek-V4 introduces Anticipatory Routing, a training-stability technique rather than a communication one: when the model detects a loss spike (which the paper ties specifically to outlier activations inside MoE layers), it starts computing routing decisions from a slightly stale copy of the router's own weights rather than the current ones, decoupling how fast the router changes from how fast the rest of the model changes until the instability passes.

## 9. Today's configurations, side by side

| Model | Total params | Active params | Routed experts | Shared experts | Top-K | Gating |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | 671B | 37B | 256 | 1 | 8 | Sigmoid |
| DeepSeek-V4-Flash | 284B | 13B | 256 | 1 | 6 | Sqrt(Softplus) |
| DeepSeek-V4-Pro | 1.6T | 49B | 384 | 1 | 6 | Sqrt(Softplus) |
| GLM-5.2 | 753B | 40B | 256 | 1 | 8 | Sigmoid |
| MAI-Thinking-1 | 962B | 34.7B | 512 | 0 | 8 | Softmax |
| Kimi K3 | 2.8T | not disclosed | 896 | yes, count undisclosed | 16 | not disclosed |
| LongCat-2.0 | 1.6T | ~48B | dynamic pool | 3 task groups instead | dynamic | router picks real vs. zero-compute |

![interactive:moe-sparsity](#)

![interactive:moe-lookup](#)

## 10. What's converged, and what's still an open argument

Fine-grained routed experts plus one isolated shared expert has stopped being a debated choice within about two years of DeepSeekMoE's publication — it's the substrate this entire post starts from. Sigmoid-family gating has similarly displaced softmax nearly everywhere, with MAI-Thinking-1's reversion to softmax being the one exception, and notably the one architecture that also doesn't run MoE at every layer.

Load balancing is still visibly being reinvented every few months: a learned per-expert bias term, that same bias term plus a sequence-level loss, a fully global-batch aggregated traditional loss, and quantile-derived allocation with no separate hyperparameter at all are four different answers shipping in the same summer, without a controlled comparison anywhere in the public record to say which wins.

The most consequential open fork is between two different ways of widening the gap between total and active parameters. LatentMoE keeps a static top-K but shrinks the effective cost of a larger K and larger N by moving computation into a compressed latent space — more experts, more of them active, but each one cheaper. LongCat-2.0's zero-computation experts instead keep the pool's nominal size fixed but make the number of "real" experts activated a dynamic, token-dependent quantity. These are not mutually exclusive, and it would not be surprising to see a future report combine both: latent-space compression to make a bigger pool affordable, and adaptive per-token activation to avoid spending that budget uniformly on tokens that don't need it.

## 11. How to read an MoE spec sheet

**Active parameters, not total parameters, predict inference cost per token** — a 2.8T-parameter model with a small top-K can be cheaper to serve than a 300B dense model, and total parameter counts are usually the headline precisely because they're the more impressive-sounding number, not the more operationally relevant one.

**A shared expert is not a routed expert that always wins the router's vote** — it's architecturally separate, with its own weights, that never participates in the top-K competition at all. Its presence or absence is itself a design choice: MAI-Thinking-1 shipping without one, specifically because its interleaved dense/sparse layout made a shared expert redundant, is a reminder that the standard recipe is a default, not a law.

**Bigger expert pools cost more in communication, not just in weight storage** — every routed token has to physically travel to whichever accelerator holds its selected expert. Hash routing's revival and LatentMoE's compressed communication payload are both responses to that same cost from different angles.

**Treat undisclosed numbers as undisclosed, not as zero** — Kimi K3's blog is genuinely silent on active-parameter count and exact shared-expert count, and LongCat-2.0's balancing mechanics for its zero-computation experts aren't public yet. A missing number is a fact about the disclosure, not necessarily a fact about the architecture.

*Where a claim above is drawn from a model's own release material (Kimi K3, MAI-Thinking-1, DeepSeek-V4) versus third-party technical coverage (GLM-5.2, LongCat-2.0), it's noted inline in the sections above. As with any snapshot across simultaneously-released reports, some of what reads as a stable convergence here may look different in six months.*
