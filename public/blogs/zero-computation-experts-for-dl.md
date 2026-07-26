---
title: "Zero-Computation Experts: How MoE Learned to Skip Work"
date: "2026/7/26"
---

The [MoE field guide](#/blog?id=moe-field-guide-for-dl) covers routing, gating, load balancing, granularity, and latent compression, all of them changes to *which* experts run or *how much they cost per parameter*. Every one of those still assumes a fixed top-K: every token activates the same number of experts, whether it's a punctuation mark or a hard reasoning step. This post is about a different lever entirely: making some of the "experts" in the pool do no computation at all, so that the router can hand a simple token an expert slot that costs nothing. It's a short lineage — two research papers eighteen months apart, then one lab shipping the idea at trillion-parameter scale.

## 1. The problem: top-K spends the same compute on every token

A standard MoE layer picks the same number of experts, K, for every token, and each of those K experts runs its full feed-forward computation regardless of whether the token needed it. An end-of-sentence period and a genuinely ambiguous word both cost exactly the same. That's a real waste: not every token needs K experts' worth of computation, and forcing a uniform budget on all of them means either K is set high enough for the hard tokens, wasting compute on the easy ones, or set low enough for the easy tokens, starving the hard ones.

Zero-computation experts attack this directly, by putting a cheap option inside the expert pool itself, so the router can choose it for tokens that don't need real computation.

## 2. What "zero computation" means

A zero-computation expert sits in the same routing pool as ordinary FFN experts and gets selected the same way, through the router's normal top-K mechanism. The difference is what happens once it's selected: instead of running a feed-forward block, it does one of a small number of near-free operations, and the token's contribution from that slot costs close to nothing.

```python
def zero_computation_expert(x, kind, c=None):
    if kind == "zero":
        return 0        # discard: contributes nothing
    elif kind == "copy":
        return x         # skip: passes the input straight through
    elif kind == "constant":
        return c         # replace: a small trainable vector, independent of x
```

This is a different move from expert pruning. Pruning removes an expert from the model; a zero-computation expert stays in the routing pool permanently, as a standing option the router can pick per token, per layer, per step.

![Three ways to do (almost) nothing](blogs/images/moe-zero-computation-types.svg?v=1)

## 3. Where it started: AdaMoE's null experts

**AdaMoE** (Zeng et al., submitted June 2024, EMNLP 2024 Findings) is the origin of this idea in its explicit form. Standard MoE routing forces every token through the same top-K, which the paper points out is arguably wrong on its face: a token like `<EOS>` and a token like `apple` don't obviously need the same number of experts. AdaMoE's fix is a small, almost trivial addition to vanilla top-K routing: add a fixed number of **null experts** to the expert pool, and increase K to make room for them. A null expert consumes zero FLOPs. The router isn't forced to fill every one of its K slots with a null expert or a real expert in fixed proportion — a load-balancing loss just keeps the *average* usage of null experts in check across the batch, so different tokens end up using different numbers of real experts without anything explicit telling them to.

Applied to fine-tuning Mixtral-8x7B, AdaMoE reduced average FLOPs by 14.5% on the ARC-C benchmark while accuracy went up by 1.69%, and average FLOPs dropped 15.21% across six benchmarks with performance still ahead of the fine-tuned baseline. The mechanism is small enough to retrofit onto an already-trained MoE model rather than requiring pretraining from scratch.

## 4. Generalizing it: MoE++'s three operations

**MoE++** (Jin et al., submitted October 2024, ICLR 2025 Oral) takes AdaMoE's null expert and treats it as one instance of a broader category. Instead of one type of zero-computation expert, MoE++ defines three, corresponding to three different near-free operations: the **zero expert** (discard the input, contribute nothing — the same operation AdaMoE's null expert performs), the **copy expert** (pass the input straight through unchanged, effectively a free residual connection through that slot), and the **constant expert** (replace the input with a small trainable vector, independent of what the token actually is).

The reason to have three instead of one is that they cover different failure modes for a "how much does this token need" decision: a token that genuinely needs nothing gets the zero expert, one whose representation is already fine as-is gets the copy expert, and one that needs a fixed, learned adjustment regardless of its specific content gets the constant expert.

MoE++ also solves a deployment problem that a naive zero-computation-expert design runs into. Real FFN experts have to be split across accelerators and communicated to over an interconnect (the [distributed-training post](#/blog?id=distributed-training-for-dl) covers why that communication is expensive at scale). Zero-computation experts have negligible parameters, so MoE++ just puts a full copy of all of them on every single GPU. A token routed to a zero-computation expert never has to leave the device it's already on, and never contributes to expert-parallel load imbalance, since these "experts" don't have any real capacity to overflow.

At matched model size, MoE++ delivers 1.1–2.1x the expert forward throughput of a vanilla MoE model, while also outperforming it — letting simple tokens use fewer real FFN experts frees up capacity for the real experts to specialize harder on the tokens that actually need them.

## 5. Going to production: LongCat-Flash and LongCat-2.0

AdaMoE and MoE++ are both academic demonstrations at model scales far below frontier size. **LongCat-Flash** (Meituan, technical report submitted September 2025, 560B parameters) is the first public case of zero-computation experts shipped inside a pretrained-from-scratch, production-scale model, and **LongCat-2.0** (June 2026, 1.6T parameters) continues the same design at a larger scale.

The mechanism is the same idea as MoE++'s zero expert, but the framing is different: rather than "the router picks some real experts and some no-op experts," Meituan describes it as **dynamic computational budget allocation**. A token's active-parameter count is no longer a fixed number decided by K alone; it's a range, and where a given token lands in that range depends on how much real expert capacity the router decides it needs. LongCat-Flash's reported range is 18.6B–31.3B active parameters per token (27B on average); LongCat-2.0's is 33B–56B. A simple token — punctuation, a repeated pattern — can route mostly to zero-computation experts and cost close to the low end of that range; a token the router judges harder pulls in more real expert capacity and costs closer to the high end.

Keeping that average inside a target band during training isn't automatic — nothing in the training objective on its own guarantees the average lands where the designers want it, rather than drifting toward either extreme. LongCat-Flash addresses this with a **PID controller** that continuously adjusts the routing bias toward or away from zero-computation experts, the same kind of per-expert bias mechanism DeepSeek-V3 uses for load balancing (covered in the [field guide's section 3](#/blog?id=moe-field-guide-for-dl)), except here the target being tracked is the average active-parameter count rather than per-expert load.

![From null experts to a trillion-parameter model](blogs/images/moe-zero-computation-timeline.svg?v=1)

## 6. Scattered follow-up work

A handful of newer papers extend this idea in different directions without forming a single coherent second wave yet. One line takes the zero/copy/constant framing and applies it to multimodal models rather than text-only ones. A separate line skips the MoE-specific framing entirely: instead of training a model with zero-computation experts from the start, it takes an already-trained dense model's MLP and deterministically decomposes it into a static set of branches, some of which end up doing near-nothing, without any additional training at all. These are worth keeping distinct from a third, older line of MoE efficiency work — expert pruning, which ranks experts by how often they're actually used and removes the least-used ones outright. Pruning and zero-computation experts both reduce wasted compute in an MoE model, but they're solving it from opposite directions: pruning shrinks the pool permanently, while zero-computation experts keep the pool's nominal size fixed and let the router opt out per token.

## 7. What's still open

The most direct open question is the training incentive itself: if a zero-computation expert is free and a real FFN expert costs the model nothing extra to route to during training (there's no compute-cost term in the loss by default), what stops the router from routing everything to the free option to minimize the language-modeling loss with no downside? LongCat's PID controller is the concrete answer in production — it holds the average active-parameter count inside a target band by adjusting routing bias — but that's a control mechanism reacting to the symptom, not a change to the underlying incentive. None of the public material makes the stronger claim that the incentive problem is solved rather than managed.

The other open question is how this idea composes with LatentMoE, the field guide's section 8 mechanism that shrinks the cost of *every* expert by moving computation into a compressed latent space rather than changing how many experts run. The two are answers to different halves of the same total-compute equation — LatentMoE lowers the cost per active expert, zero-computation experts lower the number of experts that need to be genuinely active — and nothing about them is mutually exclusive. No public report has combined both in the same model yet.

Conceptually, the closest ancestor to zero-computation experts isn't anything in the MoE lineage at all — it's **Mixture-of-Depths** (2024), which let tokens skip entire transformer blocks based on a learned per-token decision. Zero-computation experts are the same idea, applied inside the expert dimension of a single MoE layer instead of the depth dimension of the whole network.
