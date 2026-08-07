---
title: "Guess and Check: How Speculative Decoding Buys Speed for Free"
date: "2026/8/6"
---

Generating one token from a large model reads every weight in the model and does almost no arithmetic with them. At batch size 1 the GPU spends its time waiting on memory while its arithmetic units sit mostly idle. That imbalance is the whole opening for speculative decoding: if memory traffic is what costs you and compute is nearly free, then checking several tokens at once costs barely more than checking one.

So here is the idea. Have a small, cheap drafter guess the next γ tokens, then hand the whole span to the big model for a single forward pass. That pass gives you the target's distribution at every position, and you use it to accept or reject each candidate in turn. Every token accepted is one you did not have to run a separate forward pass for.

What makes this more than a heuristic is that those guesses can be accepted in a way that leaves the output distribution *exactly* unchanged. That result comes from two papers written concurrently and independently, [Leviathan et al.](https://arxiv.org/abs/2211.17192) at ICML 2023 and [Chen et al.](https://arxiv.org/abs/2302.01318) at DeepMind, both reporting roughly 2–3× speedups with output identical to what the target would have produced on its own. Everything since has been an attempt to make the guessing better.

## 1. The rule, and what it buys

Write `p` for the target's distribution at some position and `q` for the drafter's. The drafter proposes a candidate; the target decides whether to keep it:

```python
x = sample(q)                            # the draft proposes a token
if uniform() < min(1, p[x] / q[x]):      # the target checks it
    emit(x)
else:
    emit(sample(normalize(maximum(p - q, 0))))
```

Two lines carry the argument. The acceptance probability `min(1, p[x]/q[x])` keeps a token whenever the target likes it at least as much as the draft did, and otherwise keeps it with the ratio of how much less. And when a token is rejected, the replacement is not drawn from `p`. It is drawn from the *residual*, the part of `p` that `q` failed to cover, renormalized.

That residual is the step people skip, and it is the step that makes the whole thing exact. Accepting on the ratio leaves you having sampled from `min(p, q)`, which falls short of `p` by exactly `(p − q)⁺`. Sampling the rejections from that shortfall puts back precisely what was missing.

![The accept and reject rule](blogs/images/specdec-accept-rule.svg?v=1)

The green mass is kept directly and sums to α, the acceptance rate. The blue mass is what the residual makes up. Read that way, α has a clean closed form: `α = Σ min(p, q) = 1 − TV(p, q)`, one minus the total variation distance between draft and target, which is Corollary 3.6 in the original paper. Put differently, a drafter whose tokens are accepted half the time is a drafter sitting 0.5 TV away from the target, and no architectural detail enters into it.

Exactness proofs are easy to nod along to and easy to get subtly wrong in code, so it is worth watching happen. I took a target `p` over eight tokens, deliberately picked a bad draft `q` sitting 0.34 in total variation away from it, and ran two million single-token speculative steps.

![Empirical distribution after two million speculative steps](blogs/images/specdec-lossless.svg?v=1)

The draft is visibly wrong, putting its mode on the wrong token. The output is not: measured total variation between what came out and `p` is 0.00046, against a sampling-noise floor of about 0.00071 at two million draws. The draft's own error has vanished rather than been reduced. This is what makes speculative decoding unusual among inference optimizations. Quantization and pruning trade quality for speed; this trades nothing. A bad draft costs you speed, never correctness.

Now the accounting. A round drafts γ tokens and verifies them in one target pass. If the first rejection lands at position k you emit k+1 tokens, the k accepted ones plus the resampled replacement; if all γ survive you emit γ+1, the extra being a free sample from the target's own distribution at the final position, which the verification pass already computed. Under the paper's stated assumption that per-token acceptance is i.i.d. with rate α, the yield of a round is a capped geometric, `(1 − α^(γ+1))/(1 − α)`, which holds to within sampling error at every α and γ I tried. Drafting four at a time at α = 0.75, twenty-one tokens took six target passes instead of twenty-one:

![Target forward passes, autoregressive versus speculative](blogs/images/specdec-timeline.svg?v=1)

That figure counts only target passes, which is the honest way to show the mechanism and an optimistic way to show the speedup, because drafting is not free.

So put that cost back in. With `c` the cost of one draft pass as a fraction of one target pass, the walltime improvement becomes

$$
\text{speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}
$$

and the tension is visible. Raising γ raises the numerator with diminishing returns, since the i-th draft token only pays off if every token before it was accepted, while the denominator grows linearly. There is an optimum, and it moves with α:

![Speedup against draft length for several acceptance rates](blogs/images/specdec-speedup.svg?v=1)

At α = 0.9 you want to draft seven at a time for about 2.4×. At α = 0.5 the best you can do is draft one, for 1.25×, and drafting more actively hurts. Acceptance rate, not draft model size, is the number that governs everything: α sets both how much you can win and how aggressively you are allowed to play.

One piece of folklore is worth killing here. Corollary 3.9 says an improvement exists as long as **α > c**. With a typical α of 0.6 to 0.8 that means the draft needs to be roughly 1.3 to 1.7 times faster than the target, not the ten-to-thirty times often quoted. The same paper's Table 4 includes a working configuration at c = 0.11, a draft about nine times faster, still returning 1.7–2.2×. Published setups land around c ≈ 0.02–0.13, and both original papers describe the sweet spot in parameter count rather than latency ratio.

## 2. Designing the drafter

Everything above treats the draft as a black box producing `q`. Since α and c are the only two things that matter, every drafter design is a move in one plane, and the plane is worth looking at directly:

![The design space of drafters](blogs/images/specdec-design-space.svg?v=1)

The dashed line is α = c. The two arrows are the only two levers anyone has: push up by making the draft agree more often, push left by making it cheaper. Almost every paper in this area is one or the other, and the interesting ones notice that the levers interact.

The original proposal is a second, smaller model from the same family. It needs no changes to the target and works with any pair, but you need a small model trained on similar data, and its α is whatever it happens to be. The problem is structural: an independent model has to reconstruct from scratch a context that the target has already encoded perfectly well internally.

So the dominant line replaced the separate model with a head that reads the target's own internals. EAGLE conditions a small autoregressive head on the target's hidden features, which means it guesses with the benefit of the target's own representation of the context rather than its own weaker one. This is a pure α play, and it works because the features are already computed, so the head can be tiny.

A third family attacks c instead, and attacks it in a way the formula makes obvious. If drafting γ tokens autoregressively costs γ draft passes, then `c` in the denominator is really `γ · c_pass`, and the optimum γ stays low no matter how good your drafter is. Parallel drafters break that coupling by producing a whole block in one pass, so `c` stops depending on γ at all. [DFlash](https://arxiv.org/abs/2602.06036) does this with a small block-diffusion model conditioned on the target's context features, a natural fit because diffusion is natively parallel across positions.

Then there are model-free drafters, which learn nothing at all. SuffixDecoding builds a suffix tree over the prompt and previous outputs and drafts by matching against it, putting it at the far left of the plane with `c` near zero. Its α is entirely workload-dependent, near-useless on open-ended prose and remarkably good in agentic settings where the model quotes long spans of its own earlier output.

## 3. Training the drafter

Pushing α up is a training problem, and it has one dominant failure mode: the drafter is trained on a distribution it will never see at inference time.

The naive objective trains the draft head to predict the target's next token given a true prefix. But at inference the drafter runs several steps forward, conditioned on *its own* previous guesses, which are sometimes wrong. By position three it is being asked to continue a prefix that never appeared in training. This is ordinary exposure bias, and it shows up as acceptance falling off along the block.

[EAGLE-3](https://arxiv.org/abs/2503.01840) fixes it with what the authors call a training-time test: during training, simulate the multi-step drafting process, so the head is trained on the distribution it will actually face. The same paper makes a second change, dropping feature prediction in favour of predicting tokens directly, and reports that the combination lets the drafter keep improving with more training data, which the feature-prediction formulation did not.

Parallel drafters have a related but distinct problem. Since every position in the block is predicted independently, there is nothing stopping position 4 from contradicting position 3, and agreement decays toward the end of the block. The shape of that decay matters more than it looks:

![Acceptance along the block](blogs/images/specdec-suffix-decay.svg?v=1)

The yield of a round is `1 + Σ_k Π_{i≤k} α_i`, a product, so early positions are weighted enormously more than late ones and a decaying tail costs more than the average acceptance suggests. In the three profiles above, the decaying one averages a respectable acceptance across the block but yields 3.81 tokens per round against 4.33 for the flat one.

Repairing that decay is what [DSpark](https://arxiv.org/abs/2607.05147) adds: a very light sequential head on top of the parallel backbone, reinjecting dependence between positions inside the block. Its Markov variant is a low-rank factorization of a first-order transition matrix, which is to say it adds a rank-r bias to the parallel model's marginal logits in order to reconstruct part of the joint. Flattening the decay is worth more than raising the average, which is why the third profile above yields 5.32.

## 4. Designing the verification

The verification step spent three years as a fixed rule applied uniformly, and it is now where a lot of the remaining headroom is.

Start with the fact that nothing forces the draft to be a single chain. If the drafter offers several candidates at a position, the target can check all of them in the same pass, since the pass was going to be memory-bound anyway. The verification rule generalizes: try the first candidate; if it is rejected, update the residual and try the next against that.

![Chain versus tree, and acceptance against candidate count](blogs/images/specdec-tree-verify.svg?v=1)

With the same p and q as before, one candidate accepts 66% of the time and six candidates accept 87%. Every one of those is still exactly lossless, which is the surprising part: offering more guesses does not bias the output, it only reduces how often you fall back to the residual. This is the idea behind tree-structured drafting, and EAGLE-2 pushed it further by making the tree shape dynamic rather than static, spending the candidate budget where the drafter is least confident.

Verification length need not be a constant either. A long draft is only worth verifying if it is likely to survive, and under load the target model's batch capacity is a contended resource that speculation is competing for against real requests. DSpark's other half schedules verification length per request, using a calibrated estimate of prefix survival probability together with a throughput profile of the engine, so that a busy server stops spending target capacity on speculation that will probably be thrown away.

And then there is the uncomfortable one: you can verify less strictly and go faster. [Medusa](https://arxiv.org/abs/2401.10774) attaches several decoding heads to the target and verifies with what it calls typical acceptance, keeping a token when the target's probability for it clears an entropy-dependent threshold. The paper is explicit rather than quiet about the trade: "we ascertain that it is typically unnecessary to match the distribution of the original model", and in the appendix, "we do not insist on an exact correspondence between the output and language model distribution". There is a tuning knob, and larger values buy speed by accepting more aggressively.

Worth knowing that the founding paper got there first and decided otherwise. Its Appendix A.5 describes exactly such a relaxation under the name lenience, reports that it reaches 5× on T5-XXL against 3× for the strict version, and then quarantines it: everything else in the paper uses the strictest form and allows no lenience at all. The same idea, one paper treating it as a contaminant and the other as the default. At temperature 0 the distinction collapses, since typical acceptance degenerates to greedy. It does mean "lossless" needs reading carefully: it is a precise claim about the rejection rule, and a method that changes the rule does not inherit it.

## 5. Getting it into an engine

Everything so far is about one request in isolation. Putting a drafter inside a real serving engine breaks several things that engine had already settled.

Start with the shape of the batch. A normal decode step feeds the target B sequences of one token each. Verification feeds it B sequences of γ+1 tokens, and if the draft is a tree rather than a chain, of however many nodes the tree has. That is a different kernel launch shape every round, and it collides with CUDA graphs, which want fixed shapes, and with chunked prefill, which is already competing for the same token budget.

Then the attention mask. Verifying a chain is easy, since the drafted tokens are causally ordered and an ordinary causal mask works. Verifying a *tree* is not: two sibling candidates at the same position must not see each other, while both must see their shared ancestors. This needs a custom mask encoding the tree topology:

![Causal mask versus tree mask](blogs/images/specdec-tree-mask.svg?v=1)

On this five-node tree a plain causal mask wrongly allows four pairs, letting `b` attend to its sibling `a` and letting `d` attend to both `b` and `c`. Each of those would condition a candidate on a token that is not on its own path, which quietly corrupts the verification. Building the right mask from the tree topology is the actual systems contribution of [SpecInfer](https://arxiv.org/abs/2305.09781) (ASPLOS '24), which introduced tree-based parallel decoding so an entire token tree could be verified in a single forward pass, reporting 1.5–2.8× for distributed inference and 2.6–3.5× for offloading-based inference.

Then the KV cache. Drafted tokens that end up rejected still had cache slots written for them, so the engine has to roll those slots back, and it has to do so without disturbing the block tables of every other sequence in the batch. Engines with paged caches inherit this cheaply, because the rollback is bookkeeping in a block table rather than data movement, which is one of the quieter arguments for paging.

In practice the two open-source engines have converged on similar structures. [vLLM](https://docs.vllm.ai/en/stable/features/speculative_decoding/) exposes a proposer interface behind a `speculative_config`, with `ngram`, `suffix`, `draft_model`, `mtp`, `eagle3` and `dflash` as first-class methods, and a rejection sampler with a convergence test in its suite that checks the sampler's output distribution actually matches the target's. That test is the practical form of section 1's guarantee: losslessness is a property you can regress against, and an engine that means it will have a test for it.

SGLang runs [speculative workers](https://docs.sglang.ai/advanced_features/speculative_decoding.html) alongside its overlap scheduler, which tries to hide CPU scheduling work behind GPU execution. That combination is genuinely hard, and the limitations are documented rather than hidden: overlap currently supports only `--speculative-eagle-topk 1`, meaning the chain case, with wider trees on the roadmap. It is a nice illustration that tree drafting and scheduler overlap both want to be the thing that fills the GPU's idle time, and they can get in each other's way.

The last piece is knowing when to stop. Since speculation competes with real requests for target-model batch capacity, an engine under load should be able to turn it off. vLLM can do this by driving the proposed length to zero based on the size of the running queue, though as of an [open issue](https://github.com/vllm-project/vllm/issues/25112) the automatic version does not always disable at the configured batch size. This is the same idea as DSpark's confidence scheduling, arrived at from the operations side rather than the modelling side: below some load speculation is nearly free, above it you are stealing capacity from requests that would have been served anyway.

## 6. Where the gains go

The premise in the opening paragraph was that the GPU has idle arithmetic capacity. That premise weakens as you fill the machine, and speculative decoding weakens with it.

![Reported speedup against batch size](blogs/images/specdec-batch-size.svg?v=1)

[One study](https://arxiv.org/abs/2310.18813) measured it directly: the same setup giving 2.73× at batch size 1 gives 1.31× at batch size 32, because "large batch sizes already fully utilize the underlying GPU computational resources". EAGLE-3's own published tables show the same shape, falling to 1.38× at batch 64 in SGLang and to 1.01× at batch 56 in vLLM. The technique is a latency optimization that looks like a throughput optimization only when the machine is empty.

The qualifier that matters in 2026 is sequence length, which is the fourth line on that chart. [MagicDec](https://arxiv.org/abs/2408.11049) points out that at long context the KV cache scales with batch times sequence length, so loading it keeps the workload memory-bound even at large batch, and reports up to 2.51× for Llama-3.1-8B at batch sizes from 32 to 256. The memory-bound premise is not repealed by batching; it is repealed by short sequences and small caches.

It can also go negative. A [llama.cpp issue on Apple Silicon](https://github.com/ggml-org/llama.cpp/issues/23752) reports MTP speculative decoding as a net loss at every configuration tested on an M1 Max, going from 25.3 tokens/s down to 19.3, because "the draft evaluation overhead on Metal exceeds the speculative gain". One user's report on unconfirmed status, but a concrete reminder that c is a property of your hardware and kernels, not of the parameter counts on paper.

So all of it is boxed in by one inequality. Raise α by conditioning the drafter on the target and training it against the distribution it will really face; lower c by drafting a whole block in one pass; and then hand the result to an engine that knows when the machine is too full for any of it to be worth doing. The published maxima are worth reading as what a method achieves in its single most favourable configuration, with independently measured numbers landing closer to 1.5–3×. The interesting part was never the multiplier. It is that you can get it without changing a single token of what the model would have said.
