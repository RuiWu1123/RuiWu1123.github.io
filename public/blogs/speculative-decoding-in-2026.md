---
title: "Guess and Check: How Speculative Decoding Buys Speed for Free"
date: "2026/8/6"
---

Generating one token from a large model reads every weight in the model and does almost no arithmetic with them. At batch size 1 the GPU spends its time waiting on memory, and its arithmetic units sit mostly idle. That imbalance is the whole opening for speculative decoding: if compute is free and memory traffic is what costs you, then checking several tokens at once costs barely more than checking one.

So here is the bet. Let something cheap guess the next few tokens. Then run the big model once over all of those guesses at the same time and ask, for each position, whether it would have produced that token itself. Every guess it agrees with is a token you got without paying for a separate forward pass.

The part that makes this more than a heuristic is that you can accept those guesses in a way that leaves the output distribution *exactly* unchanged. Not approximately, not "close enough in practice". The tokens coming out are drawn from precisely the distribution the big model would have produced on its own. That result is due to [Leviathan, Kalman and Matias](https://arxiv.org/abs/2211.17192) (ICML 2023), and concurrently and independently to [Chen et al. at DeepMind](https://arxiv.org/abs/2302.01318); both report roughly 2–3× on their respective models, with identical outputs.

This post is about why that works, what determines how much you actually gain, and what the frontier looks like in 2026. Every figure below comes from a simulator I wrote for this post rather than from a drawing tool, so the numbers in them are measured rather than asserted.

## 1. The rule that makes it exact

Write `p` for the distribution the target model would produce at some position, and `q` for the draft's distribution at the same position. The draft proposes a token; the target then decides whether to keep it:

```python
x = sample(q)                            # the draft proposes a token
if uniform() < min(1, p[x] / q[x]):      # the target checks it
    emit(x)
else:
    emit(sample(normalize(maximum(p - q, 0))))
```

Two lines carry the whole argument. The acceptance probability `min(1, p[x]/q[x])` keeps a token whenever the target likes it at least as much as the draft did, and otherwise keeps it with the ratio of how much less. And when a token is rejected, the replacement is not drawn from `p`. It is drawn from the *residual*, the part of `p` that `q` failed to cover, normalized back to a distribution.

That residual is the piece people skip, and it's the piece that makes the whole thing exact. Accepting on the ratio leaves you having sampled from `min(p, q)`, which is short of `p` by exactly `(p − q)⁺`. Sampling the rejections from that shortfall puts back precisely what was missing.

![The accept and reject rule](blogs/images/specdec-accept-rule.svg?v=1)

The green mass is what gets kept directly, and it sums to α, the acceptance rate. The blue mass is what the residual has to make up. Read that way, α has a clean closed form: `α = Σ min(p, q) = 1 − TV(p, q)`, one minus the total variation distance between draft and target. Leviathan et al. give this as Corollary 3.6. A draft that agrees with the target half the time is a draft whose distribution is half a TV unit away, and nothing about the architecture enters into it.

## 2. Checking that it is really lossless

Exactness proofs are easy to nod along to and easy to get subtly wrong in code, so it is worth watching it happen. I took a target distribution `p` over eight tokens, deliberately picked a bad draft `q` (total variation distance 0.34 from `p`, so it disagrees about a third of the time), and ran two million single-token speculative steps.

![Empirical distribution after two million speculative steps](blogs/images/specdec-lossless.svg?v=1)

The draft is visibly wrong: it puts its mode on the wrong token and badly misweights several others. The output is not. Measured total variation between what came out and `p` is 0.00046, against a sampling-noise floor of about 0.00071 at two million draws. The draft's own error, 0.34, has vanished completely rather than been reduced.

This is the property that makes speculative decoding unusual among inference optimizations. Quantization trades quality for speed. Pruning trades quality for speed. This one does not trade anything: a bad draft costs you speed, never correctness. If the draft is garbage, α collapses toward zero, nearly everything is rejected, and you have paid for draft passes that bought you nothing. What you have not done is change the answer.

## 3. What you actually gain

A round drafts γ tokens and verifies them in one target pass. If the first rejection happens at position k, you emit k+1 tokens: the k accepted ones plus the resampled replacement. If all γ are accepted you emit γ+1, the extra one being a free sample from the target's own distribution at the final position, which the verification pass already computed.

Under the paper's stated simplifying assumption that per-token acceptance is i.i.d. with rate α, the expected yield of a round is a capped geometric:

$$
\mathbb{E}[\text{tokens per round}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

My simulator reproduces this to within sampling error at every α and γ I tested, which is a useful check that the implementation and the algebra agree. Here is one round of it playing out, drafting four at a time at α = 0.75:

![Target forward passes, autoregressive versus speculative](blogs/images/specdec-timeline.svg?v=1)

Twenty-one tokens took twenty-one target passes autoregressively and six speculatively, or 3.5 tokens per pass. That figure counts only target passes, which is the honest way to show the mechanism but an optimistic way to show the speedup, because drafting is not free.

Put the draft cost back in. Leviathan et al. define `c` as the cost of one draft pass as a fraction of one target pass, which makes the walltime improvement

$$
\text{speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}
$$

Now the tension is visible. Raising γ raises the numerator with diminishing returns, since each extra draft token only pays off if every token before it was accepted, while the denominator grows linearly. There is an optimum, and it moves with α:

![Speedup against draft length for several acceptance rates](blogs/images/specdec-speedup.svg?v=1)

At α = 0.9 you want to draft seven at a time and you get about 2.4×. At α = 0.5 the best you can do is draft one, for about 1.25×, and drafting more actively hurts. This is why acceptance rate, not draft model size, is the number that matters: α sets both how much you can win and how aggressively you are allowed to play.

The break-even condition falls out of the same formula and is worth stating precisely, because a piece of folklore has grown up around it. Leviathan et al.'s Corollary 3.9 says an improvement exists as long as **α > c**. With a typical α of 0.6 to 0.8, that means the draft needs to be somewhere around 1.3 to 1.7 times faster than the target, not the ten-to-thirty times you sometimes see quoted. Their own Table 4 includes a working configuration at c = 0.11, a draft roughly nine times faster, still returning 1.7–2.2×. Published setups land in the range c ≈ 0.02–0.13, and both original papers describe the sweet spot in parameter count rather than latency ratio, around two orders of magnitude smaller than the target. The balance you are managing is between α and c, and there is no fixed speed multiple that decides it.

## 4. Where the gains go

The premise in the opening paragraph was that the GPU has idle arithmetic capacity. That premise weakens as you fill the machine, and speculative decoding weakens with it.

[Su, Giannoula and Pekhimenko](https://arxiv.org/abs/2310.18813) measured this directly: the same adaptive speculative decoding setup that gives 2.73× at batch size 1 gives 1.31× at batch size 32, because "large batch sizes already fully utilize the underlying GPU computational resources". EAGLE-3's own published tables show the same shape from the other direction, falling to 1.38× at batch size 64 in SGLang and to 1.01× at batch size 56 in vLLM. The technique is a latency optimization that happens to look like a throughput optimization when the machine is empty.

The qualifier that matters in 2026 is sequence length. [MagicDec](https://arxiv.org/abs/2408.11049) points out that at long context the KV cache itself scales with batch times sequence length, so loading it keeps the workload memory-bound even at large batch, and reports up to 2.51× for Llama-3.1-8B at batch sizes from 32 to 256. The memory-bound premise is not repealed by batching; it is repealed by short sequences and small caches.

It can also go negative, which is easier to hit than the marketing suggests. A [llama.cpp issue on Apple Silicon](https://github.com/ggml-org/llama.cpp/issues/23752) reports MTP speculative decoding as a net loss at every configuration tested on an M1 Max, going from 25.3 tokens/s baseline down to 19.3, because "the draft evaluation overhead on Metal exceeds the speculative gain". One user's report on unconfirmed status, but a concrete reminder that c is a property of your hardware and kernels, not of the model sizes on paper.

## 5. Where drafts come from

Everything above treats the draft as a black box producing `q`. The last three years of work are almost entirely about what to put in that box, and the designs sort into a few families.

The original proposal is the obvious one: a second, smaller model from the same family. It needs no changes to the target and works with any pair, but you have to have a small model that was trained on similar data, and its α is whatever it is.

The dominant academic line replaced the separate model with a head that reads the target's own internals. EAGLE conditions a small autoregressive head on the target's hidden features, so it is guessing with the benefit of the target's own representation of the context. [EAGLE-3](https://arxiv.org/abs/2503.01840) (Li, Wei, Zhang and Zhang; Peking University, Microsoft Research, Waterloo and Vector Institute; NeurIPS 2025) made two changes that define the current baseline: it dropped feature prediction in favour of predicting tokens directly, and it added what the authors call a training-time test, simulating the multi-step drafting process during training so that the draft head is trained on the distribution it will actually face rather than only on one-step-ahead prediction.

A third family gives up sequential drafting entirely. If drafting γ tokens autoregressively costs γ draft passes, the draft cost `c` scales with γ and the optimum in the curve above stays low. Parallel drafters produce a whole block in one pass, so `c` stops depending on γ. [DFlash](https://arxiv.org/abs/2602.06036) (Chen, Liang and Liu, UC San Diego's Z Lab, ICML 2026) does this with a small block-diffusion model conditioned on the target's context features, which is a good fit because diffusion is natively parallel across positions. The cost is that positions within a block are predicted independently, so they can disagree with each other, and acceptance decays toward the end of the block.

Fixing that decay is what [DSpark](https://arxiv.org/abs/2607.05147) (Cheng et al., Peking University and DeepSeek-AI) adds: a very light sequential head on top of the parallel backbone, which reinjects dependence between positions inside the block. Its Markov variant is a low-rank factorization of a first-order transition matrix, which is to say it adds a rank-r bias to the parallel model's marginal logits to reconstruct some of the joint. It pairs that with scheduling verification length per request based on estimated survival probability, so that under load the engine stops spending target-model batch capacity on speculation that is unlikely to be accepted.

Finally there are model-free drafters, which do not learn anything. SuffixDecoding builds a suffix tree over the prompt and previous outputs and drafts by matching against it. In agentic workloads, where the model quotes long spans of its own earlier output or of the prompt, this is unreasonably effective for something with no parameters.

## 6. The exactness everyone quietly gives up

Section 1's guarantee holds for the rejection rule as written. Several popular methods do not use that rule.

[Medusa](https://arxiv.org/abs/2401.10774) (Cai et al., ICML 2024) attaches several decoding heads to the target and verifies candidates with what it calls typical acceptance, keeping a token when the target's probability for it clears an entropy-dependent threshold. The paper is explicit about the trade rather than quiet about it: "we ascertain that it is typically unnecessary to match the distribution of the original model", and in the appendix, "we do not insist on an exact correspondence between the output and language model distribution". There is a tuning knob, and larger values buy speed by accepting more aggressively.

Worth knowing that the founding paper got there first and decided otherwise. Leviathan et al.'s Appendix A.5 describes exactly such a relaxation under the name lenience, reports that it reaches 5× on T5-XXL against 3× for the strict version, and then quarantines it: everything else in the paper uses the strictest form and allows no lenience at all. The same idea, one paper treating it as a contaminant and the other as the default.

None of this makes Medusa wrong, and at temperature 0 the distinction collapses, since typical acceptance degenerates to greedy. It does mean "lossless" needs reading carefully. It is a precise claim about the rejection rule, and a method that changes the rule does not inherit it.

## 7. Reading the 2026 numbers

The speedup figures attached to recent work need more care than usual, and the gap between reported and reproduced is consistently large.

EAGLE-3's headline is "up to 6.5×", which is one cell of one table at temperature 0 and batch size 1. Independent measurement lands well below: [Red Hat's vLLM evaluation](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding) reports up to 2.5× on A100 and notes it performs poorly on translation, and a Thoughtworks reproduction on H100 measured 1.25× against 1.32× published. DFlash's abstract says over 6× on Qwen3-8B, while the average across its own benchmark table is 4.86× and the chat benchmark is 2.75×; Baseten's independent implementation reports about 3× at concurrency 16 on B200. Note also that DFlash's EAGLE-3 comparison uses a third-party checkpoint, since no official EAGLE-3 Qwen3 checkpoint exists, while DFlash's own drafters were trained by its authors, so some of that gap may be training data rather than architecture.

DSpark's 60–85% deserves the most caution, and to the authors' credit the paper says so itself. That range is for the V4-Flash variant only, measured against DeepSeek's own prior production baseline, on DeepSeek's live user traffic, on undisclosed hardware, and read off a fitted performance frontier rather than a controlled A/B test. On its largest throughput number the paper explicitly declines to call it a representative speedup. There is no independent reproduction and there structurally cannot be one from outside: the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) release (MIT licensed, and it does contain real training and evaluation code for DSpark, DFlash and EAGLE-3) does not include the production serving engine. LMSYS's SGLang integration is careful to say it reproduces the mechanism and the shape of the curve, not the numbers.

None of which means the work is not real. It means the honest summary of the field is that speculative decoding reliably delivers something in the neighbourhood of 1.5× to 3× in independently measured conditions, that the ceiling is set by acceptance rate and by how full your GPU already is, and that the published maxima are best understood as what the method can do in its most favourable single configuration.

What has genuinely changed in the last year is not the size of the numbers but where the effort goes. The draft is no longer a small model you happen to have lying around; it is a component co-designed with the target, reading its internals, and increasingly producing whole blocks in one pass. The verification step, which for three years was a fixed rule applied uniformly, is becoming something the scheduler reasons about per request. Both of those are consequences of the same arithmetic in section 3: everything is a fight to raise α without raising c.

Sources: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) · [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) · [EAGLE-3](https://arxiv.org/abs/2503.01840) · [Medusa](https://arxiv.org/abs/2401.10774) · [DFlash](https://arxiv.org/abs/2602.06036) · [DSpark](https://arxiv.org/abs/2607.05147) · [The Synergy of Speculative Decoding and Batching](https://arxiv.org/abs/2310.18813) · [MagicDec](https://arxiv.org/abs/2408.11049)
