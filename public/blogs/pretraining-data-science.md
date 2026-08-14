---
title: "Reading the Setup: What Pretraining Data Experiments Actually Show"
date: "2026/8/12"
---

Three well-regarded papers report three different verdicts on repeating pretraining data. Anthropic's 2022 study finds that repeating 0.1% of the corpus a hundred times degrades an 800M-parameter model to the level of a 400M one. Muennighoff et al. find that four full epochs cost essentially nothing. Pythia trains for roughly 1.5 epochs on a deduplicated Pile and reports "no evidence that the second epoch negatively impacts evaluation scores," alongside the flatter claim that deduplication showed "no clear benefit."

None of these is wrong, and none of them contradicts the others. They measured different things under different constraints, and each conclusion is exactly as wide as its setup.

That gap between what a paper found and what a headline says it found is the subject here. What follows is a survey of empirical findings about pretraining data, organised so that every finding arrives with the experiment that produced it: how many models, at what sizes, for how many tokens, against which baseline, measured with which metric. The findings are grouped into three: the machinery the field built to make data claims checkable at all, what is known about repeated data, and what is known about data mixtures.

Figures are of two kinds and the text always says which. Some are evaluated here from a published functional form, so you can check the arithmetic. The rest are published numbers redrawn in a consistent style, with the source table named beside them; I do not reproduce the original figures.

## 1. The coordinate system, and the correction that data constraints force on it

[Chinchilla](https://arxiv.org/abs/2203.15556) fits the loss surface as `L(N, D) = E + A/N^α + B/D^β` and reports `E = 1.69`, `A = 406.4`, `B = 410.7`, `α = 0.34`, `β = 0.28` from over 400 models between 70M and 16B parameters trained on 5B to 500B tokens. Its conclusion, that under a fixed compute budget parameters and tokens should scale in equal proportion, is the coordinate system everything else is stated against.

The assumption hiding inside it is that tokens can be bought at constant quality in whatever quantity the optimum calls for. [Muennighoff et al.](https://arxiv.org/abs/2305.16264) removed that assumption and re-derived the surface. Their setup is worth stating precisely, because the paper is quoted far more often than it is read: over 400 models from 10M to 9B parameters, up to 900B total tokens, up to 1500 epochs, all on subsets of C4, GPT-2 architecture and tokenizer, cosine schedule decaying 10× over training, and deliberately no early stopping so that overfitting could be observed rather than avoided. The subsets are nested, so a run with less unique data always sees a subset of what a larger run saw. Loss is reported on a held-out test set, unlike Chinchilla, which reported training loss.

Their model introduces two quantities. With `U_D` unique tokens and `R_D = D/U_D − 1` repetitions, the *effective* data, meaning the amount of fresh data that would have produced the same loss, is

$$
D' \;=\; U_D + U_D\, R_D^{*}\left(1 - e^{-R_D / R_D^{*}}\right)
$$

with a symmetric expression for effective parameters `N'` under excess parameters `R_N`. Fitting 182 runs gives `R_D* = 15.39` and `R_N* = 5.31`, which the authors call the half-lives of repeated data and of excess parameters.

![What an extra epoch is worth](blogs/images/pdsci-repetition.svg?v=1)

Computed here from that expression. Four epochs return 3.73 unique corpora worth of value, which is where the widely quoted "up to four epochs is nearly free" comes from; the paper's own instance of it is an 8.7B model on 44B unique tokens finishing four epochs at 0.5% higher validation loss than the same model on 178B unique tokens. Sixteen epochs return 10.58. The curve has a hard ceiling at `1 + R_D* = 16.4`: no amount of repetition makes a corpus worth more than 16.4 times itself. The marginal column is derived rather than quoted, since the exponential gives the k-th epoch a value of `e^{−(k−1)/15.39}` relative to a fresh token; the paper states the ceiling and the half-life but never tabulates per-epoch values.

Two consequences follow that are less often repeated. Because `R_N* < R_D*`, excess parameters lose value faster than repeated data, so under a data constraint you should spend surplus compute on epochs before parameters, which directly inverts Chinchilla's equal-scaling advice. They confirm this at scale: at 9.3×10²¹ FLOPs with 25B unique tokens, a model with 27% fewer parameters than Chinchilla prescribes achieves better loss.

And the ceiling can be raised by changing what counts as the corpus. Mixing in Python from The Stack, they find no degradation up to 50% code even when evaluating only on natural language, which makes code worth a 2× increase in effective tokens. The composite recommendation is explicit: double the data with code, then repeat four times, for "8x more training tokens that are expected to be just as good as having had 8x more unique data." Two of their nineteen evaluation tasks improve the moment any code is added at all: bAbI goes from 0.0 to 12.5 at 10% code, and WebNLG from 4.8 to 9.5, which the authors attribute to code teaching long-range state tracking.

## 2. What makes a data experiment believable

Every claim in the rest of this post rests on an inference from small experiments to large ones. That inference is itself an object of study, and the results are sobering.

[Pythia](https://arxiv.org/abs/2304.01373) exists to make the inference auditable. Sixteen models, eight sizes from 70M to 12B, two data variants (Pile and Pile-deduped), every model trained on exactly 299,892,736,000 tokens, and, unusually, **all models trained on the same data in the same order**, with an identical batch of 1024 sequences × 2048 tokens at every size. There are 154 checkpoints per model: one at initialisation, log-spaced points at steps 1 through 512, then every 1000 steps. A reproducible dataloader ships with it, so any checkpoint can be tied to the exact tokens seen up to that point. Compare that to the 1 to 30 checkpoints released with GPT-2, GPT-3, OPT, T5 or BLOOM. The suite is not a model release; it is an instrument.

The instrument immediately produced a negative result about a widely believed mechanism. The intuition that data seen later in training is memorised more would let a practitioner suppress memorisation by ordering data carefully. Testing it with (32, 32)-memorisation over the first 64 tokens of each context, they find that "a Poisson model fits the data extremely well," meaning training order has little effect: memorised sequences are not denser at the beginning or the end. The practical advice they draw from this is not what anyone wanted. You cannot reorder data to reduce memorisation, but you can place the sequences you are worried about early, so that if memorisation happens you find out before the run ends.

[DataDecide](https://arxiv.org/abs/2504.11393) attacks the extrapolation question head-on with the largest controlled sweep in the area: 25 pretraining corpora, 14 model sizes from 4M to 1B, 3 seeds, **1,050 models** and more than 30,000 checkpoints, at roughly 820,000 H100-hours. Every model is trained at 100 tokens per parameter, five times Chinchilla, deliberately overtrained. The 25 corpora are not toys: Dolma 1.7 plus four domain ablations, C4, FineWeb-Pro, FineWeb-Edu, Falcon, five Falcon+CC quality-filter variants, six DCLM-Baseline variants, and three λ-mixes of DCLM with Dolma. Success is measured by *decision accuracy*, the fraction of corpus pairs whose ordering at the small scale matches their ordering at 1B.

Their headline is that ranking corpora at a single small size, around 150M parameters, gets about 80% of the pairwise comparisons right at the 1B target. Their second result is the one that should change behaviour: **no scaling-law method among eight baselines exceeds the compute-to-decision-accuracy frontier of that single-scale baseline.** Fitting a curve through several model sizes, which is what most teams do, buys nothing over training one small model and reading off the order.

Two further findings from the same sweep matter for how you read everyone else's numbers. Continuous likelihood metrics beat accuracy at small scale, and specifically the *raw* likelihood of the correct answer, not the normalised versions that penalise probability mass on wrong answers; with that switch, code benchmarks go "from trivial to 80%" predictable because small models climb above the noise floor while the metric still tracks large-scale accuracy. And the noise floor is not small: at the 1B, 5× Chinchilla scale, the standard deviation between seeds "can be as high as 2% points of accuracy" for some recipes on most tasks. Any reported data-curation gain under two points is inside the noise of a single run.

The complementary practitioner-side measurement comes from [Olmix](https://arxiv.org/abs/2602.12237), which asks how small the proxy can be before the ordering it produces stops matching the target.

![How small a proxy can be](blogs/images/pdsci-proxy.svg?v=1)

Published numbers from their Figure 3, redrawn. Proxies at or above 15M parameters reach Spearman rank correlation above 0.89 with the ordering of the same mixtures at 1B; at 1M the correlation falls to 0.73. They settle on 30M parameters and 3B tokens, at ρ = 0.896. The target for that comparison is a 1B Olmo 2 trained on 100B tokens over 24 topic domains, scored as macro-average bits-per-byte over 52 downstream tasks. Two independent groups, using different corpora and different targets, land in the same place: the proxy can be four orders of magnitude smaller than the target, and making it bigger is not where the returns are.

## 3. Repeated data, first setup: a small subset repeated many times

[Hernandez et al.](https://arxiv.org/abs/2205.10487) ran the experiment that produces the alarming numbers, and the construction is very specific. Every model trains on exactly 100B tokens, a fixed budget, pre-Chinchilla. Of those, 90% are drawn without repetition and 10% are repeats of a small random subset of the same corpus, so repeats *replace* unique tokens rather than adding to them. The repeated subset can be tiny and repeated enormously, like 0.01% of the tokens repeated a thousand times, or larger and repeated less. They scan repeat fractions of 1%, 3%, 10%, 20%, 50% and 90%, model sizes from 1.5M to 1B, and repeat counts across two orders of magnitude. Test loss is measured on held-out data containing none of the repeated material.

The abstract's claim is that "performance of an 800M parameter model can be degraded to that of a 2x smaller model (400M params) by repeating 0.1% of the data 100 times, despite the other 90% of the training tokens remaining unique." The body of the paper puts the same data point at "nearly to that of a 340M parameter model," so the paper is internally inconsistent by about 0.35× and the conservative reading is the body's.

What makes this more than an anecdote is that the degradation is not monotonic in the amount of repetition. It is worst in a band, and they fit the band's edges as power laws in model size, `E = k·N^α`, with `(k, α) = (5.1×10⁷, −0.50)` on the right edge and `(4.2×10⁶, −0.56)` on the left.

![The region where repetition hurts](blogs/images/pdsci-degradation.svg?v=1)

Computed here by evaluating those two fits. The band slopes down and to the right: as models grow, the number of repeats that does the most damage falls. The mechanism the authors propose reads directly off the geometry. Repeat a lot of data a couple of times and the model lacks the capacity to memorise it, so nothing pathological happens. Repeat a tiny sliver ten thousand times and the model memorises it, but the sliver is so small that memorising it consumes little capacity. In between there is a region where the repeated data is *both* memorisable *and* large enough that memorising it costs a meaningful fraction of the model. Their arithmetic for why the model takes that trade: an 800M model reaches about 2.0 nats per token and a 400M model about 2.2, and memorised data costs 0, so `0.9 × 2.2 + 0.1 × 0 = 1.98 < 2.0`. Memorising is locally the better deal.

Push the repeat fraction up and the effect stops being a dip and becomes literal double descent: at 90% repeated data with 100× to 10,000× repeats, loss falls, rises, and falls again. At the peak of that curve the effective-parameter penalties are severe, 10× at 50% repeated and 73× at 90%.

The extrapolation the authors offer, and hedge, is the line most often quoted out of context: the fitted boundaries suggest "significant degradation from repeating data as little as 2x on state-of-the-art language models with hundreds of billions of parameters." That is a conditional statement about their fixed-100B-token regime with a randomly chosen repeated subset. It is not a statement about a modern run that repeats a whole curated corpus twice, and section 4 is about why the difference matters.

## 4. What repetition damages, and why three papers disagree without conflicting

The most interesting part of the Anthropic study is not the loss number but the finding that the loss number understates the damage.

![Damage is concentrated, not spread](blogs/images/pdsci-damage.svg?v=1)

Published numbers from their Figures 5 and 6, redrawn. At only 3% repeated data, at the worst repeat count, copying a repeated paragraph loses 3× of effective model size while overall test loss loses at most 1.15×. Prefix matching, the standard behavioural probe for induction heads, loses 1.47×. The damage lands disproportionately on the mechanisms associated with generalisation rather than being spread evenly across what the model can do.

The mechanistic evidence is unusually concrete for a scaling paper. In a one-layer attention-only model, on the last appearances of "Potters," the control model puts more than 97% of its probability on "ters" given "Pot"; the repeated-data model puts under 4%. In a two-layer model on "Dursley," control gives 92% where the repeated-data model gives 10%, and on the constructed "unDursleyish" the gap is 68% against 0.4%. The authors' reading is that "the pressure to memorize the repeated dataset has led a skip tri-gram head to replace the induction head entirely." Consistent with that, in-context learning degrades most: at the double-descent peak, loss on the eleventh copy of a repeated paragraph does not decrease at all with additional copies. They are careful about what they could not show, reporting that logit attribution failed to separate "the induction head is less active" from "other heads are interfering."

Now put the three verdicts side by side, because the reconciliation is the payload.

Anthropic repeats a **random small subset** while holding the **total token budget fixed**, so repeats displace unique tokens, and they measure at repeat counts up to 10,000×. Under those conditions repetition is destructive, and worst in the middle of the range.

Muennighoff et al. repeat the **entire available corpus** because there is no more of it, under a **compute** budget rather than a token budget, at epoch counts mostly in the single and double digits. Under those conditions the first four epochs are nearly free. There is no contradiction: 4 whole-corpus epochs is not the same experiment as 100 repeats of 0.1% of the data, and the Anthropic band puts an 800M model's damage peak at roughly 100 repeats, more than an order of magnitude away from four.

Pythia runs about 1.5 epochs on a MinHash-deduplicated Pile and sees nothing, which the band also predicts, and separately reports that deduplication brought "no clear benefit on language modeling performance." That last claim is the one to handle carefully: it is qualitative, the supporting plots live in an appendix, and the authors themselves offer the deflationary explanation, that the Pile's own upsampling of curated subsets may not behave like the incidental duplication other papers studied, or that "the general tendency of deduplicated data to outperform non-deduplicated data is primarily a statement about the quality of the data used in other works."

Muennighoff et al. reach the same conclusion from a different direction and pin down the condition. On C4 they find perplexity filtering effective and deduplication not: keeping the lowest-perplexity 25% raises their rescaled 19-task average from 22.2 to 25.3 at 4.2B parameters, while deduplication lowers it from 22.2 to 20.5. On OSCAR, which is noisier, both help. The generalisation they state is that "data filtering is primarily effective for noisy datasets," which is also the most plausible reading of Pythia's null result, since the Pile was curated to begin with.

So the honest summary is not "repetition is fine" or "repetition is dangerous" but a conditional: repeating a whole corpus a handful of times is close to free; repeating a small slice of it hundreds of times is not; and deduplication pays in proportion to how dirty the corpus was before you started.

## 5. Data mixtures: the laws, and what they are laws about

Mixing is the one part of data work shaped like a clean optimisation problem, and it has accumulated the most formalism. The formalism divides into three claims that should be evaluated separately: that loss is a predictable function of the mixture, that the function can be fit cheaply, and that optimising the fitted function produces a better model.

The first serious attack was [DoReMi](https://arxiv.org/abs/2305.10429), which trains a 280M proxy with group distributionally robust optimization over domains, the model minimising worst-case excess loss against a reference and the domain weights emerging from the inner maximisation, then transfers those weights to a model 30× larger. It reports improving "average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights" and reaching baseline accuracy in 75k steps against 200k. The result that should reshape intuition is elsewhere in the paper: perplexity improves on all 22 Pile domains *including the ones it downweights*, and the reweighting is drastic, with Pile-CC going from 0.1121 to 0.6057 and ArXiv from 0.1052 to 0.0036. Cutting arXiv to a thirtieth of its weight improved arXiv perplexity. What a specialised domain most needs is a model that has learned language, and web text is the cheapest route there.

[Data Mixing Laws](https://arxiv.org/abs/2403.16952) proposed the functional form that the later literature argues about. With `r_j` the proportion of training domain `j` and `L_i` the loss on validation domain `i`,

$$
L_i(r_1, \ldots, r_M) \;=\; c_i + k_i \exp\Big(\sum_{j=1}^{M} t_{ij}\, r_j\Big)
$$

where negative `t_ij` means `j` helps `i`. The exponential sits outside the sum, which keeps the parameter count linear rather than quadratic in the number of domains. Composed with separate laws in training steps and model size, it predicts a mixture for a 1B model on 100B RedPajama tokens that matches the default mixture trained for 48% more steps. [RegMix](https://arxiv.org/abs/2407.01492) declines the functional form entirely, training 512 models of 1M parameters on 1B tokens each with Dirichlet-sampled mixtures and fitting a regressor; LightGBM beats linear regression at Spearman 97.1 against 88.0 when predicting the ranking of 1B models, which is itself evidence that the surface is not linear.

[Scaling Laws for Optimal Data Mixtures](https://arxiv.org/abs/2507.09404) extends the form to depend on `N` and `D` jointly and validates it on three modalities. Their LLM fit uses 160 mixtures over 7 SlimPajama domains at 412M to 1.4B parameters, validated at 3B and 7B; the multimodal and vision fits use 127 and 95 mixtures. The structural point is sharper than the accuracy numbers: in their *additive* law the cross-derivatives `∂²L/∂N∂h` and `∂²L/∂D∂h` vanish, so the optimal mixture is scale-invariant, while their *joint* law makes the optimum compute-dependent. Only one of those can be right, and they report the joint law fitting better. Their validation error is a mean relative error of 1.21% to 1.30% for LLMs, and they never report R², so cross-paper comparisons on fit quality have to use MRE.

That leaves the question of why any such law should hold. [Dai and Zheng](https://arxiv.org/abs/2606.08167) give the first mechanistic account. They assume skills within a domain follow a power law and that domains share a head of fundamental skills while their tails are disjoint. Finite capacity then forces an allocation across domains under a shared budget, and that shared constraint is precisely what couples the domains' losses, producing the interaction term that everyone had been fitting empirically. But capacity competition alone is degenerate: it predicts the optimal training mixture equals the target mixture, which is not what anyone observes. Adding a second term for one-pass-SGD noise, which depends only on a domain's own weight, breaks the symmetry and shifts weight toward domains that are harder to learn.

![How well the forms fit](blogs/images/pdsci-lawfit.svg?v=1)

Published numbers from their Table 2, redrawn: mean relative error when fitting the 64 one-billion-parameter runs over 17 Pile domains that RegMix released. The two-term account reaches 1.533% with 5K parameters, against 2.209% for the additive law at K(2K+1) parameters and 6.480% for RegMix's own regression. The comparison is on RegMix's data, fit by the authors of the winning method, so read it as an internal consistency check rather than an independent benchmark.

## 6. Whether any of it beats sampling in proportion

The three claims separate cleanly at this point. The laws fit. The fits are cheap. The remaining question is whether optimising them produces a better model than not bothering, and the most careful answer is negative.

[Aioli](https://arxiv.org/abs/2411.05735) evaluates six mixing methods against stratified sampling, which just samples each group equally. The setup is narrow and should be stated: a single model size, 160M parameters in the Pythia architecture, trained from scratch on SlimPajama carved into six domain settings of two, three and seven groups, for 5,000 steps at the small settings and 40,000 at the largest, five seeds at the small settings and three at the largest, scored as average test perplexity per group. Methods are given a budget of up to ten extra training runs to learn their mixture; stratified sampling and Aioli get zero.

![Nothing consistently beats the baseline](blogs/images/pdsci-aioli.svg?v=1)

Published numbers from their Table 2, redrawn as change against the stratified baseline, where negative is better. The finding is blunt: "no existing method consistently outperforms a simple stratified sampling baseline in terms of average test perplexity," and every method loses on at least one setting, by up to 6.9 perplexity points. DoReMi, the most cited method in the set, loses on three of six and is 6.898 worse on one of them. DoGE wins on one of six.

What saves this from being purely destructive is their diagnosis. They show that the existing methods are all instances of one parametric form, in which the next step's loss is `L^{t+1}(p) = σ(A^t p^t)` with `A_ij` encoding how training on group `j` affects loss on group `i`, and that offline methods use a static log-linear version while online methods use a linear dynamic version solved by exponentiated gradient descent. Then they test the pieces separately. The parameterisation is high fidelity: fitting the laws to sweeps over the simplex gives R² of 0.991 for the static log-linear form and 0.947 for the linear dynamic one. The solver is high fidelity: greedy descent recovers the optimal dynamic proportions in two of three settings. What is wrong is the *parameters*. Scoring each method by the similarity between its estimated `A` and an approximation of the true one, similarity correlates with improvement over stratified at R² = 0.491. The methods are solving the right problem with the right tool and the wrong numbers.

Aioli's own contribution follows from that diagnosis: estimate `A` from the run's own loss trajectory, by spending a small fraction of each round on one-hot mixtures and reading off the per-group loss deltas, at zero extra training runs. It beats stratified on all six settings, by an average of 0.274 perplexity. That is a small number, and the paper's claim is consistency rather than dominance; Aioli is the best method in only two of the six columns. In a more realistic setting where every method gets only 0.5 extra runs instead of ten, initialising from a method's mixture and continuing with Aioli improves 28 of 30 method-setting pairs, by an average of 1.202 perplexity.

The honest reading of the whole section is that a 160M model on SlimPajama is a narrow bed to test on, and the authors say the offline methods may be handicapped by the ten-run budget at seven groups. But the result stands as the most direct comparison anyone has run, and its verdict is that mixture optimisation as of 2025 buys less than its literature suggests.

## 7. What practice converged on

[Olmix](https://arxiv.org/abs/2602.12237) is the counterweight, in that it is the framework actually used to pretrain Olmo 3, and it treats mixture search as an engineering problem with a measured design space rather than a method to be defended. Their study fixes a target of a 1B Olmo 2 on 100B tokens over 24 DCLM topic domains scored on 52 tasks, then sweeps seven design questions.

Two of the answers overturn assumptions embedded in earlier work. Sample complexity in the number of domains is **linear, not quadratic**: error curves for 6, 12, 18 and 24 domains collapse when plotted against `c` in `K = c(m+1)` runs, and they recommend `K ≥ 3(m+1)`. Several earlier mixing-law papers assumed quadratic and sized their swarms accordingly. And the regression family question is confounded by swarm size: BiMix wins at 25 runs, LightGBM needs more than 118, and log-linear regression is best overall at ρ = 0.80 once the swarm is large enough. A comparison of regressors at a fixed small swarm measures something other than what it claims to.

Their headline engineering result addresses a problem the academic literature does not model at all, which is that the domain set keeps changing. They simulate a five-update development sequence going from 24 domains to 64: add fifteen Stack-Edu languages, add six more sources, revise one, remove one, partition one into twenty.

![Reusing the swarm across domain updates](blogs/images/pdsci-olmix.svg?v=1)

Published numbers from their Figure 10, redrawn. Recomputing the mixture from scratch at every update costs 832 proxy runs and yields 12.2% improvement in bits-per-byte over the natural distribution. Freezing the ratios among unaffected domains and solving in the reduced space reaches 11.6%, which is 95% of that, for 216 runs, a 74% saving. A partial variant reaches 12.0% for 272 runs.

One caveat belongs with those numbers: the only end-to-end baseline is the natural distribution, meaning weights proportional to domain sizes. There is no head-to-head comparison against DoReMi, RegMix, or a uniform mixture as complete methods. The "12% better" is relative to doing no mixture optimisation at all, which is the right baseline for an engineering decision and the wrong one for a claim about which method wins.

## 8. What the survey adds up to

Read together, the measurement papers and the method papers point the same direction, which is not the direction the field's headlines point.

The apparatus works and is cheap. A 30M-parameter proxy orders mixtures at ρ ≈ 0.9 against a 1B target; a 150M proxy gets 80% of corpus comparisons right; fitting scaling laws across several sizes does not beat either. What the apparatus keeps finding is that the effects are small relative to the noise. DataDecide measures a seed-to-seed standard deviation of up to two accuracy points at 1B, and Aioli measures the best available mixture method as worth 0.27 perplexity against sampling in proportion. A large fraction of published data-curation gains sit in that range.

The conditionals are doing more work than the claims. Repetition is nearly free or catastrophic depending on whether you repeat the whole corpus a few times or a slice of it a hundred times. Deduplication helps or does nothing depending on whether the corpus was dirty. The optimal mixture is scale-invariant or not depending on which of two laws from the same paper you fit. In each case the disagreement in the literature is smaller than it looks, and resolving it requires reading the setup rather than the abstract.

The most useful thing to take from all of this is a habit rather than a number. When a data result arrives, the questions that determine whether it transfers are: what was held fixed, compute or tokens; what was the baseline, a tuned alternative or no intervention at all; how many seeds, and what is the spread between them; and at what scale was the claim measured against the scale at which it will be used. Every finding above changes meaning under at least one of those, and most of the apparent contradictions in this field are two papers answering different questions in the same words.
