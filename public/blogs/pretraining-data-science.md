---
title: "Reading the Setup: What Pretraining Data Experiments Actually Show"
date: "2026/8/12"
---

Repeat 0.1% of a pretraining corpus a hundred times and an 800M-parameter model ends up performing like a 400M one. Repeat the entire corpus four times and the loss goes up by 0.5%. Train 1.5 epochs on a deduplicated Pile and the deduplication turns out to have bought nothing measurable.

All three are real results, from Hernandez et al., Muennighoff et al. and Pythia.

The same thing happens on the other main question, which is how to weight the domains a corpus is made of. DoReMi reports 6.5 points of downstream accuracy over the Pile's default domain weights. Aioli puts six mixing methods, DoReMi included, against sampling every group equally, and none of them wins consistently. Olmix, the framework behind Olmo 3, reports 12% better bits-per-byte from mixture search, measured against doing no mixture search at all.

Both sets of numbers are real too. The disagreements come from the experiments: what was repeated, what was held fixed, what the baseline was, how many seeds. Each number is only as general as the run that produced it, so every finding below comes with its setup: how many models, at what sizes, for how many tokens, against which baseline, measured with which metric.

Repetition and mixing also rest on the same assumption, that a small run predicts a large one. That assumption has itself been measured.

## 1. The coordinate system, and the correction that data constraints force on it

[Chinchilla](https://arxiv.org/abs/2203.15556) fits the loss surface as

$$
L(N, D) \;=\; E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

where `N` is the parameter count, `D` the number of training tokens, and `L` the cross-entropy in nats per token on held-out text. The three terms are a decomposition of the risk. `E` is what an ideal generative process on the data distribution would still pay, the entropy of natural text. $A/N^{\alpha}$ is the extra loss a perfectly trained transformer with only `N` parameters pays for having a finite hypothesis space. $B/D^{\beta}$ is the extra loss from stopping after `D` tokens instead of training to convergence.

The five constants come from a fit. Chinchilla takes the final loss of every run in its first two approaches and minimises a Huber loss with $\delta = 10^{-3}$ between predicted and observed *log* loss, using L-BFGS started from a grid over `α, β ∈ {0, 0.5, …, 2}` and the log-scale versions of `A`, `B` and `E`. Working in log space turns the sum of three terms into a log-sum-exp, which keeps the fit from being dominated by the largest-loss runs. The result is `E = 1.69`, `A = 406.4`, `B = 410.7`, `α = 0.34`, `β = 0.28`, over 400 models between 70M and 16B parameters trained on 5B to 500B tokens.

The rule everyone quotes falls out of those five numbers in one step. Take the usual approximation `C ≈ 6ND` for training FLOPs, substitute `D = C/(6N)` into the surface, and minimise over `N`: the two variable terms become $A/N^{\alpha}$ and $B(6N/C)^{\beta}$, one falling and one rising in `N`, and setting the derivative to zero gives $N_{\mathrm{opt}} \propto C^{\beta/(\alpha+\beta)}$ and $D_{\mathrm{opt}} \propto C^{\alpha/(\alpha+\beta)}$. With `α = 0.34` and `β = 0.28`, `β/(α+β) = 0.28/0.62 = 0.45` and `α/(α+β) = 0.55`; the paper reports 0.46 and 0.54 from the unrounded fit, and its other two approaches, which never write down a functional form, give 0.50/0.50 and 0.49/0.51. Those exponents being near a half is the entire content of "scale parameters and tokens in equal proportion."

The assumption hiding inside it is that tokens can be bought at constant quality in whatever quantity the optimum calls for. [Muennighoff et al.](https://arxiv.org/abs/2305.16264) removed that assumption and re-derived the surface. Their setup: over 400 models from 10M to 9B parameters, up to 900B total tokens, up to 1500 epochs, all on subsets of C4, GPT-2 architecture and tokenizer, cosine schedule decaying 10× over training, and deliberately no early stopping so that overfitting shows up. The subsets are nested, so a run with less unique data always sees a subset of what a larger run saw. Loss is reported on a held-out test set, unlike Chinchilla, which reported training loss.

Their model splits Chinchilla's `D` in two. $U_D = \min\{D_C, D\}$ is how many unique tokens the budget `D_C` actually supplies, and $R_D = D/U_D - 1$ is the number of repetitions, so a single epoch is $R_D = 0$. What they want is *effective* data `D'`, the amount of fresh data that would have produced the same loss.

The derivation is a geometric series. Assume each pass over a token extracts a fixed fraction of whatever information is still left in it, so the `k`-th pass is worth $(1-\delta)^{k}$ times the first. Summing over `R_D` repeats,

$$
\begin{aligned}
D' \;&=\; U_D + (1-\delta)U_D + (1-\delta)^{2}U_D + \cdots + (1-\delta)^{R_D}U_D \\[2pt]
     &=\; U_D + \frac{(1-\delta)\,U_D\left(1 - (1-\delta)^{R_D}\right)}{\delta}
\end{aligned}
$$

Now set $R_D^{*} = (1-\delta)/\delta$. For small $\delta$ this makes $1/R_D^{*} \approx \delta$, and $(1-\delta) \approx e^{-\delta} \approx e^{-1/R_D^{*}}$, so the sum collapses to

$$
D' \;=\; U_D + U_D\, R_D^{*}\left(1 - e^{-R_D / R_D^{*}}\right)
$$

So $R_D^{*}$ is the per-repeat decay rate $\delta$ in different clothes, written as $(1-\delta)/\delta$ because that form reads as a half-life. The fitted $\delta \approx 6\times 10^{-2}$ is what gives $R_D^{*} = 15.39$. Effective parameters get the same treatment, with $U_N = \min\{N_{\mathrm{opt}}, N\}$ and $R_N = N/U_N - 1$.

The fit is layered on Chinchilla's. They first refit the Chinchilla form on C4, using 54 of Chinchilla's own data points with `α` tied to `β`, which gives $L = 1.87 + 521/N^{0.353} + 1488/D^{0.353}$. Those constants are then frozen, and only $R_D^{*}$ and $R_N^{*}$ are fitted, on 182 of their own runs from 7M to 9B parameters and 1 to 500 epochs, with the same Huber-in-log-space objective and L-BFGS from a grid over `{0, 4, …, 20}²`. That gives $R_D^{*} = 15.39$ and $R_N^{*} = 5.31$, which the authors call the half-lives of repeated data and of excess parameters.

![What an extra epoch is worth](blogs/images/pdsci-repetition.svg?v=1)

Computed here from that expression. Four epochs return 3.73 unique corpora worth of value, which is where the widely quoted "up to four epochs is nearly free" comes from; the paper's own instance of it is an 8.7B model on 44B unique tokens finishing four epochs at 0.5% higher validation loss than the same model on 178B unique tokens. Sixteen epochs return 10.58. The curve's ceiling is $1 + R_D^{*} = 16.4$: no amount of repetition makes a corpus worth more than 16.4 times itself. The marginal column is my own arithmetic: the exponential gives the k-th epoch a value of $e^{-(k-1)/15.39}$ relative to a fresh token. The paper states the ceiling and the half-life and never tabulates per-epoch values.

Two more consequences follow. Because $R_N^{*} < R_D^{*}$, excess parameters lose value faster than repeated data, so under a data constraint you should spend surplus compute on epochs before parameters, which directly inverts Chinchilla's equal-scaling advice. They confirm this at scale: at 9.3×10²¹ FLOPs with 25B unique tokens, their frontier calls for 6.34B parameters on 242B total tokens, against Chinchilla's 8.67B on 178B, and the smaller model reaches lower loss. That is 27% fewer parameters spent on 36% more passes over the same 25B unique tokens.

And the ceiling can be raised by changing what counts as the corpus. Mixing in Python from The Stack, they find no degradation up to 50% code even when evaluating only on natural language, which makes code worth a 2× increase in effective tokens. The composite recommendation is explicit: double the data with code, then repeat four times, for "8x more training tokens that are expected to be just as good as having had 8x more unique data." Two of their nineteen evaluation tasks improve the moment any code is added at all: bAbI goes from 0.0 to 12.5 at 10% code, and WebNLG from 4.8 to 9.5, which the authors attribute to code teaching long-range state tracking.

## 2. What makes a data experiment believable

[Pythia](https://arxiv.org/abs/2304.01373) exists to make the inference auditable. Sixteen models, eight sizes from 70M to 12B, two data variants (Pile and Pile-deduped), every model trained on exactly 299,892,736,000 tokens, and, unusually, **all models trained on the same data in the same order**, with an identical batch of 1024 sequences × 2048 tokens at every size. There are 154 checkpoints per model: one at initialisation, log-spaced points at steps 1 through 512, then every 1000 steps. A reproducible dataloader ships with it, so any checkpoint can be tied to the exact tokens seen up to that point. Compare that to the 1 to 30 checkpoints released with GPT-2, GPT-3, OPT, T5 or BLOOM.

Pythia knocked out a widely believed mechanism almost as soon as it was released. The intuition that data seen later in training is memorised more would let a practitioner suppress memorisation by ordering data carefully. Testing it with (32, 32)-memorisation over the first 64 tokens of each context, they find that "a Poisson model fits the data extremely well," meaning training order has little effect: memorised sequences are not denser at the beginning or the end. Their practical advice follows: you cannot reorder data to reduce memorisation, but you can place the sequences you are worried about early, so that if memorisation happens you find out before the run ends.

[DataDecide](https://arxiv.org/abs/2504.11393) attacks the extrapolation question head-on with the largest controlled sweep in the area: 25 pretraining corpora, 14 model sizes from 4M to 1B, 3 seeds, **1,050 models** and more than 30,000 checkpoints, at roughly 820,000 H100-hours. Every model is trained at 100 tokens per parameter, five times Chinchilla, deliberately overtrained. The corpora cover the Dolma, C4, FineWeb, Falcon and DCLM lines, along with their quality-filter variants and a few λ-mixes. Success is measured by *decision accuracy*. Over all pairs of recipes `(A, B)`, with `y_A, y_B` the observed means at the target scale and `ŷ_A, ŷ_B` the small-scale predictions, it is the fraction of pairs for which `sign(ŷ_A − ŷ_B) = sign(y_A − y_B)`, which is Kendall's τ rescaled to run from 0 to 1. The target-scale winner is fixed by mean accuracy over the three 1B seeds, macro-averaged over ten OLMES tasks.

Their headline is that ranking corpora at a single small size, around 150M parameters, gets about 80% of the pairwise comparisons right at the 1B target. Their second result: **no scaling-law method among eight baselines exceeds the compute-to-decision-accuracy frontier of that single-scale baseline.** Fitting a curve through several model sizes, which is what most teams do, buys nothing over training one small model and reading off the order.

Two further findings from the same sweep matter for how you read everyone else's numbers. Continuous likelihood metrics beat accuracy at small scale, and specifically two of them. CORRECT PROB is the mean over items of `P(correct continuation | context)`. TOTAL PROB is the mean of `Σ_c P(c | context)` summed over every continuation the item offers, right and wrong alike. Both are length-normalised per character. What the two share is that neither penalises probability mass on wrong answers, which is what separates them from NORM CORRECT PROB, which divides the correct continuation's probability by the sum over the answer set, and from MARGIN, which subtracts the best wrong continuation. Those two track discrete accuracy and inherit its problem: a small model is near chance, and a metric that only credits being right has no signal left to give. TOTAL PROB goes the other way and rewards mass on plausible wrong answers, which the authors read as evidence the model has seen the domain at all. With that switch, code benchmarks go "from trivial to 80%" predictable, because small models climb above the noise floor while the metric still tracks large-scale accuracy. And the noise floor is not small: at the 1B, 5× Chinchilla scale, the standard deviation between seeds "can be as high as 2% points of accuracy" for some recipes on most tasks. So a curation gain under two points is the same size as the spread between seeds.

The complementary practitioner-side measurement comes from [Olmix](https://arxiv.org/abs/2602.12237), which asks how small the proxy can be before the ordering it produces stops matching the target.

![How small a proxy can be](blogs/images/pdsci-proxy.svg?v=1)

Published numbers from their Figure 3, redrawn. Proxies at or above 15M parameters reach Spearman rank correlation above 0.89 with the ordering of the same mixtures at 1B; at 1M the correlation falls to 0.73. They settle on 30M parameters and 3B tokens, at ρ = 0.896. The target for that comparison is a 1B Olmo 2 trained on 100B tokens over 24 topic domains, scored as macro-average bits-per-byte over 52 downstream tasks. Two groups using different corpora and different targets both end up around four orders of magnitude below the target.

## 3. Repeated data, first setup: a small subset repeated many times

[Hernandez et al.](https://arxiv.org/abs/2205.10487) produced the largest reported degradation numbers, and their setup is very specific. Every model trains on exactly 100B tokens, a fixed budget, pre-Chinchilla. Of those, 90% are drawn without repetition and 10% are repeats of a small random subset of the same corpus, so repeats displace unique tokens. The repeated subset can be tiny and repeated enormously, like 0.01% of the tokens repeated a thousand times, or larger and repeated less. They scan repeat fractions of 1%, 3%, 10%, 20%, 50% and 90%, model sizes from 1.5M to 1B, and repeat counts across two orders of magnitude. Test loss is measured on held-out data containing none of the repeated material.

The abstract's claim is that "performance of an 800M parameter model can be degraded to that of a 2x smaller model (400M params) by repeating 0.1% of the data 100 times, despite the other 90% of the training tokens remaining unique." The body of the paper puts the same data point at "nearly to that of a 340M parameter model," so the paper is internally inconsistent by about 0.35×, and I take the body's number.

The degradation is not monotonic in the amount of repetition. It is worst in a band, whose edges they fit as power laws in model size, $E = k\,N^{\alpha}$, with `(k, α) = (5.1×10⁷, −0.50)` on the right edge and `(4.2×10⁶, −0.56)` on the left.

![The region where repetition hurts](blogs/images/pdsci-degradation.svg?v=1)

Computed here by evaluating those two fits. The band slopes down and to the right: as models grow, the number of repeats that does the most damage falls. The mechanism the authors propose reads directly off the geometry. Repeat a lot of data a couple of times and the model lacks the capacity to memorise it, so nothing pathological happens. Repeat a tiny sliver ten thousand times and the model memorises it, but the sliver is so small that memorising it consumes little capacity. In between there is a region where the repeated data is *both* memorisable *and* large enough that memorising it costs a meaningful fraction of the model. Their arithmetic for why the model takes that trade: an 800M model reaches about 2.0 nats per token and a 400M model about 2.2, and memorised data costs 0, so `0.9 × 2.2 + 0.1 × 0 = 1.98 < 2.0`. 
Push the repeat fraction up and the loss curve against repeat count runs a full double descent: at 90% repeated data with 100× to 10,000× repeats, loss falls, rises, and falls again. At the peak of that curve the effective-parameter penalties are severe, 10× at 50% repeated and 73× at 90%.

The authors offer one extrapolation, and hedge it: the fitted boundaries suggest "significant degradation from repeating data as little as 2x on state-of-the-art language models with hundreds of billions of parameters." That is a conditional statement about their fixed-100B-token regime with a randomly chosen repeated subset. It is not a statement about a modern run that repeats a whole curated corpus twice, and section 4 is about why the difference matters.

## 4. What repetition damages, and why three papers disagree without conflicting

The same experiments measured several other things, and all of them degraded further than the loss did.

![Where the damage lands](blogs/images/pdsci-damage.svg?v=1)

Published numbers from their Figures 5 and 6, redrawn. At only 3% repeated data, at the worst repeat count, copying a repeated paragraph loses 3× of effective model size while overall test loss loses at most 1.15×. Prefix matching, the standard behavioural probe for induction heads, loses 1.47×. The damage concentrates on the mechanisms associated with generalisation.

In a one-layer attention-only model, on the last appearances of "Potters," the control model puts more than 97% of its probability on "ters" given "Pot"; the repeated-data model puts under 4%. In a two-layer model on "Dursley," control gives 92% where the repeated-data model gives 10%, and on the constructed "unDursleyish" the gap is 68% against 0.4%. The authors' reading is that "the pressure to memorize the repeated dataset has led a skip tri-gram head to replace the induction head entirely." Consistent with that, in-context learning degrades most: at the double-descent peak, loss on the eleventh copy of a repeated paragraph does not decrease at all with additional copies. They are careful about what they could not show, reporting that logit attribution failed to separate "the induction head is less active" from "other heads are interfering."

Now put the three verdicts side by side.

Anthropic repeats a **random small subset** while holding the **total token budget fixed**, so repeats displace unique tokens, and they measure at repeat counts up to 10,000×. Under those conditions repetition is destructive, and worst in the middle of the range.

Muennighoff et al. repeat the **entire available corpus** because there is no more of it, under a **compute** budget rather than a token budget, at epoch counts mostly in the single and double digits. Under those conditions the first four epochs are nearly free. There is no contradiction: 4 whole-corpus epochs is not the same experiment as 100 repeats of 0.1% of the data, and the Anthropic band puts an 800M model's damage peak at roughly 100 repeats, more than an order of magnitude away from four.

Pythia runs about 1.5 epochs on a MinHash-deduplicated Pile and sees nothing, which the band also predicts, and separately reports that deduplication brought "no clear benefit on language modeling performance." That claim is qualitative, the supporting plots live in an appendix, and the authors themselves offer the deflationary explanation, that the Pile's own upsampling of curated subsets may not behave like the incidental duplication other papers studied, or that "the general tendency of deduplicated data to outperform non-deduplicated data is primarily a statement about the quality of the data used in other works."

Muennighoff et al. reach the same conclusion from a different direction and pin down the condition. On C4 they find perplexity filtering effective and deduplication not: keeping the lowest-perplexity 25% raises their rescaled 19-task average from 22.2 to 25.3 at 4.2B parameters, while deduplication lowers it from 22.2 to 20.5. On OSCAR, which is noisier, both help. The generalisation they state is that "data filtering is primarily effective for noisy datasets," which is also the most plausible reading of Pythia's null result, since the Pile was curated to begin with.

The three results hold under three different conditions: a whole corpus repeated a few times, a small slice of it repeated hundreds of times, and deduplication run on a corpus that was already clean.

## 5. Data mixtures: the laws, and what they are laws about

Mixing is the one part of data work shaped like a clean optimisation problem, and it has accumulated the most formalism. The formalism divides into three claims that should be evaluated separately: that loss is a predictable function of the mixture, that fitting the function is cheap, and that optimising the fitted function produces a better model.

The first serious attack was [DoReMi](https://arxiv.org/abs/2305.10429), which trains a 280M proxy with group distributionally robust optimization over domains, the model minimising worst-case excess loss against a reference and the domain weights emerging from the inner maximisation, then transfers those weights to a model 30× larger. It reports improving "average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights" and reaching baseline accuracy in 75k steps against 200k. Perplexity improves on all 22 Pile domains *including the ones it downweights*, and the reweighting is drastic, with Pile-CC going from 0.1121 to 0.6057 and ArXiv from 0.1052 to 0.0036. Cutting arXiv to a thirtieth of its weight improved arXiv perplexity. One explanation that fits is that a specialised domain still needs general language ability first, and web text is the cheapest source of it.

[Data Mixing Laws](https://arxiv.org/abs/2403.16952) proposed the functional form that the later literature argues about. With `M` training domains, `r_j` the proportion of domain `j`, and `L_i` the validation loss on domain `i`,

$$
L_i(r_1, \ldots, r_M) \;=\; c_i + k_i \exp\Big(\sum_{j=1}^{M} t_{ij}\, r_j\Big)
$$

`c_i`, `k_i` and the `M` interaction terms `t_ij` are fitted per validation domain, `M + 2` numbers in all. Because `k_i > 0` the exponential is strictly positive, so `L_i > c_i` for every mixture and `c_i` is the part of domain `i`'s loss that no reweighting can touch. A negative `t_ij` means training on `j` lowers the loss on `i`.

The form came out of a two-domain pilot. On GitHub and Pile-CC at 70M and 160M they found that after subtracting a constant, the log of a domain's loss is linear in that domain's proportion, which is $L_i = c_i + k_i \exp(t_{ii} r_i)$. Going from two domains to `M` was a guess, constrained two ways: the form has to collapse back to the two-domain case, and it has to be symmetric under relabelling domains, since there is no reason to privilege one. Four candidates survive both constraints. The runner-up puts one exponential per training domain and adds them, $c_i + \sum_j k_{ij}\exp(t_{ij} r_j)$, and fits about as well, but it needs `2M + 1` coefficients. Exponential-of-a-sum won on parsimony.

The fitting procedure is unusual. The law composed with a learnable weighting over validation domains is exactly a one-hidden-layer network with an exponential activation and a softmax output layer, so they fit it as one, with AdaBoost over several such fits to damp the variance. They also decline to fit Chinchilla's two-variable surface directly, calling it unstable with that many free parameters at once, and instead run a law in steps $L(S) = E_1 + B/S^{\beta}$ and a law in size $L(N) = E_2 + A/N^{\alpha}$ in sequence before the mixing law.

Composed that way, it predicts a mixture for a 1B model on 100B RedPajama tokens that reaches the default mixture's loss in 73% of the steps. Elsewhere the paper writes the same thing as the default mixture needing 48% more steps to catch up; that number is an extrapolation along the fitted step law, not a run anyone trained. [RegMix](https://arxiv.org/abs/2407.01492) declines the functional form entirely, training 512 models of 1M parameters on 1B tokens each with Dirichlet-sampled mixtures and fitting a regressor; LightGBM beats linear regression at Spearman 97.1 against 88.0 when predicting the ranking of 1B models, which is itself evidence that the surface is not linear.

[Scaling Laws for Optimal Data Mixtures](https://arxiv.org/abs/2507.09404) extends the form to depend on `N` and `D` jointly and validates it on three modalities. Their LLM fit uses 160 mixtures over 7 SlimPajama domains at 412M to 1.4B parameters, validated at 3B and 7B. I did not look closely at the multimodal and vision fits, so what follows is the LLM one. In their additive law the cross-derivatives $\partial^2 L/\partial N \partial h$ and $\partial^2 L/\partial D \partial h$ vanish, so the optimal mixture is scale-invariant, while the joint law makes the optimum compute-dependent. Only one of those can be right, and they report the joint law fitting better. Their validation error is a mean relative error of 1.21% to 1.30% for LLMs, and they never report R², so cross-paper comparisons on fit quality have to use MRE.

[Dai and Zheng](https://arxiv.org/abs/2606.08167) give the first mechanistic account of why such a law holds. They assume skills within a domain follow a power law and that domains share a head of fundamental skills while their tails are disjoint. Finite capacity then forces an allocation across domains under a shared budget, and that shared constraint is precisely what couples the domains' losses, producing the interaction term that everyone had been fitting empirically. But capacity competition alone is degenerate: it predicts the optimal training mixture equals the target mixture, which is not what anyone observes. Adding a second term for one-pass-SGD noise, which depends only on a domain's own weight, breaks the symmetry and shifts weight toward domains that are harder to learn.

![How well the forms fit](blogs/images/pdsci-lawfit.svg?v=1)

Published numbers from their Table 2, redrawn: mean relative error when fitting the 64 one-billion-parameter runs over 17 Pile domains that RegMix released. The two-term account reaches 1.533% with 5K parameters, against 2.209% for the additive law at K(2K+1) parameters and 6.480% for RegMix's own regression. I would not read this table as a benchmark. The data is RegMix's, the fit was done by the authors of the winning method, and what it establishes is internal consistency.

## 6. Whether any of it beats sampling in proportion

The laws fit and the fits are cheap. Whether optimising them produces a better model is a separate question, and the most careful answer is negative.

[Aioli](https://arxiv.org/abs/2411.05735) evaluates six mixing methods against stratified sampling, which just samples each group equally. The setup is narrow: a single model size, 160M parameters in the Pythia architecture, trained from scratch on SlimPajama carved into six domain settings of two, three and seven groups, for 5,000 steps at the small settings and 40,000 at the largest, five seeds at the small settings and three at the largest, scored as average test perplexity per group. Methods are given a budget of up to ten extra training runs to learn their mixture; stratified sampling and Aioli get zero.

![Nothing consistently beats the baseline](blogs/images/pdsci-aioli.svg?v=1)

Published numbers from their Table 2, redrawn as change against the stratified baseline, where negative is better. Their finding: "no existing method consistently outperforms a simple stratified sampling baseline in terms of average test perplexity," and every method loses on at least one setting, by up to 6.9 perplexity points. DoReMi loses on three of six and is 6.898 worse on one of them. DoGE wins on one of six.

They also diagnose why. They show that the existing methods are all instances of one parametric form. Split training into `T` rounds and let $p^{t} \in \Delta^{m}$ be the mixture used in round `t`, over `m` groups. Then

$$
L^{t+1}_{\text{val},i}(p) \;=\; c^t_i + b^t_i\,\sigma\Big(\sum_{j=1}^{m} -A^t_{ij}\, p^t_j\Big)
$$

$A^{t}$ is an `m × m` matrix for that round, with $A^{t}_{ij}$ the amount by which training on group `j` moves the loss on group `i`; $b^{t}$ and $c^{t}$ are per-group scale and offset; `σ` is either the identity or `exp`. A published method is then three choices: the pair `(T, σ)`, a rule for estimating $A^{t}$, and a way to solve for `p`. Offline methods take `T = 1` and `σ = exp`, sweep at least `m + 1` static mixtures to fit `A`, and solve directly. Online methods take `T > 1` and `σ = Id`, and run exponentiated gradient descent, $p^{t+1}_j \propto p^{t}_j \exp\!\big(\eta \sum_i b^{t}_i A^{t}_{ij}\big)$. Where they really differ is $A^{t}$: DoReMi makes it diagonal with $A^{t}_{ii} = \min\{L^{t}_{\mathrm{train},i}(p) - L_{\mathrm{train},i}(f_{\mathrm{ref}}),\, 0\}$, so a group only gets weight while the model is still behind the reference on it, and DoGE sets $A^{t}_{ij} = \langle \nabla L^{t}_{\mathrm{val},i},\, \nabla L^{t}_{\mathrm{train},j} \rangle$, the inner product between the validation gradient on `i` and the training gradient on `j`. Then they test the pieces separately. The parameterisation is high fidelity: fitting the laws to sweeps over the simplex gives R² of 0.991 for the static log-linear form and 0.947 for the linear dynamic one. The solver is high fidelity: greedy descent recovers the optimal dynamic proportions in two of three settings. Scoring each method by the similarity between its estimated `A` and an approximation of the true one, similarity correlates with improvement over stratified at R² = 0.491. Of the three pieces, the first two hold up and the estimate of $A^{t}$ is what fails.

Aioli's own contribution follows from that diagnosis. Reading $A^{t}$ off directly would mean `m` extra training runs, one per one-hot mixture, so instead it spends a fraction $\delta$ of each round and chops that fraction into `K = mk` intervals. The mixtures used in those intervals are smoothed one-hot, $p^{t,i} = (1-\varepsilon)\mathbf{1}_i + \varepsilon\,\mathrm{Unif}(m)$, and their order is a random interleaving of `k` copies of each, so the model spends time on every one-hot mixture spread across the whole of $\delta$ rather than in one block at the end. Recording how much each group's validation loss drops after each interval builds a matrix `β` satisfying $\beta_i = P A^{t}_i$, where `P` stacks the `m` mixtures, so one matrix solve per group recovers $A^{t}$. Extra training runs: zero. It beats stratified on all six settings, by an average of 0.274 perplexity. 0.274 is small, and the claim is consistency; Aioli is still the best method in only two of the six settings. In a more realistic setting where every method gets only 0.5 extra runs instead of ten, initialising from a method's mixture and continuing with Aioli improves 28 of 30 method-setting pairs, by an average of 1.202 perplexity.

A 160M model on SlimPajama is a narrow condition to test under, and the authors say the offline methods may be handicapped by the ten-run budget at seven groups. Even so this is the most direct comparison anyone has run, and the gain it measures is smaller than this literature usually implies.

## 7. What practice converged on

[Olmix](https://arxiv.org/abs/2602.12237) is the framework actually used to pretrain Olmo 3, and it treats mixture search as an engineering problem and measures the design space item by item. Their study fixes a target of a 1B Olmo 2 on 100B tokens over 24 DCLM topic domains scored on 52 tasks, then sweeps seven design questions.

Two of the answers overturn assumptions in earlier work. Sample complexity in the number of domains is linear: error curves for 6, 12, 18 and 24 domains collapse when plotted against `c` in `K = c(m+1)` runs, and they recommend `K ≥ 3(m+1)`. Several earlier mixing-law papers assumed quadratic and sized their swarms accordingly. And the regression family question is confounded by swarm size: BiMix wins at 25 runs, LightGBM needs more than 118, and log-linear regression is best overall at ρ = 0.80 once the swarm is large enough. So a comparison of regression families at a fixed small swarm is really a comparison at that sample size.

The domain set keeps changing, which the academic literature does not model at all. They simulate a five-update development sequence going from 24 domains to 64: add fifteen Stack-Edu languages, add six more sources, revise one, remove one, partition one into twenty.

![Reusing the swarm across domain updates](blogs/images/pdsci-olmix.svg?v=1)

Published numbers from their Figure 10, redrawn. Recomputing the mixture from scratch at every update costs 832 proxy runs and yields 12.2% improvement in bits-per-byte over the natural distribution. Freezing the ratios among unaffected domains and solving in the reduced space reaches 11.6%, which is 95% of that, for 216 runs, a 74% saving. A partial variant reaches 12.0% for 272 runs.

I first read that 12% as "Olmix beats DoReMi and the rest by 12%." It is not. The only end-to-end baseline in the paper is the natural distribution, meaning weights proportional to domain sizes, and DoReMi, RegMix and a uniform mixture are never run head-to-head as complete methods. For an engineering decision the natural distribution is the right baseline, but it cannot answer which method is better.

## 8. The size of the effects

The apparatus is cheap and it works. A 30M-parameter proxy orders mixtures at ρ ≈ 0.9 against a 1B target, and a 150M proxy gets 80% of corpus comparisons right, while fitting scaling laws across several sizes does neither. What the apparatus keeps finding is that the effects are small relative to the noise. DataDecide measures a seed-to-seed standard deviation of up to two accuracy points at 1B, and Aioli measures the best available mixture method as worth 0.27 perplexity against sampling in proportion. A large fraction of published data-curation gains sit in that range.

The conditionals carry most of the weight. Repeating a whole corpus a few times is nearly free; repeating a slice of it a hundred times is destructive. Whether deduplication helps depends on how dirty the corpus was. As for whether the optimal mixture is scale-invariant, two laws in the same paper disagree. Resolving any of these means reading the setup: what was held fixed, compute or tokens; what the baseline was, a tuned alternative or no intervention at all; how many seeds, and what the spread between them was; and at what scale the claim was measured against the scale at which it will be used.
