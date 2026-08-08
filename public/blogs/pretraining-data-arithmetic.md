---
title: "Running Out of Internet: The Arithmetic of Pretraining Data"
date: "2026/8/7"
---

There is a number that reframes almost every open question in pretraining data. [Villalobos et al.](https://arxiv.org/abs/2211.04325) estimate the effective stock of text in the indexed web at around 4×10¹⁴ tokens, and frontier training sets, sitting near 10¹³ tokens in 2024, have been growing about 0.38 orders of magnitude a year. Extrapolating that growth against that ceiling puts the crossing in 2028, which is the middle of the paper's own 2026–2032 range.

![The token budget](blogs/images/pdata-budget.svg?v=1)

The dashed line is not the amount of text on the web. Raw indexed web text is estimated at 510T tokens and Common Crawl at 130T; the 4×10¹⁴ figure is what survives quality filtering, at an assumed 10–40% retention, multiplied by a 5× allowance for repeating data. Almost every term in that product is something a data team can move. Once the supply is fixed, every technique in this field is an answer to one question: how much usable signal can you extract per token you already have.

Filtering, deduplication and quality classifiers decide *which* tokens are worth spending compute on. Mixing decides in what *proportion*. Repetition decides how many *times*. Curriculum decides *when*. Synthetic data tries to move the ceiling itself. Contamination is the bill that arrives when you get any of it wrong without noticing.

## 1. What the corpus generations actually changed

The open web corpora form a sequence where each generation added one mechanism, and the sizes tell you where the effort went.

![Published sizes of open pretraining corpora](blogs/images/pdata-corpora.svg?v=1)

C4 came out of the T5 work in 2019 as roughly 750GB of English text, built by pointing heuristics at a single April 2019 Common Crawl snapshot: drop lines that do not end in terminal punctuation, drop documents containing words from a blocklist, keep only what a language detector calls English, remove duplicated lines. That set the template, and the [documentation paper](https://arxiv.org/abs/2104.08758) two years later is the reason anyone still argues about it. Applying a dialect classifier to what the blocklist removed, the authors found African American English removed at 42% and Hispanic-aligned English at 32%, against 6.2% for White American English. The surviving corpus is 97.8% WAE and 0.07% AAE. A filter designed to remove pornography removed speech communities, and nobody noticed for two years because nobody measured.

[CCNet](https://arxiv.org/abs/1911.00359) contributed the idea that quality can be scored rather than matched. Train a KenLM n-gram model on Wikipedia, score each Common Crawl document's perplexity under it, and split the corpus into head, middle and tail buckets. This is the ancestor of everything in section 3: it replaces a rule about what text looks like with a model of what good text looks like.

The Pile went the other way, assembling 825 GiB from 22 curated subsets, and its lasting contribution turned out to be the domain labels rather than the text. Because every document carries a source, The Pile became the standard testbed for mixing research, and its default 22-domain weights are the baseline that every method in section 4 reports against.

[RefinedWeb](https://arxiv.org/abs/2306.01116) made the argument that curation was unnecessary: "properly filtered and deduplicated web data alone can lead to powerful models; even significantly outperforming models from the state-of-the-art trained on The Pile." Its pipeline is the modern shape, in order: URL filtering against an adult-content blocklist and a scoring system, text extraction with trafilatura, fastText language identification, the Gopher heuristics plus their own line-wise corrections, then MinHash fuzzy dedup followed by exact-substring dedup. Out of Common Crawl they obtained 5T tokens and released 600B.

Then the transparency generation. RedPajama reproduced the Llama 1 recipe at 1.21T tokens; SlimPajama ran global MinHashLSH over it at Jaccard threshold 0.8 on 13-grams and removed 49.6% of the bytes, leaving 627B. [Dolma](https://arxiv.org/abs/2402.00159) published 3T tokens curated down from roughly 200TB of raw text to 11TB, together with the toolkit and the ablations. [FineWeb](https://arxiv.org/abs/2406.17557) processed 96 Common Crawl dumps from summer 2013 to April 2024 into 15T tokens, later extended past 18.5T. [DCLM](https://arxiv.org/abs/2406.11794) standardised 240T tokens of Common Crawl into a benchmark with 53 downstream evaluations and model scales from 412M to 7B, turning curation into something with a leaderboard.

The pools are enormous and the corpora people actually train on are one to two orders of magnitude smaller. That gap is the subject of the next three sections.

## 2. Deduplication, and why MinHash has an S-curve

Duplication in web text is not a rounding error. Running near-duplicate detection over C4, [Lee et al.](https://arxiv.org/abs/2107.06499) found 3.04% of training examples had a near-duplicate elsewhere in training, rising to 13.63% for RealNews, and the extreme cases are stranger than the averages: they removed "a single 61 word English sentence that is repeated over 60,000 times." Train-test overlap "affects over 4% of the validation set" of standard datasets, so duplication silently inflates evaluation. And deduplication "allows us to train models that emit memorized text ten times less frequently," so duplication is also what makes models regurgitate.

Exact duplicate removal is a hash table, and exact *substring* removal is a suffix array over the concatenated corpus, which finds every repeated span above a length threshold without any notion of document boundaries. Neither catches the interesting case: two documents that differ by a boilerplate header, a timestamp, or one rewritten sentence.

The standard answer is MinHash with locality-sensitive hashing. Represent each document by its set of k-gram shingles. For a random permutation π of the shingle universe, the probability that the minimum of π over set A equals the minimum over set B is exactly the Jaccard similarity J(A,B), because the argmin over A ∪ B is equally likely to be any element of the union, and the two minima agree precisely when it lands in the intersection. So a signature of m independent permutations gives m Bernoulli(J) samples: comparing signatures estimates Jaccard without ever comparing documents.

LSH turns that estimate into a retrieval structure. Cut the m-row signature into `b` bands of `r` rows each, m = b·r, and index each band separately. Two documents collide in a given band when all `r` of its rows agree, which happens with probability `s^r` at true similarity `s`, so the probability of colliding in at least one band is

$$
P(\text{candidate}) \;=\; 1 - (1 - s^{r})^{b}
$$

This is a sigmoid in `s` whose inflection sits near `(1/b)^{1/r}`, and that expression is the actual knob. It is not a similarity threshold you set; it is a threshold you *induce* by choosing how to factor your hash budget.

![The LSH S-curve, measured and predicted](blogs/images/pdata-lsh.svg?v=1)

The curve is the formula and the dots are measured: I built document pairs with controlled overlap, computed real MinHash signatures over 5-gram shingles with 180 permutations, indexed them in 20 bands of 9 rows, and recorded how often each pair actually surfaced as a candidate. The right panel factors the same 180 hashes five different ways. Nothing about the corpus changes across those five rows; 45 bands of 4 rows will pull in pairs at Jaccard 0.39, and 5 bands of 36 rows will not look at anything below 0.96. A dedup pipeline's aggressiveness is set by an integer factorisation, which is not where most people look for it.

Two further choices matter as much as `(b, r)`. Granularity: document-level, paragraph-level and line-level dedup remove very different things, and SlimPajama's ablations showed that global dedup across sources and local dedup within them are not interchangeable. And semantics: [SemDeDup](https://arxiv.org/abs/2303.09540) embeds documents with a pretrained model, clusters, and removes near-neighbours inside clusters, catching pairs that share no shingles at all. On a LAION subset it removes "50% of the data with minimal performance loss, effectively halving training time." Its own C4 result is more sober, beating prior deduplication with 15% efficiency gains, and the authors attribute the smaller margin to C4 already being partially curated.

## 3. Quality filtering, from rules to classifiers

The Gopher pipeline established the rule-based standard: filter on document statistics like word count, mean word length, symbol-to-word ratio and the fraction of lines that are bullets or end in ellipsis, strip repeated text inside a document, deduplicate across documents, and remove documents overlapping the test sets. Every modern pipeline reimplements some version of these, with thresholds that differ between reimplementations. What matters more is what happens when the filters compose, which Dolma measured.

![What survives a composed pipeline](blogs/images/pdata-funnel.svg?v=1)

Those rates are published per stage and the compounding is mine: Gopher's full rule set tagged 15.23% of UTF-8 characters for removal, C4's no-punctuation rule tagged 22.73%, fuzzy plus exact deduplication removes roughly half, and a model-based classifier keeping its top decile removes another 90%. Multiply and about 3% of the extracted text survives. That is the real shape of a web pipeline: not a filter, a survival probability.

C4's NoPunc rule *on its own* outperformed both the full C4 rule set and the full Gopher rule set, on perplexity and downstream tasks; the best configuration was Gopher-All plus C4-NoPunc. More rules is not more quality, and the single crudest rule in the collection carried much of the benefit.

The 2024 consensus moved from rules to learned classifiers, and the two reference implementations differ in an instructive way. DCLM trains a fastText classifier with instruction-formatted text as positives, specifically OpenHermes 2.5 and r/ExplainLikeImFive, against random web pages as negatives. FineWeb-Edu instead has Llama-3-70B-Instruct score 460k pages for educational value, distils that into a small classifier reaching F1 82% at threshold 3, and applies it to all 15T tokens at a cost of 6,000 H100 hours. Both work. FineWeb-Edu reports that a 1.71B model trained on 350B tokens moves MMLU from 33% to 37% and ARC from 46% to 57%. DCLM's headline is stronger and states the general principle: "model-based filtering is key," with DCLM-Baseline giving a 7B model 64% 5-shot MMLU on 2.6T tokens, 6.6 points over the previous best open-data model at 40% less compute, and 6.6× less compute than Llama 3 8B.

Neither team defined quality; both picked a proxy corpus that they believed correlated with it, and the filter inherits every property of that choice. "Text that looks like an ELI5 answer" and "text a 70B model calls educational" are different targets, and there is no reason to expect either to be the target you want for a model that will be post-trained into an agent. The [pretrainer's guide](https://arxiv.org/abs/2305.13169) makes the accompanying trade explicit: filtering toxicity out of pretraining reduces the model's ability to *detect* toxicity, so quality and capability point in opposite directions for at least one axis, and there is no reason to think it is the only one.

## 4. Mixing, as an optimization problem

Once you have domains, you have proportions, and proportions are the one part of data work that looks like an optimization problem with a real objective. The methods differ mainly in what they are willing to assume.

DoReMi treats it as a minimax problem. Train a small proxy model with [group distributionally robust optimization](https://arxiv.org/abs/2305.10429) over domains, in which the model minimises worst-case excess loss relative to a reference model and the domain weights come out of the inner maximisation, then use those weights to train a model 30× larger. From a 280M proxy to an 8B target, this "improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps," which is 75k steps against 200k. Perplexity improves on all 22 Pile domains, *including the ones the method downweighted*. The learned weights are not gentle. Pile-CC goes from 0.1121 to 0.6057; ArXiv goes from 0.1052 to 0.0036; PubMed Central goes from 0.1071 to 0.0046. Cutting a domain to a thirtieth of its weight made that domain's own perplexity better, because what a domain needs most is a model that has learned language, and the fastest route to that is more web text.

[DoGE](https://arxiv.org/abs/2310.15393) replaces the robustness objective with a first-order one. Score each domain by the inner product of its gradient with the summed gradient across domains, up-weight domains whose gradient aligns with the direction that helps everything, and update the weights by mirror descent. The score decomposes into cross-domain alignment plus the domain's own gradient norm, so "difficulty" and "usefulness" are entangled in it, but the framing connects mixing to influence functions directly. It reports 1.7 accuracy points over uniform on 5-shot reasoning with an 82M proxy for a 684M target.

[RegMix](https://arxiv.org/abs/2407.01492) declines to model the mechanism at all. Train 512 models of 1M parameters on 1B tokens each with mixtures sampled from a Dirichlet tilted toward extremes, fit a regressor from mixture to validation loss, and optimize the regressor. LightGBM beats linear regression badly here (Spearman 97.1 against 88.0 when predicting 1B-model rankings), which by itself says the loss surface over the simplex is not close to linear. The predicted-best mixture ranked first among 64 candidate 1B models trained on 25B tokens, and the 512 proxies cost about 2% of the FLOPs of a single 1B run. Two of its findings generalise. Web corpora, not the sources everyone calls high-quality, correlate most strongly with downstream performance, echoing DoReMi's Pile-CC result from a completely different method; and "data mixture effects transcend scaling laws," meaning the ranking of mixtures found at small scale survives to large scale even where absolute losses do not.

[Data Mixing Laws](https://arxiv.org/abs/2403.16952) puts a functional form on the surface. With `r_j` the proportion of training domain `j` and `L_i` the validation loss on domain `i`,

$$
L_i(r_1, \ldots, r_M) \;=\; c_i + k_i \exp\Big(\sum_{j=1}^{M} t_{ij}\, r_j\Big)
$$

with `c_i` the loss no mixture can remove, `k_i > 0` a scale, and `t_ij` the interaction between training domain `j` and validation domain `i`; negative `t_ij` means `j` helps `i`. The exponential sits outside the sum, so the parameter count grows linearly in the number of domains rather than quadratically. The overall objective is `Σ_i s_i L_i` with `s_i` the validation mixture. Fitting this at small scale and composing it with separate scaling laws in training steps and model size lets them predict large-run losses from small runs; the payoff is a mixture for a 1B model on 100B RedPajama tokens "reaching a performance comparable to the one trained for 48% more steps on the default mixture."

Below is a three-domain instance evaluated on the whole simplex, with plausible interaction signs and one domain weighted more heavily in validation.

![The loss surface over a three-domain simplex](blogs/images/pdata-mixing.svg?v=1)

Two features generalise. The optimum is interior and it is not the uniform mixture, which is the entire reason automated mixing exists. And the surface is shallow near the optimum and steep near the edges: the gap between uniform and optimal here is 0.0129 nats, which under the C4 scaling law is worth about 21% more tokens at a 100B-token budget, while walking to a corner of the simplex costs vastly more. Mixing is worth doing and it is not worth agonising over the third decimal place.

[CLIMB](https://arxiv.org/abs/2504.13161) addresses the assumption underneath all of the above, which is that you have domain labels. Common Crawl does not come with them. CLIMB embeds documents, clusters them, and runs the proxy-plus-predictor search over cluster weights instead of source weights, then continues training a 1B model on 400B tokens with the discovered mixture to beat Llama-3.2-1B by 2.0%. Clusters are not domains: nothing in the mixing formalism requires the partition to be human-meaningful.

## 5. Repetition: how many times can you use the same token

Everything so far assumes each token is seen once. The moment the pool is fixed, that assumption is a choice, and [Muennighoff et al.](https://arxiv.org/abs/2305.16264) measured what it costs across more than 400 runs from 10M to 9B parameters, up to 900B tokens and up to 1500 epochs.

Let `U_D` be the number of unique tokens and `R_D = D/U_D − 1` the number of repetitions, so `R_D = 0` is a single epoch. Then the *effective* data, meaning the amount of fresh data that would have produced the same loss, is

$$
D' \;=\; U_D + U_D\, R_D^{*}\left(1 - e^{-R_D / R_D^{*}}\right)
$$

with a symmetric expression for effective parameters `N'` under excess parameters `R_N`. Fitting gives `R_D* = 15.39` and `R_N* = 5.31`, which the authors describe as the half-life of repeated data and of excess parameters. Everything about repetition follows from the shape of that exponential.

![What repeated epochs are worth](blogs/images/pdata-repetition.svg?v=1)

Four epochs return 3.73 unique tokens' worth of value, or 93% of what four fresh epochs would have given, which is the quantitative form of the paper's "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data." Sixteen epochs return 10.58, or 66%. And the curve has a hard ceiling at `1 + R_D* = 16.4`: no amount of repetition of a fixed corpus is worth more than 16.4 times that corpus, ever. By epoch 44 you have collected 15.45 of those 16.4, and the marginal epoch is worth 0.063 of a fresh token, a sixteenth of what the first epoch was worth. Note that the paper observes 44-epoch models actually diverging, so the model's late tail describes a regime you would not operate in.

The widely repeated claim that returns hit zero around 40 epochs does not appear in the paper. What the paper says is that meaningful gains extend to about 16 epochs, "beyond which returns diminish extremely fast." The 40 is a secondary-source paraphrase; the formula above is the primary object, and it says something slightly different and much more useful.

## 6. Where the compute should go when data is fixed

The repetition curve becomes a decision once you put it back into a compute budget. Muennighoff et al. plug `N'` and `D'` into a Chinchilla-style form refit on C4, giving

$$
L(N', D') \;=\; 1.87 + \frac{521}{N'^{\,0.353}} + \frac{1488}{D'^{\,0.353}}
$$

which lets you ask a question the original [Chinchilla](https://arxiv.org/abs/2203.15556) analysis could not. Chinchilla assumes unlimited unique data and concludes that parameters and tokens should scale together, roughly 20 tokens per parameter. Under a fixed unique-token budget that advice stops applying, because "more tokens" is no longer available at constant quality.

So fix `U_D` at 1T unique tokens, sweep the compute budget `C = 6ND`, and for each budget search over model size, letting the number of epochs fall out as `D / U_D`.

![Optimal epochs against compute at a fixed unique-token budget](blogs/images/pdata-allocation.svg?v=1)

Below 10²³·⁵ FLOPs the optimizer never repeats anything: it spends every additional FLOP on parameters and stays at one epoch. Past that point parameters have been pushed far enough past `U_N` that the `R_N` penalty bites harder than the `R_D` penalty, and the optimum starts buying epochs instead. At 10²⁵ FLOPs it wants a 229B-parameter model and 7.3 passes over the data.

That threshold moves with the data you have, and the scaling is clean: 100B unique tokens starts repeating at 10²¹·⁵, 1T at 10²³·⁵, 10T at 10²⁵·⁵. Extrapolate to the whole effective stock of public human text, 4×10¹⁴ tokens, and repetition starts paying at about 10²⁷ FLOPs. Villalobos et al. put the compute at which models exhaust the stock at around 5×10²⁸ FLOPs. Those two numbers sit within about two orders of magnitude of each other, which is the actual content of the phrase "data wall": not a cliff, but the point past which additional compute buys epochs and parameters rather than experience.

## 7. Curriculum: not all tokens should arrive at the same time

Mixing as posed in section 4 asks for one set of proportions for the whole run. Two independent industrial results say that is leaving value on the table.

Llama 3 anneals the learning rate linearly to zero over the final 40M tokens while upsampling very high quality sources, and averages checkpoints from the annealing phase. Annealing improved an 8B model "on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively," with the ablation annealing on those benchmarks' *training* sets, and the improvement on the 405B model was negligible. The mechanism is presumably that a small model needs the demonstration and a large one has already generalised to it.

The same paper also turns annealing into a measurement instrument. To evaluate a candidate dataset, take a 50%-trained 8B checkpoint, anneal to zero over 40B tokens with 30% weight on the candidate and 70% on the default mix, and read off the change. That costs one short run per candidate instead of a scaling-law sweep per candidate.

[OLMo 2](https://arxiv.org/abs/2501.00656) formalises the same instinct into a named stage. Stage 1 is about 3.9T tokens, over 95% web. Stage 2, which they call midtraining, draws from a high-quality pool of roughly 843B tokens assembled from filtered web, academic text, Q&A, instruction data and synthetic mathematics, sampled down to 50B, 100B or 300B with DCLM-filtered web still supplying about half the budget. For the 7B they run three 50B anneals differing in data order and average the weights.

![Midtraining, before and after](blogs/images/pdata-midtrain.svg?v=1)

Those are published numbers, not measurements of mine. GSM8K goes from 24.1 to 67.5 for the 7B and 37.3 to 75.1 for the 13B; DROP goes from 40.7 to 60.8; the ten-benchmark average moves 10.6 points for the 7B and 10.3 for the 13B. A tenth of the token budget, spent last, moves the average by more than most architectural changes do. They also name the evaluation protocol, *microannealing*, which is the Llama 3 measurement run once per candidate dataset.

Which makes a single global mixture the wrong object. These results describe a schedule, and none of the mixing methods optimize over schedules; they optimize over a point.

## 8. Synthetic data, and the shape of its failure mode

If the stock of human text is fixed, the obvious move is to manufacture more. The evidence here is genuinely two-sided and the two sides are usually quoted against each other when they are actually about different things.

On the productive side, [phi-1](https://arxiv.org/abs/2306.11644) trained a 1.3B model on 6B tokens of filtered code plus about 1B tokens of GPT-3.5-generated textbooks and exercises, on 8 A100s for 4 days, reaching 50.6% pass@1 on HumanEval and 55.5% on MBPP; a 350M version still reached 45%. Read the composition carefully, because it is routinely misdescribed: the 6B is filtered code from The Stack and StackOverflow, not general web text, and the synthetic portion is roughly a seventh of the tokens. [WRAP](https://arxiv.org/abs/2401.16380) makes the more directly relevant claim for a data-constrained world, using an instruction-tuned model to rephrase web documents into different styles and reporting roughly 3× faster pretraining on C4, better than 50% perplexity improvement across Pile subsets at matched compute, and over 2% zero-shot gains across 13 tasks. Rephrasing spends compute to buy data that behaves like unique data, which is precisely what section 5 says is scarce.

On the failure side is the Nature result on [model collapse](https://www.nature.com/articles/s41586-024-07566-y): "indiscriminate use of model-generated content in training causes irreversible defects in the resulting models, in which tails of the original content distribution disappear." The mechanism is simpler than the name suggests. Fit a Gaussian by maximum likelihood to `n` samples, draw `n` fresh samples from the fit, refit, and repeat. Each individual step is an unbiased estimate of the mean. But the maximum-likelihood variance is biased low by exactly `(n−1)/n`, so

$$
\mathbb{E}[\sigma_g^2] \;=\; \left(\frac{n-1}{n}\right)^{g}\sigma_0^2
$$

and the variance is a non-negative martingale, which must converge; the limit is zero.

![Variance under recursive refitting](blogs/images/pdata-collapse.svg?v=1)

With `n = 100` and 300 independent chains, the measured mean variance tracks that geometric decay exactly, and the practical consequence is the green line: the probability mass the generation-`g` model assigns outside the *original* distribution's ±2σ falls from 4.6% at generation 0 to 0.08% by generation 100 and is numerically zero by generation 200. Nothing pathological happens at any single step. Collapse is what finite samples do when they are iterated, and no amount of care inside one generation prevents it.

Which is why the two sides are not in conflict. Collapse is a statement about the closed loop; phi and WRAP are statements about a single generation of synthesis anchored to real data. The condition that separates them is whether real data keeps entering the loop, and that condition is checkable rather than philosophical.

## 9. Contamination, and why n-gram decontamination does not work

Every claim in this post is a benchmark number, so the last question is whether benchmark numbers mean anything. The standard industrial answer is n-gram decontamination: remove any training document sharing an n-gram with a test item, with `n` usually between 8 and 13.

[Yang et al.](https://arxiv.org/abs/2311.04850) showed that this is trivially evaded, and not by an adversary. Rephrase a test item, or translate it, and the n-gram overlap goes to zero while the item remains, for evaluation purposes, the same item. Their headline is that "a 13B model can easily overfit a test benchmark and achieve drastically high performance, on par with GPT-4"; on rephrased MMLU, Llama-2 7B goes from 45.3 to 88.5 and the 13B from 54.8 to 89.9.

![n-gram overlap between a benchmark item and its paraphrase](blogs/images/pdata-contamination.svg?v=1)

Those are measured overlaps between one arithmetic word problem and a paraphrase of it that any competent model would answer identically. Unigram overlap is 38%, bigram 15%, trigram 3%, and from 4-grams up it is exactly zero. A decontamination filter at n = 13 does not merely miss this pair; it cannot see it at all, and the same holds at n = 8 and n = 5. The n that would catch paraphrases is small enough to delete most of the corpus.

Contamination is already measured in real corpora rather than hypothesised. The same work found 8.5% of HumanEval overlapping RedPajama-Data-1T, 15.9% overlapping StarCoder-Data and 18.9% overlapping The Stack. Those are the numbers n-gram matching *can* see.

[Min-K% Prob](https://arxiv.org/abs/2310.16789) asks whether a given text was in the pretraining data without access to the corpus or a reference model, by taking the k% lowest-probability tokens in the text and averaging their log-likelihood, on the theory that unseen text contains outlier tokens the model finds surprising. On WIKIMIA, a benchmark built by using Wikipedia edit timestamps to construct ground-truth membership labels, it reaches 0.72 average AUC against 0.67 for the best prior method, an improvement the paper reports as 7.4% and which is relative, not 7.4 AUC points. An AUC of 0.72 is a real signal and a poor test. We cannot currently verify the cleanliness of an evaluation on a corpus we did not build, and most published comparisons are between models whose contamination status is unknown in both directions.

## 10. What the arithmetic implies

Put the numbers in one place and a shape emerges. The effective stock is around 4×10¹⁴ tokens and dataset sizes reach it around 2028. Composed filtering keeps a few percent of extracted text, and the classifier stage is doing most of that work and is defined by an arbitrary choice of positive examples. Deduplication's aggressiveness is set by an integer factorisation most teams never tune. Mixing is worth roughly 20% of a token budget and no more. Repetition is worth at most 16.4× your unique corpus, and starts being the right call at 10²³·⁵ FLOPs for a 1T-token pool and 10²⁷ for the entire web. A tenth of the budget spent last is worth ten points of benchmark average. Synthesis works exactly as long as real data keeps entering the loop. And none of these measurements can currently be validated against a benchmark we know to be clean.

Four of the five biggest levers, filtering positives, dedup granularity, mixture and the midtraining schedule, are choices someone made once and rarely revisited. The one that gets all the attention, corpus size, is the one hitting a wall. There is a version of the next few years where data work stops looking like acquisition and starts looking like measurement: not what else can we get, but what is the stuff we already have actually worth, and how would we know.
