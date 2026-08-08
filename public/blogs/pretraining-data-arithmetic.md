---
title: "Running Out of Internet: The Arithmetic of Pretraining Data"
date: "2026/8/7"
---

Pretraining a frontier model today means reading something on the order of ten to twenty trillion tokens. Llama 3 was trained on about 15T. DeepSeek-V3 on 14.8T. Nearly all of that text comes from Common Crawl, a non-profit that crawls the public web every month or two and releases the result for free; each such crawl is called a snapshot, and one snapshot is on the order of a few billion pages and tens of terabytes of raw HTML. The largest openly released web corpus, FineWeb, reaches 18.5T tokens after processing 96 of those snapshots, covering eleven years of the web. These numbers have been roughly doubling every year and a half for most of a decade, and for most of that decade the response to wanting a better model was to find more text.

It is worth asking how much longer that works. The web is large but it is not infinite, and the fraction of it that is text, in a language you care about, not already a copy of something else, and not machine-generated boilerplate, is a good deal smaller than the fraction that is bytes.

[Villalobos et al.](https://arxiv.org/abs/2211.04325) tried to put a number on it. The indexed web means the portion of public pages a search engine can reach, excluding anything behind a login, private databases, and most dynamically generated pages. Their estimate of the raw stock of text on it is around 510 trillion tokens, with Common Crawl covering about 130T of that. But raw stock is not what you can train on. After assuming that quality filtering keeps somewhere between 10% and 40% of deduplicated web text, and allowing a factor of roughly 5 for how much value repeating data can add, they arrive at an *effective* stock of about 4×10¹⁴ tokens. Meanwhile training-set sizes have been growing at roughly 0.38 orders of magnitude per year, starting from about 10¹³ tokens in 2024.

![The token budget](blogs/images/pdata-budget.svg?v=1)

Extrapolating that growth rate against that ceiling puts the crossing in 2028, comfortably inside the paper's own stated window of 2026 to 2032. At that point "get more data" stops being a strategy that works on its own.

Look again at how the 4×10¹⁴ was constructed. It is raw stock, times a filtering retention rate, times a repetition multiplier. Every term in that product except the first is something a data team controls. Better filtering moves the retention rate. Better use of repetition moves the multiplier. And a smarter allocation of a fixed corpus across domains and training stages changes how much a given token is worth in the first place, which does not show up in the product at all but shows up in the model.

So the field has quietly stopped being about acquisition and become about extraction: how much usable signal can you get out of a token you already have. Every technique below is an answer to some version of that question. Filtering, deduplication and quality classifiers decide *which* tokens are worth spending compute on. Mixing decides in what *proportion*. Repetition decides how many *times*. Curriculum decides *when*. Synthetic data tries to move the ceiling itself. And contamination is the bill that arrives when you get any of it wrong without noticing.

## 1. What the corpus generations actually changed

The open web corpora form a fairly clean sequence in which each generation adds exactly one mechanism. Reading them in order is the fastest way to understand why a modern pipeline has the stages it has.

![Published sizes of open pretraining corpora](blogs/images/pdata-corpora.svg?v=1)

C4 came out of the T5 work in 2019 as roughly 750 GB of English text. The recipe was heuristics pointed at a single April 2019 Common Crawl snapshot: drop lines that do not end in terminal punctuation, drop documents containing words from a blocklist, keep only what a language detector calls English, remove duplicated lines. Every one of those rules is defensible on its own, and together they turned an unusable pile of HTML-stripped garbage into something you could train on. C4 established the template that everything since has refined.

It also established the field's most instructive failure. Two years later a [documentation paper](https://arxiv.org/abs/2104.08758) went back and asked what the filters had actually removed. Applying a dialect classifier to the discarded documents, the authors found African American English removed at 42% and Hispanic-aligned English at 32%, against 6.2% for White American English and 7.2% for everything else. The surviving corpus is 97.8% WAE and 0.07% AAE. The blocklist was a list of words considered obscene, and it does not take much imagination to see how a list like that interacts with a dialect in which some of those words are ordinary vocabulary. Nobody intended this. Nobody noticed for two years, because between building the corpus and auditing it there was no step where anyone measured what came out. A filter is a hypothesis about your data, and an unmeasured hypothesis is just a preference.

[CCNet](https://arxiv.org/abs/1911.00359) introduced the idea that quality could be *scored* rather than matched. Train a KenLM n-gram model on Wikipedia, KenLM being fast enough to train and score at corpus scale, then score each Common Crawl document by its perplexity under that model, and sort the corpus into head, middle and tail buckets. The reasoning is that text a Wikipedia-trained model finds unsurprising is probably text with the surface statistics of decent prose. This is a proxy and everyone knew it was a proxy, but it is the ancestor of everything in section 3: it replaced a rule about what bad text looks like with a model of what good text looks like, and models generalise where rules do not.

The Pile went the other way entirely, assembling 825 GiB from 22 curated subsets: PubMed, arXiv, GitHub, books, StackExchange, a Common Crawl slice, and so on. The bet was on diversity, and the bet paid off for a while. But its lasting contribution turned out to be something the authors probably considered bookkeeping. Because every document carries a source label, The Pile is the only large corpus on which you can ask "what happens if I use more of this and less of that." It became the standard testbed for mixing research, and its default 22-domain weights are the baseline that every method in section 4 reports against.

[RefinedWeb](https://arxiv.org/abs/2306.01116) then argued that all the curation was unnecessary: "properly filtered and deduplicated web data alone can lead to powerful models; even significantly outperforming models from the state-of-the-art trained on The Pile." This was a genuinely contrarian position in 2023, when the received wisdom was that books and papers were where the good stuff lived. Their pipeline is essentially the modern shape, in order: URL filtering against an adult-content blocklist plus a scoring system, text extraction with trafilatura, an open-source tool that strips navigation, ads and footers to leave the article body; language identification with fastText, a lightweight linear classifier over word vectors that is fast enough to run over tens of billions of documents and is therefore the default for both language ID and quality scoring at pipeline scale; the Gopher heuristics plus their own line-wise corrections, which section 3 unpacks; then MinHash fuzzy deduplication followed by exact-substring deduplication, which section 2 unpacks. Out of Common Crawl they obtained 5T tokens and released an extract of 600B. Falcon was trained on it, and the web-only line has been dominant ever since.

Then came the transparency generation, which is where the field started publishing its ablations instead of just its artifacts. RedPajama reproduced the Llama 1 recipe at 1.21T tokens. SlimPajama ran global MinHashLSH over RedPajama at Jaccard threshold 0.8 on 13-grams and removed 49.6% of the bytes, leaving 627B, which is a startling number to sit with: half of a corpus assembled by a competent team from already-filtered sources was duplicate. [Dolma](https://arxiv.org/abs/2402.00159) published 3T tokens curated down from roughly 200 TB of raw text to 11 TB, and released the toolkit and the ablations alongside. [FineWeb](https://arxiv.org/abs/2406.17557) processed 96 Common Crawl dumps spanning summer 2013 to April 2024 into 15T tokens, later extended past 18.5T, with every ablation model published. [DCLM](https://arxiv.org/abs/2406.11794) standardised 240T tokens of Common Crawl into a benchmark with 53 downstream evaluations and controlled model scales from 412M to 7B, turning data curation into something you could compete at.

The pools, at 240T and 18.5T, are what is available; the corpora people actually train on are one to two orders of magnitude smaller. Everything in the next three sections lives in that gap.

## 2. Deduplication, and why MinHash has an S-curve

Duplication in web text is not a rounding error.

[Lee et al.](https://arxiv.org/abs/2107.06499) ran near-duplicate detection over several standard corpora. In C4, 3.04% of training examples have a near-duplicate elsewhere in training; in RealNews the figure is 13.63%. The extreme cases are stranger than the averages. They report removing from C4 "a single 61 word English sentence that is repeated over 60,000 times." That is one sentence, sixty thousand copies, in a corpus that had already been through line-level deduplication.

Two consequences follow, and they are independent of each other. First, evaluation: train-test overlap "affects over 4% of the validation set" of standard datasets, so duplicates quietly inflate the numbers you use to make decisions. Second, memorization: deduplication "allows us to train models that emit memorized text ten times less frequently." A model that has seen a passage sixty thousand times has effectively been instructed to memorise it, and it obliges.

Exact duplicate removal is a hash table. Exact *substring* removal is a suffix array built over the concatenated corpus, which finds every repeated span above a length threshold regardless of where document boundaries fall, and catches the very common situation where two documents share a long block of boilerplate but differ elsewhere. Neither of these touches the case that actually dominates: two documents that differ by a header, a timestamp, a navigation menu, or one rewritten sentence.

For that you need approximate matching at corpus scale, which means you cannot compare documents pairwise, because there are 10¹⁰ of them and pairwise is 10²⁰ comparisons. The standard answer is MinHash with locality-sensitive hashing, whose central parameter is one almost nobody tunes deliberately.

Cut each document into overlapping runs of k consecutive words, called shingles. A document becomes the set of its shingles, and therefore a subset of a very large universe. The similarity you want is Jaccard: the size of the intersection over the size of the union. Now take a random permutation π of the shingle universe and, for each document, record the minimum of π over that document's shingles. The key fact is that for two documents A and B,

$$
\Pr\big[\min \pi(A) = \min \pi(B)\big] \;=\; \frac{|A \cap B|}{|A \cup B|} \;=\; J(A, B)
$$

The element of A ∪ B with the smallest π value is equally likely to be any element of the union, since π is a uniformly random permutation. The two minima coincide exactly when that element happens to lie in the intersection. So the probability is |A ∩ B| / |A ∪ B|, which is the definition of Jaccard.

That gives you an estimator: a signature of m independent permutations produces m Bernoulli(J) samples, so the fraction of agreeing positions estimates Jaccard to within about 1/√m. You have replaced set comparison with integer comparison, and the signatures are tiny compared to the documents.

LSH is what turns the estimator into a retrieval structure so you never look at most pairs at all. Cut the m-row signature into `b` bands of `r` rows each, with m = b·r, and hash each band separately into a table. Two documents become *candidates* if they collide in at least one band. Colliding in a given band requires all `r` rows of that band to agree, which happens with probability `s^r` at true similarity `s`, and the bands are independent, so

$$
P(\text{candidate}) \;=\; 1 - (1 - s^{r})^{b}
$$

This is a sigmoid in `s`, and its inflection sits near `(1/b)^{1/r}`. That expression is the actual knob controlling how aggressive your deduplication is. It is not a similarity threshold you set anywhere in the code. It is a threshold you *induce* by choosing how to factor your hash budget, and it is very easy to inherit it from whatever configuration you copied.

![The LSH S-curve, measured and predicted](blogs/images/pdata-lsh.svg?v=1)

The curve is the formula and the dots are measured. I built document pairs with controlled overlap, computed real MinHash signatures over 5-gram shingles using 180 permutations implemented as affine maps modulo a Mersenne prime, indexed them in 20 bands of 9 rows, and recorded how often each pair actually surfaced as a candidate. Theory and measurement agree to within sampling noise.

The right panel factors the same budget of 180 hashes five different ways. Nothing about the corpus, the shingle size, or the amount of computation changes across those five rows. But 45 bands of 4 rows will pull in pairs down at Jaccard 0.39, treating documents that share barely a third of their content as duplicates, while 5 bands of 36 rows will not look at anything below 0.96 and will miss almost every real near-duplicate. Between those extremes lies every reasonable configuration, and the difference between them is an integer factorisation. If you have ever wondered why two teams running "MinHash deduplication at threshold 0.8" get very different removal rates, this is usually where it lives.

Two further choices matter about as much. Granularity: document-level, paragraph-level and line-level deduplication remove very different things, and SlimPajama's ablations showed that global deduplication across sources and local deduplication within each source are not interchangeable, because the cross-source duplicates are exactly the documents that appear in multiple curated collections and therefore look important. And semantics: [SemDeDup](https://arxiv.org/abs/2303.09540) embeds documents with a pretrained model, clusters the embeddings, and removes near-neighbours within clusters, catching pairs that share no shingles at all because one is a paraphrase or a translation of the other. On a LAION subset it removes "50% of the data with minimal performance loss, effectively halving training time." Its own result on C4 is considerably more sober, beating prior deduplication methods with efficiency gains around 15%, and the authors attribute the smaller margin to C4 already being partially curated. Semantic deduplication has enormous headroom on raw multimodal data and much less on text that someone has already cleaned.

## 3. Quality filtering, from rules to classifiers

Deduplication removes text that is redundant. Quality filtering removes text that was never worth having, which is a much harder thing to define and correspondingly easier to get wrong.

The Gopher pipeline established the rule-based standard, and every modern pipeline reimplements some version of it: filter on document statistics like word count, mean word length, symbol-to-word ratio, and the fraction of lines that are bullets or end in an ellipsis; strip repeated text within a document; deduplicate across documents; and remove documents overlapping the test sets. The specific thresholds differ between reimplementations, and chasing them is mostly a waste of time. What happens when the filters compose, Dolma measured directly.

![What survives a composed pipeline](blogs/images/pdata-funnel.svg?v=1)

Each per-stage rate here is published and the compounding is mine. Gopher's full rule set tagged 15.23% of UTF-8 characters for removal. C4's no-punctuation rule tagged 22.73%. Fuzzy plus exact deduplication removes roughly half. A model-based classifier keeping its top decile removes another 90%. Multiply and about 3% of the extracted text survives.

That number changes what kind of object a pipeline is. Nobody designs a system intending to discard 97% of its input. Each stage was designed and evaluated on its own, each looks conservative on its own, and the composition is brutal. It also means the last stage in the chain, the classifier, is doing most of the work in absolute terms while receiving the least scrutiny, since by the time you reach it the earlier stages have already made the corpus look clean.

C4's NoPunc rule *on its own* outperformed both the full C4 rule set and the full Gopher rule set, on perplexity and on downstream tasks; the best configuration overall was Gopher-All plus C4-NoPunc. More rules is not more quality. The single crudest rule in the collection, throw away lines that do not end in punctuation, carried much of the benefit, presumably because it is an extremely effective detector of navigation menus, tag clouds and product listings, which is what most of the web actually is.

The 2024 consensus moved from rules to learned classifiers, and the two reference implementations chose very different things to learn from.

DCLM trains a fastText classifier with instruction-formatted text as the positive class, specifically OpenHermes 2.5 and r/ExplainLikeImFive (ELI5), against random web pages as negatives. The implicit theory is that text resembling a good explanation is text worth training on. FineWeb-Edu takes the more expensive route: have Llama-3-70B-Instruct score 460,000 pages for educational value, distil those judgements into a small classifier that reaches F1 82% at a threshold of 3, then apply that classifier to all 15T tokens at a cost of 6,000 H100 hours. The implicit theory is that a strong model already knows what good text is and you only need to make its opinion cheap enough to apply at scale.

Both work, and both work well. FineWeb-Edu reports that a 1.71B model trained on 350B tokens moves MMLU from 33% to 37% and ARC from 46% to 57%. DCLM's headline is stronger and states the general principle outright, "model-based filtering is key": DCLM-Baseline gives a 7B model 64% 5-shot MMLU on 2.6T tokens, which is 6.6 points over the previous best open-data model while using 40% less compute, and reaches that with a sixth of the compute of Llama 3 8B.

"Model-based filtering works" is often read as "we now know how to measure data quality," and that is not what happened. Neither team defined quality. Both picked a proxy corpus they believed correlated with it, and the resulting filter inherits every property of that choice. "Text that looks like an ELI5 answer" and "text a 70B model calls educational" are different targets. Neither is obviously the right target for a model that will subsequently be post-trained into a coding agent, and nobody has shown that the ranking between filters is stable across downstream uses.

The [pretrainer's guide](https://arxiv.org/abs/2305.13169), which pretrained 28 models at 1.5B to study these trade-offs systematically, makes one instance of the problem explicit: filtering toxicity out of pretraining reduces the model's ability to *detect* toxicity later. Quality and capability point in opposite directions along that axis. There is no particular reason to believe it is the only axis where they do, and very little work has looked.

## 4. Mixing, as an optimization problem

Once your corpus has domains, you have proportions, and proportions are the one part of data work that looks like a well-posed optimization problem: a low-dimensional continuous parameter, a differentiable-ish objective, and an obvious evaluation. Predictably, this is where the most technically interesting work has gone.

The difficulty is that you cannot evaluate a mixture without training a model on it, a frontier training run costs millions of dollars, and the number of mixtures is uncountable. Every method below is really a method for making the evaluation cheap enough to search over.

DoReMi attacks it as a minimax problem. Train a small proxy model with [group distributionally robust optimization](https://arxiv.org/abs/2305.10429) over domains: the model minimises worst-case excess loss relative to a fixed reference model, and the domain weights are what the inner maximisation produces. Then take those weights and use them to train a model 30× larger. From a 280M proxy to an 8B target, this "improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps," which in absolute terms is 75k steps against 200k. The proxy costs about 8% of the target run's FLOPs.

Perplexity improves on all 22 Pile domains, *including the domains the method downweighted*. And the reweighting is not gentle. Pile-CC goes from 0.1121 to 0.6057. ArXiv goes from 0.1052 to 0.0036. PubMed Central goes from 0.1071 to 0.0046.

The method cut arXiv to a thirtieth of its original weight, and arXiv perplexity *improved*. The natural reading is that what a specialised domain most needs is not more of itself but a model that has learned language properly, and the cheapest route to that is web text. This is deeply counterintuitive if you think of pretraining as teaching subjects, and unsurprising if you think of it as learning a distribution of which the subjects are conditional slices.

[DoGE](https://arxiv.org/abs/2310.15393) swaps the robustness objective for a first-order one that connects mixing to influence functions. Score each domain by the inner product of its gradient with the summed gradient across all domains: a domain whose gradient points in the same direction as the aggregate is a domain whose data helps everything, and it gets weighted up; weights update by mirror descent. The score decomposes into cross-domain alignment plus the domain's own gradient norm, so "hard" and "useful" are entangled in it, which is a real limitation rather than a technicality. With an 82M proxy for a 684M target it reports 1.7 accuracy points over a uniform mixture on 5-shot reasoning.

[RegMix](https://arxiv.org/abs/2407.01492) declines to model the mechanism at all, and gets further for it. Train 512 models of 1M parameters on 1B tokens each, with mixtures sampled from a Dirichlet distribution deliberately tilted toward extremes. A Dirichlet samples directly over non-negative proportions summing to one, the set of which is the simplex; lowering its concentration parameter pushes the samples toward the corners, where one domain dominates, rather than clustering near the uniform mixture. That is what makes the regression's training set cover the space. Fit a regressor from mixture to validation loss. Optimize the regressor. The whole thing costs about 2% of the FLOPs of a single 1B run.

Two details are more useful than the headline. First, LightGBM beats linear regression badly, Spearman 97.1 against 88.0 when predicting the ranking of 1B models, which tells you the loss surface over the simplex is meaningfully non-linear and that anyone fitting a linear model to mixture effects is leaving accuracy on the table. Second, the predicted-best mixture ranked first among 64 candidate 1B models trained on 25B tokens each, so the small-scale ranking transferred. The paper states this as "data mixture effects transcend scaling laws," meaning the *ordering* of mixtures found at tiny scale survives to large scale even though the absolute losses do not. That is the property that makes proxy-model search viable at all, and it is empirical rather than derived.

RegMix also independently reproduced DoReMi's most surprising finding from a completely different direction: web corpora, not the sources everyone calls high-quality, correlate most strongly with downstream performance. Two methods with nothing in common methodologically both concluded that you should use much more Common Crawl than a human would choose.

[Data Mixing Laws](https://arxiv.org/abs/2403.16952) puts a functional form on the surface itself, which lets you extrapolate rather than only interpolate. With `r_j` the proportion of training domain `j` and `L_i` the validation loss on domain `i`,

$$
L_i(r_1, \ldots, r_M) \;=\; c_i + k_i \exp\Big(\sum_{j=1}^{M} t_{ij}\, r_j\Big)
$$

Here `c_i` is the loss no mixture can remove, `k_i > 0` is a scale, and `t_ij` captures the interaction between training domain `j` and validation domain `i`, with negative `t_ij` meaning `j` helps `i`. The authors compared four candidate forms and selected this one because it fits as well as the most flexible alternative with fewer coefficients. The structural choice that makes it practical is that the exponential sits *outside* the sum: one exponential of a linear combination, so the parameter count grows linearly in the number of domains rather than quadratically. The overall objective is `Σ_i s_i L_i` with `s_i` the validation mixture.

The extrapolation machinery is what they call nested scaling laws. Fit the mixing law at small model size and few steps; fit separate laws for loss against training steps and loss against model size; compose the three to predict what a large model trained for a long time on an unseen mixture will do. The payoff is a mixture for a 1B model on 100B RedPajama tokens "reaching a performance comparable to the one trained for 48% more steps on the default mixture," or equivalently reaching the default mixture's final performance in 73% of the steps.

Since the form is explicit, we can just evaluate it. Below is a three-domain instance with plausible interaction signs, evaluated on the entire simplex, with one domain weighted more heavily in the validation mixture.

![The loss surface over a three-domain simplex](blogs/images/pdata-mixing.svg?v=1)

Two features of this picture generalise. The optimum is interior and it is not the uniform mixture, which is the entire reason automated mixing exists rather than being a solved problem with an obvious answer. And the surface is shallow near the optimum and steep near the edges: the gap between uniform and optimal here is 0.0129 nats, a nat being the unit of a cross-entropy loss measured in natural logarithms, about 1.44 bits, which under the C4 scaling law is worth about 21% more tokens at a 100B-token budget, while walking to a corner of the simplex costs many times that.

Mixing is worth doing, because 21% of a token budget is enormous when tokens are the constraint. And mixing is not worth agonising over past the first significant figure, because the surface near the optimum is flat enough that the difference between a good mixture and the best mixture is smaller than the noise in most evaluations.

[CLIMB](https://arxiv.org/abs/2504.13161) addresses the assumption underneath everything above, which is that you have domain labels at all. The Pile has them because it was assembled from labelled sources. Common Crawl does not. CLIMB embeds documents, clusters them in embedding space, and runs a proxy-plus-predictor search over cluster weights instead of source weights, iteratively bootstrapping toward better mixtures. Continued training of a 1B model on 400B tokens with the discovered mixture beats Llama-3.2-1B by 2.0 points, and they release ClimbLab, a 1.2T-token corpus organised into 20 clusters, as a research substrate.

Clusters are not domains. Nothing in the mixing formalism requires the partition to be human-meaningful, and once you stop requiring it, the number of possible partitions becomes another thing to optimize over.

## 5. Repetition: how many times can you use the same token

Everything so far assumes each token is seen once. The moment the pool is fixed, that stops being an assumption and becomes a choice, and it is the choice with the cleanest available theory.

[Muennighoff et al.](https://arxiv.org/abs/2305.16264) ran more than 400 training runs from 10M to 9B parameters, up to 900B tokens and up to 1500 epochs, specifically to characterise what repeated data is worth. Their answer is a two-parameter model that is easy to state and easy to use.

Let `U_D` be the number of unique tokens available and `R_D = D/U_D − 1` the number of repetitions, so that `R_D = 0` means a single epoch. Define the *effective* data `D'` as the amount of fresh data that would have produced the same loss. Then

$$
D' \;=\; U_D + U_D\, R_D^{*}\left(1 - e^{-R_D / R_D^{*}}\right)
$$

with a symmetric expression for effective parameters `N'` under excess parameters `R_N`. Fitting against their runs gives `R_D* = 15.39` and `R_N* = 5.31`, quantities the authors describe as the half-life of repeated data and of excess parameters respectively.

As `R_D → 0` the exponential expands to `R_D`, so `D' ≈ U_D(1 + R_D) = D`: the first repetitions are worth their face value, and repetition is free. As `R_D → ∞` the exponential vanishes and `D' → U_D(1 + R_D*)`: there is a hard ceiling, and no amount of repetition can push a corpus past `1 + R_D* = 16.4` times its own size. Everything about repetition follows from where you sit on that curve.

![What repeated epochs are worth](blogs/images/pdata-repetition.svg?v=1)

Four epochs return 3.73 unique tokens' worth of value, or 93% of what four fresh epochs would have given. That is the quantitative content of the paper's much-quoted claim that "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data." Sixteen epochs return 10.58, or 66%; you are now paying a third of your compute for nothing, which may still be the right call if fresh tokens are simply unavailable. By epoch 44 you have collected 15.45 of the 16.4 available, and the marginal epoch is worth 0.063 of a fresh token, about a sixteenth of what the first epoch was worth.

Two caveats belong with those numbers. The paper observes that models trained for 44 epochs actually diverge, so the far tail of the curve describes a regime you would not operate in even if the arithmetic said to. And the widely repeated claim that returns hit zero around 40 epochs does not appear in the paper at all. What the paper says is that meaningful gains extend to roughly 16 epochs, "beyond which returns diminish extremely fast." The 40 is a secondary-source paraphrase that has propagated widely; the formula above is the primary object, and it is both more precise and more useful.

## 6. Where the compute should go when data is fixed

The repetition curve stops being a curiosity and becomes a decision the moment you put it back inside a compute budget, because it changes the answer to the most basic question in scaling.

[Chinchilla](https://arxiv.org/abs/2203.15556) fitted the loss surface as `L(N, D) = E + A/N^α + B/D^β` with `E = 1.69`, `A = 406.4`, `B = 410.7`, `α = 0.34` and `β = 0.28`, and concluded that under a fixed compute budget parameters and tokens should be scaled in equal proportion, which works out to roughly 20 tokens per parameter. That advice assumes unique data is available in whatever quantity you want. Under a fixed unique-token budget it stops applying, because "more tokens" is no longer something you can buy at constant quality.

Muennighoff et al. handle this by substituting the effective quantities into a Chinchilla-style form refit on C4:

$$
L(N', D') \;=\; 1.87 + \frac{521}{N'^{\,0.353}} + \frac{1488}{D'^{\,0.353}}
$$

Now the optimization has real structure. Spending compute on parameters runs into the `R_N` penalty once the model is much larger than the data justifies; spending it on passes over the data runs into the `R_D` penalty once you have repeated a lot. So fix `U_D` at 1T unique tokens, sweep the compute budget `C = 6ND`, and for each budget search over model size, letting the number of epochs fall out as `D / U_D`.

![Optimal epochs against compute at a fixed unique-token budget](blogs/images/pdata-allocation.svg?v=1)

Below 10²³·⁵ FLOPs the optimum never repeats anything. Every additional FLOP goes into parameters and the epoch count stays pinned at one, which is exactly the Chinchilla regime: data is not the binding constraint, so behave as though it were infinite. Past that threshold the model has been pushed far enough beyond the size that 1T tokens justifies that the `R_N` penalty exceeds the `R_D` penalty, and the optimum starts buying epochs instead. At 10²⁵ FLOPs it wants a 229B-parameter model making 7.3 passes over the data.

The threshold moves with the data you have, and the scaling is clean: 100B unique tokens starts repeating at 10²¹·⁵, 1T at 10²³·⁵, 10T at 10²⁵·⁵, each decade of data buying two decades of compute before repetition becomes correct.

Extrapolate to the entire effective stock of public human text, 4×10¹⁴ tokens, and repetition starts paying at about 10²⁷ FLOPs. Villalobos et al. put the compute at which models exhaust that stock at around 5×10²⁸ FLOPs. Those two numbers are within about two orders of magnitude of each other, which is the actual content of the phrase "data wall." It is not a cliff where training stops working. It is the point past which additional compute buys epochs and parameters rather than new information, and the returns to compute change character accordingly.

## 7. Curriculum: not all tokens should arrive at the same time

Mixing as posed in section 4 asks for a single set of proportions to be used for an entire training run. Two independent industrial results say that this is leaving a great deal of value on the table, and they arrived at essentially the same technique from different directions.

Llama 3 anneals the learning rate linearly to zero over the final 40M tokens while upsampling very high quality sources, and averages checkpoints from the annealing phase. The reported effects are large: annealing improved an 8B model "on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively." Two caveats travel with that number and both matter. The ablation annealed on those benchmarks' *training* sets, so it measures how much a targeted anneal can move a targeted benchmark rather than general capability. And the improvement on the 405B model was negligible, which suggests the mechanism is a small model needing the demonstration that a large model has already generalised to.

The same paper also turns annealing into a measurement instrument. To evaluate a candidate dataset, take a checkpoint at 50% of training, anneal the learning rate to zero over 40B tokens with 30% weight on the candidate and 70% on the default mix, and read off the change in benchmark scores. That costs one short run per candidate, against a full scaling-law sweep per candidate under the older methodology. It converts data evaluation from a research project into something you can run on every dataset someone proposes.

[OLMo 2](https://arxiv.org/abs/2501.00656) formalises the same instinct into a named training stage and reports it in more detail than anyone else has. Stage 1 is about 3.9T tokens, over 95% of it web data. Stage 2, which they call midtraining, draws from a high-quality pool of roughly 843B tokens assembled from filtered web, academic text, question-answer data, instruction data and synthetic mathematics, sampled down to 50B, 100B or 300B tokens, with DCLM-filtered web still supplying about half the budget so the model is not yanked entirely off-distribution. For the 7B they run three separate 50B anneals differing only in data ordering and average the resulting weights.

![Midtraining, before and after](blogs/images/pdata-midtrain.svg?v=1)

These are published numbers rather than measurements of mine. GSM8K goes from 24.1 to 67.5 for the 7B and from 37.3 to 75.1 for the 13B. DROP goes from 40.7 to 60.8. The ten-benchmark average moves 10.6 points for the 7B and 10.3 for the 13B.

Stage 2 is between 1% and 8% of the token budget depending on which sample you use, and it moves the average by ten points, which is more than most architectural changes have ever delivered. Whatever is happening in midtraining, it is not "more training." It is closer to the model finally being shown what the tokens were for.

They also name the per-dataset evaluation protocol, *microannealing*, which is the Llama 3 instrument run once per candidate dataset. Two labs converged independently on the technique, and independently on using it as a measurement device.

Which makes section 4's single global mixture the wrong object to be optimizing. What these results describe is a *schedule*, a mixture that varies over training, and none of the mixing methods in section 4 optimize over schedules. They optimize over a point. Data Mixing Laws gets closest, since it can predict the mixture that avoids catastrophic forgetting during continued training, but the general problem of optimizing a trajectory through the simplex is essentially open.

## 8. Synthetic data, and the shape of its failure mode

If the stock of human text is fixed, the obvious move is to manufacture more. The evidence here is genuinely two-sided, and the two sides are usually deployed against each other when they are actually describing different situations.

On the productive side, [phi-1](https://arxiv.org/abs/2306.11644) trained a 1.3B model on 6B tokens of filtered code plus about 1B tokens of GPT-3.5-generated textbooks and exercises, on 8 A100s for 4 days, reaching 50.6% pass@1 on HumanEval and 55.5% on MBPP; a 350M version still reached 45%. The composition is routinely misdescribed, so it is worth being precise: the 6B is filtered *code* from The Stack and StackOverflow, not general web text, and the synthetic portion is roughly a seventh of the tokens. The claim being made is not that synthetic data replaces real data. It is that a small amount of well-targeted synthetic data on top of aggressively filtered real data beats a much larger amount of unfiltered data.

[WRAP](https://arxiv.org/abs/2401.16380) makes the claim that matters most for a data-constrained world. Instead of generating new content, it uses an instruction-tuned model to rephrase existing web documents into different styles, and reports roughly 3× faster pretraining on C4, better than 50% perplexity improvement across Pile subsets at matched compute, and over 2% zero-shot gains across 13 tasks. Rephrasing spends compute to buy something that behaves like unique data, which is precisely the currency section 5 identified as scarce. If repetition caps out at 16.4× your corpus, and rephrasing produces something that is not quite fresh but is not quite a repeat either, then the interesting question is where rephrased data sits on that curve, and nobody has measured it.

On the failure side is the Nature result on [model collapse](https://www.nature.com/articles/s41586-024-07566-y): "indiscriminate use of model-generated content in training causes irreversible defects in the resulting models, in which tails of the original content distribution disappear." The result is usually invoked as though it were a mysterious emergent pathology. It is not, and the mechanism is elementary.

Fit a Gaussian by maximum likelihood to `n` samples. Draw `n` fresh samples from the fitted distribution. Refit. Repeat. Each individual step is unbiased in the mean and every step is done correctly. But the maximum-likelihood variance estimator is biased low by exactly `(n−1)/n`, since it divides by `n` rather than `n−1`, so across generations

$$
\mathbb{E}[\sigma_g^2] \;=\; \left(\frac{n-1}{n}\right)^{g}\sigma_0^2
$$

and more strongly, `σ_g²` is a non-negative martingale, so it converges almost surely. The only value it can converge to is zero. The distribution contracts to a point, and it does so no matter how carefully any individual generation is executed.

![Variance under recursive refitting](blogs/images/pdata-collapse.svg?v=1)

With `n = 100` and 300 independent chains, the measured mean variance tracks that geometric prediction exactly, which it should, since the prediction is not an approximation. The practically important line is the green one: the probability mass the generation-`g` model assigns outside the *original* distribution's ±2σ falls from 4.6% at generation 0 to 0.08% by generation 100, and is numerically zero by generation 200. The tails go first, and the tails are where the rare knowledge lives.

Which is why the two sides of the synthetic-data argument are not actually in conflict. Collapse is a statement about a *closed loop*, where each generation's only input is the previous generation's output. phi-1 and WRAP are statements about a single generation of synthesis anchored to real data. The condition separating them is whether fresh real data keeps entering the loop, and unlike most disagreements about synthetic data, that condition is checkable rather than philosophical. The uncomfortable part is that as more of the web becomes model-generated, the loop closes whether or not anyone decides to close it, and nobody currently has a reliable way to measure how far along that process is.

## 9. Contamination, and why n-gram decontamination does not work

Every claim in this post is a benchmark number. The last question is whether benchmark numbers mean anything.

The standard industrial defence is n-gram decontamination: remove from training any document sharing an n-gram with a test item, with `n` usually between 8 and 13. It is cheap, it is easy to verify, and it catches literal copies of benchmarks that ended up in a crawl. Every major lab does some version of it.

[Yang et al.](https://arxiv.org/abs/2311.04850) showed that it is trivially evaded, and the striking part is that evasion requires no adversary. Rephrase a test item, or translate it into another language and back, and the n-gram overlap goes to zero while the item remains, for every purpose that matters, the same item. Their headline result is that "a 13B model can easily overfit a test benchmark and achieve drastically high performance, on par with GPT-4." On rephrased MMLU, Llama-2 7B goes from 45.3 to 88.5 and the 13B from 54.8 to 89.9, with standard n-gram decontamination detecting nothing.

![n-gram overlap between a benchmark item and its paraphrase](blogs/images/pdata-contamination.svg?v=1)

Those are the measured overlaps between an arithmetic word problem and a paraphrase of it that any competent model would answer identically. Unigram overlap is 38%, bigram 15%, trigram 3%, and from 4-grams up it is exactly zero. A decontamination filter at n = 13 does not merely miss this pair, it cannot see it at all; the same holds at n = 8 and at n = 5. And the n that *would* catch paraphrases, somewhere around 2 or 3, would delete most of the corpus, since common trigrams appear everywhere. There is no setting of the parameter that works.

This is not hypothetical contamination. The same work measured 8.5% of HumanEval overlapping RedPajama-Data-1T, 15.9% overlapping StarCoder-Data, and 18.9% overlapping The Stack. Those are the overlaps that n-gram matching *can* see, in corpora that many open models were trained on, for a benchmark that is quoted in essentially every code-model release.

The detection literature is candid about how far it gets. [Min-K% Prob](https://arxiv.org/abs/2310.16789) asks whether a given text was in a model's pretraining data without access to the corpus and without a reference model, which is the situation you are actually in when evaluating someone else's model. It takes the k% lowest-probability tokens in the text and averages their log-likelihoods, on the theory that text the model has never seen will contain outlier tokens to which it assigns very low probability, while memorised text will not. Evaluated on WIKIMIA, a benchmark that uses Wikipedia edit timestamps to establish which texts were certainly in the training data and which certainly were not, it reaches 0.72 average AUC against 0.67 for the best prior method, an improvement the paper reports as 7.4% and which is relative, not 7.4 AUC points.

An AUC of 0.72 is a real signal and a poor test. It is enough to support a statistical claim about a large sample of documents and nowhere near enough to adjudicate whether a specific benchmark was in a specific corpus. So the position we are actually in is this: we cannot verify the cleanliness of an evaluation run on a corpus we did not build, most published model comparisons involve at least one model whose data is undisclosed, and the contamination status is therefore unknown in both directions rather than known to be fine.

## 10. What the arithmetic implies

The effective stock of public human text is around 4×10¹⁴ tokens, and dataset sizes reach it around 2028. Composed filtering keeps a few percent of extracted text, with the classifier stage doing most of the removing and defined by an essentially arbitrary choice of positive examples. Deduplication's aggressiveness is set by an integer factorisation that most teams inherit rather than choose. Mixing is worth roughly 20% of a token budget and not much more, but the surface is flat enough near the optimum that a decent mixture captures most of it. Repetition is worth at most 16.4× your unique corpus, and becomes the right allocation at 10²³·⁵ FLOPs for a 1T-token pool and 10²⁷ for the entire web. A tenth of the budget spent last is worth ten points of benchmark average. Synthesis works exactly as long as real data keeps entering the loop, and fails in a way that is provable rather than empirical when it does not. And none of these measurements can currently be validated against a benchmark anyone knows to be clean.

Four of the five biggest levers in that list, the classifier's positive examples, the deduplication granularity, the mixture and the midtraining schedule, are choices someone made once, early, often by copying a previous project, and rarely revisited. The one that gets nearly all the attention, corpus size, is the one running into a ceiling.

There is a version of the next few years in which data work stops looking like acquisition and starts looking like measurement. Not what else can we get, but what is the stuff we already have actually worth, which parts of it are doing the work, and how would we know if we were wrong. How many entries on that list are numbers nobody has measured for their own corpus, and how few of them would be expensive to measure.
