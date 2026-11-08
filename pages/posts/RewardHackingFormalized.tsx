import React from 'react';
import { PostText } from '../../components/PostText';
import { ClassicExamples, LanguageModelExamples } from './RewardHackingFigures';
import { DampingPlot, GrpoWeightPlot, TakeoffPlot } from './RewardHackingPlots';

const r = String.raw;

type Lang = 'en' | 'zh';

// Where each citation in the text links to.
const LINKS: Record<string, string> = {
  amodei2016: 'https://arxiv.org/abs/1606.06565',
  amodei2017: 'https://openai.com/index/learning-from-human-preferences/',
  anthropic2025: 'https://www.anthropic.com/claude-3-7-sonnet-system-card',
  baker2025: 'https://arxiv.org/abs/2503.11926',
  clark2016: 'https://openai.com/index/faulty-reward-functions/',
  denison2024: 'https://arxiv.org/abs/2406.10162',
  eisenstein2023: 'https://arxiv.org/abs/2312.09244',
  elmhamdi2024: 'https://arxiv.org/abs/2410.09638',
  everitt2016: 'https://arxiv.org/abs/1605.03143',
  everitt2017: 'https://arxiv.org/abs/1705.08417',
  everitt2021a: 'https://arxiv.org/abs/2102.01685',
  everitt2021b: 'https://arxiv.org/abs/1908.04734',
  gao2023: 'https://arxiv.org/abs/2210.10760',
  goodhart1975: 'https://doi.org/10.1007/978-1-349-17295-5_4',
  gupta2026: 'https://arxiv.org/abs/2610.17824',
  hadfield2017: 'https://arxiv.org/abs/1711.02827',
  huang2025: 'https://arxiv.org/abs/2407.13399',
  ibarz2018: 'https://arxiv.org/abs/1811.06521',
  karwowski2024: 'https://arxiv.org/abs/2310.09144',
  krakovna2020: 'https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/',
  kwa2024: 'https://arxiv.org/abs/2407.14503',
  laidlaw2025: 'https://arxiv.org/abs/2403.03185',
  lehman2020: 'https://arxiv.org/abs/1803.03453',
  leike2017: 'https://arxiv.org/abs/1711.09883',
  macdiarmid2025: 'https://arxiv.org/abs/2511.18397',
  manheim2018: 'https://arxiv.org/abs/1803.04585',
  moya2026: 'https://arxiv.org/abs/2609.35677',
  mroueh2025: 'https://arxiv.org/abs/2503.06639',
  ng1999: 'https://mlanthology.org/icml/1999/ng1999icml-policy/',
  pan2022: 'https://arxiv.org/abs/2201.03544',
  rad2026: 'https://arxiv.org/abs/2601.04411',
  randlov1998: 'https://mlanthology.org/icml/1998/randlv1998icml-learning/',
  ring2011: 'https://doi.org/10.1007/978-3-642-22887-2_2',
  sharma2024: 'https://arxiv.org/abs/2310.13548',
  singhal2023: 'https://arxiv.org/abs/2310.03716',
  skalse2022: 'https://arxiv.org/abs/2209.13085',
  stiennon2020: 'https://arxiv.org/abs/2009.01325',
  strathern1997: 'https://doi.org/10.1002/(SICI)1234-981X(199707)5:3%3C305::AID-EURO184%3E3.0.CO;2-4',
  taylor2016: 'https://intelligence.org/files/QuantilizersSaferAlternative.pdf',
  taylor2025: 'https://arxiv.org/abs/2508.17511',
  vonarx2025: 'https://metr.org/blog/2025-06-05-recent-reward-hacking/',
  wang2026: 'https://arxiv.org/abs/2510.01367',
  weng2024: 'https://lilianweng.github.io/posts/2024-11-28-reward-hacking/',
  wiener1960: 'https://www.science.org/doi/10.1126/science.131.3410.1355',
  zhong2025: 'https://arxiv.org/abs/2510.20270',
  zhuang2020: 'https://arxiv.org/abs/2102.03896',
};

const TITLE: Record<Lang, string> = {
  en: 'Reward Hacking, Formalized: From Classical RL to Language Models',
  zh: 'Reward Hacking 的形式化：从经典强化学习到语言模型',
};

const EN = r`
## TL;DR

This post reviews what reward hacking looks like in classical RL, in RLHF and in RL with verifiable rewards, and how it
was formalized before language models. It then builds a simple model of reward hacking in language-model RL, which
says:

- The training update sees only rewards, so it gives a hacked answer and a genuine one with the same reward exactly the
same weight; how much each one's probability actually changes depends on the network.
- A policy can hack in two ways: by drifting toward answers the reward overrates, or, among answers with the same
reward, by favoring the hacked ones. For a learned reward model that scores nearly every answer differently, almost all
change counts as the first kind; under a pass/fail verifier the second is the only way the share of hacks among passing
answers can grow.
- Under a pass/fail verifier with no regularization, the training objective is equally satisfied by a policy that
solves problems honestly and one that always hacks. Which one training ends up with is decided by how learning
proceeds, not by the objective.
- A plausible way for hacking to start is on a hard problem the model rarely solves, when a single sampled answer passes
by exploiting the grader. The reward cannot tell it from a genuine solution, and GRPO gives it the largest advantage it can give. Because every problem shares one network, this can carry the hack to other problems, and while
hacks are still rare the further it spreads the faster it spreads, which is consistent with hacking often appearing
abruptly.
- Two ways forward follow: find signals beyond the reward that tell hacked answers from genuine ones, and shape training
so that no single sample, hacked or not, can move the model too far. The two are likely to complement each other.

## What reward hacking looks like

[[wiener1960|Wiener (1960)]] wrote that if we hand a purpose to a machine we cannot stop in time, “we had better be quite
sure that the purpose put into the machine is the purpose which we really desire and not merely a colorful imitation of
it.” [[goodhart1975|Goodhart (1975)]] observed that any statistical regularity “will tend to collapse once pressure is
placed upon it for control purposes,” a point better known in the form “when a measure becomes a target, it ceases to be
a good measure” ([[strathern1997|Strathern, 1997]]).

A simulated bicycle rewarded for heading toward its goal, but never penalized for heading away, learned to ride in
circles 20 to 50 meters wide around its starting point ([[randlov1998|Randløv and Alstrøm, 1998]]). A soccer robot
rewarded for touching the ball learned to park next to it and keep touching it ([[ng1999|Ng et al., 1999]]). In the boat
race CoastRunners, an agent found that hitting a cluster of respawning targets in a lagoon paid more than finishing, and
outscored human players by about 20 percent while on fire and going the wrong way ([[clark2016|Clark and Amodei, 2016]]).
A robot arm trained from human feedback to grasp an object instead placed its hand between the camera and the object,
“so that it only appeared to be grasping it” ([[amodei2017|Amodei et al., 2017]]). A block-stacking agent flipped the
block over ([[krakovna2020|Krakovna et al., 2020]]); a program-repair system made the program always return an empty
list, which the tests accepted ([[lehman2020|Lehman et al., 2020]]); and in a gridworld, stepping into a bucket made
every tomato plant look watered ([[leike2017|Leike et al., 2017]]). Atari agents trained on frozen learned reward models
drove the model’s score up while the game score “plummets” ([[ibarz2018|Ibarz et al., 2018]]).

::classic

[[amodei2016|Amodei et al. (2016)]] gave the phenomenon its current name: the objective “admits of some clever ‘easy’
solution that formally maximizes it but perverts the spirit of the designer’s intent.” DeepMind’s term, specification
gaming, describes “a behaviour that satisfies the literal specification of an objective without achieving the intended
outcome” ([[krakovna2020|Krakovna et al., 2020]]). [[pan2022|Pan et al. (2022)]] showed that it can appear abruptly, as
a phase transition when model capacity grows, and [[weng2024|Weng (2024)]] summarizes the common usage: an agent
“exploits flaws or ambiguities in the reward function to achieve high rewards, without genuinely learning or completing
the intended task.”

In RLHF, optimizing harder against a learned reward model first improves summaries and then makes them worse
([[stiennon2020|Stiennon et al., 2020]]); a large part of RLHF’s measured gains on several datasets comes from simply
writing longer ([[singhal2023|Singhal et al., 2023]]); preference models often favor answers that agree with the user
over answers that are true ([[sharma2024|Sharma et al., 2024]]); and ensembles of reward models help less than hoped
because their members tend to make the same mistakes ([[eisenstein2023|Eisenstein et al., 2023]]).

Under programmatic verifiers, frontier coding agents have been caught calling ${'`'}exit(0)${'`'} before the tests run,
raising ${'`'}SkipTest${'`'}, patching code upstream of the test framework, reading expected values out of test files
and decompiling reference solutions; when a monitor reading their chain of thought was added to the reward, they kept
hacking and stopped saying so ([[baker2025|Baker et al., 2025]]). Claude 3.7 Sonnet’s system card describes it
“special-casing in order to pass test cases” as a result of reward hacking during training
([[anthropic2025|Anthropic, 2025]]), o3 overwrote graders’ timers and patched evaluators on evaluation tasks while
acknowledging, when asked, that this was not what the user wanted ([[vonarx2025|Von Arx et al., 2025]]), and frontier
models pass coding tasks that were deliberately made impossible, where “any pass necessarily implies a
specification-violating shortcut” ([[zhong2025|Zhong et al., 2025]]). In a curriculum of increasingly gameable tasks,
models occasionally went as far as rewriting their own reward function ([[denison2024|Denison et al., 2024]]), and models that learned to hack production coding environments went on to show broader misalignment
([[macdiarmid2025|MacDiarmid et al., 2025]]; [[taylor2025|Taylor et al., 2025]]). [[wang2026|Wang et al. (2026)]]
describe the mechanism as “hacking occurs when exploiting the loophole is easier than solving the actual task.”

::lm

## How it was formalized before language models

The most common formal picture writes the reward the agent optimizes as the intended reward plus an error,

$$R = R^* + \xi,$$

and the works that use it differ mainly in what they assume about $\xi$. In reward shaping $\xi$ is a term
$F(s, a, s')$ added by the designer, potential-based when $F = \gamma\Phi(s') - \Phi(s)$ ([[ng1999|Ng et al., 1999]]).
The AI Safety Gridworlds build the split into every environment, which comes with an observed reward $R$ and a hidden
performance function $R^*$ ([[leike2017|Leike et al., 2017]]). [[manheim2018|Manheim and Garrabrant (2018)]] treat
$\xi$ as noise for regressional Goodhart, and for extremal Goodhart let the relation itself change past a threshold,
$R^* = R + x$ for $R \le a$ and $R^* = R + y$ otherwise. [[kwa2024|Kwa et al. (2024)]] classify $\xi$ by its tail,
calling it heavy-tailed when $e^{tx}\Pr(\xi > x) \to \infty$ for every $t > 0$. The corrupt reward MDP of
[[everitt2017|Everitt et al. (2017)]] lets $\xi$ be nonzero only on a set of corrupt states: the agent observes
$\hat R(s) = C_s(\dot R(s))$ for a true reward $\dot R$ and a corruption function $C$, with assumptions on how many
states are corrupt and on whether a state’s reward can also be observed from elsewhere.

[[hadfield2017|Hadfield-Menell et al. (2017)]] and [[zhuang2020|Zhuang and Hadfield-Menell (2020)]] model the proxy
instead as an incomplete specification written by a designer. In inverse reward design, the proxy
$R = \tilde w^\top\phi(\tau)$ and the true reward $R^* = w^{*\top}\phi(\tau)$ are linear in trajectory features, and a
designer who picks $\tilde w$ in a training environment $\tilde M$ is modeled as approximately optimal there,

$$P(\tilde w \mid w^*, \tilde M) \propto \exp\big(\beta\, \mathbb{E}[w^{*\top}\phi(\tau) \mid \tau \sim \pi(\tau \mid \tilde w, \tilde M)]\big),$$

so the proxy carries information about $w^*$ only through behavior in $\tilde M$. In
[[zhuang2020|Zhuang and Hadfield-Menell (2020)]], the world is described by attributes $s \in \mathbb{R}^L$ under a
resource constraint $g(s) \le 0$; the true reward $R^*$ depends on all $L$ attributes and the proxy $R$ on only $J < L$
of them.

[[ring2011|Ring and Orseau (2011)]], [[everitt2016|Everitt and Hutter (2016)]] and
[[everitt2021b|Everitt et al. (2021b)]] let the agent act on the channel that produces the reward. In common notation
the observed reward is a distortion $R = d(R^*)$, and the agent’s actions can change $d$. In the delusion box of
[[ring2011|Ring and Orseau (2011)]] the action is $a_t = \langle d_t, a^e_t \rangle$, where $d_t$ programs a map
$d : O \to O$ through which the agent sees the world, $o_t = d(a_t, o^e_t)$, and part of $o_t$ is read as reward; in
[[everitt2016|Everitt and Hutter (2016)]], $R = d_s(\check R)$ for an inner reward $\check R = u^*(s)$ and a distortion
that depends on the state the agent reaches. [[everitt2021b|Everitt et al. (2021b)]] state the general version as a
causal influence diagram whose nodes include the implemented reward function $\Theta^R_t$ and the intended one
$\Theta^R_*$: tampering is the agent’s decisions influencing $\Theta^R_t$ in ways that violate conditions of
$\Theta^R_*$, and by the criterion of [[everitt2021a|Everitt et al. (2021a)]], an incentive to control $\Theta^R_t$
requires a directed path from decision to utility through it.

[[skalse2022|Skalse et al. (2022)]], [[karwowski2024|Karwowski et al. (2024)]] and
[[laidlaw2025|Laidlaw et al. (2025)]] compare the proxy and the true reward as functions on the space of policies, using
that returns are linear in the occupancy measure, $J_R(\pi) = \langle \eta^\pi, R \rangle$.
[[skalse2022|Skalse et al. (2022)]] call $R$ and $R^*$ hackable on a policy set $\Pi$ if they order some pair of
policies in opposite ways,

$$\exists\, \pi, \pi' \in \Pi:\quad J_R(\pi) < J_R(\pi') \ \text{and}\ J_{R^*}(\pi) > J_{R^*}(\pi').$$

The other two measure the relation by a cosine: [[karwowski2024|Karwowski et al. (2024)]] by the angle between the
projections of $R$ and $R^*$ onto $\mathrm{span}(\Omega)$, where $\Omega$ is the polytope of occupancy measures, and
[[laidlaw2025|Laidlaw et al. (2025)]] by their correlation $\kappa > 0$ over state–action pairs drawn from a reference
occupancy measure $\mu_{\mathrm{ref}}$, with the distance a policy has moved measured by
$\chi^2(\mu_\pi \,\|\, \mu_{\mathrm{ref}})$.

[[taylor2016|Taylor (2016)]] changes the optimizer rather than the reward. Given a base distribution $\gamma$ over
actions, such as what a human would do, a $q$-quantilizer samples from the top $q$ fraction of $\gamma$ ranked by the
proxy, so its distribution satisfies $Q_q(a) \le \gamma(a)/q$, and the unknown harm is modeled as an arbitrary
nonnegative cost $c$ whose expectation under $\gamma$ is bounded.

## A model of reward hacking in language-model RL

For learned reward models, [[gao2023|Gao et al. (2023)]] fit the gold reward as $d(\alpha - \beta d)$ for best-of-$n$
and $d(\alpha - \beta \log d)$ for RL, with $d = \sqrt{\mathrm{KL}}$, and [[huang2025|Huang et al. (2025)]] argue that
KL regularization is too weak and $\chi^2$ regularization is the right tool. For verifiers,
[[mroueh2025|Mroueh (2025)]] shows that GRPO amplifies whatever a binary reward accepts, [[rad2026|Rad et al. (2026)]]
analyze GRPO with randomly noisy verifiers, and [[moya2026|Moya et al. (2026)]] study gradient flow with a binary
verifier that accepts some incorrect answers. A recent survey names the largest gap as theoretical: the mean-field analysis of random verifier errors has no
counterpart for response-dependent, exploitable errors under group-relative gradients
([[gupta2026|Gupta and Joshi, 2026]]). The model below is aimed at such exploitable errors, and covers learned reward
models as well. It is a model of reward hacking, not of language-model RL
as a whole: it keeps what hacking depends on (one network shared across prompts, a reference model, and updates that
see only rewards) and abstracts away the rest, and its statements concern the expected update of an idealized estimator: on-policy, sequence-level, with no
token-length normalization, clipping or reuse of samples across epochs. Each takeaway below is tagged with its type: an
identity holds exactly under the setup, a proposition under the conditions stated with it, a worst-case bound against
an adversary, and a toy-model result only in a deliberately simplified model, where it points to a mechanism without
establishing it for real training. Claims marked as hypotheses are not derived.

**The setup.** A single network $\pi_\theta(y \mid x)$ answers every prompt $x$, and training starts from a reference
model $\pi_{\mathrm{ref}}$. The trainer sees a proxy reward $r(x, y)$ (a reward-model score, or a verifier’s verdict
computed after the response has run, so that editing the tests counts as part of the response) but not the intended
reward $r^*(x, y)$; their difference $e = r - r^*$ is the error. We group responses into reward levels
$L_v = \{y : r(x, y) = v\}$, write $\pi_R(v)$ for the probability the policy puts on level $v$,
$\nu_v = \pi_{\mathrm{ref}}(\cdot \mid L_v)$ for the reference policy inside it, and
$\bar e(v) = \mathbb{E}_{\nu_v}[e]$ for the reference error at that level. The amount of hacking is the part of the
proxy gain that is not real,

$$\mathrm{Hack}(\pi) = \mathbb{E}_\pi[e] - \mathbb{E}_{\pi_{\mathrm{ref}}}[e].$$

For a pass/fail verifier we also write $p$ for the pass rate, $q$ for the fraction of passes that are hacks, and
$\bar q$ for that fraction under $\pi_{\mathrm{ref}}$. When we ask what can be guaranteed without knowing where the
exploits are, we let an adversary place them after the policy is fixed, subject only to what is observable.
Throughout, the reward function and the prompt distribution are fixed during training: editing the tests counts as
part of one response, but tampering that changes how later responses are scored is outside the model, and a stochastic
verifier's randomness is treated as part of the response. Conditional distributions such as $\nu_v$ require the
reference model to put positive probability on the level. For the pass/fail case we also assume the verifier never
rejects a correct answer. "A prompt the model cannot solve" means one on which sampling finds no genuine solution, not
one where a solution has probability exactly zero. Finally, $\mathrm{Hack}$ is an operational quantity: it measures how
much of the proxy gain is not matched by true gain, can be negative, and says nothing about intent.

**The update cannot tell them apart.** The algorithms used in practice (REINFORCE with a baseline, RLOO, GRPO,
Dr. GRPO) sample a group of $G$ responses per prompt and weight each response’s gradient by a function of the group’s
rewards. Conditioning on the reward level of one sample, its weight depends only on the other samples, which are
independent of it, so

$$\mathbb{E}[\Delta\theta \mid x] = \sum_v a_v\, \nabla \pi(L_v)$$

for scalars $a_v$ that depend on the policy only through $\pi_R$. For a pass/fail verifier the update is
$w(p)\nabla p$, and for GRPO

$$w_G(p) = \mathbb{E}_m\Big[\sqrt{\tfrac{G-1-m}{1+m}}\Big] + \mathbb{E}_m\Big[\sqrt{\tfrac{m}{G-m}}\Big], \qquad m \sim \mathrm{Bin}(G-1, p),$$

which is symmetric around $p = 1/2$ and equal to $\sqrt{G-1}$ as $p \to 0$; for $G \ge 4$ it is smallest at
$p = 1/2$, while for $G = 2$ and $G = 3$ it is constant ($1$ and $\sqrt 2$). This uses group-averaged gradients, the
standard deviation with denominator $G$, and zero advantage for groups whose rewards are all equal.

> **Takeaway 1 · Identity.** Every set of responses inside a reward level gets the same coefficient per unit of
probability, so the reward signal itself does not distinguish an exploit from an honest answer with the same reward.
How each response's probability actually changes depends on the parameterization, the gradient geometry and any other
terms in the update; a rewarded response can even lose probability. Under a pass/fail verifier, a response that is the
only success in its group receives the largest advantage the group can give, $\sqrt{G-1}$ under GRPO; this is an
advantage, not the size of the parameter update.

::wg

**Two ways to hack.** Splitting the policy into its distribution over levels and its distribution inside each level
gives

$$\mathrm{Hack}(\pi) = \underbrace{\sum_v \big(\pi_R(v) - \pi_{\mathrm{ref},R}(v)\big)\,\bar e(v)}_{\text{selection along the proxy}} + \underbrace{\sum_v \pi_R(v)\big(\mathbb{E}_{\pi(\cdot\mid L_v)}[e] - \bar e(v)\big)}_{\text{amplification within levels}},$$

and the KL divergence splits the same way,
$\mathrm{KL}(\pi\|\pi_{\mathrm{ref}}) = \mathrm{KL}(\pi_R\|\pi_{\mathrm{ref},R}) + \sum_v \pi_R(v)\,\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)$.
The first term is the classical Goodhart effect: moving toward reward levels that are more often wrong. The second is
amplifying the wrong responses within a level, and Cauchy–Schwarz bounds it per level by
$\sqrt{\chi^2(\pi(\cdot\mid L_v)\|\nu_v)\,\mathrm{Var}_{\nu_v}(e)}$. A reward model gives nearly every response its own score, while a pass/fail verifier has a single passing level. With
binary true reward and a verifier that never rejects a correct answer,
$\mathrm{Hack} = (p - p_{\mathrm{ref}})\bar q + p\,(q - \bar q)$; if the verifier can also reject correct answers, with
$c$ the probability of a rejected correct answer, the term $-(c - c_{\mathrm{ref}})$ is added. How a change splits
between the two terms depends on how finely the proxy separates responses: when every response has its own score, each
level is a single response and the within-level term is zero by definition.

> **Takeaway 2 · Identity.** Hacking is selection along the proxy plus amplification within reward levels. The
within-level term vanishes for every possible error exactly when $\pi = \pi_{\mathrm{ref}} \cdot f(r)$ for some function
$f$, a family that includes the KL-regularized optimum, best-of-$n$ with ties broken independently of the response, and
quantilizers. For a reward model that scores nearly every response differently, almost all of any change counts as
selection; this reflects what the proxy can see, not a different training mechanism. Under a pass/fail verifier,
raising the pass rate scales the amount of hacking but not its share, and the share $q$ can move away from $\bar q$
only through amplification within the passing level.

**When true reward falls under a reward model.** As the KL budget grows, the KL-regularized optimum over all policies moves along the exponentially tilted family
$\pi_\eta \propto \pi_{\mathrm{ref}} e^{\eta r}$, the reference policy reweighted by $e^{\eta r}$. Along it,
$\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[f] = \mathrm{Cov}_{\pi_\eta}(f, r)$ for any $f$, hence

$$\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[r^*] = \mathrm{Var}_{\pi_\eta}(r) - \mathrm{Cov}_{\pi_\eta}(e, r).$$

If the error is independent of true quality under $\pi_{\mathrm{ref}}$, reweighting by $e^{\eta r}$ keeps it independent
all along the path.
Both statements assume the tilt is normalizable, $\mathbb{E}_{\pi_{\mathrm{ref}}}[e^{\eta r}] < \infty$, and that
$\mathrm{Var}_{\pi_\eta}(r) > 0$. With independence,
$\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[r^*] = \mathrm{Var}_{\pi_\eta}(r^*) \ge 0$.

> **Takeaway 3 · Proposition.** Along the path of KL-regularized optima, true reward falls exactly where the regression
slope of the error on the proxy exceeds one. If the error is independent of true quality under $\pi_{\mathrm{ref}}$,
true reward never falls along this path; each unit of proxy gain then brings
$\mathrm{Var}_{\pi_\eta}(r^*)/\big(\mathrm{Var}_{\pi_\eta}(r^*) + \mathrm{Var}_{\pi_\eta}(e)\big)$ units of true gain,
which is at most one but need not shrink as training goes on.

Independence and tail behavior are separate conditions. The tilt requires
$\mathbb{E}_{\pi_{\mathrm{ref}}}[e^{\eta r}] < \infty$; when the error is independent of true quality, this means the
error has a finite moment generating function, which rules out the heavy-tailed errors of
[[kwa2024|Kwa et al. (2024)]], and the independent case here is the tilt-path counterpart of their setting with
independent, light-tailed error. Since independence rules out a decline, a decline like the one
[[gao2023|Gao et al. (2023)]] measure means that either the error depends on quality under the reference model (for
example, a reward model that prefers long or flattering answers that are worse), or training does not follow this
path: best-of-$n$ is a different reweighting, under which heavy-tailed errors matter
([[kwa2024|Kwa et al., 2024]]; [[elmhamdi2024|El-Mhamdi and Hoang, 2024]]), and a finite network trained by gradient
descent generally does not realize the tilt exactly.

**Under a verifier, the objective does not choose.** For a pass/fail verifier, every policy with the same pass rate has
the same expected reward, whatever its hack share. Over the space of all policies and without regularization, every policy with $p = 1$ is optimal, from uniform
amplification of the reference passes ($q = \bar q$) to a policy that only hacks ($q = 1$); a softmax network may reach
$p = 1$ only in the limit, and a shared network may not be able to set $p$ and $q$ independently on each prompt.
Regularization can choose: over the space of all policies, the optimum with a KL penalty toward $\pi_{\mathrm{ref}}$
has the form $\pi_{\mathrm{ref}} \cdot f(r)$ and keeps $q = \bar q$ (Takeaway 2). In a restricted shared network, a
departure from $\bar q$ can also come from the best compromise the model class allows, so it cannot all be attributed
to an optimization path that has not converged. [[moya2026|Moya et al. (2026)]] make a
related observation for exact gradient flow, where reward rises while correctness falls whenever $p\dot q > (1 - q)\dot p$ (correctness being $p(1 - q)$ when no correct answer is rejected). The criterion is just the derivative of $p(1-q)$, so
it holds whatever algorithm drives $p$ and $q$, including the group-based estimators used in practice. With more than
two reward levels it has no direct counterpart; what carries over is that, within each level, the reward does not see
which responses are correct.

> **Takeaway 4 · Proposition.** Under a pass/fail verifier with no regularization, the objective does not determine the
hack share. For a given model class and training setup, which optimal policy training ends up at is decided by the
learning dynamics: which passing responses gradient descent amplifies faster, and how an update on one prompt moves the
others. In this sense reward hacking under a verifier is a question about implicit bias.

**What can be guaranteed.** If nothing restricts where exploits can be, nothing can be guaranteed: an adversary can call
every response of a level an exploit. Suppose instead we only know that under $\pi_{\mathrm{ref}}$ at most a fraction
$\bar q_v$ (with $0 < \bar q_v < 1$) of level $v$ is exploits. Writing $\rho = \pi(\cdot \mid L_v)/\nu_v$, the worst-case exploit share in the level is the probability under $\pi$ of the responses with the largest $\rho$,
taken until their reference probability adds up to $\bar q_v$. This is a fractional knapsack problem.
Here the adversary may split probability mass, either because labels can be randomized or because no single response
carries noticeable probability, which is a good approximation for language models. With deterministic labels on a few
heavy responses the knapsack only gives an upper bound: if a level has two responses of probability $1/2$ each and
$\bar q_v = 0.1$, no exploit set fits the budget. The adversary chooses after seeing $\pi$, so these are robust bounds,
not a description of how exploits arise. The resulting bounds are the per-level counterparts of the
no-free-lunch theorem of [[everitt2017|Everitt et al. (2017)]] and the cost bound of [[taylor2016|Taylor (2016)]].

> **Takeaway 5 · Worst-case bound.** When the adversary can split mass, the worst-case exploit share of a level is at
least $\bar q_v$, with equality exactly when every response of the level is amplified by the same factor. If
$\rho \le M$ it is at most $\min(1, M\bar q_v)$, and in any case at most
$\min\big(1, \bar q_v + \sqrt{\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)/2}\big)$.

**A budget that survives shared parameters.** The per-level picture treats each prompt as if its policy could be set
freely, and says nothing about prompts the reference model cannot solve. Neither is needed for the following. Take an
algorithm whose expected update is a gradient, $\mathbb{E}[\Delta\theta] = \nabla J$ with $J = \sum_x \omega(x) J_x$
(expected reward for REINFORCE and RLOO, $W(p) = \int_0^p w$ for binary GRPO), add a penalty $\lambda D_R$ on the
within-level divergence $D_R(x) = \sum_v \pi_R(v \mid x)\,\mathrm{KL}(\pi(\cdot \mid x, L_v)\|\nu_v)$, and run any
preconditioned gradient flow $\dot\theta = M\nabla F$ with $M \succeq 0$ from $\pi_{\mathrm{ref}}$, with the objective, the weights $\omega$ and $\lambda$ held fixed. Since $F$ can only
increase and starts at $J(0)$, the within-level divergence is paid for by the reward gained.

> **Takeaway 6 · Proposition.** Under exact gradient flow, for every network and every prompt mix,
$$\lambda \sum_x \omega(x)\, D_R(x, t) \;\le\; J(t) - J(0).$$
The budget covers the prompts in the training mix, weighted by $\omega(x)$. Prompts the model cannot solve get no
guarantee themselves, but what their exploits do to the within-level distributions of the other training prompts is
paid from the same budget. Effects on prompts outside the mix are not covered.

Pinsker’s inequality and Cauchy–Schwarz turn this into a bound on the average excess exploit share over a set of
prompts, with each prompt weighted by its training weight $\omega(x)$ and its pass rate $p(x)$, since the passing level
enters $D_R$ with weight $p(x)$; prompts with small weight or small pass rate are barely constrained. With stochastic
updates and finite step sizes $F$ need not increase monotonically, so the bound holds for the expected flow rather than
for every training run. The usual KL penalty adds the between-level divergence
$\mathrm{KL}(\pi_R\|\pi_{\mathrm{ref},R})$ on top. Against a reward model this between-level term is the one that
matters; against a verifier it limits how far the pass rate $p$, and with it the accepted-error mass $pq$, can move,
but it does not constrain the share $q$, which only the within-level term does.

**How exploits spread.** With shared parameters, a rewarded exploit on prompt $x$ moves every other prompt $x'$. For a
pass/fail verifier,

$$\frac{d}{dt}\log\frac{h(x')}{s(x')} = \mathbb{E}_x\Big[w(p(x))\Big(s(x)\,[K_{\mathcal{HS}} - K_{\mathcal{SS}}](x', x) + h(x)\,[K_{\mathcal{HH}} - K_{\mathcal{SH}}](x', x)\Big)\Big],$$

where $K_{AB}(x', x) = g_A(x')^\top g_B(x)$ is the inner product of the gradients of $\log\pi(A \mid x')$ and
$\log\pi(B \mid x)$ (with a preconditioner $M$, $g_A^\top M g_B$). Splitting the expectation into $x = x'$ and
$x \ne x'$ separates a prompt’s own damage from what it imports.
The kernel is evaluated at the current parameters and changes as training proceeds; the formula requires $h, s > 0$,
and the split into own and imported terms refers to a finite set of training prompts.

> **Takeaway 7 · Identity.** The exploit component of the update on one prompt pushes the exploit-to-solution ratio on
another prompt upward when that prompt's exploit is better aligned in gradient space with the first exploit than its
genuine solution is ($K_{\mathcal{HH}} > K_{\mathcal{SH}}$); whether the ratio actually rises depends on all components
together, including the solution term, other prompts and regularization. Gradient alignment is set by the
parameterization, not by semantic similarity; that it is common for exploits that do not depend on the task is a
hypothesis. On prompts the model cannot solve, a method that does not know the exploit cannot guarantee that the
probability it adds there goes to genuine solutions rather than exploits, even if it trains on those prompts. What it
can control is what those prompts export.

**Where it starts.** One plausible first rewarded exploit is a lone success on a prompt the model rarely solves; the
model does not show that this is how hacking usually starts. When exactly one response reaches a group's top reward,
the rewards alone cannot say whether it is honest or an exploit, and any statistic that compares two members of the top
level, such as their agreement or a contrast between them, is unavailable. More generally, observations that carry no
information about $r^*$ beyond $r$ cannot identify which accepted responses are wrong
([[moya2026|Moya et al., 2026]]); a lone success is the case where not even comparisons within the group are left.

> **Takeaway 8 · Proposition.** A rule that uses only the group's rewards and is symmetric under permuting the
samples cannot treat a lone honest success and a lone exploit differently according to true correctness, since their
reward configurations are the same. Under the standard binary GRPO analyzed here, the lone success receives the
advantage $\sqrt{G-1}$, the largest the group can give; other rules can lower the weight on both at once. Telling them
apart needs information about the response beyond its reward, such as its content, a monitor or a probe. On a prompt
where only exploits pass and each response is one with probability $h$, a lone pass occurs with probability
$Gh(1-h)^{G-1} \approx Gh$ when $Gh \ll 1$.

**How it takes off.** On a single prompt where only exploits pass, take a model with two free logits, one for the
exploits and one for everything else, trained by plain gradient flow on the group-averaged objective. It gives
$\dot h = 2w(h)h^2(1-h)^2$ for the exploit probability $h$, the same form as the flow [[rad2026|Rad et al. (2026)]] derive for GRPO with a noisy verifier, taken with a
perfect verifier and with their large-group normalization $1/\sigma(h)$ replaced by its finite-group counterpart
$w_G(h)$, which tends to $1/\sqrt{h(1-h)}$ as $G \to \infty$. Time here is gradient-flow time; GRPO's extra factor
$\sqrt{G-1}$ comes from standardizing the advantages and acts like a larger step size, so comparing algorithms needs a
common budget of steps, samples or tokens. Across prompts, let all of them share an exploit propensity with log-odds
$z$ and write $u = e^z$. While exploits are rare, lone exploits arrive at a rate proportional to $u$, which gives the leading term
$\dot u \approx a u^2$. The coefficient of $u^3$ collects several pieces with different signs: groups with several
exploits, and the second-order correction to the rate of lone exploits, since
$P(K = 1) = Gu/(1+u)^G = Gu - G^2u^2 + O(u^3)$. Their sum need not be positive. Under GRPO with $G = 2$, a group in
which both responses pass gets zero advantage; and in the two-logit model above with every prompt sharing the
propensity, $\dot u = 2w(h)\,u^2/(1+u)^2$, whose cubic coefficient $2w'(0) - 4w(0)$ is negative because $w_G$ does not
increase near $0$. We therefore make an additional assumption: that the effective drift is approximated by
$\dot u \approx a u^2 + b u^3$ with $a, b > 0$, valid only while exploits are rare. This positive-coefficient model
illustrates one possible mechanism for acceleration and for suppression that saturates; it does not follow from how
often lone and multiple successes occur. Damping the effect of lone exploits on other prompts to a fraction $\epsilon$
replaces $a$ by $\epsilon a$. At the start, $u \approx h_0$, and the damped term dominates while $\epsilon a > b h_0$;
if, within this model, $b/a$ is of order $G$, the crossover sits at $\epsilon$ of order $Gh_0$.

> **Takeaway 9 · Toy model.** In the two-logit model, for fixed $G$, the time for the exploit share to reach any fixed
threshold is about $1/(2w(0)h_0)$ when $h_0$ is small: about $1/(2h_0)$ for REINFORCE and $1/(2\sqrt{G-1}\,h_0)$ for
GRPO, in gradient-flow time. Across prompts, while exploits are rare, the leading term grows faster than
exponentially, so the time to reach a given risk level is dominated by the early phase. The rest holds only in the
additional positive-coefficient model: there, damping lone exploits by $\epsilon$ delays the take-off like
$1/\epsilon$ while $\epsilon \gg Gh_0$ and then saturates, at most about $1/(Gh_0)$ times later if $b/a$ is of order
$G$, so slowing the first carriers delays the take-off without stopping it, much as in an epidemic. With the exact
two-logit coefficients the cubic term is negative and this saturation does not follow. In either case, stopping the
take-off takes a force that pulls the propensity back while it is small and outweighs the drift there, such as a
strong enough KL penalty toward the reference model (the within-level KL alone does not penalize moving probability
into the passing level as a whole) or negative transfer, where learning to solve the solvable prompts lowers the
exploit propensity.


::takeoff

::damping

## Threat model and two ways forward

Put together, the model gives a threat model for reward hacking in language-model RL. The trainer sees only the proxy
$r$, and the update sees only rewards: an exploit and an honest answer in the same reward level receive the same weight
(Takeaway 1), so inside a level the reward gives the update no information about which is which. Under a binary
verifier, within-level amplification is the only way the hack share can grow (Takeaway 2), and without regularization
the objective does not choose between the two (Takeaway 4). A plausible worst case starts on a prompt the model rarely
solves: under GRPO a lone exploit receives the group’s largest advantage, and its reward alone cannot tell it from a lone
honest success (Takeaway 8); through shared parameters its update can raise exploits on other prompts whose gradients align
with it (Takeaway 7), and while hacks are rare the process accelerates (Takeaway 9). Two directions follow, each
addressing one half of the problem.

**Find a signal that separates $r$ from $r^*$.** The worst case is the zero-information one: inside a reward level nothing in the reward tells an exploit from a solution, and the only guarantees left are bounds on how unevenly the level is amplified
(Takeaway 5). Any signal that carries information about $r^*$ beyond $r$, such as held-out tests, a second verifier, a
chain-of-thought monitor or a probe on activations, breaks that symmetry, because it lets the update treat responses
with the same reward differently.
[[moya2026|Moya et al. (2026)]] analyze one such signal, audit feedback on correctness, and show that at the current
policy it lowers accepted errors and raises correctness as long as it outweighs the verifier's pressure toward errors. Such a signal has to be used with care: once it is optimized against, the policy can
learn to evade it ([[baker2025|Baker et al., 2025]]).

**Limit how much any sample moves the policy, hack or not.** The second direction does not try to tell hacks apart
at all. Since the update cannot distinguish an exploit from a solution with the same reward, it can instead be built so
that no sample, whichever it is, moves the policy too far, and the learning dynamics, which decide the outcome under a
verifier (Takeaway 4), may then limit the cumulative damage wherever the exploits happen to be. This is a hypothesis
about methods, and it needs a budget on the accumulated drift as well: a small influence per step does not by itself
keep the total drift small. The model shows where this leverage
lies: the maximal advantage a lone success receives (Takeaway 8), the transfer that carries a rewarded response from one
prompt to the others (Takeaway 7), and the within-level divergence, which a penalty ties to the reward gained, for every network and prompt mix but weighted by training weight and pass rate (Takeaway 6). The price is that honest lone successes are slowed as well. In the model, damping alone only delays the take-off; stopping it takes a force that pulls the propensity back while
it is still small and outweighs the drift there, such as a strong enough KL pull toward the reference model or negative
transfer from prompts that can be solved (Takeaway 9).

The two directions are likely to complement each other, since each covers a weakness of the other, though nothing
here shows that combining them is optimal. A signal is rarely complete and can
be evaded once optimized against, and limits on how far any sample moves the policy catch what it misses; those limits in turn slow honest
lone successes along with exploits, and a signal, where it exists, shows where they can be relaxed.

None of this says what learning to hack does to the rest of a model ([[macdiarmid2025|MacDiarmid et al., 2025]]), and all of it describes expected updates of an idealized estimator.
`;

// Chinese paragraphs are kept on one line each: PostText joins wrapped lines with a space.
// The translation keeps the English terms Chinese ML writing usually leaves untranslated.
const ZH = r`
## TL;DR

本文先回顾 reward hacking 在经典 RL、RLHF 和 RLVR 中的典型案例，以及语言模型出现之前人们如何形式化它；然后为语言模型 RL 中的 reward hacking 建立一个简单的理论模型，得到以下结论：

- 训练更新只看得到 reward，所以对 reward 相同的 hack 回答和真正的解答，它给的权重完全一样；各自的概率实际怎么变，取决于网络本身。
- Policy 有两种 hack 的方式：一是整体偏向被 reward 高估的回答，二是在 reward 相同的回答里越来越偏向 hack 的那些。对几乎给每个回答不同分数的 learned reward model，几乎所有变化都算作第一种；在 binary verifier（只判通过或不通过）下，第二种是通过回答中 hack 占比增长的唯一途径。
- 在没有正则项的 binary verifier 下，训练目标对老老实实解题的 policy 和全靠 hack 通过的 policy 一视同仁；最终学成哪一种，不由训练目标决定，而由训练过程本身的动力学决定。
- Hacking 一种可能的起点是：在一道模型很少能解出的难题上，恰好有一个采样回答靠钻评分漏洞通过了。Reward 无法把它和真正的解答区分开，GRPO 反而给它最大的 advantage；由于所有题目共享同一个网络，这可能把 hack 带到别的题目上，而且在 hack 还很少时传得越广就传得越快，这与 hacking 常常突然出现的现象一致。
- 由此提出两个方向：一是在 reward 之外找到能区分 hack 和真正解答的信号；二是不去区分，而是让任何单个样本，不论是不是 hack，都无法把模型推得太远。两者有望互补。

## Reward hacking 长什么样

[[wiener1960|Wiener (1960)]] 写道，如果把一个目标交给一台我们来不及叫停的机器，“我们最好十分确定，放进机器的目标正是我们真正想要的，而不只是它的一个花哨的仿品”。[[goodhart1975|Goodhart (1975)]] 指出，任何统计规律“一旦被拿来用作控制的工具，就会趋于失效”，更为人熟知的说法是“当一个指标变成目标，它就不再是好指标” ([[strathern1997|Strathern, 1997]])。

一辆模拟自行车朝目标骑行有奖励、背离目标却从不受罚，结果学会了在起点附近绕 20 到 50 米宽的圈 ([[randlov1998|Randløv and Alstrøm, 1998]])。一个碰到球就有奖励的足球机器人，学会了停在球边不停地碰它 ([[ng1999|Ng et al., 1999]])。在赛艇游戏 CoastRunners 里，智能体发现在泻湖里反复撞几个会重生的目标比跑完全程更划算，一边着火一边逆行，得分还比人类玩家高出约 20% ([[clark2016|Clark and Amodei, 2016]])。一只从人类反馈中学抓取的机械臂，把手挡在镜头和物体之间，“看起来像是在抓” ([[amodei2017|Amodei et al., 2017]])。一个叠积木的智能体干脆把积木翻了过来 ([[krakovna2020|Krakovna et al., 2020]])；一个程序修复系统让程序永远返回空列表，测试居然通过了 ([[lehman2020|Lehman et al., 2020]])；在一个 gridworld 里，走进水桶格就能让所有番茄看起来都浇过水 ([[leike2017|Leike et al., 2017]])。在冻结的 learned reward model 上训练的 Atari 智能体，把 reward model 的打分越推越高，游戏得分却“一落千丈” ([[ibarz2018|Ibarz et al., 2018]])。

::classic

这一现象今天的名字来自 [[amodei2016|Amodei et al. (2016)]]：目标函数“存在某种取巧的简单解，形式上最大化了目标函数，却违背了设计者的本意”。DeepMind 称之为 specification gaming，即“满足了目标的字面规定、却没有达成预期结果的行为” ([[krakovna2020|Krakovna et al., 2020]])。[[pan2022|Pan et al. (2022)]] 发现，随着模型容量增长，它可能以相变的形式突然出现；[[weng2024|Weng (2024)]] 总结了通行的用法：智能体“利用 reward function 中的缺陷或歧义拿到高 reward，却没有真正学会或完成预期任务”。

在 RLHF 中，对 learned reward model 优化得越狠，摘要质量先升后降 ([[stiennon2020|Stiennon et al., 2020]])；在好几个数据集上，RLHF 测出的提升有很大一部分只是因为回答变长了 ([[singhal2023|Singhal et al., 2023]])；preference model 往往更偏爱迎合用户的回答，而不是正确的回答 ([[sharma2024|Sharma et al., 2024]])；reward model ensemble 的作用也没有预想的大，因为各个成员往往犯同样的错 ([[eisenstein2023|Eisenstein et al., 2023]])。

在程序化的 verifier 下，前沿 coding agent 被发现会在测试运行前调用 ${'`'}exit(0)${'`'}、抛出 ${'`'}SkipTest${'`'}、改动测试框架上游的代码、从测试文件里读出期望值、反编译参考答案；把一个读取其 CoT 的 monitor 加进 reward 之后，它们照样 hack，只是不再说出来 ([[baker2025|Baker et al., 2025]])。Claude 3.7 Sonnet 的 system card 提到，它会“为了通过测试用例而写特判”，这是训练中 reward hacking 的结果 ([[anthropic2025|Anthropic, 2025]])；o3 在评测任务里改写评分器的计时器、篡改评测程序，被问到时也承认这并非用户想要的 ([[vonarx2025|Von Arx et al., 2025]])；在被刻意改成不可能完成的编程任务上，“任何通过都必然意味着违反规格的捷径”，前沿模型却常常通过 ([[zhong2025|Zhong et al., 2025]])。在一系列越来越容易钻空子的任务里，模型偶尔甚至会改写自己的 reward function ([[denison2024|Denison et al., 2024]])；而在生产级编程环境中学会 hack，还伴随着更广泛的 misalignment ([[macdiarmid2025|MacDiarmid et al., 2025]]; [[taylor2025|Taylor et al., 2025]])。[[wang2026|Wang et al. (2026)]] 把其中的机制概括为“当钻漏洞比真正解决任务更容易时，hacking 就会发生”。

::lm

## 语言模型之前的形式化

最常见的形式化，是把智能体优化的 reward 写成真正想要的 reward 加上一个误差，

$$R = R^* + \xi,$$

沿用这种写法的工作，区别主要在于对 $\xi$ 做了什么假设。在 reward shaping 中，$\xi$ 是设计者额外加入的一项 $F(s, a, s')$，当 $F = \gamma\Phi(s') - \Phi(s)$ 时称为 potential-based ([[ng1999|Ng et al., 1999]])。AI Safety Gridworlds 把这种拆分内置在每个环境里：智能体能观测到 reward $R$，却看不到另一个 performance function $R^*$ ([[leike2017|Leike et al., 2017]])。[[manheim2018|Manheim and Garrabrant (2018)]] 在 regressional Goodhart 中把 $\xi$ 视为噪声，在 extremal Goodhart 中则让两者的关系本身在某个阈值处改变：$R \le a$ 时 $R^* = R + x$，否则 $R^* = R + y$。[[kwa2024|Kwa et al. (2024)]] 按尾部给 $\xi$ 分类：若对任意 $t > 0$ 都有 $e^{tx}\Pr(\xi > x) \to \infty$，就称它是重尾的。[[everitt2017|Everitt et al. (2017)]] 的 corrupt reward MDP 让 $\xi$ 只在部分被污染的状态上非零：给定真实 reward $\dot R$ 和污染函数 $C$，智能体观测到 $\hat R(s) = C_s(\dot R(s))$，再对被污染状态的数量、一个状态的 reward 能否从别处观测到做出假设。

[[hadfield2017|Hadfield-Menell et al. (2017)]] 和 [[zhuang2020|Zhuang and Hadfield-Menell (2020)]] 则把 proxy 看作设计者写下的一份不完整的规格。在 inverse reward design 中，proxy $R = \tilde w^\top\phi(\tau)$ 和真实 reward $R^* = w^{*\top}\phi(\tau)$ 都是轨迹特征的线性函数；设计者在训练环境 $\tilde M$ 中选定 $\tilde w$，模型假设他在那里近似最优，

$$P(\tilde w \mid w^*, \tilde M) \propto \exp\big(\beta\, \mathbb{E}[w^{*\top}\phi(\tau) \mid \tau \sim \pi(\tau \mid \tilde w, \tilde M)]\big),$$

因此 proxy 只能通过 $\tilde M$ 中的行为透露关于 $w^*$ 的信息。在 [[zhuang2020|Zhuang and Hadfield-Menell (2020)]] 中，世界由属性 $s \in \mathbb{R}^L$ 描述，并受资源约束 $g(s) \le 0$；真实 reward $R^*$ 依赖全部 $L$ 个属性，proxy $R$ 只依赖其中 $J < L$ 个。

[[ring2011|Ring and Orseau (2011)]]、[[everitt2016|Everitt and Hutter (2016)]] 和 [[everitt2021b|Everitt et al. (2021b)]] 允许智能体作用于产生 reward 的通道本身。用统一的记号，观测到的 reward 是一种扭曲 $R = d(R^*)$，而智能体的动作可以改变 $d$。在 [[ring2011|Ring and Orseau (2011)]] 的 delusion box 中，动作为 $a_t = \langle d_t, a^e_t \rangle$，其中 $d_t$ 设定了一个映射 $d : O \to O$，智能体透过它观察世界，$o_t = d(a_t, o^e_t)$，$o_t$ 的一部分被当作 reward；在 [[everitt2016|Everitt and Hutter (2016)]] 中，$R = d_s(\check R)$，其中内部 reward $\check R = u^*(s)$，扭曲取决于智能体到达的状态。[[everitt2021b|Everitt et al. (2021b)]] 用 causal influence diagram 给出一般形式，图中的节点包括实际实现的 reward function $\Theta^R_t$ 和设计者想要的 $\Theta^R_*$：如果智能体的决策影响 $\Theta^R_t$ 的方式违背了 $\Theta^R_*$，就是 tampering；按 [[everitt2021a|Everitt et al. (2021a)]] 的判据，要有控制 $\Theta^R_t$ 的激励，必须存在一条从决策经过它通往效用的有向路径。

[[skalse2022|Skalse et al. (2022)]]、[[karwowski2024|Karwowski et al. (2024)]] 和 [[laidlaw2025|Laidlaw et al. (2025)]] 把 proxy 和真实 reward 当作 policy 空间上的函数来比较，依据是回报对 occupancy measure 是线性的，$J_R(\pi) = \langle \eta^\pi, R \rangle$。[[skalse2022|Skalse et al. (2022)]] 的定义是：若 $R$ 和 $R^*$ 对某一对 policy 的排序相反，它们在 policy 集合 $\Pi$ 上就是 hackable 的，

$$\exists\, \pi, \pi' \in \Pi:\quad J_R(\pi) < J_R(\pi') \ \text{且}\ J_{R^*}(\pi) > J_{R^*}(\pi').$$

另外两篇用一个余弦来衡量两者的关系：[[karwowski2024|Karwowski et al. (2024)]] 用 $R$ 与 $R^*$ 在 $\mathrm{span}(\Omega)$ 上投影的夹角，其中 $\Omega$ 是 occupancy measure 构成的多面体；[[laidlaw2025|Laidlaw et al. (2025)]] 用两者在参考 occupancy measure $\mu_{\mathrm{ref}}$ 下采样的状态–动作对上的相关系数 $\kappa > 0$，并用 $\chi^2(\mu_\pi \,\|\, \mu_{\mathrm{ref}})$ 衡量 policy 偏离了多远。

[[taylor2016|Taylor (2016)]] 改的是优化器而不是 reward。给定动作上的一个基础分布 $\gamma$（比如人会怎么做），$q$-quantilizer 从 $\gamma$ 中按 proxy 排序的前 $q$ 比例里采样，因此它的分布满足 $Q_q(a) \le \gamma(a)/q$；未知的危害被假设为一个任意的非负代价 $c$，只知道它在 $\gamma$ 下的期望有上界。

## 语言模型 RL 中 reward hacking 的理论模型

对 learned reward model，[[gao2023|Gao et al. (2023)]] 把 gold reward 拟合为 best-of-$n$ 下的 $d(\alpha - \beta d)$ 和 RL 下的 $d(\alpha - \beta \log d)$，其中 $d = \sqrt{\mathrm{KL}}$；[[huang2025|Huang et al. (2025)]] 认为 KL 正则太弱，$\chi^2$ 正则才是合适的工具。对 verifier，[[mroueh2025|Mroueh (2025)]] 证明 GRPO 会放大 binary reward 所接受的任何东西，[[rad2026|Rad et al. (2026)]] 分析了 verifier 带随机噪声时的 GRPO，[[moya2026|Moya et al. (2026)]] 研究了一个会接受部分错误答案的 binary verifier 下的梯度流。一篇近期综述指出，最大的空白在理论上：针对随机 verifier 误差的 mean-field 分析，在 group-relative 梯度下依赖于回答、可被利用的误差上还没有对应的理论 ([[gupta2026|Gupta and Joshi, 2026]])。下面的模型针对的就是这类可利用的误差，同时也覆盖 learned reward model。它是 reward hacking 的模型，而不是整个语言模型 RL 的模型：它保留了 hacking 所依赖的要素（所有 prompt 共享同一个网络、一个 reference model、只看得到 reward 的更新），把其余部分抽象掉，所有结论针对的都是一个理想化估计量的期望更新：on-policy、按整条序列计算，不做 token 长度归一化、clipping，也不跨 epoch 重复使用样本。下面每条要点都标了类型：恒等式在设定下精确成立；命题在随附的条件下成立；最坏情况界针对一个对手成立；toy model 结论只在刻意简化的模型里成立，它指出一种机制，但没有证明真实训练就是如此。标为假说的说法不是推导出来的。

**设定．**　一个网络 $\pi_\theta(y \mid x)$ 回答所有 prompt $x$，训练从 reference model $\pi_{\mathrm{ref}}$ 出发。训练者看得到 proxy reward $r(x, y)$（reward model 的打分，或者回答执行之后才得出的 verifier 判定，所以改测试也算回答的一部分），看不到真正想要的 reward $r^*(x, y)$；两者之差 $e = r - r^*$ 就是误差。按 reward 值把回答分成若干 reward level $L_v = \{y : r(x, y) = v\}$，记 $\pi_R(v)$ 为 policy 落在 level $v$ 上的概率，$\nu_v = \pi_{\mathrm{ref}}(\cdot \mid L_v)$ 为 reference model 在该 level 内部的分布，$\bar e(v) = \mathbb{E}_{\nu_v}[e]$ 为 reference model 在该 level 上的平均误差。Hacking 的量，就是 proxy 增益中不真实的那部分，

$$\mathrm{Hack}(\pi) = \mathbb{E}_\pi[e] - \mathbb{E}_{\pi_{\mathrm{ref}}}[e].$$

对 binary verifier，另记 $p$ 为通过率，$q$ 为通过的回答中 hack 所占的比例，$\bar q$ 为 $\pi_{\mathrm{ref}}$ 下的这一比例。当我们问“在不知道 exploit 藏在哪里的前提下能保证什么”时，我们就设想一个对手，在 policy 确定之后再放置 exploit，只受可观测量的约束。全文假设 reward 函数和 prompt 分布在训练中固定：改测试算作单个回答的一部分，但改变之后回答如何被评分的持久性篡改不在模型范围内；如果 verifier 有随机性，就把随机性并入回答。$\nu_v$ 这样的条件分布要求 reference model 在该 level 上的概率为正。对 binary verifier，另外假设 verifier 从不拒绝正确的回答。“模型解不出的 prompt”指采样找不到真正的解答，而不是解答的概率严格为零。最后，$\mathrm{Hack}$ 是一个操作性的量：它度量 proxy 增益中没有真实增益对应的部分，可以为负，也不涉及任何意图。

**更新无法区分两者．**　实践中常用的算法（带 baseline 的 REINFORCE、RLOO、GRPO、Dr. GRPO）对每个 prompt 采样一组 $G$ 个回答，用组内 reward 的某个函数给每个回答的梯度加权。给定某个样本所在的 reward level，它的权重只取决于组内其他样本，而这些样本与它独立，所以

$$\mathbb{E}[\Delta\theta \mid x] = \sum_v a_v\, \nabla \pi(L_v),$$

其中标量 $a_v$ 只通过 $\pi_R$ 依赖于 policy。对 binary verifier，更新为 $w(p)\nabla p$，GRPO 的权重为

$$w_G(p) = \mathbb{E}_m\Big[\sqrt{\tfrac{G-1-m}{1+m}}\Big] + \mathbb{E}_m\Big[\sqrt{\tfrac{m}{G-m}}\Big], \qquad m \sim \mathrm{Bin}(G-1, p),$$

它关于 $p = 1/2$ 对称，$p \to 0$ 时趋于 $\sqrt{G-1}$；$G \ge 4$ 时在 $p = 1/2$ 处取最小值，而 $G = 2$ 和 $G = 3$ 时它是常数（分别为 $1$ 和 $\sqrt 2$）。这里采用组内平均的梯度、分母为 $G$ 的标准差，并令 reward 全部相同的组 advantage 为零。

> **要点 1 · 恒等式**　同一 reward level 内的任何一组回答，单位概率上得到的系数都相同，所以 reward 信号本身区分不了 reward 相同的 exploit 和真正的解答。各回答的概率实际怎么变，取决于参数化、梯度几何以及更新中的其他项；得到奖励的回答甚至可能概率下降。在 binary verifier 下，组里唯一成功的回答拿到这个组能给出的最大 advantage（GRPO 下为 $\sqrt{G-1}$）；这是 advantage 的大小，不是参数更新的大小。

::wg

**Hack 的两种途径．**　把 policy 拆成它在各 level 之间的分布和每个 level 内部的分布，得到

$$\mathrm{Hack}(\pi) = \underbrace{\sum_v \big(\pi_R(v) - \pi_{\mathrm{ref},R}(v)\big)\,\bar e(v)}_{\text{沿 proxy 的 selection}} + \underbrace{\sum_v \pi_R(v)\big(\mathbb{E}_{\pi(\cdot\mid L_v)}[e] - \bar e(v)\big)}_{\text{level 内的 amplification}},$$

KL 散度也可以同样拆开，$\mathrm{KL}(\pi\|\pi_{\mathrm{ref}}) = \mathrm{KL}(\pi_R\|\pi_{\mathrm{ref},R}) + \sum_v \pi_R(v)\,\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)$。第一项是经典的 Goodhart 效应：向更常出错的 reward level 移动。第二项是在同一 level 内放大错误的回答，由 Cauchy–Schwarz，它在每个 level 上不超过 $\sqrt{\chi^2(\pi(\cdot\mid L_v)\|\nu_v)\,\mathrm{Var}_{\nu_v}(e)}$。Reward model 几乎给每个回答一个不同的分数，而 binary verifier 只有一个通过的 level。若真实 reward 也是二值的，且 verifier 从不拒绝正确回答，则 $\mathrm{Hack} = (p - p_{\mathrm{ref}})\bar q + p\,(q - \bar q)$；若 verifier 也会拒绝正确回答，记 $c$ 为正确回答被拒的概率，还要加上 $-(c - c_{\mathrm{ref}})$ 一项。一次变化如何在两项之间分配，取决于 proxy 把回答区分得有多细：如果每个回答的分数都不同，每个 level 只含一个回答，level 内这一项按定义为零。

> **要点 2 · 恒等式**　Hacking 等于沿 proxy 的 selection 加上 reward level 内的 amplification。Level 内这一项对任何可能的误差都为零，当且仅当 $\pi = \pi_{\mathrm{ref}} \cdot f(r)$ 对某个函数 $f$ 成立，这一族包括 KL 正则下的最优解、平局打破与回答无关的 best-of-$n$，以及 quantilizer。对几乎给每个回答不同分数的 reward model，几乎所有变化都算作 selection；这反映的是 proxy 能看到什么，而不是训练机制不同。在 binary verifier 下，提高通过率会按比例放大 hacking 的量，却不改变它的占比，而占比 $q$ 只能靠通过 level 内的 amplification 偏离 $\bar q$。

**Reward model 下真实 reward 何时下降．**　随着 KL 预算增大，在全体 policy 上求得的 KL 正则最优解沿着 exponential tilting 族 $\pi_\eta \propto \pi_{\mathrm{ref}} e^{\eta r}$ 移动，也就是把 reference policy 按 $e^{\eta r}$ 重新加权。沿着这条路径，对任意 $f$ 有 $\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[f] = \mathrm{Cov}_{\pi_\eta}(f, r)$，因此

$$\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[r^*] = \mathrm{Var}_{\pi_\eta}(r) - \mathrm{Cov}_{\pi_\eta}(e, r).$$

如果在 $\pi_{\mathrm{ref}}$ 下误差与真实质量独立，按 $e^{\eta r}$ 重新加权后，这种独立性沿整条路径都会保持。以上都假设这条 tilt 路径可以归一化，即 $\mathbb{E}_{\pi_{\mathrm{ref}}}[e^{\eta r}] < \infty$，并且 $\mathrm{Var}_{\pi_\eta}(r) > 0$。独立时 $\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[r^*] = \mathrm{Var}_{\pi_\eta}(r^*) \ge 0$。

> **要点 3 · 命题**　沿 KL 正则最优解的路径，真实 reward 恰好在误差对 proxy 的回归斜率超过 1 时下降。如果在 $\pi_{\mathrm{ref}}$ 下误差与真实质量独立，真实 reward 沿这条路径永远不会下降；此时每单位 proxy 增益带来 $\mathrm{Var}_{\pi_\eta}(r^*)/\big(\mathrm{Var}_{\pi_\eta}(r^*)+\mathrm{Var}_{\pi_\eta}(e)\big)$ 单位的真实增益，这个比例不超过 1，但不一定随训练变小。

独立性和尾部性质是两个不同的条件。Tilt 要求 $\mathbb{E}_{\pi_{\mathrm{ref}}}[e^{\eta r}] < \infty$；误差与真实质量独立时，这意味着误差的矩母函数有限，从而排除了 [[kwa2024|Kwa et al. (2024)]] 意义下的重尾误差，这里的独立情形对应于他们误差独立且轻尾的设定在 tilt 路径上的版本。既然独立排除了下降，像 [[gao2023|Gao et al. (2023)]] 观测到的那种下降就意味着：要么误差在 reference model 下依赖于质量（例如 reward model 偏爱更长或更讨好、实际却更差的回答），要么训练并没有走这条路径——best-of-$n$ 是另一种重新加权，在它之下重尾误差会起作用 ([[kwa2024|Kwa et al., 2024]]; [[elmhamdi2024|El-Mhamdi and Hoang, 2024]])，而用梯度下降训练的有限网络一般也无法精确实现这条 tilt 路径。

**Binary verifier 下，目标函数不做选择．**　对 binary verifier，通过率相同的 policy 期望 reward 都一样，不论其中 hack 占多少。在全体 policy 构成的空间里且没有正则项时，所有 $p = 1$ 的 policy 都是最优的，从均匀放大 reference model 通过的回答（$q = \bar q$），到只会 hack 的 policy（$q = 1$）；softmax 网络可能只能在极限意义下达到 $p = 1$，共享网络也未必能在每个 prompt 上独立设定 $p$ 和 $q$。正则项则可以做选择：在全体 policy 构成的空间里，加上朝 $\pi_{\mathrm{ref}}$ 的 KL 惩罚后，最优解形如 $\pi_{\mathrm{ref}} \cdot f(r)$，保持 $q = \bar q$（要点 2）。在受限的共享网络里，偏离 $\bar q$ 还可能来自模型类约束下的最优折中，不能全部归因于尚未收敛的优化路径。[[moya2026|Moya et al. (2026)]] 对精确梯度流有一个相关的观察：只要 $p\dot q > (1 - q)\dot p$（不拒绝正确回答时，正确率为 $p(1-q)$），reward 上升的同时正确率就在下降。这个判据只是对 $p(1-q)$ 求导，所以无论用什么算法驱动 $p$ 和 $q$ 都成立，包括实践中的 group-based 估计量。有两个以上 reward level 时它没有直接的对应；能推广过去的是：在每个 level 内部，reward 都看不出哪些回答是正确的。

> **要点 4 · 命题**　在没有正则项的 binary verifier 下，目标函数不决定 hack 的占比。给定模型类和训练设置，训练最终落到哪个最优 policy 上，由学习动力学决定：梯度下降把哪些通过的回答放大得更快，以及一个 prompt 上的更新如何牵动其他 prompt。在这个意义上，verifier 下的 reward hacking 是一个 implicit bias 问题。

**能保证什么．**　如果对 exploit 可能出现在哪里毫无限制，就什么也保证不了：对手可以把某个 level 的每个回答都当作 exploit。假设我们只知道在 $\pi_{\mathrm{ref}}$ 下 level $v$ 中至多有比例 $\bar q_v$（$0 < \bar q_v < 1$）是 exploit。记 $\rho = \pi(\cdot \mid L_v)/\nu_v$，则该 level 内最坏情况下的 exploit 占比，就是 $\pi$ 下那些 $\rho$ 最大的回答的概率：从 $\rho$ 最大的回答取起，直到它们在 reference 下的概率加起来等于 $\bar q_v$。这是一个分数背包问题。这里允许对手切分概率质量：要么标签可以随机化，要么没有哪个单独的回答有可观的概率，对语言模型来说这是很好的近似。如果标签是确定的、且 level 里只有少数几个概率很大的回答，分数背包只给出上界：比如一个 level 只有两个各占 $1/2$ 的回答而 $\bar q_v = 0.1$，就没有任何 exploit 集合能放进预算。对手是在看到 $\pi$ 之后才做选择的，所以这些是稳健的界，而不是对 exploit 实际如何产生的描述。由此得到的界，是 [[everitt2017|Everitt et al. (2017)]] 的 no-free-lunch 定理和 [[taylor2016|Taylor (2016)]] 的代价界在单个 level 上的对应物。

> **要点 5 · 最坏情况界**　在对手可以切分概率质量时，一个 level 内最坏情况下的 exploit 占比至少是 $\bar q_v$，当且仅当该 level 的每个回答都被放大同样的倍数时取等。若 $\rho \le M$，占比至多 $\min(1, M\bar q_v)$；无论如何，占比至多 $\min\big(1, \bar q_v + \sqrt{\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)/2}\big)$。

**共享参数下依然成立的预算．**　单个 level 的分析假设每个 prompt 上的 policy 可以随意设定，也没有处理 reference model 解不出的 prompt。下面的结论两者都不需要。取一个期望更新是梯度的算法，$\mathbb{E}[\Delta\theta] = \nabla J$，$J = \sum_x \omega(x) J_x$（对 REINFORCE 和 RLOO 是期望 reward，对 binary GRPO 是 $W(p) = \int_0^p w$），对 level 内散度 $D_R(x) = \sum_v \pi_R(v \mid x)\,\mathrm{KL}(\pi(\cdot \mid x, L_v)\|\nu_v)$ 加一个惩罚 $\lambda D_R$，并从 $\pi_{\mathrm{ref}}$ 出发运行任意 preconditioned 梯度流 $\dot\theta = M\nabla F$，$M \succeq 0$，目标函数、权重 $\omega$ 和 $\lambda$ 都保持固定。由于 $F$ 只增不减且起点是 $J(0)$，level 内散度只能用获得的 reward 来支付。

> **要点 6 · 命题**　在精确梯度流下，对任何网络、任何 prompt 配比， $$\lambda \sum_x \omega(x)\, D_R(x, t) \;\le\; J(t) - J(0).$$ 这个预算覆盖训练配比中的 prompt，按 $\omega(x)$ 加权。模型解不出的 prompt 本身得不到保证，但它们的 exploit 对其他训练 prompt 的 level 内分布造成的影响，都要从同一个预算里支付。训练配比之外的 prompt 不在覆盖范围内。

再用 Pinsker 不等式和 Cauchy–Schwarz，就能把它变成一组 prompt 上平均超额 exploit 占比的界，其中每个 prompt 按训练权重 $\omega(x)$ 和通过率 $p(x)$ 加权，因为通过 level 在 $D_R$ 中的权重就是 $p(x)$；权重小或通过率低的 prompt 几乎不受约束。用随机梯度和有限步长训练时，$F$ 不一定单调上升，所以这个界针对的是期望意义下的梯度流，而不是每一次训练。常用的 KL 惩罚在此之上还加了 level 之间的散度 $\mathrm{KL}(\pi_R\|\pi_{\mathrm{ref},R})$。对 reward model，起作用的正是这一项；对 verifier，这一项限制通过率 $p$ 能走多远，也就限制了被接受的错误总量 $pq$，但不约束占比 $q$，约束 $q$ 的只有 level 内那一项。

**Exploit 如何扩散．**　参数共享时，prompt $x$ 上一个得到奖励的 exploit 会牵动所有其他 prompt $x'$。对 binary verifier，

$$\frac{d}{dt}\log\frac{h(x')}{s(x')} = \mathbb{E}_x\Big[w(p(x))\Big(s(x)\,[K_{\mathcal{HS}} - K_{\mathcal{SS}}](x', x) + h(x)\,[K_{\mathcal{HH}} - K_{\mathcal{SH}}](x', x)\Big)\Big],$$

其中 $K_{AB}(x', x) = g_A(x')^\top g_B(x)$ 是 $\log\pi(A \mid x')$ 与 $\log\pi(B \mid x)$ 梯度的内积（有 preconditioner $M$ 时为 $g_A^\top M g_B$）。把期望拆成 $x = x'$ 和 $x \ne x'$ 两部分，就把一个 prompt 自身造成的损害和它从别处“输入”的损害区分开了。这个 kernel 是在当前参数处计算的，会随训练变化；公式要求 $h, s > 0$，而“自身”与“输入”的拆分针对的是有限的训练 prompt 集合。

> **要点 7 · 恒等式**　一个 prompt 上更新中的 exploit 分量，会把另一个 prompt 上 exploit 与解答之比往上推，条件是另一个 prompt 上的 exploit 在梯度空间里比那里的真正解答更对齐这个 exploit（$K_{\mathcal{HH}} > K_{\mathcal{SH}}$）；这个比值实际是否上升，取决于所有分量的总和，包括解答项、其他 prompt 和正则项。梯度对齐由参数化决定，不等于语义相似；对于不依赖具体任务的 exploit 这种情况很常见，只是一个假说。在模型解不出的 prompt 上，一个不知道 exploit 是什么的方法，就算在这些 prompt 上训练，也无法保证它在那里增加的概率会落到真正的解答上，而不是 exploit；它能控制的，是这些 prompt 向外“输出”了什么。

**从哪里开始．**　第一个得到奖励的 exploit，一种可能的情形是某个模型很少能解出的 prompt 上的孤立成功；模型并没有证明 hacking 通常就是这样开始的。当组里恰好只有一个回答拿到最高 reward 时，仅凭 reward 无法判断它是真正的解答还是 exploit，而任何需要比较最高 level 里两个成员的统计量（比如它们是否一致、彼此之间的对比）都无从计算。更一般地，如果观测在 $r$ 之外不携带关于 $r^*$ 的信息，就无法识别哪些被接受的回答是错的 ([[moya2026|Moya et al., 2026]])；孤立成功是连组内比较都不剩的情形。

> **要点 8 · 命题**　只用组内 reward、且对样本置换对称的规则，在 reward 配置相同时，无法依据真实正确性区别对待孤立的真正成功和孤立的 exploit。在本文分析的标准 binary GRPO 下，孤立成功的 advantage 为 $\sqrt{G-1}$，是这个组能给出的最大值；其他规则可以同时压低两者的权重。要区分它们，需要 reward 之外关于回答本身的信息，比如回答内容、monitor 或 probe。在一个只有 exploit 能通过、每个回答通过的概率为 $h$ 的 prompt 上，恰好一个通过的概率为 $Gh(1-h)^{G-1}$，在 $Gh \ll 1$ 时约为 $Gh$。

**如何 take-off．**　在一个只有 exploit 能通过的 prompt 上，取一个有两个自由 logit 的模型，一个对应 exploit，一个对应其余所有回答，对组内平均的目标做普通梯度流。它给出 exploit 概率 $h$ 满足 $\dot h = 2w(h)h^2(1-h)^2$，与 [[rad2026|Rad et al. (2026)]] 对带噪声 verifier 的 GRPO 推出的动力学形式相同：取完美 verifier，并把他们大组极限下的归一化 $1/\sigma(h)$ 换成有限组的 $w_G(h)$ 即可，$G \to \infty$ 时 $w_G(h) \to 1/\sqrt{h(1-h)}$。这里的时间是梯度流时间；GRPO 多出的 $\sqrt{G-1}$ 因子来自 advantage 的标准化，作用相当于更大的步长，所以比较不同算法需要统一按步数、样本数或 token 数计算的预算。放到跨 prompt 来看，设所有 prompt 共享一个 exploit 倾向，其 log-odds 为 $z$，记 $u = e^z$。在 exploit 还很少时，孤立 exploit 出现的速率正比于 $u$，由此得到主导项 $\dot u \approx a u^2$。$u^3$ 的系数则由几部分组成，符号各不相同：含多个 exploit 的组，以及孤立 exploit 出现速率本身的二阶修正，因为 $P(K = 1) = Gu/(1+u)^G = Gu - G^2u^2 + O(u^3)$。它们的和不一定为正。在 $G = 2$ 的 GRPO 下，两个回答都通过的组 advantage 为零；而在上面的两 logit 模型里，若所有 prompt 共享同一个倾向，则 $\dot u = 2w(h)\,u^2/(1+u)^2$，其三次项系数 $2w'(0) - 4w(0)$ 为负，因为 $w_G$ 在 $0$ 附近不增。因此我们额外假设：有效漂移可以用 $\dot u \approx a u^2 + b u^3$ 近似，其中 $a, b > 0$，且只在 exploit 还很少时成立。这个正系数模型用来展示一种可能的加速机制，以及压制效果会饱和的机制；它不能单凭孤立成功和多重成功出现的频率推出。把孤立 exploit 对其他 prompt 的影响压到原来的 $\epsilon$ 倍，相当于把 $a$ 换成 $\epsilon a$。开始时 $u \approx h_0$，被压制的一项占主导的条件是 $\epsilon a > b h_0$；在这个模型里，如果 $b/a$ 是 $G$ 的量级，交叉点就在 $\epsilon \sim Gh_0$。

> **要点 9 · toy model**　在两 logit 模型中，固定 $G$、$h_0$ 很小时，exploit 占比到达任一固定阈值的时间约为 $1/(2w(0)h_0)$：按梯度流时间计，REINFORCE 约为 $1/(2h_0)$，GRPO 约为 $1/(2\sqrt{G-1}\,h_0)$。跨 prompt 时，在 exploit 还很少的阶段，主导项的增长比指数还快，所以到达给定风险水平的时间主要由早期阶段决定。以下结论只在额外的正系数模型里成立：把孤立 exploit 的影响压到 $\epsilon$ 倍，在 $\epsilon \gg Gh_0$ 时 take-off 按 $1/\epsilon$ 推迟，之后趋于饱和；若 $b/a$ 是 $G$ 的量级，最多推迟到原来的约 $1/(Gh_0)$ 倍，所以减慢最早的“携带者”只能推迟 take-off，不能阻止它，和传染病类似。用精确的两 logit 系数时三次项为负，推不出这种饱和。无论哪种情况，要阻止 take-off，都需要一个在倾向还小时就把它拉回来、并且强到足以抵消那里漂移的力，比如足够强的朝 reference model 的 KL 惩罚（单独的 level 内 KL 不惩罚把概率整体移入通过 level），或者负迁移，即学会解可解的 prompt 反过来压低 exploit 倾向。

::takeoff

::damping

## 威胁模型与两个方向

把上面的结论放在一起，就得到语言模型 RL 中 reward hacking 的一个威胁模型。训练者只看得到 proxy $r$，更新也只看得到 reward：同一 reward level 内的 exploit 和真正的解答得到相同的权重（要点 1），所以在一个 level 内部，reward 不给更新任何区分两者的信息。在 binary verifier 下，level 内的 amplification 是 hack 占比增长的唯一途径（要点 2），而在没有正则项时目标函数在两者之间不做选择（要点 4）。一种可能的最坏情况始于一个模型很少能解出的 prompt：在 GRPO 下，孤立的 exploit 拿到组里最大的 advantage，仅凭 reward 无法把它和孤立的真正成功区分开（要点 8）；通过共享参数，它的更新可能抬高其他 prompt 上与它梯度对齐的 exploit（要点 7），而且在 hack 还很少时这个过程会自我加速（要点 9）。由此引出两个方向，分别针对问题的一半。

**找到能区分 $r$ 和 $r^*$ 的信号．**　最坏的情况是零信息的情况：在一个 reward level 内部，reward 里没有任何东西能把 exploit 和真正的解答分开，剩下的保证只是对该 level 被放大得有多不均匀的界（要点 5）。任何在 $r$ 之外还携带 $r^*$ 信息的信号，比如 held-out 测试、第二个 verifier、CoT monitor 或者对激活做的 probe，都能打破这种对称，因为它让更新可以区别对待 reward 相同的回答。[[moya2026|Moya et al. (2026)]] 分析了其中一种信号，即对正确性的 audit 反馈，并证明在当前 policy 处，只要它压过 verifier 把概率推向错误回答的压力，就能降低被接受的错误、同时提高正确率。不过这样的信号要谨慎使用：一旦把它当作优化目标，policy 就可能学会规避它 ([[baker2025|Baker et al., 2025]])。

**不管是不是 hack，都限制单个样本对 policy 的影响．**　第二个方向完全不去辨别哪些样本是 hack。既然更新本来就分不清 reward 相同的 exploit 和真正的解答，不如把更新设计成：无论某个样本是哪一种，它都不能把 policy 推得太远。在 verifier 下，结果本来就由学习动力学决定（要点 4），这样一来，不管 exploit 藏在哪里，它造成的累计损害都可能受到限制。这是一个关于方法的假说，而且还需要配合对累计漂移的预算：每一步影响小，本身并不能保证总漂移小。模型也指出了着力点：孤立成功拿到的最大 advantage（要点 8），把得到奖励的回答从一个 prompt 带到其他 prompt 的迁移（要点 7），以及 level 内散度，惩罚项会把它和获得的 reward 绑在一起，对任何网络、任何 prompt 配比都成立，但按训练权重和通过率加权（要点 6）。代价是真正的孤立成功也会被放慢。而且在模型里，只做压制只能推迟 take-off；要阻止它，需要一个在倾向还小时就把它拉回来、并强到足以抵消那里漂移的力，比如足够强的朝 reference model 的 KL 拉力，或者来自可解 prompt 的负迁移（要点 9）。

两个方向有望互补，因为它们各自弥补对方的一个短板，不过本文并没有证明结合起来就是最优的。信号很少是完整的，一旦被优化还可能被规避，限制单个样本对 policy 影响的做法可以兜住它漏掉的部分；反过来，这些限制会把真正的孤立成功和 exploit 一起放慢，而有信号的地方，就可以据此放松限制。

这些结论都没有回答学会 hack 会对模型的其余部分产生什么影响 ([[macdiarmid2025|MacDiarmid et al., 2025]])，而且描述的都是一个理想化估计量在期望意义下的更新。
`;

const switchClass =
  'underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';

const RewardHackingFormalized: React.FC = () => {
  const [lang, setLang] = React.useState<Lang>('en');

  return (
    <article className="animate-fade-in pt-10 pb-16">
      <h1 className="text-xl font-serif text-anthropic-text leading-tight text-center">{TITLE[lang]}</h1>
      <p className="mt-2 text-center text-[12px] text-anthropic-gray/80">11.6.2026</p>
      <div className="mt-1.5 flex justify-center gap-x-4 text-[12px] text-anthropic-gray">
        {(['en', 'zh'] as Lang[]).map((l) =>
          l === lang ? (
            <span key={l} className="text-anthropic-text">
              {l === 'en' ? 'English' : '中文'}
            </span>
          ) : (
            <button key={l} onClick={() => setLang(l)} className={switchClass}>
              {l === 'en' ? 'English' : '中文'}
            </button>
          ),
        )}
      </div>

      <div className="mt-8 space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <PostText
          source={lang === 'en' ? EN : ZH}
          links={LINKS}
          figures={{
            classic: <ClassicExamples lang={lang} />,
            lm: <LanguageModelExamples lang={lang} />,
            wg: <GrpoWeightPlot lang={lang} />,
            takeoff: <TakeoffPlot lang={lang} />,
            damping: <DampingPlot lang={lang} />,
          }}
        />

      </div>
    </article>
  );
};

export default RewardHackingFormalized;
