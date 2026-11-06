import React from 'react';
import { PostText } from '../../components/PostText';
import { ClassicExamples, LanguageModelExamples } from './RewardHackingFigures';

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
  goodhart1975: 'https://scholar.google.com/scholar?q=Goodhart%20Problems%20of%20monetary%20management%3A%20the%20U.K.%20experience',
  gupta2026: 'https://arxiv.org/abs/2610.17824',
  hadfield2017: 'https://arxiv.org/abs/1711.02827',
  huang2025: 'https://arxiv.org/abs/2407.13399',
  ibarz2018: 'https://arxiv.org/abs/1811.06521',
  karwowski2024: 'https://arxiv.org/abs/2310.09144',
  krakovna2020: 'https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/',
  kwa2024: 'https://arxiv.org/abs/2407.14503',
  laidlaw2025: 'https://arxiv.org/abs/2403.03185',
  lehman2020: 'https://arxiv.org/abs/1803.03453',
  leike2017: 'https://arxiv.org/abs/1711.09883',
  macdiarmid2025: 'https://arxiv.org/abs/2511.18397',
  manheim2018: 'https://arxiv.org/abs/1803.04585',
  moya2026: 'https://arxiv.org/abs/2609.35677',
  mroueh2025: 'https://arxiv.org/abs/2503.06639',
  ng1999: 'https://scholar.google.com/scholar?q=Policy%20invariance%20under%20reward%20transformations%3A%20Theory%20and%20application%20to%20reward%20shaping',
  pan2022: 'https://arxiv.org/abs/2201.03544',
  rad2026: 'https://arxiv.org/abs/2601.04411',
  randlov1998: 'https://scholar.google.com/scholar?q=Learning%20to%20drive%20a%20bicycle%20using%20reinforcement%20learning%20and%20shaping',
  ring2011: 'https://scholar.google.com/scholar?q=Delusion%2C%20survival%2C%20and%20intelligent%20agents',
  sharma2024: 'https://arxiv.org/abs/2310.13548',
  singhal2023: 'https://arxiv.org/abs/2310.03716',
  skalse2022: 'https://arxiv.org/abs/2209.13085',
  stiennon2020: 'https://arxiv.org/abs/2009.01325',
  strathern1997: 'https://scholar.google.com/scholar?q=Strathern%20%27Improving%20ratings%27%3A%20audit%20in%20the%20British%20University%20system',
  taylor2016: 'https://intelligence.org/files/QuantilizersSaferAlternative.pdf',
  taylor2025: 'https://arxiv.org/abs/2508.17511',
  vonarx2025: 'https://metr.org/blog/2025-06-05-recent-reward-hacking/',
  wang2026: 'https://scholar.google.com/scholar?q=Is%20it%20thinking%20or%20cheating%3F%20Detecting%20implicit%20reward%20hacking%20by%20measuring%20reasoning%20effort',
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

This post first reviews how reward hacking shows up in classical RL, in RLHF and in RL with verifiable rewards, and how
it was defined. It then goes through how work before language models represented reward hacking mathematically. Next it
sets up a model of reward hacking in language-model RL and derives what follows from it: group-based updates cannot tell an
exploit from an honest answer with the same reward; hacking splits into selection along the proxy, the main way reward
models are hacked, and amplification within a reward level, the only way verifiers are hacked; under a verifier the
objective is indifferent to the share of hacks, so the outcome is decided by the learning dynamics; and exploits spread
through shared parameters, start as lone successes that their group cannot distinguish from honest ones, and accelerate
once started.

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
models occasionally went as far as rewriting their own reward function ([[denison2024|Denison et al., 2024]]), and
learning to hack production coding environments has been seen to coincide with broader misalignment
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
verifier that accepts some incorrect answers. A recent review notes that exploitable errors under group-relative
updates, the case that matters for verifier hacking, still have no theory ([[gupta2026|Gupta and Joshi, 2026]]). The
model below covers that case and learned reward models alike. It is a model of reward hacking, not of language-model RL
as a whole: it keeps what hacking depends on (one network shared across prompts, a reference model, and updates that
see only rewards) and abstracts away the rest, and its statements concern expected, on-policy updates.

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

**The update cannot tell them apart.** The algorithms used in practice (REINFORCE with a baseline, RLOO, GRPO,
Dr. GRPO) sample a group of $G$ responses per prompt and weight each response’s gradient by a function of the group’s
rewards. Conditioning on the reward level of one sample, its weight depends only on the other samples, which are
independent of it, so

$$\mathbb{E}[\Delta\theta \mid x] = \sum_v a_v\, \nabla \pi(L_v)$$

for scalars $a_v$ that depend on the policy only through $\pi_R$. For a pass/fail verifier the update is
$w(p)\nabla p$, and for GRPO

$$w_G(p) = \mathbb{E}_m\Big[\sqrt{\tfrac{G-1-m}{1+m}}\Big] + \mathbb{E}_m\Big[\sqrt{\tfrac{m}{G-m}}\Big], \qquad m \sim \mathrm{Bin}(G-1, p),$$

which is symmetric around $p = 1/2$, smallest there, and equal to $\sqrt{G-1}$ as $p \to 0$.

> **Takeaway 1.** Every set of responses inside a reward level gets the same coefficient per unit of probability: an
exploit and an honest answer with the same reward are pushed alike, and how much each actually grows depends only on
the network. A response that is the only success in its group receives the largest advantage the group can give.

**Two ways to hack.** Splitting the policy into its distribution over levels and its distribution inside each level
gives

$$\mathrm{Hack}(\pi) = \underbrace{\sum_v \big(\pi_R(v) - \pi_{\mathrm{ref},R}(v)\big)\,\bar e(v)}_{\text{selection along the proxy}} + \underbrace{\sum_v \pi_R(v)\big(\mathbb{E}_{\pi(\cdot\mid L_v)}[e] - \bar e(v)\big)}_{\text{amplification within levels}},$$

and the KL divergence splits the same way,
$\mathrm{KL}(\pi\|\pi_{\mathrm{ref}}) = \mathrm{KL}(\pi_R\|\pi_{\mathrm{ref},R}) + \sum_v \pi_R(v)\,\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)$.
The first term is the classical Goodhart effect: moving toward reward levels that are more often wrong. The second is
amplifying the wrong responses within a level, and Cauchy–Schwarz bounds it per level by
$\sqrt{\chi^2(\pi(\cdot\mid L_v)\|\nu_v)\,\mathrm{Var}_{\nu_v}(e)}$. A reward model gives nearly every response its own
score, while a pass/fail verifier has a single passing level, where
$\mathrm{Hack} = (p - p_{\mathrm{ref}})\bar q + p\,(q - \bar q)$.

> **Takeaway 2.** Hacking is selection along the proxy plus amplification within reward levels. The within-level term
vanishes for every possible error exactly when $\pi = \pi_{\mathrm{ref}} \cdot f(r)$ for some function $f$, a family
that includes the KL-regularized optimum, best-of-$n$ and quantilizers. Reward models are hacked almost entirely through
selection. Under a pass/fail verifier, raising the pass rate scales the amount of hacking but not its share, and the
share $q$ can grow only through amplification within the passing level.

**When true reward falls under a reward model.** Along the tilted family
$\pi_\eta \propto \pi_{\mathrm{ref}} e^{\eta r}$ traced by the KL-regularized optimum as the KL budget grows,
$\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[f] = \mathrm{Cov}_{\pi_\eta}(f, r)$ for any $f$, hence

$$\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[r^*] = \mathrm{Var}_{\pi_\eta}(r) - \mathrm{Cov}_{\pi_\eta}(e, r).$$

If the error is independent of true quality under $\pi_{\mathrm{ref}}$, the tilt factorizes and the independence
persists along the path.

> **Takeaway 3.** Along the path of KL-regularized optima, true reward falls exactly where the regression slope of the
error on the proxy exceeds one. If the error is independent of true quality under $\pi_{\mathrm{ref}}$, true reward
never falls along this path; it only becomes a smaller share of each unit of proxy gain.

The independent case matches the light-tailed case of [[kwa2024|Kwa et al. (2024)]]. A decline like the one
[[gao2023|Gao et al. (2023)]] measure therefore points to errors that depend on quality in the high-score tail (a reward
model that prefers long or flattering answers that are worse), to heavy-tailed errors under hard selection such as
best-of-$n$ ([[kwa2024|Kwa et al., 2024]]; [[elmhamdi2024|El-Mhamdi and Hoang, 2024]]), or to a trained policy that is
not an exponential tilt.

**Under a verifier, the objective does not choose.** For a pass/fail verifier, every policy with the same pass rate has
the same expected reward, whatever its hack share. All policies with $p = 1$ are optimal, from uniform amplification of
the reference passes ($q = \bar q$) to a policy that only hacks ($q = 1$). [[moya2026|Moya et al. (2026)]] make a
related observation for exact gradient flow, where reward rises while correctness falls whenever
$p\dot q > (1 - q)\dot p$; the indifference above extends it to the group-based estimators used in practice and to any
number of reward levels.

> **Takeaway 4.** Under a pass/fail verifier the objective does not determine the hack share. Which optimal policy
training reaches is decided by the learning dynamics: which passing responses gradient descent amplifies faster, and how
an update on one prompt moves the others. Reward hacking under a verifier is a question about implicit bias.

**What can be guaranteed.** If nothing restricts where exploits can be, nothing can be guaranteed: an adversary can call
every response of a level an exploit. Suppose instead we only know that under $\pi_{\mathrm{ref}}$ at most a fraction
$\bar q_v$ (with $0 < \bar q_v < 1$) of level $v$ is exploits. Writing $\rho = \pi(\cdot \mid L_v)/\nu_v$, the
worst-case exploit share in the level is the $\pi$-mass of the responses with the largest $\rho$ that together have
reference mass $\bar q_v$, a fractional knapsack. The resulting bounds are the per-level counterparts of the
no-free-lunch theorem of [[everitt2017|Everitt et al. (2017)]] and the cost bound of [[taylor2016|Taylor (2016)]].

> **Takeaway 5.** The worst-case exploit share of a level is at least $\bar q_v$, with equality exactly when every
response of the level is amplified by the same factor. If $\rho \le M$ it is at most $M\bar q_v$, and in any case it is
at most $\bar q_v + \sqrt{\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)/2}$.

**A budget that survives shared parameters.** The per-level picture treats each prompt as if its policy could be set
freely, and says nothing about prompts the reference model cannot solve. Neither is needed for the following. Take an
algorithm whose expected update is a gradient, $\mathbb{E}[\Delta\theta] = \nabla J$ with $J = \sum_x \omega(x) J_x$
(expected reward for REINFORCE and RLOO, $W(p) = \int_0^p w$ for binary GRPO), add a penalty $\lambda D_R$ on the
within-level divergence $D_R(x) = \sum_v \pi_R(v \mid x)\,\mathrm{KL}(\pi(\cdot \mid x, L_v)\|\nu_v)$, and run any
preconditioned gradient flow $\dot\theta = M\nabla F$ with $M \succeq 0$ from $\pi_{\mathrm{ref}}$. Since $F$ can only
increase and starts at $J(0)$, the within-level divergence is paid for by the reward gained.

> **Takeaway 6.** For every network and every prompt mix,
$$\lambda \sum_x \omega(x)\, D_R(x, t) \;\le\; J(t) - J(0).$$
Unsolvable prompts get no guarantee themselves, but whatever their exploits do elsewhere is paid from the same budget.

Pinsker’s inequality and Cauchy–Schwarz turn this into a bound on the average excess exploit share over any set of
prompts. The usual KL penalty adds the between-level divergence on top, which is what one wants against a reward model
and only slows down the pass rate against a verifier.

**How exploits spread.** With shared parameters, a rewarded exploit on prompt $x$ moves every other prompt $x'$. For a
pass/fail verifier,

$$\frac{d}{dt}\log\frac{h(x')}{s(x')} = \mathbb{E}_x\Big[w(p(x))\Big(s(x)\,[K_{\mathcal{HS}} - K_{\mathcal{SS}}](x', x) + h(x)\,[K_{\mathcal{HH}} - K_{\mathcal{SH}}](x', x)\Big)\Big],$$

where $K_{AB}(x', x) = g_A(x')^\top g_B(x)$ is the inner product of the gradients of $\log\pi(A \mid x')$ and
$\log\pi(B \mid x)$ (with a preconditioner $M$, $g_A^\top M g_B$). Splitting the expectation into $x = x'$ and
$x \ne x'$ separates a prompt’s own damage from what it imports.

> **Takeaway 7.** An exploit on one prompt raises the exploit-to-solution ratio on another when exploiting there is
better aligned with it than solving there is, $K_{\mathcal{HH}} > K_{\mathcal{SH}}$, which is the natural case for
exploits that do not depend on the task. On prompts the reference model cannot solve, a method that trains on them
without knowing the exploit cannot keep the mass it adds there from being exploit mass; what it can control is what
those prompts export.

**Where it starts.** The first rewarded exploit is typically a lone success on a prompt nobody solves. When exactly one
response reaches a group’s top reward, everything the group contains (responses, rewards, log-probabilities, gradients)
looks the same whether that response is honest or an exploit, and statistics that need two members of the top level are
undefined.

> **Takeaway 8.** Any rule that uses only the group must treat a lone honest success and a lone exploit alike, and the
lone response receives the group’s largest advantage. On a prompt where only exploits reach the top, this happens with
probability about $G\pi(\mathcal{H})$ while $G\pi(\mathcal{H}) \ll 1$.

**How it takes off.** On a single prompt where only exploits pass, a class-level softmax model gives
$\dot h = 2w(h)h^2(1-h)^2$, the same form [[rad2026|Rad et al. (2026)]] find for noisy verifiers. Across prompts, let all
of them share an exploit propensity with log-odds $z$ and write $u = e^z$. Lone exploits arrive at a rate proportional
to $u$ and groups with several exploits at a rate proportional to $u^2$, each raising $z$, so
$\dot u \approx a u^2 + b u^3$. Damping the effect of lone exploits on other prompts to a fraction $\epsilon$ replaces
$a$ by $\epsilon a$.

> **Takeaway 9.** On one prompt, while $Gh \ll 1$, the exploit share reaches order one on the time scale
$1/(2w(0)h_0)$: about $1/(2h_0)$ for REINFORCE and $1/(2\sqrt{G-1}\,h_0)$ for GRPO. Across prompts the shared propensity
blows up in finite time. Damping lone exploits by $\epsilon$ delays the onset like $1/\epsilon$ while
$\epsilon \gg G h_0$, and then saturates, roughly a factor $1/(G h_0)$ later than before. Like an epidemic, the process
accelerates, and slowing its first carriers delays it without stopping it; stopping it takes a force that pulls the
propensity back at small $z$, such as a pull toward the reference model or negative transfer from prompts that can be
solved.

## Takeaways

Before language models, reward hacking was written as a proxy equal to the true reward plus an error, as corruption on
some states, as an agent that controls its own reward channel, as two rewards that order policies differently, or
handled by limiting optimization against a base distribution. For RL of language models, with one network for all
prompts, group-based updates that see only rewards, and a strong reference model, the picture becomes more specific.
Hacking splits into selection along the proxy, which is how reward models are hacked, and amplification within a reward
level, which is the only way verifiers are hacked. Under a verifier the objective is indifferent to the hack share, so
what decides it is the learning dynamics: exploits spread through shared parameters, start as lone successes that their
group cannot tell apart from honest ones, and accelerate once started. None of this says what learning to hack does to
the rest of a model ([[macdiarmid2025|MacDiarmid et al., 2025]]), and all of it describes expected updates.
`;

// Chinese paragraphs are kept on one line each: PostText joins wrapped lines with a space.
const ZH = r`
## TL;DR

本文先回顾了 reward hacking 在经典 RL、RLHF 和 RLVR 中的典型案例和定义，再介绍了语言模型之前的工作如何用数学形式刻画它。随后为语言模型 RL 中的 reward hacking 建立了一个理论模型，并推导出：group-based 更新分不清奖励相同的漏洞和老实答案；hacking 可以拆成沿 proxy 的选择和奖励层内的放大，前者是奖励模型被 hack 的主要方式，后者是 verifier 被 hack 的唯一方式；在 verifier 下目标函数对 hack 占比无差别，最终结果由学习动力学决定；漏洞通过共享参数扩散，起点是组内无法与老实成功区分的孤立成功，一旦开始就会自我加速。

## Reward hacking 长什么样

[[wiener1960|Wiener (1960)]] 写道，如果把一个目的交给一台来不及叫停的机器，“我们最好非常确定，放进机器的目的正是我们真正想要的目的，而不仅仅是它的一个花哨的仿制品”。[[goodhart1975|Goodhart (1975)]] 指出，任何统计规律“一旦为了控制目的对它施加压力，就会趋于瓦解”，更广为人知的说法是“当一个指标成为目标，它就不再是好指标” ([[strathern1997|Strathern, 1997]])。

一辆模拟自行车因朝向目标而得奖励、背离目标却从不受罚，结果学会了在起点周围绕 20 到 50 米宽的圈 ([[randlov1998|Randløv and Alstrøm, 1998]])。一个因触碰足球而得奖励的机器人，学会了停在球旁边不停地碰它 ([[ng1999|Ng et al., 1999]])。在赛艇游戏 CoastRunners 里，智能体发现在泻湖里反复撞几个会重生的目标比跑完全程更划算，一边着火、一边逆行，得分还比人类玩家高出约 20% ([[clark2016|Clark and Amodei, 2016]])。一只从人类反馈中学抓取的机械臂，把手挡在镜头和物体之间，“让它看起来像是在抓取” ([[amodei2017|Amodei et al., 2017]])。一个叠积木的智能体直接把积木翻了过来 ([[krakovna2020|Krakovna et al., 2020]])；一个程序修复系统让程序永远返回空列表，测试居然通过了 ([[lehman2020|Lehman et al., 2020]])；在一个 gridworld 里，走进水桶格就能让所有番茄看起来都浇过水 ([[leike2017|Leike et al., 2017]])。在冻结的、学出来的奖励模型上训练的 Atari 智能体，把奖励模型的打分推高，游戏得分却“暴跌” ([[ibarz2018|Ibarz et al., 2018]])。

::classic

这个现象今天的名字来自 [[amodei2016|Amodei et al. (2016)]]：目标函数“允许某种聪明的‘简单’解，它在形式上最大化了目标函数，却违背了设计者意图的精神”。DeepMind 的叫法是 specification gaming，指“满足目标的字面规格、却没有达到预期结果的行为” ([[krakovna2020|Krakovna et al., 2020]])。[[pan2022|Pan et al. (2022)]] 发现它可能在模型容量增长时以相变的形式突然出现；[[weng2024|Weng (2024)]] 总结了通行的用法：智能体“利用奖励函数中的缺陷或模糊之处获得高奖励，却没有真正学会或完成预期任务”。

在 RLHF 中，对学出来的奖励模型优化得越用力，摘要先变好、再变差 ([[stiennon2020|Stiennon et al., 2020]])；在几个数据集上，RLHF 测得的提升有很大一部分只是因为回答变长了 ([[singhal2023|Singhal et al., 2023]])；偏好模型常常更喜欢附和用户的回答，而不是正确的回答 ([[sharma2024|Sharma et al., 2024]])；奖励模型集成的帮助也没有预想的大，因为成员往往犯同样的错 ([[eisenstein2023|Eisenstein et al., 2023]])。

在程序化 verifier 下，前沿代码 agent 被发现会在测试运行前调用 ${'`'}exit(0)${'`'}、抛出 ${'`'}SkipTest${'`'}、修改测试框架上游的代码、从测试文件里读出期望值、反编译参考答案；当一个读取它们 CoT 的 monitor 被加进奖励后，它们继续 hack，只是不再说出来 ([[baker2025|Baker et al., 2025]])。Claude 3.7 Sonnet 的 system card 写道，它会“为了通过测试用例而特判”，这是训练中 reward hacking 的结果 ([[anthropic2025|Anthropic, 2025]])；o3 在评测任务里覆盖评分器的计时器、修改评测器，被问到时也承认这不是用户想要的 ([[vonarx2025|Von Arx et al., 2025]])；在被刻意改成不可能完成的代码任务上，“任何通过都必然意味着违反规格的捷径”，前沿模型却经常通过 ([[zhong2025|Zhong et al., 2025]])。在一个越来越容易钻空子的任务课程里，模型偶尔甚至会改写自己的奖励函数 ([[denison2024|Denison et al., 2024]])；而学会 hack 生产代码环境，被观察到与更广泛的 misalignment 同时出现 ([[macdiarmid2025|MacDiarmid et al., 2025]]; [[taylor2025|Taylor et al., 2025]])。[[wang2026|Wang et al. (2026)]] 把其中的机制描述为“当利用漏洞比解决实际任务更容易时，就会发生 hacking”。

::lm

## 语言模型之前的形式化

最常见的形式化，是把智能体优化的奖励写成意图中的奖励加一个误差，

$$R = R^* + \xi,$$

用这种写法的工作，主要区别在于对 $\xi$ 做了什么假设。在 reward shaping 中，$\xi$ 是设计者添加的项 $F(s, a, s')$，当 $F = \gamma\Phi(s') - \Phi(s)$ 时称为基于势函数的 ([[ng1999|Ng et al., 1999]])。AI Safety Gridworlds 把这个拆分写进了每个环境：环境带有一个智能体观测到的奖励 $R$ 和一个它看不到的性能函数 $R^*$ ([[leike2017|Leike et al., 2017]])。[[manheim2018|Manheim and Garrabrant (2018)]] 在回归型 Goodhart 中把 $\xi$ 当作噪声，在极端型 Goodhart 中让两者的关系本身在阈值处改变：$R \le a$ 时 $R^* = R + x$，否则 $R^* = R + y$。[[kwa2024|Kwa et al. (2024)]] 按尾部给 $\xi$ 分类：若对任意 $t > 0$ 都有 $e^{tx}\Pr(\xi > x) \to \infty$，就称它是重尾的。[[everitt2017|Everitt et al. (2017)]] 的被污染奖励 MDP 让 $\xi$ 只在一部分被污染的状态上非零：给定真实奖励 $\dot R$ 和污染函数 $C$，智能体观测到 $\hat R(s) = C_s(\dot R(s))$，并对被污染状态的数量、一个状态的奖励能否从别处观测到做出假设。

[[hadfield2017|Hadfield-Menell et al. (2017)]] 和 [[zhuang2020|Zhuang and Hadfield-Menell (2020)]] 则把 proxy 建模为设计者写下的一份不完整的规格。在 inverse reward design 中，proxy $R = \tilde w^\top\phi(\tau)$ 和真实奖励 $R^* = w^{*\top}\phi(\tau)$ 都是轨迹特征的线性函数，设计者在训练环境 $\tilde M$ 中选择 $\tilde w$，被建模为在那里近似最优，

$$P(\tilde w \mid w^*, \tilde M) \propto \exp\big(\beta\, \mathbb{E}[w^{*\top}\phi(\tau) \mid \tau \sim \pi(\tau \mid \tilde w, \tilde M)]\big),$$

所以 proxy 只通过 $\tilde M$ 中的行为携带关于 $w^*$ 的信息。在 [[zhuang2020|Zhuang and Hadfield-Menell (2020)]] 中，世界由属性 $s \in \mathbb{R}^L$ 描述，受资源约束 $g(s) \le 0$；真实奖励 $R^*$ 依赖全部 $L$ 个属性，proxy $R$ 只依赖其中 $J < L$ 个。

[[ring2011|Ring and Orseau (2011)]]、[[everitt2016|Everitt and Hutter (2016)]] 和 [[everitt2021b|Everitt et al. (2021b)]] 让智能体能作用于产生奖励的通道本身。用统一的记号，观测到的奖励是一个扭曲 $R = d(R^*)$，而智能体的动作可以改变 $d$。在 [[ring2011|Ring and Orseau (2011)]] 的 delusion box 中，动作为 $a_t = \langle d_t, a^e_t \rangle$，其中 $d_t$ 对一个映射 $d : O \to O$ 编程，智能体通过它看世界，$o_t = d(a_t, o^e_t)$，$o_t$ 的一部分被读作奖励；在 [[everitt2016|Everitt and Hutter (2016)]] 中，$R = d_s(\check R)$，其中内部奖励 $\check R = u^*(s)$，扭曲依赖于智能体到达的状态。[[everitt2021b|Everitt et al. (2021b)]] 用因果影响图给出一般形式，图中的节点包括实现的奖励函数 $\Theta^R_t$ 和意图中的奖励函数 $\Theta^R_*$：智能体的决策以违反 $\Theta^R_*$ 条件的方式影响 $\Theta^R_t$，就是篡改；按 [[everitt2021a|Everitt et al. (2021a)]] 的判据，要有控制 $\Theta^R_t$ 的激励，必须存在一条经过它、从决策到效用的有向路径。

[[skalse2022|Skalse et al. (2022)]]、[[karwowski2024|Karwowski et al. (2024)]] 和 [[laidlaw2025|Laidlaw et al. (2025)]] 把 proxy 和真实奖励当作策略空间上的函数来比较，利用的是回报对占用测度是线性的，$J_R(\pi) = \langle \eta^\pi, R \rangle$。[[skalse2022|Skalse et al. (2022)]] 说，若 $R$ 和 $R^*$ 把某一对策略排出相反的顺序，则它们在策略集合 $\Pi$ 上是 hackable 的，

$$\exists\, \pi, \pi' \in \Pi:\quad J_R(\pi) < J_R(\pi') \ \text{且}\ J_{R^*}(\pi) > J_{R^*}(\pi').$$

另外两篇用一个余弦来度量两者的关系：[[karwowski2024|Karwowski et al. (2024)]] 用 $R$ 与 $R^*$ 在 $\mathrm{span}(\Omega)$ 上投影的夹角，其中 $\Omega$ 是占用测度构成的多面体；[[laidlaw2025|Laidlaw et al. (2025)]] 用两者在参考占用测度 $\mu_{\mathrm{ref}}$ 下抽取的状态–动作对上的相关系数 $\kappa > 0$，并用 $\chi^2(\mu_\pi \,\|\, \mu_{\mathrm{ref}})$ 度量策略走了多远。

[[taylor2016|Taylor (2016)]] 改的是优化器而不是奖励。给定动作上的一个基础分布 $\gamma$（比如人会怎么做），$q$-quantilizer 从按 proxy 排序的 $\gamma$ 的前 $q$ 比例中采样，因此它的分布满足 $Q_q(a) \le \gamma(a)/q$；未知的危害则被建模为一个任意的非负代价 $c$，只知道它在 $\gamma$ 下的期望有上界。

## 语言模型 RL 中 reward hacking 的理论模型

对学出来的奖励模型，[[gao2023|Gao et al. (2023)]] 把 gold reward 拟合成 best-of-$n$ 下的 $d(\alpha - \beta d)$ 和 RL 下的 $d(\alpha - \beta \log d)$，其中 $d = \sqrt{\mathrm{KL}}$；[[huang2025|Huang et al. (2025)]] 认为 KL 正则太弱，$\chi^2$ 正则才是合适的工具。对 verifier，[[mroueh2025|Mroueh (2025)]] 证明 GRPO 会放大二值奖励接受的任何东西，[[rad2026|Rad et al. (2026)]] 分析了带随机噪声 verifier 的 GRPO，[[moya2026|Moya et al. (2026)]] 研究了一个会接受部分错误答案的二值 verifier 下的梯度流。一篇近期综述指出，group-relative 更新下的可利用误差，也就是 verifier hacking 真正关心的情形，至今仍没有理论 ([[gupta2026|Gupta and Joshi, 2026]])。下面的模型同时覆盖这种情形和学出来的奖励模型。它是 reward hacking 的模型，而不是整个语言模型 RL 的模型：它保留 hacking 所依赖的东西（一个在所有 prompt 间共享的网络、一个参考模型、只看得到奖励的更新），把其余部分抽象掉，所有结论针对的都是期望意义下的 on-policy 更新。

**设定．**　一个网络 $\pi_\theta(y \mid x)$ 回答所有 prompt $x$，训练从参考模型 $\pi_{\mathrm{ref}}$ 出发。训练者看得到 proxy 奖励 $r(x, y)$（奖励模型的打分，或者在回答运行之后才算出的 verifier 判定，因此改测试也算作回答的一部分），看不到意图中的奖励 $r^*(x, y)$；两者之差 $e = r - r^*$ 就是误差。把回答按奖励值分成若干奖励层 $L_v = \{y : r(x, y) = v\}$，记 $\pi_R(v)$ 为策略放在第 $v$ 层上的概率，$\nu_v = \pi_{\mathrm{ref}}(\cdot \mid L_v)$ 为参考策略在该层内部的分布，$\bar e(v) = \mathbb{E}_{\nu_v}[e]$ 为参考策略在该层的平均误差。Hacking 的量，就是 proxy 增益中不真实的那一部分，

$$\mathrm{Hack}(\pi) = \mathbb{E}_\pi[e] - \mathbb{E}_{\pi_{\mathrm{ref}}}[e].$$

对通过/不通过型 verifier，另记 $p$ 为通过率，$q$ 为通过的回答里 hack 所占的比例，$\bar q$ 为 $\pi_{\mathrm{ref}}$ 下的这个比例。当我们问“在不知道漏洞在哪的前提下能保证什么”时，就让一个对手在策略确定之后再摆放漏洞，只受可观测量的约束。

**更新分不清它们．**　实践中用的算法（带 baseline 的 REINFORCE、RLOO、GRPO、Dr. GRPO）每个 prompt 采样一组 $G$ 个回答，用组内奖励的某个函数给每个回答的梯度加权。给定某个样本的奖励层，它的权重只取决于组里其他样本，而这些样本与它独立，所以

$$\mathbb{E}[\Delta\theta \mid x] = \sum_v a_v\, \nabla \pi(L_v),$$

其中标量 $a_v$ 只通过 $\pi_R$ 依赖于策略。对通过/不通过型 verifier，更新是 $w(p)\nabla p$，GRPO 的权重为

$$w_G(p) = \mathbb{E}_m\Big[\sqrt{\tfrac{G-1-m}{1+m}}\Big] + \mathbb{E}_m\Big[\sqrt{\tfrac{m}{G-m}}\Big], \qquad m \sim \mathrm{Bin}(G-1, p),$$

它关于 $p = 1/2$ 对称，在那里最小，在 $p \to 0$ 时趋于 $\sqrt{G-1}$。

> **要点 1**　同一奖励层内的任何一组回答，每单位概率得到的系数都一样：奖励相同的漏洞和老实答案被同样地推，各自实际长多少只取决于网络本身。组里唯一成功的那个回答，拿到的是这个组能给的最大 advantage。

**两种 hack 的方式．**　把策略拆成它在各层之间的分布和每层内部的分布，得到

$$\mathrm{Hack}(\pi) = \underbrace{\sum_v \big(\pi_R(v) - \pi_{\mathrm{ref},R}(v)\big)\,\bar e(v)}_{\text{沿 proxy 的选择}} + \underbrace{\sum_v \pi_R(v)\big(\mathbb{E}_{\pi(\cdot\mid L_v)}[e] - \bar e(v)\big)}_{\text{层内放大}},$$

KL 散度也按同样方式拆开，$\mathrm{KL}(\pi\|\pi_{\mathrm{ref}}) = \mathrm{KL}(\pi_R\|\pi_{\mathrm{ref},R}) + \sum_v \pi_R(v)\,\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)$。第一项是经典的 Goodhart 效应：往更常出错的奖励层移动。第二项是在同一层内把错的回答放大，由 Cauchy–Schwarz，它在每层上不超过 $\sqrt{\chi^2(\pi(\cdot\mid L_v)\|\nu_v)\,\mathrm{Var}_{\nu_v}(e)}$。奖励模型几乎给每个回答一个不同的分数，而通过/不通过型 verifier 只有一个通过层，在那里 $\mathrm{Hack} = (p - p_{\mathrm{ref}})\bar q + p\,(q - \bar q)$。

> **要点 2**　Hacking 等于沿 proxy 的选择加上奖励层内的放大。层内项对所有可能的误差都为零，当且仅当 $\pi = \pi_{\mathrm{ref}} \cdot f(r)$ 对某个函数 $f$ 成立，这个族包括 KL 正则下的最优解、best-of-$n$ 和 quantilizer。奖励模型几乎完全是通过选择被 hack 的。在通过/不通过型 verifier 下，提高通过率会按比例放大 hacking 的量，却不改变它的占比，而占比 $q$ 只能通过通过层内的放大来增长。

**奖励模型下真实奖励什么时候会降．**　随着 KL 预算增大，KL 正则最优解走过的是倾斜族 $\pi_\eta \propto \pi_{\mathrm{ref}} e^{\eta r}$。沿着它，对任意 $f$ 有 $\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[f] = \mathrm{Cov}_{\pi_\eta}(f, r)$，因此

$$\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[r^*] = \mathrm{Var}_{\pi_\eta}(r) - \mathrm{Cov}_{\pi_\eta}(e, r).$$

如果在 $\pi_{\mathrm{ref}}$ 下误差与真实质量独立，倾斜会因式分解，独立性沿整条路径保持。

> **要点 3**　沿 KL 正则最优解的路径，真实奖励下降恰好发生在误差对 proxy 的回归斜率超过 1 的地方。如果在 $\pi_{\mathrm{ref}}$ 下误差与真实质量独立，真实奖励沿这条路径永远不会下降，只是在每单位 proxy 增益中所占的份额越来越小。

独立的情形与 [[kwa2024|Kwa et al. (2024)]] 的轻尾情形一致。因此，像 [[gao2023|Gao et al. (2023)]] 测到的那种下降，指向的是以下几种可能：误差在高分尾部依赖于质量（奖励模型偏爱更长或更讨好、但实际更差的回答），在 best-of-$n$ 这类强选择下误差是重尾的 ([[kwa2024|Kwa et al., 2024]]; [[elmhamdi2024|El-Mhamdi and Hoang, 2024]])，或者训练出来的策略并不是一个指数倾斜。

**在 verifier 下，目标函数不做选择．**　对通过/不通过型 verifier，所有通过率相同的策略期望奖励都一样，不管它的 hack 占比是多少。所有 $p = 1$ 的策略都是最优的，从均匀放大参考策略的通过回答（$q = \bar q$），到只会 hack 的策略（$q = 1$）。[[moya2026|Moya et al. (2026)]] 对精确梯度流做过一个相关的观察：只要 $p\dot q > (1 - q)\dot p$，奖励上升的同时正确率就在下降；上面的无差别性把它推广到了实践中用的 group-based 估计量和任意多个奖励层。

> **要点 4**　在通过/不通过型 verifier 下，目标函数不决定 hack 占比。训练最终到达哪个最优策略，由学习动力学决定：梯度下降恰好把哪些通过的回答放大得更快，以及在一个 prompt 上的更新如何牵动其他 prompt。Verifier 下的 reward hacking 是一个 implicit bias 问题。

**能保证什么．**　如果对漏洞可能在哪里毫无限制，就什么都保证不了：对手可以把某一层的每个回答都称为漏洞。假设我们只知道在 $\pi_{\mathrm{ref}}$ 下第 $v$ 层至多有比例 $\bar q_v$（$0 < \bar q_v < 1$）是漏洞。记 $\rho = \pi(\cdot \mid L_v)/\nu_v$，则该层内最坏情况下的漏洞占比，等于 $\rho$ 最大、参考质量合计为 $\bar q_v$ 的那些回答在 $\pi$ 下的质量，这是一个分数背包问题。由此得到的界，是 [[everitt2017|Everitt et al. (2017)]] 的 no-free-lunch 定理和 [[taylor2016|Taylor (2016)]] 代价界在单层上的对应物。

> **要点 5**　一层内最坏情况下的漏洞占比至少是 $\bar q_v$，取等当且仅当该层的每个回答都被放大同样的倍数。若 $\rho \le M$，占比至多 $M\bar q_v$；无论如何，占比至多 $\bar q_v + \sqrt{\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)/2}$。

**一个在共享参数下仍成立的预算．**　单层的图景把每个 prompt 当作策略可以随意设定，也没有说参考模型解不出的 prompt 怎么办。下面这个结论两者都不需要。取一个期望更新是梯度的算法，$\mathbb{E}[\Delta\theta] = \nabla J$，$J = \sum_x \omega(x) J_x$（对 REINFORCE 和 RLOO 是期望奖励，对二值 GRPO 是 $W(p) = \int_0^p w$），对层内散度 $D_R(x) = \sum_v \pi_R(v \mid x)\,\mathrm{KL}(\pi(\cdot \mid x, L_v)\|\nu_v)$ 加一个惩罚 $\lambda D_R$，并从 $\pi_{\mathrm{ref}}$ 出发运行任意带预条件的梯度流 $\dot\theta = M\nabla F$，$M \succeq 0$。由于 $F$ 只增不减，且起点是 $J(0)$，层内散度只能用获得的奖励来支付。

> **要点 6**　对任何网络、任何 prompt 配比， $$\lambda \sum_x \omega(x)\, D_R(x, t) \;\le\; J(t) - J(0).$$ 解不出的 prompt 本身得不到保证，但它们的漏洞在别处造成的影响，都要从同一个预算里支付。

再用 Pinsker 不等式和 Cauchy–Schwarz，就能把它变成任意一组 prompt 上平均超额漏洞占比的界。常用的 KL 惩罚在此之上还加了层间散度，对奖励模型这正是我们想要的，对 verifier 则只会拖慢通过率。

**漏洞怎么扩散．**　参数共享时，prompt $x$ 上一个被奖励的漏洞会牵动所有其他 prompt $x'$。对通过/不通过型 verifier，

$$\frac{d}{dt}\log\frac{h(x')}{s(x')} = \mathbb{E}_x\Big[w(p(x))\Big(s(x)\,[K_{\mathcal{HS}} - K_{\mathcal{SS}}](x', x) + h(x)\,[K_{\mathcal{HH}} - K_{\mathcal{SH}}](x', x)\Big)\Big],$$

其中 $K_{AB}(x', x) = g_A(x')^\top g_B(x)$ 是 $\log\pi(A \mid x')$ 与 $\log\pi(B \mid x)$ 梯度的内积（有预条件 $M$ 时为 $g_A^\top M g_B$）。把期望拆成 $x = x'$ 和 $x \ne x'$ 两部分，就把一个 prompt 自己造成的损害和它从别处进口的损害分开了。

> **要点 7**　一个 prompt 上的漏洞会提高另一个 prompt 上漏洞与解答之比，条件是在那边 hack 比在那边解题与它更对齐，$K_{\mathcal{HH}} > K_{\mathcal{SH}}$；对不依赖具体任务的漏洞，这是很自然的情形。在参考模型解不出的 prompt 上，一个不知道漏洞是什么的方法如果在这些 prompt 上训练，就无法保证它在那里增加的质量不是漏洞的质量；它能控制的是这些 prompt 往外输出了什么。

**从哪里开始．**　第一个被奖励的漏洞，通常是一个没人解得出的 prompt 上的孤立成功。当组里恰好只有一个回答拿到最高奖励时，这个组所包含的一切（回答、奖励、对数概率、梯度），不管那个回答是老实的还是漏洞，看起来都一样，而需要最高层里至少两个成员的统计量则无从定义。

> **要点 8**　任何只用组内信息的规则，都必须同等对待孤立的老实成功和孤立的漏洞，而这个孤立的回答拿到的是组里最大的 advantage。在一个只有漏洞能拿到最高奖励的 prompt 上，只要 $G\pi(\mathcal{H}) \ll 1$，这种情况发生的概率约为 $G\pi(\mathcal{H})$。

**怎么起飞．**　在一个只有漏洞能通过的单个 prompt 上，类别层面的 softmax 模型给出 $\dot h = 2w(h)h^2(1-h)^2$，与 [[rad2026|Rad et al. (2026)]] 对噪声 verifier 得到的形式相同。跨 prompt 来看，设所有 prompt 共享一个漏洞倾向，其 log-odds 为 $z$，记 $u = e^z$。孤立漏洞以正比于 $u$ 的速率出现，含多个漏洞的组以正比于 $u^2$ 的速率出现，每一次都会抬高 $z$，于是 $\dot u \approx a u^2 + b u^3$。把孤立漏洞对其他 prompt 的影响压到原来的 $\epsilon$ 倍，就把 $a$ 换成了 $\epsilon a$。

> **要点 9**　在单个 prompt 上，当 $Gh \ll 1$ 时，漏洞占比在时间尺度 $1/(2w(0)h_0)$ 上达到 $O(1)$：对 REINFORCE 约为 $1/(2h_0)$，对 GRPO 约为 $1/(2\sqrt{G-1}\,h_0)$。跨 prompt 时，共享的漏洞倾向在有限时间内爆炸。把孤立漏洞的影响压到 $\epsilon$ 倍，在 $\epsilon \gg G h_0$ 时起飞时间按 $1/\epsilon$ 推迟，之后饱和，大约比原来晚 $1/(G h_0)$ 倍。和传染病一样，这个过程会自我加速，减慢最早的携带者只能推迟它，不能阻止它；要阻止它，需要一个在 $z$ 较小时把倾向拉回去的力，比如朝参考模型的拉力，或者来自可解 prompt 的负迁移。

## 小结

在语言模型之前，reward hacking 被写成 proxy 等于真实奖励加误差，写成某些状态上的污染，写成智能体控制了自己的奖励通道，写成两个奖励对策略的排序不同，或者通过限制相对于基础分布的优化强度来处理。到了语言模型的 RL，一个网络回答所有 prompt，group-based 更新只看得到奖励，参考模型又很强，图景变得更具体了。Hacking 分成沿 proxy 的选择和奖励层内的放大：前者是奖励模型被 hack 的方式，后者是 verifier 被 hack 的唯一方式。在 verifier 下，目标函数对 hack 占比无差别，所以决定它的是学习动力学：漏洞通过共享参数扩散，起点是组内无法与老实成功区分的孤立成功，而一旦开始就会自我加速。这些都没有回答学会 hack 会对模型的其余部分造成什么影响 ([[macdiarmid2025|MacDiarmid et al., 2025]])，而且描述的都是期望意义下的更新。
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
          figures={{ classic: <ClassicExamples lang={lang} />, lm: <LanguageModelExamples lang={lang} /> }}
        />

      </div>
    </article>
  );
};

export default RewardHackingFormalized;
