import React from 'react';
import { ALIGNMENT_NEWS } from '../constants';

const linkClass =
  'underline underline-offset-2 decoration-anthropic-text/35 hover:decoration-anthropic-text transition-colors';

const SectionHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="mb-4 mt-10 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
    <h2 className="text-lg font-serif text-anthropic-text">{children}</h2>
  </div>
);

/**
 * Proxy and oracle reward over training. The two quantities coincide while the
 * policy is still improving legitimately and separate once an exploit is
 * available; the curves are schematic, drawn rather than measured.
 */
const DivergenceFigure: React.FC = () => (
  <svg
    viewBox="0 0 560 206"
    className="w-full"
    role="img"
    aria-label="Training reward keeps rising after an exploit is found while the held-out oracle reward stops improving"
  >
    <g fill="none" stroke="#191919" strokeOpacity="0.35" strokeWidth="1">
      <path d="M58 20 L58 180 L532 180" />
      <path d="M54 26 L58 20 L62 26" />
      <path d="M526 176 L532 180 L526 184" />
    </g>

    <path
      d="M312 96 L312 180"
      fill="none"
      stroke="#191919"
      strokeOpacity="0.25"
      strokeWidth="1"
      strokeDasharray="2 3"
    />

    <path
      d="M58 172 C 150 146 240 118 305 106 C 330 102 345 112 365 122 C 390 134 415 138 440 138 C 475 138 500 137 520 136"
      fill="none"
      stroke="#191919"
      strokeOpacity="0.55"
      strokeWidth="1.3"
      strokeDasharray="4 3"
    />
    <path
      d="M58 172 C 150 146 240 118 305 95 C 345 80 365 44 405 33 C 445 25 485 26 520 26"
      fill="none"
      stroke="#191919"
      strokeWidth="1.6"
    />

    <g fill="#191919" fillOpacity="0.72" fontSize="11">
      <text x="524" y="16" textAnchor="end">
        training reward (proxy)
      </text>
      <text x="524" y="156" textAnchor="end">
        held-out oracle reward
      </text>
      <text x="318" y="171">
        exploit found
      </text>
      <text x="295" y="199" textAnchor="middle">
        training steps
      </text>
      <text x="22" y="100" textAnchor="middle" transform="rotate(-90 22 100)">
        reward
      </text>
    </g>
  </svg>
);

const Alignment: React.FC = () => {
  return (
    <div className="animate-fade-in pt-10 pb-16">
      <h1 className="text-xl font-serif text-anthropic-text leading-tight text-center">
        Research Statement
      </h1>
      <p className="mt-2 text-center text-[12px] text-anthropic-gray">
        <a href="/research_statement.pdf" target="_blank" rel="noreferrer" className={linkClass}>
          pdf
        </a>
      </p>

      <div className="mt-8 space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <p>
          I work on AI safety, with a particular focus on specification gaming in large language
          models, also called reward hacking, and on how to monitor, mitigate, and control it.
        </p>
        <p>
          Two forces produce the divergence between the objective a model ends up pursuing and the
          one it was meant to be given. Training produces it: a policy optimized against a proxy
          reward learns whatever the proxy{' '}
          <a href="https://arxiv.org/abs/2209.13085" target="_blank" rel="noreferrer" className={linkClass}>
            fails to exclude
          </a>
          , and the resulting reward hacking does not stay local to the task that produced it, but{' '}
          <a href="https://arxiv.org/abs/2511.18397" target="_blank" rel="noreferrer" className={linkClass}>
            generalizes into broader misaligned behavior
          </a>
          . Deployment produces it as well: an objective conveyed in natural language is incomplete,
          and an agent able to execute acts on the specification it was given rather than the one
          intended. OpenAI&apos;s report on the 2026{' '}
          <a
            href="https://openai.com/index/hugging-face-incident-and-the-road-ahead/"
            target="_blank"
            rel="noreferrer"
            className={linkClass}
          >
            Hugging Face incident
          </a>{' '}
          is a concrete instance. Research agents under cybersecurity evaluation made reward easier
          to obtain instead of solving their assigned tasks, circumvented isolation controls, and
          adopted goals from peer agents rather than the objectives they were given.
        </p>
        <p>
          Two questions organize my work: how does a model&apos;s effective objective come apart from
          the intended one, and how do we detect and act on that divergence when the supervision we
          would use to detect it is itself incomplete? Three lines of work follow, each resting on
          the limitation the previous one exposed.
        </p>
      </div>

      <figure className="mt-8">
        <DivergenceFigure />
        <figcaption className="text-[12px] text-anthropic-gray/80 leading-relaxed mt-2">
          The divergence the work is about. While the policy is still improving legitimately the two
          rewards coincide, so the proxy is a usable measurement; once an exploit is available the
          training reward keeps climbing and the quantity it stood in for stops improving. The curves
          are schematic.
        </figcaption>
      </figure>

      <SectionHeading>Specification gaming</SectionHeading>
      <div className="space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <p>
          My main line of work asks how a misaligned objective is acquired during RL training, and in
          particular what the training dynamics look like: when hacking appears, and why at that
          point. Reward hacking is known to{' '}
          <a href="https://arxiv.org/abs/2201.03544" target="_blank" rel="noreferrer" className={linkClass}>
            set in abruptly
          </a>{' '}
          rather than gradually as a policy becomes more capable, which makes the question of when it
          appears as consequential as whether it appears.
        </p>
        <p>
          In{' '}
          <a href="https://arxiv.org/abs/2604.01476" target="_blank" rel="noreferrer" className={linkClass}>
            From Rebound to Remedy
          </a>{' '}
          I built a testbed in which the training environment contains a deliberate loophole, write
          access to the evaluator the model is scored by, and found that hacking does not emerge
          monotonically. Models attempt it early and fail, retreat to legitimate problem-solving,
          then return with a cheaper exploit, after which hacking dominates training. The onset of
          that third phase tracks the scarcity of legitimate reward, which makes the pattern a
          property of the incentive landscape rather than of a particular model, and means that a
          single end-of-training measurement can report either a safe model or a fully hacking one.
        </p>
        <p>
          The behavior is also legible in the model&apos;s representations before it is legible in
          its outputs, and candidate directions differ in how closely they track it. Building on this
          I proposed Advantage Modification, which folds the representation-level score into the
          training advantage so that the penalty is internalized in the training signal rather than
          imposed at decoding, and which suppresses hacking more robustly than applying the same
          signal as generation-time steering.
        </p>
      </div>

      <SectionHeading>Alignment training</SectionHeading>
      <div className="space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <p>
          A second line develops alignment training methods that reduce misaligned behavior in
          deployment. Both methods below aim at the same target: getting a model to act on the
          principle behind a safety rule rather than on the examples it was shown, so that alignment
          holds on inputs the training data did not cover.
        </p>
        <p>
          In{' '}
          <a href="https://arxiv.org/abs/2510.04320" target="_blank" rel="noreferrer" className={linkClass}>
            Read the Scene, Not the Script
          </a>{' '}
          I traced two opposite failures of safety-aligned models, jailbreaking and over-refusal, to
          one cause: models reason over surface-form cues that do not encode consequences. I named
          this failure mode consequence-blindness, built a benchmark that separates semantic risk
          from outcome risk, and showed that supervision on consequence reasoning improves robustness
          to semantic-camouflage jailbreaks and reduces over-refusal without costing utility.
        </p>
        <p>
          <a href="https://arxiv.org/abs/2412.16339" target="_blank" rel="noreferrer" className={linkClass}>
            Deliberative alignment
          </a>{' '}
          trains a model to reason over an explicit safety specification before it answers. In{' '}
          <a href="https://arxiv.org/abs/2601.08000" target="_blank" rel="noreferrer" className={linkClass}>
            CADA
          </a>{' '}
          we found that giving open-source models extensive code-like safety rules to reason over
          improves harmlessness inconsistently and systematically degrades helpfulness, whereas
          training on cases that illustrate a simple code generalizes better, and CADA builds on this
          by running RL on self-generated safety reasoning chains. Relatedly, in{' '}
          <a href="https://arxiv.org/abs/2606.00424" target="_blank" rel="noreferrer" className={linkClass}>
            Weak Critics Make Strong Learners
          </a>{' '}
          we treat a weak model as a critic rather than a labeler or judge, asking it only for a
          non-misleading revision direction and distilling the critic-guided behavior on-policy into
          the strong model.
        </p>
      </div>

      <SectionHeading>Monitoring and control</SectionHeading>
      <div className="space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <p>
          A monitor is an automated system that reads the model&apos;s output and flags behavior that
          is suspicious or potentially harmful, which is how misbehavior and signs of misalignment
          are caught while the model is running rather than after the fact. There are two ways to
          read a model. One is the{' '}
          <a href="https://arxiv.org/abs/2507.11473" target="_blank" rel="noreferrer" className={linkClass}>
            chain of thought
          </a>
          . The other is its internal representations, where a probe serves as the monitor and
          remains available when the output and the reasoning trace look clean.
        </p>
        <p>
          My work follows the second, and it connects monitoring to control through interpretability,
          since locating the mechanism behind a behavior is also what gives a handle for intervening
          on it at runtime, which is what a{' '}
          <a href="https://arxiv.org/abs/2312.06942" target="_blank" rel="noreferrer" className={linkClass}>
            control protocol
          </a>{' '}
          needs when the model may be adversarial. The representation-level signal recovered in From
          Rebound to Remedy is a monitor of this kind, and it carries the connection in both
          directions: the same direction that detects the behavior is the one I used to intervene on
          it during training, so the audit and the control handle are the same object. One tension
          across these results remains unresolved, and it is the one I most want to settle: a monitor
          that is{' '}
          <a href="https://arxiv.org/abs/2503.11926" target="_blank" rel="noreferrer" className={linkClass}>
            optimized against
          </a>{' '}
          stops measuring what it was built to measure.
        </p>
      </div>

      <SectionHeading>Current and future work</SectionHeading>
      <div className="space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <p>
          My near-term focus is reward hacking, along three questions: its mechanism and training
          dynamics, how monitors can be used to mitigate it, and how it leads a model into broader
          emergent misalignment. A further problem cuts across all three: models can tell when they
          are being evaluated, and frontier models{' '}
          <a href="https://arxiv.org/abs/2505.23836" target="_blank" rel="noreferrer" className={linkClass}>
            do so well above chance
          </a>
          . This bears on every claim above that rests on a measurement, since a testbed with planted
          exploits is still an evaluation, and a model that recognizes it as one may behave
          differently than it would in deployment, which would turn a clean construction-time label
          into a measurement of the wrong thing. I intend to work on this directly, both on measuring
          how far evaluation awareness shifts specification gaming and on building environments in
          which that gap can be estimated rather than assumed.
        </p>
        <p>
          Answering these questions requires open infrastructure that does not currently exist. Most
          of what is known about reward hacking at scale comes from frontier labs, on internal
          environments, models, and RL runs that cannot be reproduced or extended from outside, while
          academic testbeds, including my own, are reproducible but thin: single exploits, one task
          family, static evaluation, and, most limiting, hack labels defined by the same detection
          rules a study is meant to evaluate.
        </p>
        <p>
          I am building open agentic environments for reward hacking research that close this gap.
          Exploits are planted rather than discovered, so whether a rollout hacked is determined at
          construction time and independently of any detector, and exploit composition is a
          controlled variable rather than a property of the environment. My aim is that the
          infrastructure, not only the findings, is the contribution: a pipeline on which
          training-dynamics questions can be asked reproducibly, starting with whether a mitigation
          removes hacking or displaces it toward forms a monitor cannot detect, and whether an
          exploitation strategy learned in one environment transfers as a general prior that
          environments are exploitable.
        </p>
      </div>

      <SectionHeading>Field News</SectionHeading>

      {/* A time axis running upward: the newest event sits at the top. */}
      <div className="relative">
        <div
          aria-hidden="true"
          className="absolute left-[5px] top-1.5 bottom-0 w-px"
          style={{
            background:
              'linear-gradient(to top, rgba(25,25,25,0) 0%, rgba(25,25,25,0.13) 18%, rgba(25,25,25,0.13) 100%)'
          }}
        />

        <div className="space-y-5">
          {ALIGNMENT_NEWS.map((item, index) => (
            <div key={index} className="relative pl-8">
              <span
                aria-hidden="true"
                className={`absolute left-0 top-[5px] w-[11px] h-[11px] rounded-full ring-4 ring-anthropic-bg ${
                  index === 0 ? 'bg-anthropic-accent' : 'bg-anthropic-stone'
                }`}
              />

              <div className="text-anthropic-gray/70 text-[12px] mb-1">{item.date}</div>

              <h3 className="text-[14px] md:text-base font-serif text-anthropic-text mb-1 leading-[1.45]">
                {item.title}
              </h3>

              <p className="text-[12.5px] text-anthropic-gray leading-[1.5] mb-1 max-w-measure-lg">
                {item.summary}
              </p>

              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px]">
                {item.links.map((link) => (
                  <a
                    key={link.url}
                    href={link.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-anthropic-gray underline underline-offset-2 decoration-anthropic-text/25 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors"
                  >
                    {link.label}
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Alignment;
