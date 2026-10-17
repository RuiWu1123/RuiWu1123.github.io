import React from 'react';

const linkClass =
  'underline underline-offset-2 decoration-anthropic-text/35 hover:decoration-anthropic-text transition-colors';

const SectionHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="mb-4 mt-10 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
    <h2 className="text-lg font-serif text-anthropic-text">{children}</h2>
  </div>
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
      </div>

      <SectionHeading>Alignment training</SectionHeading>
      <div className="space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <p>
          A second line develops alignment training methods that reduce misaligned behavior in
          deployment. Both methods below aim at the same target: getting a model to act on the
          principle behind a safety rule rather than on the examples it was shown, so that alignment
          holds on inputs the training data did not cover.
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

    </div>
  );
};

export default Alignment;
