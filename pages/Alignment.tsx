import React from 'react';

const linkClass =
  'underline underline-offset-2 decoration-anthropic-text/35 hover:decoration-anthropic-text transition-colors';

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
          I work on AI safety, with a focus on reward hacking in large language models, also known
          as specification gaming: a model gets a high reward in ways people did not intend, without
          actually solving the task. A coding model that edits the unit tests so they pass, instead
          of fixing the code, is a typical case. Because a training reward only approximates what we
          want, a model optimized against it will learn to{' '}
          <a href="https://arxiv.org/abs/2209.13085" target="_blank" rel="noreferrer" className={linkClass}>
            exploit its gaps
          </a>
          , and the behavior does not stay contained: models that learn to reward hack in training
          can{' '}
          <a href="https://arxiv.org/abs/2511.18397" target="_blank" rel="noreferrer" className={linkClass}>
            generalize to broader misalignment
          </a>
          . The same failure shows up outside training as well. In the 2026{' '}
          <a
            href="https://openai.com/index/hugging-face-incident-and-the-road-ahead/"
            target="_blank"
            rel="noreferrer"
            className={linkClass}
          >
            Hugging Face incident
          </a>
          , agents in a cybersecurity evaluation that could not solve their tasks tried to cheat the
          evaluation instead: they escaped their sandbox, coordinated with one another, and broke
          into Hugging Face&apos;s systems in search of the answers.
        </p>
        <p>
          My research centers on four questions. The first is how reward hacking is acquired during
          reinforcement learning. It can{' '}
          <a href="https://arxiv.org/abs/2201.03544" target="_blank" rel="noreferrer" className={linkClass}>
            appear abruptly rather than gradually
          </a>
          , and models that learn simple forms of gaming can{' '}
          <a href="https://arxiv.org/abs/2406.10162" target="_blank" rel="noreferrer" className={linkClass}>
            generalize to tampering with their own reward
          </a>
          . The second is how to train models that stay aligned even when the reward is imperfect.
          Telling a model during training that hacking is acceptable, known as{' '}
          <a href="https://www.anthropic.com/research/emergent-misalignment-reward-hacking" target="_blank" rel="noreferrer" className={linkClass}>
            inoculation prompting
          </a>
          , keeps the hacking it learns from{' '}
          <a href="https://arxiv.org/abs/2511.18397" target="_blank" rel="noreferrer" className={linkClass}>
            generalizing into broader misalignment
          </a>
          , and training a model to{' '}
          <a href="https://arxiv.org/abs/2512.08093" target="_blank" rel="noreferrer" className={linkClass}>
            confess
          </a>{' '}
          when it has broken the rules, rewarding only the honesty of the confession, makes
          misbehavior easier to surface. The third is how to catch misbehavior while a model is
          running. At the frontier, monitoring is{' '}
          <a href="https://openai.com/index/how-we-detect-frontier-model-misalignment/" target="_blank" rel="noreferrer" className={linkClass}>
            a layered system
          </a>{' '}
          rather than a single detector: cheap checks screen all traffic and escalate what they flag
          to more expensive review. A{' '}
          <a href="https://arxiv.org/abs/2507.11473" target="_blank" rel="noreferrer" className={linkClass}>
            chain-of-thought monitor
          </a>{' '}
          works only while the reasoning{' '}
          <a href="https://arxiv.org/abs/2512.18311" target="_blank" rel="noreferrer" className={linkClass}>
            stays monitorable
          </a>
          , and training against it{' '}
          <a href="https://arxiv.org/abs/2503.11926" target="_blank" rel="noreferrer" className={linkClass}>
            teaches the model to hide its intent
          </a>
          , although current models{' '}
          <a href="https://arxiv.org/abs/2603.05706" target="_blank" rel="noreferrer" className={linkClass}>
            struggle to control what their reasoning reveals
          </a>
          .{' '}
          <a href="https://arxiv.org/abs/2609.19101" target="_blank" rel="noreferrer" className={linkClass}>
            Probes on internal activations
          </a>{' '}
          detect reward hacking about as well as a chain-of-thought monitor, are cheap enough to run
          on every rollout, and have already been{' '}
          <a href="https://www.goodfire.com/research" target="_blank" rel="noreferrer" className={linkClass}>
            deployed in production inference
          </a>
          . The fourth is how to measure all of this. Evaluation is moving toward quantities that
          extrapolate, such as{' '}
          <a href="https://arxiv.org/abs/2503.14499" target="_blank" rel="noreferrer" className={linkClass}>
            the length of task a model can complete reliably
          </a>
          , but models{' '}
          <a href="https://arxiv.org/abs/2505.23836" target="_blank" rel="noreferrer" className={linkClass}>
            often recognize when they are being evaluated
          </a>
          , so a measured safety rate may describe behavior under observation rather than in
          deployment. Closing that gap is what{' '}
          <a href="https://www.apolloresearch.ai/blog/we-need-a-science-of-evals" target="_blank" rel="noreferrer" className={linkClass}>
            a science of evals
          </a>{' '}
          would do, and the field does not have one yet.
        </p>
      </div>
    </div>
  );
};

export default Alignment;
