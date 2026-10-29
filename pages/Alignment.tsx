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

      <div className="mt-8 text-base font-sans text-anthropic-gray leading-relaxed">
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
          into Hugging Face&apos;s systems in search of the answers. My research centers on three questions.
          How and when does reward hacking emerge during reinforcement learning, given that it
          often appears{' '}
          <a href="https://arxiv.org/abs/2201.03544" target="_blank" rel="noreferrer" className={linkClass}>
            abruptly rather than gradually
          </a>
          ? How can we train models to follow the principle behind a safety rule rather than
          imitate the examples they were shown, so that they remain safe on inputs they have not
          seen? And how can we detect and intervene on misbehavior at runtime, by monitoring a
          model&apos;s{' '}
          <a href="https://arxiv.org/abs/2507.11473" target="_blank" rel="noreferrer" className={linkClass}>
            chain of thought
          </a>{' '}
          or its internal representations, even when its outputs look fine?
        </p>
      </div>
    </div>
  );
};

export default Alignment;
