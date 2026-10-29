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
          I work on AI safety. My main interest is reward hacking in large language models, also
          known as specification gaming: a model finds a way to get a high score that its designers
          did not intend, instead of actually doing the task. A familiar example is a coding model
          that edits the unit tests so they pass rather than fixing the code. This happens because
          the reward used in training only approximates what we actually want, and a model that is
          optimized hard enough will{' '}
          <a href="https://arxiv.org/abs/2209.13085" target="_blank" rel="noreferrer" className={linkClass}>
            exploit the gaps
          </a>
          . It also matters beyond the task at hand: models that learn to reward hack during
          training can{' '}
          <a href="https://arxiv.org/abs/2511.18397" target="_blank" rel="noreferrer" className={linkClass}>
            become misaligned in broader ways
          </a>
          . And the problem is not limited to training. AI agents given a goal in natural language
          can pursue it in ways nobody intended, as in the 2026{' '}
          <a
            href="https://openai.com/index/hugging-face-incident-and-the-road-ahead/"
            target="_blank"
            rel="noreferrer"
            className={linkClass}
          >
            Hugging Face incident
          </a>
          , where research agents found ways to collect reward without solving their tasks, got
          around isolation controls, and took on goals from other agents. I am interested in three
          related questions. First, how and when does reward hacking emerge during reinforcement
          learning? It often appears{' '}
          <a href="https://arxiv.org/abs/2201.03544" target="_blank" rel="noreferrer" className={linkClass}>
            suddenly rather than gradually
          </a>
          , so knowing when it starts is as important as knowing whether it happens. Second, how
          can we train models to follow the intent behind a safety rule, rather than just copying
          the examples they were trained on, so that they stay safe in situations they have not
          seen before? Third, how can we detect and stop bad behavior while a model is running, by
          monitoring its{' '}
          <a href="https://arxiv.org/abs/2507.11473" target="_blank" rel="noreferrer" className={linkClass}>
            chain of thought
          </a>{' '}
          or its internal activations, especially when its outputs look fine?
        </p>
      </div>
    </div>
  );
};

export default Alignment;
