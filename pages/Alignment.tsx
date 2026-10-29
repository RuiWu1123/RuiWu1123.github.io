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
          I work on AI safety, with a particular focus on specification gaming in large language
          models, also called reward hacking, and on how to monitor, mitigate, and control it. A
          policy optimized against a proxy reward learns whatever the proxy{' '}
          <a href="https://arxiv.org/abs/2209.13085" target="_blank" rel="noreferrer" className={linkClass}>
            fails to exclude
          </a>
          , and the resulting hacking does not stay local to the task that produced it, but{' '}
          <a href="https://arxiv.org/abs/2511.18397" target="_blank" rel="noreferrer" className={linkClass}>
            generalizes into broader misaligned behavior
          </a>
          ; deployed agents acting on an incomplete natural-language objective show the same
          divergence, as in the 2026{' '}
          <a
            href="https://openai.com/index/hugging-face-incident-and-the-road-ahead/"
            target="_blank"
            rel="noreferrer"
            className={linkClass}
          >
            Hugging Face incident
          </a>
          . Two questions organize my work: how does a model&apos;s effective objective come apart
          from the intended one, and how do we detect and act on that divergence when the
          supervision we would use to detect it is itself incomplete? I approach them along three
          lines: the training dynamics of specification gaming, since reward hacking{' '}
          <a href="https://arxiv.org/abs/2201.03544" target="_blank" rel="noreferrer" className={linkClass}>
            sets in abruptly
          </a>{' '}
          rather than gradually; alignment training that gets a model to act on the principle behind
          a safety rule rather than on the examples it was shown; and monitoring and control through
          a model&apos;s{' '}
          <a href="https://arxiv.org/abs/2507.11473" target="_blank" rel="noreferrer" className={linkClass}>
            chain of thought
          </a>{' '}
          and its internal representations.
        </p>
      </div>
    </div>
  );
};

export default Alignment;
