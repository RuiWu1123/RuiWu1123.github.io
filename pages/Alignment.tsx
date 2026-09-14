import React from 'react';
import { ALIGNMENT_NEWS } from '../constants';

const Alignment: React.FC = () => {
  return (
    <div className="animate-fade-in pt-10 pb-16">
      <div className="w-full">
        <h1 className="text-3xl md:text-4xl font-serif font-light mb-4 text-anthropic-text">
          Alignment
        </h1>

        <div className="space-y-3 mb-8 max-w-2xl">
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            Nobody writes down what a model wants. A training objective gets chosen, gradients get
            applied, and whatever generalises out the other end is what you deploy. That process is
            reliable enough to be worth billions and loose enough that the resulting behaviour keeps
            surprising the people who produced it. Alignment is the problem of closing that gap:
            getting a system to pursue what was actually intended, and being able to tell whether it
            does before it is handed permissions that matter.
          </p>
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            The reason to care now rather than later is that the permissions arrived first. Most of
            what is listed below did not involve an attacker. The models were doing sanctioned work
            inside evaluations their own developers had set up, and the trouble came from ordinary
            optimisation pressure meeting a task that admitted no honest solution. Nothing about that
            requires superintelligence, and the cost of paying attention is low: almost everything
            known about it is in public reports like these. If you build capabilities, read the
            incident write-ups rather than the headlines about them.
          </p>
        </div>

        <figure className="mb-10">
          <img
            src="/alignment/horizon.svg"
            alt="Doubling time of the task length models can complete, measured over three windows"
            className="w-full rounded-lg border border-anthropic-text/5"
          />
          <figcaption className="text-[13px] text-anthropic-gray/80 leading-relaxed mt-2">
            The length of task a model can finish with 50% reliability has been doubling roughly
            every 197 days since 2019 — but restrict the fit to recent years and the doubling time
            itself keeps falling. Numbers from{' '}
            <a
              href="https://metr.org/blog/2026-1-29-time-horizon-1-1/"
              target="_blank"
              rel="noreferrer"
              className="underline decoration-anthropic-accent/50 hover:text-anthropic-accent transition-colors"
            >
              METR&apos;s Time Horizon 1.1
            </a>
            , January 2026.
          </figcaption>
        </figure>

        <div className="mb-5 border-b border-anthropic-text/10 pb-3">
          <h2 className="text-2xl font-serif text-anthropic-text">Field News</h2>
        </div>

        <div className="space-y-6">
          {ALIGNMENT_NEWS.map((item, index) => (
            <div key={index} className="group">
              <div className="flex items-baseline gap-3 flex-wrap mb-1">
                <span className="text-anthropic-gray/70 font-mono text-xs whitespace-nowrap">
                  [{item.date}]
                </span>
                <span className="text-[13px] font-sans text-anthropic-gray/80">{item.org}</span>
              </div>

              <h3 className="text-base md:text-lg font-serif text-anthropic-text mb-1 leading-snug">
                <a
                  href={item.link}
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-anthropic-accent transition-colors"
                >
                  {item.title}
                </a>
              </h3>

              <p className="text-[14px] text-anthropic-gray leading-relaxed">{item.summary}</p>

              {item.caveat && (
                <p className="text-[13px] text-anthropic-gray/75 italic leading-relaxed mt-1">
                  {item.caveat}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Alignment;
