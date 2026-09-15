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
            The reason to care now rather than later is that the permissions arrived first. Models
            are already running shops, hiring people, writing production code and holding
            credentials, and most of what is below went wrong without anyone attacking anything: a
            model that cannot solve a task honestly finds a dishonest route, a model under evaluation
            often knows it, and a model given a goal will pursue it past the point where a person
            would stop. None of that requires superintelligence. The cost of paying attention is also
            low, because nearly everything known about it is in public write-ups like these — so read
            the incident reports rather than the headlines about them.
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

          <div className="space-y-7">
            {ALIGNMENT_NEWS.map((item, index) => (
              <div key={index} className="relative pl-8">
                <span
                  aria-hidden="true"
                  className={`absolute left-0 top-[5px] w-[11px] h-[11px] rounded-full ring-4 ring-anthropic-bg ${
                    index === 0 ? 'bg-anthropic-accent' : 'bg-anthropic-stone'
                  }`}
                />

                <div className="text-anthropic-gray/70 font-mono text-xs mb-1">{item.date}</div>

                <h3 className="text-base md:text-lg font-serif text-anthropic-text mb-1.5 leading-snug">
                  {item.title}
                </h3>

                <p className="text-[14px] text-anthropic-gray leading-relaxed mb-1.5">
                  {item.summary}
                </p>

                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
                  {item.links.map((link) => (
                    <a
                      key={link.url}
                      href={link.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-anthropic-accent hover:text-anthropic-text transition-colors"
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
    </div>
  );
};

export default Alignment;
