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
            Capability has a lever. Add compute, add data, and the curves move in a way that is now
            predictable enough to plan a company around. Alignment has no equivalent. The objective
            cannot be written down, because what we want out of a model is the kind of thing that is
            far easier to recognise than to state, and the work of checking whether a system learned
            the right thing does not get cheaper as the system gets more capable. It gets harder, for
            the same reason: a more capable model has more ways to look correct.
          </p>
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            So the underlying failure has been there the whole time, a system optimising something
            adjacent to what was intended under supervision too coarse to notice the difference. What
            alignment research has bought is frequency: models refuse more reliably, scheme less often
            under test, get caught more often when they do. But what matters is frequency multiplied
            by what a single failure costs, and that second term is set by capability, which is the
            term that compounds. A model that goes wrong once in ten thousand runs meant little when a
            run was a paragraph of text. It means something else when a run holds a corporate card, a
            shell, and eight hours.
          </p>
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            Which is why the split of effort looks wrong to me. Most of the value actually reaching
            people comes from applying models that already work, not from the next increment of
            generality, and that increment is precisely what raises the cost of every failure
            downstream. I would rather see the race toward general capability slowed, engineering
            attention moved to applications, and far more of the field&apos;s money and talent pointed
            at the part we still cannot do: saying what we want, and checking whether we got it.
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
