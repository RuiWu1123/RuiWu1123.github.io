import React from 'react';
import { ALIGNMENT_NEWS } from '../constants';

const Alignment: React.FC = () => {
  return (
    <div className="animate-fade-in pt-10 pb-16">
      <div className="w-full">
        <h1 className="text-3xl md:text-4xl font-serif font-light mb-4 text-anthropic-text">
          Alignment
        </h1>

        <div className="space-y-3 mb-12 max-w-2xl">
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            AI alignment is the problem of getting a system to pursue what its developers actually
            intended, and of being able to check that it does. It splits roughly in two: writing down
            the right objective in the first place, and making sure the model that comes out of
            training is optimising that objective rather than something merely correlated with it
            that happens to score well under whatever supervision was available.
          </p>
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            The live sub-problems are reward hacking, the misalignment that generalises out of it,
            deception and scheming, models recognising that they are being evaluated, and whether
            chain-of-thought stays a channel anyone can monitor. Below is what has moved those
            questions, newest first. Each entry links the primary source.
          </p>
        </div>

        <div className="flex items-baseline justify-between mb-5 border-b border-anthropic-text/10 pb-3">
          <h2 className="text-2xl font-serif text-anthropic-text">Field News</h2>
          <span className="text-xs font-sans text-anthropic-gray/70 uppercase tracking-widest">
            Results &amp; Incidents
          </span>
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
