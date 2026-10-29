import React from 'react';
import { Link } from 'react-router-dom';
import { THOUGHTS } from '../constants';

const Thoughts: React.FC = () => {
  return (
    <div className="animate-fade-in pt-10 pb-16">
      <h1 className="text-xl font-serif text-anthropic-text leading-tight text-center">
        Thoughts &amp; Blog Posts
      </h1>

      <div className="mt-8 border-t-[1.5px] border-anthropic-text pt-4">
        {THOUGHTS.length === 0 ? (
          <p className="text-[12.5px] text-anthropic-gray/70 text-center">Nothing here yet.</p>
        ) : (
          <div className="space-y-4">
            {THOUGHTS.map((post) => (
              <div key={post.title} className="flex gap-4">
                <span className="text-anthropic-gray/70 text-[12px] whitespace-nowrap pt-[3px] w-[4.5rem] flex-shrink-0">
                  {post.date}
                </span>
                <div>
                  <h2 className="text-[14px] md:text-base font-serif text-anthropic-text leading-[1.45]">
                    {post.link ? (
                      <Link to={post.link} className="hover:underline underline-offset-4 decoration-anthropic-text/30">
                        {post.title}
                      </Link>
                    ) : (
                      post.title
                    )}
                  </h2>
                  <p className="text-anthropic-gray text-[12.5px] leading-[1.5]">
                    <span className="italic">tl;dr</span> {post.tldr}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Thoughts;
