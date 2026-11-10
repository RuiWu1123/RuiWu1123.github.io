
import React from 'react';
import { Link } from 'react-router-dom';
import { SOCIAL_LINKS, PUBLICATIONS } from '../constants';

const contactClass =
  'underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';
const pubLinkClass =
  'text-anthropic-gray underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';
const inlineLinkClass =
  'underline underline-offset-2 decoration-anthropic-text/35 hover:decoration-anthropic-text transition-colors';

const Home: React.FC = () => {
  const copyWeChat = () => {
    navigator.clipboard.writeText('R1lastshot');
    alert('WeChat ID "R1lastshot" copied to clipboard.');
  };

  const copyEmail = () => {
    navigator.clipboard.writeText('rw761@scarletmail.rutgers.edu');
    alert('Email "rw761@scarletmail.rutgers.edu" copied to clipboard.');
  };

  return (
    <div className="animate-fade-in pb-16">
      {/* Hero: name, then contact on one line, then the introduction */}
      <section className="pt-10 pb-10">
        <h1 className="text-xl font-serif text-anthropic-text leading-tight text-center">
          Kris (Rui) Wu
        </h1>

        {/* Contact, set as words rather than icons */}
        <div className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1 text-[12px] text-anthropic-gray">
          <button onClick={copyEmail} className={contactClass} title="rw761@scarletmail.rutgers.edu (click to copy)">
            email
          </button>
          <a href={SOCIAL_LINKS.scholar} target="_blank" rel="noreferrer" className={contactClass}>
            scholar
          </a>
          <a href={SOCIAL_LINKS.linkedin} target="_blank" rel="noreferrer" className={contactClass}>
            linkedin
          </a>
          <a href="https://x.com/RuiWu560670" target="_blank" rel="noreferrer" className={contactClass}>
            X
          </a>
          <a href="https://github.com/RuiWu1123" target="_blank" rel="noreferrer" className={contactClass}>
            github
          </a>
          <button onClick={copyWeChat} className={contactClass} title="WeChat: R1lastshot (click to copy)">
            wechat
          </button>
        </div>

        <p className="mt-6 text-base font-sans text-anthropic-gray leading-relaxed">
          I&apos;m a senior undergraduate at the Computer Science Department, Rutgers
          University&ndash;New Brunswick. I do research about AI Alignment with{' '}
          <a
            href="https://www.ruixiangtang.net/"
            target="_blank"
            rel="noreferrer"
            className={inlineLinkClass}
          >
            Ruixiang Tang
          </a>
          , here is my{' '}
          <Link to="/alignment" className={inlineLinkClass}>
            research statement
          </Link>
          .
        </p>

        <p className="mt-4 text-base font-sans text-anthropic-gray leading-relaxed">
          I try to write some valuable blogs, see{' '}
          <Link to="/thoughts" className={inlineLinkClass}>
            here
          </Link>
          .
        </p>
      </section>

      {/* Publications */}
      <section>
        <div className="mb-4 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
          <h2 className="text-lg font-serif text-anthropic-text">Selected Publications</h2>
        </div>

        <div className="space-y-3.5">
          {PUBLICATIONS.map((pub) => (
            <div key={pub.id} className="group">
              <h3 className="text-[14px] md:text-base font-serif text-anthropic-text leading-[1.45]">
                {pub.link ? (
                  <a href={pub.link} target="_blank" rel="noreferrer" className="hover:underline underline-offset-4 decoration-anthropic-text/30">
                    {pub.title}
                  </a>
                ) : (
                  pub.title
                )}
              </h3>

              <p className="text-anthropic-gray text-[12.5px] leading-[1.5]">
                {pub.authors.map((author, idx) => (
                  <span key={idx} className={author.includes("Rui Wu") ? "font-bold text-anthropic-text" : ""}>
                    {author}{idx < pub.authors.length - 1 ? ", " : ""}
                  </span>
                ))}
              </p>

              <div className="text-[12px] text-anthropic-gray/90 italic leading-[1.5]">
                {pub.venue}
              </div>

              <div className="flex gap-3 text-[12px] mt-0.5">
                {pub.link && (
                  <a href={pub.link} target="_blank" rel="noreferrer" className={pubLinkClass}>
                    arxiv
                  </a>
                )}
                {pub.github && (
                  <a href={pub.github} target="_blank" rel="noreferrer" className={pubLinkClass}>
                    code
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Home;
