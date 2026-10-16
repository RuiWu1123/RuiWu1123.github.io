
import React from 'react';
import { Link } from 'react-router-dom';
import { NEWS_ITEMS, SOCIAL_LINKS, PUBLICATIONS } from '../constants';

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

  // Helper to render content with links specifically on quoted text or full text
  const renderNewsContent = (content: string, link?: string) => {
    // Check for quoted text to link specifically
    const parts = content.split('"');
    
    // If we found a quoted part (e.g., My paper "Title" is...) and have a link
    if (parts.length >= 3 && link) {
        return (
            <>
                {parts[0]}"
                <a 
                    href={link} 
                    target="_blank" 
                    rel="noreferrer" 
                    className="text-anthropic-text underline decoration-anthropic-accent/50 hover:decoration-anthropic-accent hover:text-anthropic-accent transition-colors font-medium"
                >
                    {parts[1]}
                </a>
                "{parts[2]}
            </>
        );
    }

    // Fallback: link the whole text when a news item has a link
    if (link) {
         return (
          <a 
            href={link} 
            className={link.startsWith('#') ? "hover:text-anthropic-accent transition-colors" : "hover:text-anthropic-accent transition-colors underline decoration-anthropic-text/30"} 
            target={link.startsWith('#') ? "_self" : "_blank"} 
            rel={link.startsWith('#') ? "" : "noreferrer"}
          >
            {content}
          </a>
        );
    }

    return <>{content}</>;
  };

  return (
    <div className="animate-fade-in pb-16">
      {/* Hero: portrait on the left, introduction on the right */}
      <section className="pt-8 pb-10 flex flex-col min-[460px]:flex-row min-[460px]:items-start gap-5 min-[460px]:gap-7">
        <div className="min-[460px]:w-[140px] min-[460px]:flex-shrink-0">
          <img
            src="/portrait.jpg"
            alt="Kris Rui Wu"
            className="w-[140px] object-cover aspect-[2/3]"
          />

          {/* Contact, set as words rather than icons */}
          <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-anthropic-gray">
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
        </div>

        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-serif mb-3 text-anthropic-text leading-tight">
            Hi, I&apos;m Kris (Rui) Wu.
          </h1>
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            A junior undergraduate at Rutgers University (CS), working on{' '}
            <Link to="/alignment" className={inlineLinkClass}>
              AI alignment
            </Link>
            . I also write{' '}
            <Link to="/blog" className={inlineLinkClass}>
              longer posts
            </Link>{' '}
            on how language models are trained and served. I was born and raised in Yuexi, a small
            county in Anhui, China. Outside of research, I am into fashion, and I like wandering
            through independent boutiques and vintage shops that actually have a point of view.
          </p>
        </div>
      </section>

       {/* News Section */}
      <section className="mb-8">
        <div className="mb-4 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
          <h2 className="text-lg font-serif text-anthropic-text">News</h2>
        </div>
        
        <div>
          <div className="space-y-1.5">
            {NEWS_ITEMS.map((news, index) => (
              <div key={index} className="flex gap-4 group">
                <span className="text-anthropic-gray/70 text-[12px] whitespace-nowrap pt-[2px] w-[4.5rem] flex-shrink-0">
                  {news.date}
                </span>
                <p className="text-anthropic-gray text-[12.5px] leading-[1.5] group-hover:text-anthropic-text transition-colors">
                  {renderNewsContent(news.content, news.link)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Publications */}
      <section>
        <div className="mb-4 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
          <h2 className="text-lg font-serif text-anthropic-text">Publications</h2>
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
