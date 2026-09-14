
import React from 'react';
import { Mail, Github, Twitter, Linkedin, MessageCircle, GraduationCap, FileText } from 'lucide-react';
import { NEWS_ITEMS, SOCIAL_LINKS, PUBLICATIONS } from '../constants';

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

    // Fallback: Link the whole text if a link exists (e.g. for Travel)
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
      {/* Hero Section */}
      <section className="pt-6 pb-10 md:pt-8 md:pb-12 flex flex-col sm:flex-row sm:items-start gap-6 sm:gap-8">
        {/* Left: Content */}
        <div className="flex-1 min-w-0">
          <h1 className="text-3xl md:text-4xl font-serif font-light mb-3 text-anthropic-text leading-tight">
            Hi, I'm Rui.
          </h1>
          <p className="text-base font-sans text-anthropic-gray leading-relaxed">
            A junior undergraduate at Rutgers University (CS), working on AI alignment. Currently I am
            interested in the problem that a model learns to pursue the wrong goal under imperfect
            human supervision.
          </p>

        </div>

        {/* Right: Portrait */}
        <div className="sm:w-[200px] sm:flex-shrink-0">
          <img
            src="/portrait.jpg"
            alt="Rui Wu"
            className="w-full rounded-xl object-cover aspect-[4/3] border border-anthropic-text/5 shadow-sm"
          />

          {/* Contact */}
          <div className="flex items-center justify-center sm:justify-start gap-0.5 mt-2 sm:-ml-1.5">
              <button 
                onClick={copyEmail}
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-1.5 rounded-full hover:bg-anthropic-stone/30"
                aria-label="Email"
                title="rw761@scarletmail.rutgers.edu (Click to Copy)"
              >
                <Mail size={18} strokeWidth={1.6} />
              </button>
              <a 
                href={SOCIAL_LINKS.scholar}
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-1.5 rounded-full hover:bg-anthropic-stone/30"
                aria-label="Google Scholar"
                title="Google Scholar"
              >
                <GraduationCap size={18} strokeWidth={1.6} />
              </a>
              <a 
                href={SOCIAL_LINKS.linkedin}
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-1.5 rounded-full hover:bg-anthropic-stone/30"
                aria-label="LinkedIn"
                title="LinkedIn"
              >
                <Linkedin size={18} strokeWidth={1.6} />
              </a>
              <a 
                href="https://x.com/RuiWu560670" 
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-1.5 rounded-full hover:bg-anthropic-stone/30"
                aria-label="Twitter"
                title="Twitter"
              >
                <Twitter size={18} strokeWidth={1.6} />
              </a>
              <a 
                href="https://github.com/RuiWu1123" 
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-1.5 rounded-full hover:bg-anthropic-stone/30"
                aria-label="GitHub"
                title="GitHub"
              >
                <Github size={18} strokeWidth={1.6} />
              </a>
               <button 
                onClick={copyWeChat}
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-1.5 rounded-full hover:bg-anthropic-stone/30"
                aria-label="WeChat"
                title="WeChat: R1lastshot (Click to Copy)"
              >
                <MessageCircle size={18} strokeWidth={1.6} />
              </button>
          </div>
        </div>
      </section>

       {/* News Section */}
      <section className="mb-12">
        <div className="flex items-baseline justify-between mb-5 border-b border-anthropic-text/10 pb-3">
          <h2 className="text-2xl font-serif text-anthropic-text">News</h2>
          <span className="text-xs font-sans text-anthropic-gray/70 uppercase tracking-widest">Latest Updates</span>
        </div>
        
        <div>
          <div className="space-y-3">
            {NEWS_ITEMS.map((news, index) => (
              <div key={index} className="flex gap-5 group">
                <span className="text-anthropic-gray/70 font-mono text-xs whitespace-nowrap pt-1 w-[5.5rem] flex-shrink-0">
                  [{news.date}]
                </span>
                <p className="text-anthropic-gray text-[15px] leading-relaxed group-hover:text-anthropic-text transition-colors">
                  {renderNewsContent(news.content, news.link)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Publications */}
      <section>
        <div className="flex items-baseline justify-between mb-5 border-b border-anthropic-text/10 pb-3">
          <h2 className="text-2xl font-serif text-anthropic-text">Publications</h2>
          <span className="text-xs font-sans text-anthropic-gray/70 uppercase tracking-widest">Selected Papers</span>
        </div>

        <div className="space-y-6">
          {PUBLICATIONS.map((pub) => (
            <div key={pub.id} className="group">
              <h3 className="text-base md:text-lg font-serif text-anthropic-text mb-1 leading-snug">
                {pub.link ? (
                  <a href={pub.link} target="_blank" rel="noreferrer" className="hover:text-anthropic-accent transition-colors">
                    {pub.title}
                  </a>
                ) : (
                  pub.title
                )}
              </h3>

              <p className="text-anthropic-gray text-[14px] mb-1 leading-relaxed">
                {pub.authors.map((author, idx) => (
                  <span key={idx} className={author.includes("Rui Wu") ? "font-bold text-anthropic-text" : ""}>
                    {author}{idx < pub.authors.length - 1 ? ", " : ""}
                  </span>
                ))}
              </p>

              <div className="text-[13px] text-anthropic-gray/90 italic mb-1.5">
                {pub.venue} ({pub.year})
              </div>

              <div className="flex gap-4 text-[13px]">
                {pub.link && (
                  <a
                    href={pub.link}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center text-sm font-medium text-anthropic-accent hover:text-anthropic-text transition-colors"
                  >
                    <FileText size={16} className="mr-1.5" />
                    ArXiv
                  </a>
                )}
                {pub.github && (
                  <a
                    href={pub.github}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center text-sm font-medium text-anthropic-accent hover:text-anthropic-text transition-colors"
                  >
                    <Github size={16} className="mr-1.5" />
                    Code
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
