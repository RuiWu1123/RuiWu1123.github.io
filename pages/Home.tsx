
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
    <div className="animate-fade-in pb-20">
      {/* Hero Section */}
      <section className="pt-8 pb-14 md:pt-12 md:pb-16 flex flex-col sm:flex-row sm:items-start gap-8 sm:gap-10">
        {/* Left: Content */}
        <div className="flex-1 min-w-0">
          <h1 className="text-4xl md:text-5xl font-serif font-light mb-5 text-anthropic-text leading-tight">
            Hi, I'm Rui.
          </h1>
          <p className="text-lg font-sans font-light text-anthropic-gray leading-relaxed">
            A junior undergraduate at Rutgers University (CS), working on AI alignment. Currently I am
            interested in the problem that a model learns to pursue the wrong goal under imperfect
            human supervision.
          </p>

        </div>

        {/* Right: Portrait */}
        <div className="sm:w-[250px] sm:flex-shrink-0">
          <img
            src="/portrait.jpg"
            alt="Rui Wu"
            className="w-full rounded-xl object-cover aspect-[4/3] border border-anthropic-text/5 shadow-sm"
          />

          {/* Contact */}
          <div className="flex items-center justify-center sm:justify-start gap-1 mt-3 sm:-ml-2">
              <button 
                onClick={copyEmail}
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-2 rounded-full hover:bg-anthropic-stone/30"
                aria-label="Email"
                title="rw761@scarletmail.rutgers.edu (Click to Copy)"
              >
                <Mail size={20} strokeWidth={1.5} />
              </button>
              <a 
                href={SOCIAL_LINKS.scholar}
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-2 rounded-full hover:bg-anthropic-stone/30"
                aria-label="Google Scholar"
                title="Google Scholar"
              >
                <GraduationCap size={20} strokeWidth={1.5} />
              </a>
              <a 
                href={SOCIAL_LINKS.linkedin}
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-2 rounded-full hover:bg-anthropic-stone/30"
                aria-label="LinkedIn"
                title="LinkedIn"
              >
                <Linkedin size={20} strokeWidth={1.5} />
              </a>
              <a 
                href="https://x.com/RuiWu560670" 
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-2 rounded-full hover:bg-anthropic-stone/30"
                aria-label="Twitter"
                title="Twitter"
              >
                <Twitter size={20} strokeWidth={1.5} />
              </a>
              <a 
                href="https://github.com/RuiWu1123" 
                target="_blank" 
                rel="noreferrer" 
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-2 rounded-full hover:bg-anthropic-stone/30"
                aria-label="GitHub"
                title="GitHub"
              >
                <Github size={20} strokeWidth={1.5} />
              </a>
               <button 
                onClick={copyWeChat}
                className="text-anthropic-text hover:text-anthropic-accent transition-colors p-2 rounded-full hover:bg-anthropic-stone/30"
                aria-label="WeChat"
                title="WeChat: R1lastshot (Click to Copy)"
              >
                <MessageCircle size={20} strokeWidth={1.5} />
              </button>
          </div>
        </div>
      </section>

       {/* News Section */}
      <section className="mb-20">
        <div className="flex items-baseline justify-between mb-8 border-b border-anthropic-text/10 pb-4">
          <h2 className="text-3xl font-serif text-anthropic-text">News</h2>
          <span className="text-sm font-sans text-anthropic-gray/60 uppercase tracking-widest">Latest Updates</span>
        </div>
        
        <div className="max-h-60 overflow-y-auto pr-4 scrollbar-thin scrollbar-thumb-anthropic-stone scrollbar-track-transparent">
          <div className="space-y-6">
            {NEWS_ITEMS.map((news, index) => (
              <div key={index} className="flex gap-6 group">
                <span className="text-anthropic-gray/60 font-mono text-sm whitespace-nowrap pt-1 w-24 flex-shrink-0">
                  [{news.date}]
                </span>
                <p className="text-anthropic-gray font-light leading-relaxed group-hover:text-anthropic-text transition-colors">
                  {renderNewsContent(news.content, news.link)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Publications */}
      <section>
        <div className="flex items-baseline justify-between mb-8 border-b border-anthropic-text/10 pb-4">
          <h2 className="text-3xl font-serif text-anthropic-text">Publications</h2>
          <span className="text-sm font-sans text-anthropic-gray/60 uppercase tracking-widest">Selected Papers</span>
        </div>

        <div className="space-y-8">
          {PUBLICATIONS.map((pub) => (
            <div key={pub.id} className="group border-l-4 pl-6 md:pl-8 py-4 pr-4 rounded-r-lg transition-colors duration-300 border-anthropic-stone hover:border-anthropic-accent">
              <h3 className="text-xl md:text-2xl font-serif text-anthropic-text mb-3 leading-tight">
                {pub.link ? (
                  <a href={pub.link} target="_blank" rel="noreferrer" className="hover:text-anthropic-accent transition-colors">
                    {pub.title}
                  </a>
                ) : (
                  pub.title
                )}
              </h3>

              <p className="text-anthropic-gray text-base mb-3 font-light">
                {pub.authors.map((author, idx) => (
                  <span key={idx} className={author.includes("Rui Wu") ? "font-bold text-anthropic-text" : ""}>
                    {author}{idx < pub.authors.length - 1 ? ", " : ""}
                  </span>
                ))}
              </p>

              <div className="text-sm text-anthropic-gray/80 italic mb-4">
                {pub.venue} ({pub.year})
              </div>

              <div className="flex gap-4">
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
