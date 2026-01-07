
import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ArrowLeft, Bot } from 'lucide-react';
import { BLOG_POSTS, loadBlogContent } from '../constants';

// Function to parse inline markdown (bold, italic, links)
const parseInlineMarkdown = (text: string): React.ReactNode[] => {
  const parts: React.ReactNode[] = [];
  let remainingText = text;
  let key = 0;

  while (remainingText.length > 0) {
    // Match links: [text](url)
    const linkMatch = remainingText.match(/\[([^\]]+)\]\(([^)]+)\)/);
    if (linkMatch && linkMatch.index !== undefined) {
      // Add text before link
      if (linkMatch.index > 0) {
        const before = remainingText.substring(0, linkMatch.index);
        parts.push(<span key={key++}>{before}</span>);
      }
      // Add link
      parts.push(
        <a
          key={key++}
          href={linkMatch[2]}
          target="_blank"
          rel="noreferrer"
          className="text-anthropic-accent underline hover:text-anthropic-text transition-colors"
        >
          {linkMatch[1]}
        </a>
      );
      remainingText = remainingText.substring(linkMatch.index + linkMatch[0].length);
      continue;
    }

    // Match bold: **text** or __text__
    const boldMatch = remainingText.match(/\*\*([^*]+)\*\*|__([^_]+)__/);
    if (boldMatch && boldMatch.index !== undefined) {
      if (boldMatch.index > 0) {
        const before = remainingText.substring(0, boldMatch.index);
        parts.push(<span key={key++}>{before}</span>);
      }
      parts.push(<strong key={key++} className="font-semibold">{boldMatch[1] || boldMatch[2]}</strong>);
      remainingText = remainingText.substring(boldMatch.index + boldMatch[0].length);
      continue;
    }

    // Match italic: *text* (single asterisk)
    const italicMatch = remainingText.match(/\*([^*]+)\*/);
    if (italicMatch && italicMatch.index !== undefined) {
      if (italicMatch.index > 0) {
        const before = remainingText.substring(0, italicMatch.index);
        parts.push(<span key={key++}>{before}</span>);
      }
      parts.push(<em key={key++} className="italic">{italicMatch[1]}</em>);
      remainingText = remainingText.substring(italicMatch.index + italicMatch[0].length);
      continue;
    }

    // No more markdown found, add remaining text
    parts.push(<span key={key++}>{remainingText}</span>);
    break;
  }

  return parts;
};

// Function to parse content and create text blocks with embedded images
const parseContentWithImages = (content: string) => {
  const lines = content.split('\n');
  const result: (string | { type: 'image'; src: string; alt?: string; caption?: string })[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check for standard Markdown image syntax: ![alt](path)
    const imageMatch = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imageMatch) {
      const alt = imageMatch[1].trim();
      const imagePath = imageMatch[2].trim();

      // Check if next line is a caption (starts with ^ or is italic text)
      let caption: string | undefined;
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1].trim();
        // If next line starts with ^ or is wrapped in * (italic), treat as caption
        if (nextLine.startsWith('^') || (nextLine.startsWith('*') && nextLine.endsWith('*'))) {
          caption = nextLine.replace(/^\^/, '').replace(/^\*/, '').replace(/\*$/, '');
          i++; // Skip the caption line
        }
      }

      result.push({
        type: 'image',
        src: imagePath.startsWith('/') ? imagePath : `/${imagePath}`,
        alt: alt || 'Blog illustration',
        caption
      });
    } else {
      result.push(line);
    }
  }

  return result;
};

const Blog: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [blogContent, setBlogContent] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const postId = searchParams.get('id');
  const activePost = BLOG_POSTS.find(p => p.id === postId);

  // Load blog content when activePost changes
  useEffect(() => {
    if (activePost) {
      setLoading(true);
      loadBlogContent(activePost.id).then(content => {
        setBlogContent(content);
        setLoading(false);
      });
    } else {
      setBlogContent('');
    }
  }, [activePost]);

  // Function to handle navigation
  const handlePostClick = (id: string) => {
    setSearchParams({ id });
    window.scrollTo(0, 0);
  };

  const handleBack = () => {
    setSearchParams({});
    window.scrollTo(0, 0);
  };

  return (
    <div className="animate-fade-in pt-12 pb-20">
      <div className="max-w-4xl mx-auto">
        
        {/* Detail View */}
        {activePost ? (
          <div>
            <button 
              onClick={handleBack}
              className="group flex items-center text-sm text-anthropic-gray hover:text-anthropic-accent transition-colors mb-8"
            >
              <ArrowLeft size={16} className="mr-1 group-hover:-translate-x-1 transition-transform" />
              Back to Blogs
            </button>

            <article className="animate-fade-in">
              <div className="flex flex-wrap items-center gap-4 mb-6">
                <span className="text-anthropic-accent font-mono text-sm tracking-wide">{activePost.date}</span>
                {activePost.generator && (
                  <span className="flex items-center gap-1.5 px-2 py-0.5 bg-anthropic-stone/50 rounded-full text-xs font-medium text-anthropic-text/70 border border-anthropic-text/5">
                    <Bot size={12} />
                    Generated by {activePost.generator}
                  </span>
                )}
              </div>
              
              <h1 className="text-3xl md:text-5xl font-serif text-anthropic-text mb-10 leading-tight">
                {activePost.title}
              </h1>
              
              <div className="prose prose-lg max-w-none text-anthropic-text text-lg font-normal leading-relaxed whitespace-pre-line border-t border-anthropic-text/10 pt-10">
                {loading ? (
                  <div className="text-center py-8">
                    <p className="text-anthropic-gray/60">Loading...</p>
                  </div>
                ) : (
                  parseContentWithImages(blogContent).map((item, idx) => {
                    if (typeof item === 'string') {
                      // Check if paragraph is the disclaimer (starts with *This blog)
                      if (item.trim().startsWith('*This blog')) {
                        return (
                          <div key={idx} className="bg-anthropic-stone/30 p-6 rounded-lg border border-anthropic-text/5 mb-8">
                            <p className="text-sm text-anthropic-text/80 italic m-0">
                              {item.replace(/\*/g, '')}
                            </p>
                          </div>
                        );
                      }
                      // Check for h2 heading (##)
                      if (item.trim().startsWith('## ')) {
                        return <h2 key={idx} className="text-3xl font-serif text-anthropic-text mb-6 mt-12">{parseInlineMarkdown(item.replace(/^## /, ''))}</h2>;
                      }
                      // Check for h3 heading (###)
                      if (item.trim().startsWith('### ')) {
                        return <h3 key={idx} className="text-2xl font-serif text-anthropic-text mb-4 mt-10">{parseInlineMarkdown(item.replace(/^### /, ''))}</h3>;
                      }
                      return item.trim() && <p key={idx} className="mb-6">{parseInlineMarkdown(item)}</p>;
                    } else if (item.type === 'image') {
                      return (
                        <figure key={idx} className="my-6 flex flex-col items-center">
                          <img
                            src={item.src}
                            alt={item.alt}
                            className="w-full max-w-sm md:max-w-lg lg:max-w-xl h-auto rounded-lg shadow-sm border border-anthropic-text/5 object-cover"
                            loading="lazy"
                          />
                          {item.caption && (
                            <figcaption className="mt-3 text-sm text-anthropic-gray/70 text-center max-w-sm md:max-w-lg lg:max-w-xl italic">
                              {item.caption}
                            </figcaption>
                          )}
                        </figure>
                      );
                    }
                    return null;
                  })
                )}
              </div>
            </article>
          </div>
        ) : (
          /* Directory View */
          <div>
            <h1 className="text-4xl md:text-5xl font-serif font-light mb-8 text-anthropic-text">Thoughts & Updates</h1>
            <p className="text-xl font-sans font-light text-anthropic-gray mb-16 max-w-2xl">
             Thoughts on research, alignment, and small pieces of life I want to share.
            </p>

            <div className="border-t border-anthropic-text/10">
              {BLOG_POSTS.length > 0 ? (
                <div className="divide-y divide-anthropic-text/10">
                  {BLOG_POSTS.map((post) => (
                    <div 
                      key={post.id} 
                      onClick={() => handlePostClick(post.id)}
                      className="group py-8 cursor-pointer hover:bg-anthropic-stone/20 -mx-4 px-4 rounded-lg transition-colors duration-200"
                    >
                      <div className="flex flex-col md:flex-row md:items-baseline gap-2 md:gap-8">
                        <div className="w-32 flex-shrink-0">
                          <span className="text-anthropic-gray/60 font-mono text-sm">{post.date}</span>
                        </div>
                        <div className="flex-grow">
                          <h3 className="text-2xl font-serif text-anthropic-text group-hover:text-anthropic-accent transition-colors mb-2">
                            {post.title}
                          </h3>
                          <div className="flex items-center gap-2">
                            {post.generator && (
                              <span className="inline-flex items-center gap-1 text-xs text-anthropic-gray/50 uppercase tracking-wider">
                                <Bot size={12} />
                                {post.generator}
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="hidden md:block text-anthropic-gray/30 group-hover:text-anthropic-accent transform group-hover:translate-x-2 transition-all">
                          →
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-20 bg-anthropic-stone/20 rounded-lg border border-dashed border-anthropic-text/10 mt-8">
                  <p className="text-anthropic-gray font-serif text-lg italic">
                    "Research is formalized curiosity. It is poking and prying with a purpose."
                  </p>
                  <p className="text-sm text-anthropic-gray/60 mt-4 font-sans uppercase tracking-widest">
                    Content coming soon
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Blog;
