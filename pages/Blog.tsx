
import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ArrowLeft } from 'lucide-react';
import { BLOG_POSTS, loadBlogContent } from '../constants';

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
              </div>
              
              <h1 className="text-3xl md:text-5xl font-serif text-anthropic-text mb-10 leading-tight">
                {activePost.title}
              </h1>
              
              <div className="max-w-none text-anthropic-text text-lg font-normal leading-relaxed border-t border-anthropic-text/10 pt-10">
                {loading ? (
                  <div className="text-center py-8">
                    <p className="text-anthropic-gray/60">Loading...</p>
                  </div>
                ) : (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({ children }) => (
                        <h1 className="text-4xl font-serif text-anthropic-text mb-8 mt-12 leading-tight">
                          {children}
                        </h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-3xl font-serif text-anthropic-text mb-6 mt-12 leading-tight">
                          {children}
                        </h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-2xl font-serif text-anthropic-text mb-4 mt-10 leading-tight">
                          {children}
                        </h3>
                      ),
                      h4: ({ children }) => (
                        <h4 className="text-xl font-serif text-anthropic-text mb-3 mt-8 leading-tight">
                          {children}
                        </h4>
                      ),
                      p: ({ children }) => (
                        <p className="mb-6 text-anthropic-text">
                          {children}
                        </p>
                      ),
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          target={href?.startsWith('#') ? '_self' : '_blank'}
                          rel={href?.startsWith('#') ? undefined : 'noreferrer'}
                          className="text-anthropic-accent underline decoration-anthropic-accent/50 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors"
                        >
                          {children}
                        </a>
                      ),
                      ul: ({ children }) => (
                        <ul className="list-disc pl-6 mb-6 space-y-2">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal pl-6 mb-6 space-y-2">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => (
                        <li className="pl-1">
                          {children}
                        </li>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-4 border-anthropic-accent/50 bg-anthropic-stone/30 pl-5 pr-4 py-3 my-8 italic text-anthropic-text/80">
                          {children}
                        </blockquote>
                      ),
                      code: ({ className, children }) => {
                        const isBlock = Boolean(className);
                        if (!isBlock) {
                          return (
                            <code className="px-1.5 py-0.5 rounded bg-anthropic-stone/60 text-[0.9em] font-mono text-anthropic-text">
                              {children}
                            </code>
                          );
                        }

                        return (
                          <code className={`${className} block overflow-x-auto whitespace-pre`}>
                            {children}
                          </code>
                        );
                      },
                      pre: ({ children }) => (
                        <pre className="mb-6 overflow-x-auto rounded-lg border border-anthropic-text/10 bg-[#191919] p-4 text-sm leading-relaxed text-[#F4F3EF]">
                          {children}
                        </pre>
                      ),
                      img: ({ src, alt }) => (
                        <img
                          src={src || ''}
                          alt={alt || 'Blog illustration'}
                          className="mx-auto my-6 w-full max-w-sm md:max-w-lg lg:max-w-xl h-auto rounded-lg shadow-sm border border-anthropic-text/5 object-cover"
                          loading="lazy"
                        />
                      ),
                      table: ({ children }) => (
                        <div className="mb-6 overflow-x-auto">
                          <table className="w-full border-collapse text-left text-base">
                            {children}
                          </table>
                        </div>
                      ),
                      th: ({ children }) => (
                        <th className="border border-anthropic-text/10 bg-anthropic-stone/40 px-3 py-2 font-semibold">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="border border-anthropic-text/10 px-3 py-2 align-top">
                          {children}
                        </td>
                      ),
                      hr: () => (
                        <hr className="my-10 border-anthropic-text/10" />
                      ),
                    }}
                  >
                    {blogContent}
                  </ReactMarkdown>
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
