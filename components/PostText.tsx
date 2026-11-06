import React from 'react';
import { M, MD } from './Math';

/**
 * A small markup for long-form posts, so that their text reads as text rather
 * than as JSX. Blocks are separated by blank lines:
 *
 *   ## Heading            a section heading (centred, with the rule from the home page)
 *   $$ ... $$             display math
 *   **Run-in.** text      a paragraph opening with a bold run-in heading
 *   > **Label.** text     a boxed callout; it may contain $$ ... $$ display math
 *   ::name                the figure passed in as figures[name]
 *   anything else         a paragraph
 *
 * Inside a paragraph: $...$ is inline math, `...` is code, **...** is bold, and
 * [[key|text]] is a citation that scrolls to the reference with id ref-key.
 */

export const SectionHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="!mt-10 mb-4 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
    <h2 className="text-lg font-serif text-anthropic-text">{children}</h2>
  </div>
);

const citeClass =
  'underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';

const goToReference = (key: string) => {
  const el = document.getElementById(`ref-${key}`);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.classList.add('bg-[#F5F5F3]');
  window.setTimeout(() => el.classList.remove('bg-[#F5F5F3]'), 1600);
};

const INLINE = /(\$[^$]+\$|`[^`]+`|\*\*[^*]+\*\*|\[\[[^\]|]+\|[^\]]+\]\])/g;

const renderInline = (text: string): React.ReactNode[] =>
  text.split(INLINE).map((part, i) => {
    if (part.startsWith('$') && part.endsWith('$') && part.length > 1) return <M key={i}>{part.slice(1, -1)}</M>;
    if (part.startsWith('`') && part.endsWith('`') && part.length > 1)
      return (
        <code key={i} className="font-mono text-[0.88em]">
          {part.slice(1, -1)}
        </code>
      );
    if (part.startsWith('**') && part.endsWith('**') && part.length > 3)
      return (
        <b key={i} className="text-anthropic-text">
          {part.slice(2, -2)}
        </b>
      );
    if (part.startsWith('[[') && part.endsWith(']]')) {
      const [key, label] = part.slice(2, -2).split('|');
      return (
        <a
          key={i}
          href={`#ref-${key}`}
          onClick={(e) => {
            e.preventDefault(); // the site routes on the hash, so scroll instead of navigating
            goToReference(key);
          }}
          className={citeClass}
        >
          {label}
        </a>
      );
    }
    return part;
  });

export const PostText: React.FC<{ source: string; figures?: Record<string, React.ReactNode> }> = ({
  source,
  figures = {},
}) => {
  const blocks = source
    .trim()
    .split(/\n\s*\n/)
    .map((b) => b.replace(/\s*\n\s*/g, ' ').trim())
    .filter(Boolean);

  return (
    <>
      {blocks.map((block, i) => {
        if (block.startsWith('::')) return <React.Fragment key={i}>{figures[block.slice(2)]}</React.Fragment>;
        if (block.startsWith('## ')) return <SectionHeading key={i}>{renderInline(block.slice(3))}</SectionHeading>;
        if (block.startsWith('$$') && block.endsWith('$$')) return <MD key={i}>{block.slice(2, -2).trim()}</MD>;
        if (block.startsWith('> '))
          return (
            <div key={i} className="rounded border border-anthropic-text/15 bg-[#F5F5F3] px-4 py-3">
              {block
                .slice(2)
                .split(/(\$\$[^$]+\$\$)/)
                .map((seg) => seg.trim())
                .filter(Boolean)
                .map((seg, j) =>
                  seg.startsWith('$$') ? <MD key={j}>{seg.slice(2, -2).trim()}</MD> : <p key={j}>{renderInline(seg)}</p>,
                )}
            </div>
          );
        return <p key={i}>{renderInline(block)}</p>;
      })}
    </>
  );
};
