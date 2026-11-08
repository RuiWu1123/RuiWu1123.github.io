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
 *   - item                a bulleted list, one "- " line per item
 *   +++ Title             a collapsed "technical note", closed by a line "+++";
 *                         its body is itself post markup
 *   anything else         a paragraph
 *
 * Inside a paragraph: $...$ is inline math, `...` is code, **...** is bold, and
 * [[key|text]] is a citation linking to links[key], the cited work itself.
 */

export const SectionHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="!mt-10 mb-4 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
    <h2 className="text-lg font-serif text-anthropic-text">{children}</h2>
  </div>
);

const citeClass =
  'underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';

// Bold comes first and is non-greedy, so a bold run-in may itself contain math such as $r^*$.
const INLINE = /(\*\*.+?\*\*|\$[^$]+\$|`[^`]+`|\[\[[^\]|]+\|[^\]]+\]\])/g;

const renderInline = (text: string, links: Record<string, string>): React.ReactNode[] =>
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
          {renderInline(part.slice(2, -2), links)}
        </b>
      );
    if (part.startsWith('[[') && part.endsWith(']]')) {
      const [key, label] = part.slice(2, -2).split('|');
      return (
        <a key={i} href={links[key]} target="_blank" rel="noreferrer" className={citeClass}>
          {label}
        </a>
      );
    }
    return part;
  });

export const PostText: React.FC<{
  source: string;
  links?: Record<string, string>;
  figures?: Record<string, React.ReactNode>;
}> = ({ source, links = {}, figures = {} }) => {
  // Collapsed notes are cut out first, so that their blank lines do not split them.
  const parts = source.trim().split(/^\+\+\+ (.+)\n([\s\S]*?)\n\+\+\+$/m);
  if (parts.length > 1) {
    const out: React.ReactNode[] = [];
    for (let k = 0; k < parts.length; k += 3) {
      if (parts[k].trim()) out.push(<PostText key={`t${k}`} source={parts[k]} links={links} figures={figures} />);
      if (k + 2 < parts.length)
        out.push(
          <details key={`d${k}`} className="border-l border-anthropic-text/30 pl-4">
            <summary className="cursor-pointer text-[13px] text-anthropic-gray/80 hover:text-anthropic-text">
              {renderInline(parts[k + 1], links)}
            </summary>
            <div className="mt-3 space-y-4 text-[13px]">
              <PostText source={parts[k + 2]} links={links} figures={figures} />
            </div>
          </details>,
        );
    }
    return <>{out}</>;
  }
  const join = (b: string) => b.replace(/\s*\n\s*/g, ' ').trim();
  const raw = source.trim().split(/\n\s*\n/);

  return (
    <>
      {raw.map((rawBlock, i) => {
        if (rawBlock.startsWith('- '))
          return (
            <ul key={i} className="list-disc space-y-1.5 pl-5">
              {rawBlock
                .slice(2)
                .split(/\n- /)
                .map((item, j) => (
                  <li key={j}>{renderInline(join(item), links)}</li>
                ))}
            </ul>
          );
        const block = join(rawBlock);
        if (!block) return null;
        if (block.startsWith('::')) return <React.Fragment key={i}>{figures[block.slice(2)]}</React.Fragment>;
        if (block.startsWith('## ')) return <SectionHeading key={i}>{renderInline(block.slice(3), links)}</SectionHeading>;
        if (block.startsWith('$$') && block.endsWith('$$')) return <MD key={i}>{block.slice(2, -2).trim()}</MD>;
        if (block.startsWith('> '))
          return (
            <div key={i} className="border border-anthropic-text px-4 py-3">
              {block
                .slice(2)
                .split(/(\$\$[^$]+\$\$)/)
                .map((seg) => seg.trim())
                .filter(Boolean)
                .map((seg, j) =>
                  seg.startsWith('$$') ? <MD key={j}>{seg.slice(2, -2).trim()}</MD> : <p key={j}>{renderInline(seg, links)}</p>,
                )}
            </div>
          );
        return <p key={i}>{renderInline(block, links)}</p>;
      })}
    </>
  );
};
