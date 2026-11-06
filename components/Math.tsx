import React from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const render = (tex: string, displayMode: boolean) =>
  katex.renderToString(tex, { displayMode, throwOnError: false, strict: false });

/** Inline TeX. */
export const M: React.FC<{ children: string }> = ({ children }) => (
  <span dangerouslySetInnerHTML={{ __html: render(children, false) }} />
);

/** Display TeX, scrolling sideways on narrow screens rather than overflowing the page. */
export const MD: React.FC<{ children: string }> = ({ children }) => (
  <div className="my-4 overflow-x-auto" dangerouslySetInnerHTML={{ __html: render(children, true) }} />
);
