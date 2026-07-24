import React, { useEffect, useState } from 'react';

type Lang = 'en' | 'zh';

export function slugify(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, '')
    .replace(/\s+/g, '-');
}

export interface TocItem {
  id: string;
  title: string;
}

export function extractHeadings(markdown: string): TocItem[] {
  const lines = markdown.split('\n');
  const items: TocItem[] = [];
  for (const raw of lines) {
    const line = raw.trim();
    // Only top-level numbered sections (##), not ### / #### subheadings.
    if (line.startsWith('## ')) {
      const title = line.slice(3).trim();
      if (title) items.push({ id: slugify(title), title });
    }
  }
  return items;
}

const STR: Record<Lang, { label: string }> = {
  en: { label: 'On this page' },
  zh: { label: '本页目录' },
};

function useActiveHeading(items: TocItem[]): string | null {
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    if (items.length === 0) return;
    // Give ReactMarkdown a tick to render the headings before we try to observe them.
    const timer = window.setTimeout(() => {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) setActiveId(entry.target.id);
          });
        },
        { rootMargin: '-96px 0px -70% 0px', threshold: 0 }
      );
      items.forEach((item) => {
        const el = document.getElementById(item.id);
        if (el) observer.observe(el);
      });
      return () => observer.disconnect();
    }, 50);
    return () => window.clearTimeout(timer);
  }, [items]);

  return activeId;
}

function scrollToHeading(id: string) {
  const el = document.getElementById(id);
  if (el) {
    const y = el.getBoundingClientRect().top + window.scrollY - 88;
    window.scrollTo({ top: y, behavior: 'smooth' });
  }
}

const TocList: React.FC<{ items: TocItem[]; activeId: string | null; onNavigate: (id: string) => void }> = ({
  items,
  activeId,
  onNavigate,
}) => (
  <ul className="space-y-2">
    {items.map((item) => (
      <li key={item.id}>
        <a
          href={`#${item.id}`}
          onClick={(e) => {
            e.preventDefault();
            onNavigate(item.id);
          }}
          className={`block text-sm leading-snug border-l-2 pl-3 -ml-px transition-colors ${
            activeId === item.id
              ? 'border-anthropic-accent text-anthropic-accent font-medium'
              : 'border-anthropic-text/10 text-anthropic-gray hover:text-anthropic-text hover:border-anthropic-text/30'
          }`}
        >
          {item.title}
        </a>
      </li>
    ))}
  </ul>
);

/**
 * Sticky sidebar meant to sit in a grid column next to the article. Stays
 * pinned near the top of the viewport as the reader scrolls through the
 * whole post, and highlights whichever section is currently in view.
 */
export const TableOfContentsSidebar: React.FC<{ items: TocItem[]; lang: Lang }> = ({ items, lang }) => {
  const activeId = useActiveHeading(items);
  const t = STR[lang];

  if (items.length < 3) return null;

  return (
    <nav aria-label={t.label} className="sticky top-24 max-h-[calc(100vh-7rem)] overflow-y-auto pr-2">
      <div className="text-xs uppercase tracking-wide text-anthropic-gray/60 mb-3 font-sans">{t.label}</div>
      <TocList items={items} activeId={activeId} onNavigate={scrollToHeading} />
    </nav>
  );
};

/** Collapsible "On this page" box for screens too narrow for the sidebar. */
export const TableOfContentsMobile: React.FC<{ items: TocItem[]; lang: Lang }> = ({ items, lang }) => {
  const activeId = useActiveHeading(items);
  const [open, setOpen] = useState(false);
  const t = STR[lang];

  if (items.length < 3) return null;

  return (
    <div className="not-prose mb-10 rounded-xl border border-anthropic-text/10 bg-anthropic-stone/20">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 text-sm font-medium text-anthropic-text"
      >
        <span>{t.label}</span>
        <span className={`text-anthropic-gray/60 transition-transform ${open ? 'rotate-180' : ''}`}>⌄</span>
      </button>
      {open && (
        <div className="px-5 pb-5">
          <TocList
            items={items}
            activeId={activeId}
            onNavigate={(id) => {
              scrollToHeading(id);
              setOpen(false);
            }}
          />
        </div>
      )}
    </div>
  );
};
