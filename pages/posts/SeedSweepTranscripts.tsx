import React from 'react';

interface Rollout {
  n: number;
  score: number;
  adv: number;
  label: string;
  focus: boolean;
  thinking: string;
  answer: string | null; // null when the thinking never closed, so nothing was graded
}

interface Group {
  key: string;
  where: string;
  title: string;
  intro: string;
  problem: string;
  rollouts: Rollout[];
}

const preClass =
  'overflow-x-auto whitespace-pre-wrap break-words rounded bg-[#F5F5F3] px-3.5 py-3 font-mono text-[11px] leading-[1.55] text-anthropic-text';
const toggleClass =
  'text-[12px] text-anthropic-gray underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';

const signed = (x: number) => (x > 0 ? `+${x.toFixed(2)}` : `−${Math.abs(x).toFixed(2)}`);

const RolloutRow: React.FC<{ r: Rollout; open: boolean; onToggle: () => void }> = ({ r, open, onToggle }) => {
  const [showThinking, setShowThinking] = React.useState(false);

  return (
    <div className="border-t border-anthropic-text/15 py-2">
      <button onClick={onToggle} className="flex w-full items-baseline gap-3 text-left group" aria-expanded={open}>
        <span className="w-3 flex-shrink-0 text-[11px] text-anthropic-gray/60">{open ? '−' : '+'}</span>
        <span className="text-[12.5px] leading-[1.5] text-anthropic-gray group-hover:text-anthropic-text transition-colors">
          <span className="text-anthropic-text">Rollout {r.n}</span>
          <span className="text-anthropic-gray/70">
            {' '}
            &middot; score {r.score.toFixed(1)} &middot; advantage {signed(r.adv)}
          </span>
          <span> &middot; {r.label}</span>
          {r.focus && <span className="italic text-anthropic-gray/70"> &middot; discussed in the text</span>}
        </span>
      </button>

      {open && (
        <div className="mt-2 space-y-2 pl-6">
          {r.answer === null ? (
            <p className="text-[12px] text-anthropic-gray/80">The thinking never closed, so nothing was graded.</p>
          ) : (
            <>
              <p className="text-[12px] text-anthropic-gray/80">Answer</p>
              <pre className={`${preClass} max-h-[28rem] overflow-y-auto`}>{r.answer}</pre>
            </>
          )}
          <button onClick={() => setShowThinking((v) => !v)} className={toggleClass}>
            {showThinking ? 'hide thinking' : `show thinking (${r.thinking.length.toLocaleString('en-US')} characters)`}
          </button>
          {showThinking && <pre className={`${preClass} max-h-[28rem] overflow-y-auto`}>{r.thinking}</pre>}
        </div>
      )}
    </div>
  );
};

/**
 * The groups of rollouts that GRPO compared for one prompt at one step, behind
 * the examples quoted in the post. Loaded on demand, since the full thinking
 * traces are large.
 */
const SeedSweepTranscripts: React.FC<{ src: string }> = ({ src }) => {
  const [groups, setGroups] = React.useState<Group[] | null>(null);
  const [failed, setFailed] = React.useState(false);
  const [active, setActive] = React.useState(0);
  const [openSet, setOpenSet] = React.useState<Set<number>>(new Set());

  React.useEffect(() => {
    fetch(src)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data: Group[]) => setGroups(data))
      .catch(() => setFailed(true));
  }, [src]);

  // Each group opens with the rollout the text discusses.
  React.useEffect(() => {
    if (!groups) return;
    setOpenSet(new Set(groups[active].rollouts.filter((r) => r.focus).map((r) => r.n)));
  }, [groups, active]);

  if (failed) return <p className="text-[12.5px] text-anthropic-gray/70">The transcripts could not be loaded.</p>;
  if (!groups) return <p className="text-[12.5px] text-anthropic-gray/70">Loading transcripts&hellip;</p>;

  const g = groups[active];
  const toggle = (n: number) =>
    setOpenSet((prev) => {
      const next = new Set(prev);
      next.has(n) ? next.delete(n) : next.add(n);
      return next;
    });

  return (
    <div>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5" role="tablist">
        {groups.map((x, i) => (
          <button
            key={x.key}
            role="tab"
            aria-selected={i === active}
            onClick={() => setActive(i)}
            className={
              'text-[12px] transition-colors ' +
              (i === active
                ? 'text-anthropic-text underline underline-offset-4 decoration-anthropic-text'
                : 'text-anthropic-gray/60 hover:text-anthropic-text')
            }
          >
            {x.key}. {x.where}
          </button>
        ))}
      </div>

      <div className="mt-5">
        <p className="font-serif text-anthropic-text">{g.title}</p>
        <p className="mt-1 text-[12.5px] leading-[1.5] text-anthropic-gray">{g.intro}</p>

        <details className="mt-3">
          <summary className={`${toggleClass} cursor-pointer`}>the problem, as given to the model</summary>
          <pre className={`${preClass} mt-2`}>{g.problem}</pre>
        </details>

        <div className="mt-3 border-b border-anthropic-text/15">
          {g.rollouts.map((r) => (
            <RolloutRow key={`${g.key}-${r.n}`} r={r} open={openSet.has(r.n)} onToggle={() => toggle(r.n)} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default SeedSweepTranscripts;
