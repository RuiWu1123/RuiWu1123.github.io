import React from 'react';

type Lang = 'en' | 'zh';

/**
 * Plots computed from the post's own equations. Each sweeps one parameter over
 * time instead of offering a slider; with reduced motion they hold one frame.
 */

const INK = '#191919';
const W = 400;
const H = 190;
const PAD = { l: 40, r: 14, t: 16, b: 34 };

// ---------------------------------------------------------------- math

/** GRPO weight for a pass/fail verifier: E_m[sqrt((G-1-m)/(1+m))] + E_m[sqrt(m/(G-m))], m ~ Bin(G-1, p). */
const wG = (p: number, G: number) => {
  const n = G - 1;
  let pmf = Math.pow(1 - p, n); // P(m = 0)
  let total = 0;
  for (let m = 0; m <= n; m++) {
    total += pmf * (Math.sqrt((n - m) / (1 + m)) + Math.sqrt(m / (G - m)));
    pmf *= ((n - m) / (m + 1)) * (p / (1 - p));
  }
  return total;
};

/** Time for the exploit share h to grow from h0 under dh/dt = 2 w(h) h^2 (1-h)^2, as (t, h) points. */
const takeoff = (h0: number, w: (h: number) => number) => {
  const hs: number[] = [];
  for (let i = 0; i <= 160; i++) hs.push(h0 * Math.pow(0.5 / h0, i / 160));
  for (let i = 1; i <= 80; i++) hs.push(0.5 + (0.48 * i) / 80);
  const rate = (h: number) => 2 * w(h) * h * h * (1 - h) * (1 - h);
  const pts: [number, number][] = [[0, h0]];
  let t = 0;
  for (let i = 1; i < hs.length; i++) {
    const [a, b] = [hs[i - 1], hs[i]];
    t += ((b - a) * (1 / rate(a) + 1 / rate(b))) / 2;
    pts.push([t, b]);
  }
  return pts;
};

/** Shared propensity u = e^z with du/dt = eps*a*u^2 + b*u^3, integrated in log u. */
const A = 1;
const B = 1;
const U0 = 0.01;
const blowup = (eps: number) => {
  const pts: [number, number][] = [];
  let t = 0;
  const steps = 400;
  const [l0, l1] = [Math.log(U0), Math.log(1e4)];
  const dtdl = (l: number) => {
    const u = Math.exp(l);
    return 1 / (eps * A * u + B * u * u);
  };
  for (let i = 0; i <= steps; i++) {
    const l = l0 + ((l1 - l0) * i) / steps;
    if (i > 0) t += (((l1 - l0) / steps) * (dtdl(l - (l1 - l0) / steps) + dtdl(l))) / 2;
    const u = Math.exp(l);
    pts.push([t, u / (1 + u)]);
  }
  return { pts, T: t };
};

// ---------------------------------------------------------------- drawing

const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const path = (pts: [number, number][], x: (v: number) => number, y: (v: number) => number) =>
  pts.map(([a, b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)} ${y(b).toFixed(1)}`).join(' ');

const Txt: React.FC<{ x: number; y: number; anchor?: 'start' | 'middle' | 'end'; size?: number; children: React.ReactNode }> = ({
  x,
  y,
  anchor = 'middle',
  size = 10,
  children,
}) => (
  <text x={x} y={y} textAnchor={anchor} fontSize={size} fill={INK} fillOpacity={0.75} stroke="none">
    {children}
  </text>
);

const Axes: React.FC<{ x0: number; x1: number; y0: number; y1: number; xLabel: string; yLabel: React.ReactNode }> = ({
  x0,
  x1,
  y0,
  y1,
  xLabel,
  yLabel,
}) => (
  <g stroke={INK} strokeOpacity={0.45} strokeWidth={1}>
    <path d={`M${x0} ${y1} L${x0} ${y0} L${x1} ${y0}`} fill="none" />
    <Txt x={(x0 + x1) / 2} y={y0 + 26}>
      {xLabel}
    </Txt>
    <text
      x={x0 - 28}
      y={(y0 + y1) / 2}
      fontSize={10}
      fill={INK}
      fillOpacity={0.75}
      stroke="none"
      textAnchor="middle"
      transform={`rotate(-90 ${x0 - 28} ${(y0 + y1) / 2})`}
    >
      {yLabel}
    </text>
  </g>
);

const Tick: React.FC<{ x: number; y: number; label: string; axis: 'x' | 'y' }> = ({ x, y, label, axis }) =>
  axis === 'x' ? (
    <g>
      <path d={`M${x} ${y} L${x} ${y + 3}`} stroke={INK} strokeOpacity={0.45} />
      <Txt x={x} y={y + 13} size={8.5}>
        {label}
      </Txt>
    </g>
  ) : (
    <g>
      <path d={`M${x - 3} ${y} L${x} ${y}`} stroke={INK} strokeOpacity={0.45} />
      <Txt x={x - 5} y={y + 3} anchor="end" size={8.5}>
        {label}
      </Txt>
    </g>
  );

/** Steps through `frames` back and forth, pausing at both ends; holds `still` under reduced motion. */
const useSweep = (frames: number, still: number, stepMs = 90, pause = 12) => {
  const [i, setI] = React.useState(still);
  React.useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
    const cycle = 2 * (frames - 1 + pause);
    let k = 0;
    const id = window.setInterval(() => {
      k = (k + 1) % cycle;
      const half = frames - 1 + pause;
      const pos = k < half ? Math.min(k, frames - 1) : Math.max(frames - 1 - (k - half), 0);
      setI(pos);
    }, stepMs);
    return () => window.clearInterval(id);
  }, [frames, pause, stepMs]);
  return i;
};

const Frame: React.FC<{ caption: string; children: React.ReactNode }> = ({ caption, children }) => (
  <figure className="my-6">
    <div className="mx-auto max-w-[520px]">{children}</div>
    <figcaption className="mx-auto mt-1.5 max-w-[520px] text-[12px] leading-[1.45] text-anthropic-gray/80">{caption}</figcaption>
  </figure>
);

// ---------------------------------------------------------------- 1. GRPO weight

const G_VALUES = Array.from({ length: 31 }, (_, k) => k + 2);

export const GrpoWeightPlot: React.FC<{ lang: Lang }> = ({ lang }) => {
  const curves = React.useMemo(
    () =>
      G_VALUES.map((G) => {
        const pts: [number, number][] = [];
        for (let i = 0; i <= 120; i++) {
          const p = 0.002 + (0.996 * i) / 120;
          pts.push([p, wG(p, G)]);
        }
        return pts;
      }),
    [],
  );
  const i = useSweep(G_VALUES.length, 6);
  const G = G_VALUES[i];
  const [x0, x1, y0, y1] = [PAD.l, W - PAD.r, H - PAD.b, PAD.t];
  const x = scale(0, 1, x0, x1);
  const y = scale(0, 6, y0, y1);
  const top = Math.sqrt(G - 1);

  return (
    <Frame
      caption={
        lang === 'en'
          ? 'The GRPO weight w_G(p) as the group size G grows from 2 to 32. It is symmetric about p = 1/2; for G ≥ 4 it is smallest there, while for G = 2 and 3 it is flat. As p → 0 it rises to √(G − 1), so a lone success in a group gets the largest advantage.'
          : '组大小 G 从 2 增大到 32 时 GRPO 的权重 w_G(p)。它关于 p = 1/2 对称；G ≥ 4 时在那里最小，G = 2 和 3 时是一条水平线。p → 0 时升到 √(G − 1)，所以组里唯一的成功得到最大的 advantage。'
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="GRPO weight as a function of the pass rate, for growing group size">
        <Axes x0={x0} x1={x1} y0={y0} y1={y1} xLabel={lang === 'en' ? 'pass rate p' : '通过率 p'} yLabel={
            <>
              w<tspan baselineShift="sub" fontSize={8}>G</tspan>(p)
            </>
          }
        />
        {[0, 0.5, 1].map((v) => (
          <Tick key={v} x={x(v)} y={y0} label={String(v)} axis="x" />
        ))}
        {[0, 2, 4, 6].map((v) => (
          <Tick key={v} x={x0} y={y(v)} label={String(v)} axis="y" />
        ))}
        <path d={`M${x0} ${y(top)} L${x1} ${y(top)}`} stroke={INK} strokeOpacity={0.3} strokeDasharray="3 3" />
        <Txt x={x1} y={y(top) - 4} anchor="end" size={9}>
          √(G − 1)
        </Txt>
        <path d={path(curves[i], x, y)} fill="none" stroke={INK} strokeWidth={1.6} />
        <Txt x={x(0.5)} y={y1 + 4} size={11}>
          G = {G}
        </Txt>
      </svg>
    </Frame>
  );
};

// ---------------------------------------------------------------- 2. take-off on one prompt

const H0_VALUES = Array.from({ length: 41 }, (_, k) => Math.pow(10, -1 - (2 * k) / 40)); // 0.1 → 0.001
const G_TAKEOFF = 8;

export const TakeoffPlot: React.FC<{ lang: Lang }> = ({ lang }) => {
  const curves = React.useMemo(
    () =>
      H0_VALUES.map((h0) => ({
        reinforce: takeoff(h0, () => 1),
        grpo: takeoff(h0, (h) => wG(Math.min(Math.max(h, 1e-6), 1 - 1e-6), G_TAKEOFF)),
      })),
    [],
  );
  const i = useSweep(H0_VALUES.length, 20);
  const h0 = H0_VALUES[i];
  const [x0, x1, y0, y1] = [PAD.l, W - PAD.r, H - PAD.b, PAD.t];
  const [lt0, lt1] = [-1, 4];
  const lx = scale(lt0, lt1, x0, x1);
  const x = (t: number) => lx(Math.log10(Math.max(t, Math.pow(10, lt0))));
  const y = scale(0, 1, y0, y1);
  const tR = 1 / (2 * h0);
  const tG = 1 / (2 * Math.sqrt(G_TAKEOFF - 1) * h0);

  return (
    <Frame
      caption={
        lang === 'en'
          ? `Exploit share h(t) on one prompt where only exploits pass, from dh/dt = 2w(h)h²(1 − h)², as the starting share h₀ shrinks from 0.1 to 0.001 (GRPO with G = ${G_TAKEOFF}). Both curves sit near zero, then rise abruptly; the take-off arrives on the time scale 1/(2w(0)h₀), marked by the dotted lines, about √(G − 1) times sooner under GRPO in gradient-flow time, a factor that comes from standardizing the advantages.`
          : `在只有 exploit 能通过的单个 prompt 上，由 dh/dt = 2w(h)h²(1 − h)² 得到的 exploit 占比 h(t)；初始占比 h₀ 从 0.1 缩小到 0.001（GRPO 取 G = ${G_TAKEOFF}）。两条曲线都先在零附近徘徊，然后陡然上升；take-off 发生在时间尺度 1/(2w(0)h₀) 上（虚线），按梯度流时间计，GRPO 大约早 √(G − 1) 倍，这个因子来自 advantage 的标准化。`
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Exploit share over time on one prompt, for shrinking initial share">
        <Axes
          x0={x0}
          x1={x1}
          y0={y0}
          y1={y1}
          xLabel={lang === 'en' ? 'training time (log scale)' : '训练时间（对数刻度）'}
          yLabel={lang === 'en' ? 'exploit share h' : 'exploit 占比 h'}
        />
        {[-1, 0, 1, 2, 3, 4].map((v) => (
          <Tick key={v} x={lx(v)} y={y0} label={`10${v < 0 ? '⁻¹' : ['⁰', '¹', '²', '³', '⁴'][v]}`} axis="x" />
        ))}
        {[0, 0.5, 1].map((v) => (
          <Tick key={v} x={x0} y={y(v)} label={String(v)} axis="y" />
        ))}
        <path d={`M${x(tR)} ${y0} L${x(tR)} ${y1}`} stroke={INK} strokeOpacity={0.3} strokeDasharray="1.5 2.5" />
        <path d={`M${x(tG)} ${y0} L${x(tG)} ${y1}`} stroke={INK} strokeOpacity={0.3} strokeDasharray="1.5 2.5" />
        <path d={path(curves[i].reinforce, x, y)} fill="none" stroke={INK} strokeWidth={1.6} strokeDasharray="4 3" />
        <path d={path(curves[i].grpo, x, y)} fill="none" stroke={INK} strokeWidth={1.6} />
        <g>
          <path d={`M${x0 + 8} ${y1 + 4} L${x0 + 26} ${y1 + 4}`} stroke={INK} strokeWidth={1.6} />
          <Txt x={x0 + 30} y={y1 + 7} anchor="start" size={9}>
            GRPO
          </Txt>
          <path d={`M${x0 + 8} ${y1 + 16} L${x0 + 26} ${y1 + 16}`} stroke={INK} strokeWidth={1.6} strokeDasharray="4 3" />
          <Txt x={x0 + 30} y={y1 + 19} anchor="start" size={9}>
            REINFORCE
          </Txt>
        </g>
        <Txt x={x1} y={y1 + 7} anchor="end" size={11}>
          h₀ = {h0 < 0.01 ? h0.toExponential(1) : h0.toFixed(3)}
        </Txt>
      </svg>
    </Frame>
  );
};

// ---------------------------------------------------------------- 3. damping lone exploits

const EPS_VALUES = Array.from({ length: 41 }, (_, k) => Math.pow(10, -(4 * k) / 40)); // 1 → 1e-4

export const DampingPlot: React.FC<{ lang: Lang }> = ({ lang }) => {
  const runs = React.useMemo(() => EPS_VALUES.map((eps) => blowup(eps)), []);
  const i = useSweep(EPS_VALUES.length, 20);
  const eps = EPS_VALUES[i];
  const Tmax = Math.max(...runs.map((r) => r.T));

  // left: share over time; right: onset time against eps
  const split = 214;
  const [ly0, ly1] = [H - PAD.b, PAD.t];
  const lx = scale(0, Tmax * 1.05, PAD.l, split - 18);
  const ly = scale(0, 1, ly0, ly1);
  const rx0 = split + 34;
  const rxl = scale(-4, 0, rx0, W - PAD.r);
  const lTmin = Math.log10(Math.min(...runs.map((r) => r.T)));
  const lTmax = Math.log10(Tmax);
  const ryl = scale(lTmin - 0.1, lTmax + 0.15, ly0, ly1);
  const onsetCurve: [number, number][] = runs.map((r, k) => [Math.log10(EPS_VALUES[k]), Math.log10(r.T)]);

  return (
    <Frame
      caption={
        lang === 'en'
          ? 'Damping the effect of lone exploits on other prompts by a factor ε, in du/dt = εa·u² + b·u³ for the shared propensity u (a = b = 1, u₀ = 0.01). Left: the exploit share over time as ε falls from 1 to 10⁻⁴. Right: the take-off time, which first grows like 1/ε and then saturates. In this model, damping delays the take-off but does not stop it. The polynomial approximation holds only while exploits are rare, so the curves past that point are illustrative. The constants are arbitrary, so only the shapes carry meaning.'
          : '把孤立 exploit 对其他 prompt 的影响压到 ε 倍，共享倾向 u 满足 du/dt = εa·u² + b·u³（a = b = 1，u₀ = 0.01）。左：ε 从 1 降到 10⁻⁴ 时 exploit 占比随时间的变化。右：take-off 时间先按 1/ε 增长，随后饱和。在这个模型里，压制只能推迟 take-off，不能阻止它。多项式近似只在 exploit 还很少时成立，超出这一阶段的曲线只是示意。常数是随意取的，只有曲线形状有意义。'
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Exploit share over time and take-off time, as the damping factor shrinks">
        <Axes
          x0={PAD.l}
          x1={split - 18}
          y0={ly0}
          y1={ly1}
          xLabel={lang === 'en' ? 'training time' : '训练时间'}
          yLabel={lang === 'en' ? 'exploit share' : 'exploit 占比'}
        />
        {[0, 0.5, 1].map((v) => (
          <Tick key={v} x={PAD.l} y={ly(v)} label={String(v)} axis="y" />
        ))}
        <path d={path(runs[i].pts, lx, ly)} fill="none" stroke={INK} strokeWidth={1.6} />
        <path d={`M${lx(runs[i].T)} ${ly0} L${lx(runs[i].T)} ${ly1}`} stroke={INK} strokeOpacity={0.3} strokeDasharray="1.5 2.5" />
        <Txt x={split - 18} y={ly1 + 4} anchor="end" size={11}>
          ε = {eps < 0.01 ? eps.toExponential(0) : eps.toFixed(2)}
        </Txt>

        <Axes
          x0={rx0}
          x1={W - PAD.r}
          y0={ly0}
          y1={ly1}
          xLabel={lang === 'en' ? 'damping ε (log)' : '压制系数 ε（对数）'}
          yLabel={lang === 'en' ? 'take-off time (log)' : 'take-off 时间（对数）'}
        />
        {[-4, -2, 0].map((v) => (
          <Tick key={v} x={rxl(v)} y={ly0} label={v === 0 ? '1' : `10${v === -2 ? '⁻²' : '⁻⁴'}`} axis="x" />
        ))}
        <path d={path(onsetCurve, rxl, ryl)} fill="none" stroke={INK} strokeWidth={1.2} strokeOpacity={0.55} />
        <circle cx={rxl(Math.log10(eps))} cy={ryl(Math.log10(runs[i].T))} r={3.5} fill={INK} />
      </svg>
    </Frame>
  );
};
