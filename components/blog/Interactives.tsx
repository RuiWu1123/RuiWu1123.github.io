import React, { useMemo, useState } from 'react';

type Lang = 'en' | 'zh';

/* ------------------------------------------------------------------ */
/*  Shared small building blocks                                       */
/* ------------------------------------------------------------------ */

const Card: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="not-prose my-8 rounded-xl border border-anthropic-text/10 bg-anthropic-stone/20 p-5 md:p-6">
    {children}
  </div>
);

const Pill: React.FC<{ active: boolean; onClick: () => void; children: React.ReactNode }> = ({
  active,
  onClick,
  children,
}) => (
  <button
    onClick={onClick}
    className={`px-3 py-1.5 rounded-full text-xs md:text-sm font-medium border transition-colors ${
      active
        ? 'bg-anthropic-text text-anthropic-bg border-anthropic-text'
        : 'bg-transparent text-anthropic-text border-anthropic-text/20 hover:border-anthropic-accent hover:text-anthropic-accent'
    }`}
  >
    {children}
  </button>
);

/* ------------------------------------------------------------------ */
/*  1. Roofline Explorer                                               */
/* ------------------------------------------------------------------ */

type Precision = 'fp32' | 'tf32' | 'bf16' | 'fp8';

const PRECISION_PEAK_TFLOPS: Record<Precision, number> = {
  // Dense (non-sparse) H100 SXM5 peak TFLOPS, per NVIDIA's H100 datasheet
  fp32: 67,
  tf32: 495,
  bf16: 990,
  fp8: 1979,
};

const PEAK_BW_TB_S = 3.35; // H100 SXM5 HBM3, ~3.35 TB/s

const OP_PRESETS: { key: string; label: { en: string; zh: string }; logAI: number }[] = [
  { key: 'elementwise', label: { en: 'Elementwise add / ReLU', zh: 'Elementwise 加法 / ReLU' }, logAI: Math.log10(0.25) },
  { key: 'norm', label: { en: 'LayerNorm / Softmax', zh: 'LayerNorm / Softmax' }, logAI: Math.log10(2) },
  { key: 'attn', label: { en: 'Attention, small batch', zh: '小 batch 的 Attention' }, logAI: Math.log10(20) },
  { key: 'matmul', label: { en: 'Large matmul, big batch', zh: '大 batch 的大矩阵乘法' }, logAI: Math.log10(400) },
];

const STR = {
  en: {
    rooflineTitle: 'Interactive: The Roofline Model',
    rooflineSub: 'Is a kernel limited by compute, or by data movement? Pick a precision and an operation profile.',
    precision: 'Tensor Core precision (peak, dense)',
    opProfile: 'Operation profile (sets arithmetic intensity)',
    orDrag: 'or drag the arithmetic intensity yourself:',
    ai: 'Arithmetic intensity',
    achieved: 'Achievable throughput',
    utilization: 'of peak',
    regime: 'Regime',
    memBound: 'Memory-bound',
    computeBound: 'Compute-bound',
    memBoundNote: 'Every extra byte of bandwidth buys you more FLOPs. Fusing kernels / reusing data on-chip is what helps here — more compute won’t.',
    computeBoundNote: 'You’re already saturating the Tensor Cores. Only a faster/lower-precision datatype or algorithmic FLOP reduction helps here.',
    ridge: 'Ridge point (compute = bandwidth)',
    xAxis: 'Arithmetic intensity (FLOPs / byte moved, log scale)',
    yAxis: 'Throughput (TFLOPs/s, log scale)',
  },
  zh: {
    rooflineTitle: '交互演示：Roofline 模型',
    rooflineSub: '一个 kernel 到底是被算力卡住，还是被数据搬运卡住？选一个精度和一种算子画像试试。',
    precision: 'Tensor Core 精度（峰值，dense）',
    opProfile: '算子画像（决定 arithmetic intensity）',
    orDrag: '或者自己拖动 arithmetic intensity：',
    ai: 'Arithmetic Intensity',
    achieved: '可达吞吐',
    utilization: '峰值利用率',
    regime: '所处区域',
    memBound: 'Memory-bound（受限于内存带宽）',
    computeBound: 'Compute-bound（受限于算力）',
    memBoundNote: '每多一点带宽就能多换一点 FLOPs。这里真正有用的是 fuse kernel、在片上复用数据，而不是堆更多算力。',
    computeBoundNote: '此时 Tensor Core 已经跑满了。只有更快/更低精度的数据类型，或者从算法层面减少 FLOPs，才能再提升。',
    ridge: 'Ridge point（算力 = 带宽 的临界点）',
    xAxis: 'Arithmetic Intensity（每搬运 1 byte 对应的 FLOPs，log 坐标）',
    yAxis: '吞吐（TFLOPs/s，log 坐标）',
  },
};

export const RooflineExplorer: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR[lang];
  const [precision, setPrecision] = useState<Precision>('bf16');
  const [logAI, setLogAI] = useState<number>(Math.log10(20));

  const peakCompute = PRECISION_PEAK_TFLOPS[precision]; // TFLOPs/s
  const ridgeAI = peakCompute / PEAK_BW_TB_S; // FLOPs/byte, since TB/s == TFLOPs-equivalent per byte

  const ai = Math.pow(10, logAI);
  const achieved = Math.min(peakCompute, ai * PEAK_BW_TB_S);
  const isMemBound = ai < ridgeAI;
  const utilization = (achieved / peakCompute) * 100;

  // --- plotting geometry (log-log) ---
  const xMinLog = -1; // AI = 0.1
  const xMaxLog = 3; // AI = 1000
  const yMinLog = -1; // 0.1 TFLOPs
  const yMaxLog = 3.15; // ~1400 TFLOPs, headroom above fp8 peak

  const W = 640;
  const H = 340;
  const padL = 56;
  const padR = 20;
  const padT = 20;
  const padB = 46;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xOf = (aiVal: number) => {
    const l = Math.max(xMinLog, Math.min(xMaxLog, Math.log10(aiVal)));
    return padL + ((l - xMinLog) / (xMaxLog - xMinLog)) * plotW;
  };
  const yOf = (tVal: number) => {
    const l = Math.max(yMinLog, Math.min(yMaxLog, Math.log10(tVal)));
    return padT + plotH - ((l - yMinLog) / (yMaxLog - yMinLog)) * plotH;
  };

  // Roofline path: diagonal from left edge up to ridge point, then flat to right edge
  const leftAI = Math.pow(10, xMinLog);
  const leftAchieved = leftAI * PEAK_BW_TB_S;
  const rightAI = Math.pow(10, xMaxLog);

  const diagPath = `M ${xOf(leftAI)} ${yOf(leftAchieved)} L ${xOf(ridgeAI)} ${yOf(peakCompute)}`;
  const flatPath = `M ${xOf(ridgeAI)} ${yOf(peakCompute)} L ${xOf(rightAI)} ${yOf(peakCompute)}`;

  const gridlinesX = [-1, 0, 1, 2, 3];
  const gridlinesY = [-1, 0, 1, 2, 3];

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.rooflineTitle}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.rooflineSub}</p>

      <div className="mb-3">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.precision}</div>
        <div className="flex flex-wrap gap-2">
          {(Object.keys(PRECISION_PEAK_TFLOPS) as Precision[]).map((p) => (
            <Pill key={p} active={precision === p} onClick={() => setPrecision(p)}>
              {p.toUpperCase()} · {PRECISION_PEAK_TFLOPS[p]} TFLOPs
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-3">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.opProfile}</div>
        <div className="flex flex-wrap gap-2">
          {OP_PRESETS.map((op) => (
            <Pill key={op.key} active={Math.abs(op.logAI - logAI) < 0.05} onClick={() => setLogAI(op.logAI)}>
              {op.label[lang]}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.orDrag}</div>
        <input
          type="range"
          min={-1}
          max={3}
          step={0.01}
          value={logAI}
          onChange={(e) => setLogAI(parseFloat(e.target.value))}
          className="w-full accent-anthropic-accent"
        />
      </div>

      <div className="w-full overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-2xl mx-auto" style={{ minWidth: 480 }}>
          {/* Axes */}
          <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#191919" strokeOpacity={0.4} />
          <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="#191919" strokeOpacity={0.4} />

          {/* gridlines + tick labels */}
          {gridlinesX.map((g) => (
            <g key={`x-${g}`}>
              <line x1={xOf(Math.pow(10, g))} y1={padT} x2={xOf(Math.pow(10, g))} y2={padT + plotH} stroke="#191919" strokeOpacity={0.06} />
              <text x={xOf(Math.pow(10, g))} y={padT + plotH + 16} fontSize={10} textAnchor="middle" fill="#6B6B6B">
                {Math.pow(10, g) >= 1 ? Math.pow(10, g) : Math.pow(10, g).toFixed(1)}
              </text>
            </g>
          ))}
          {gridlinesY.map((g) => (
            <g key={`y-${g}`}>
              <line x1={padL} y1={yOf(Math.pow(10, g))} x2={padL + plotW} y2={yOf(Math.pow(10, g))} stroke="#191919" strokeOpacity={0.06} />
              <text x={padL - 8} y={yOf(Math.pow(10, g)) + 3} fontSize={10} textAnchor="end" fill="#6B6B6B">
                {Math.pow(10, g) >= 1 ? Math.pow(10, g) : Math.pow(10, g).toFixed(1)}
              </text>
            </g>
          ))}

          {/* axis titles */}
          <text x={padL + plotW / 2} y={H - 6} fontSize={11} textAnchor="middle" fill="#191919">
            {t.xAxis}
          </text>
          <text x={14} y={padT + plotH / 2} fontSize={11} textAnchor="middle" fill="#191919" transform={`rotate(-90 14 ${padT + plotH / 2})`}>
            {t.yAxis}
          </text>

          {/* Roofline curve */}
          <path d={diagPath} stroke="#D97757" strokeWidth={3} fill="none" />
          <path d={flatPath} stroke="#D97757" strokeWidth={3} fill="none" />

          {/* Ridge point marker */}
          <circle cx={xOf(ridgeAI)} cy={yOf(peakCompute)} r={3.5} fill="#191919" />
          <text x={xOf(ridgeAI)} y={yOf(peakCompute) - 10} fontSize={9.5} textAnchor="middle" fill="#191919">
            {t.ridge}
          </text>

          {/* Current point */}
          <line x1={xOf(ai)} y1={padT} x2={xOf(ai)} y2={yOf(achieved)} stroke="#8DA399" strokeDasharray="3 3" />
          <line x1={xOf(ai)} y1={yOf(achieved)} x2={padL} y2={yOf(achieved)} stroke="#8DA399" strokeDasharray="3 3" />
          <circle cx={xOf(ai)} cy={yOf(achieved)} r={6} fill="#8DA399" stroke="#191919" strokeWidth={1.2} />
        </svg>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 text-sm">
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.ai}</div>
          <div className="text-anthropic-text font-mono">{ai < 1 ? ai.toFixed(2) : ai.toFixed(0)} FLOPs/B</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.achieved}</div>
          <div className="text-anthropic-text font-mono">{achieved.toFixed(0)} TFLOPs/s</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.utilization}</div>
          <div className="text-anthropic-text font-mono">{utilization.toFixed(0)}%</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.regime}</div>
          <div className={`font-medium ${isMemBound ? 'text-anthropic-leaf' : 'text-anthropic-accent'}`}>
            {isMemBound ? t.memBound : t.computeBound}
          </div>
        </div>
      </div>

      <p className="text-xs text-anthropic-gray mt-4 leading-relaxed">
        {isMemBound ? t.memBoundNote : t.computeBoundNote}
      </p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  2. Grid / Block / SM Occupancy Simulator                           */
/* ------------------------------------------------------------------ */

const STR2 = {
  en: {
    title: 'Interactive: Grid → Blocks → SMs',
    sub: 'How your launch configuration maps onto a GPU with 132 SMs (H100-scale), and how many "waves" it takes.',
    threadsPerBlock: 'Threads per block',
    totalBlocks: 'Total blocks launched',
    warpsPerBlock: 'Warps / block',
    residentBlocksPerSM: 'Resident blocks / SM',
    residentWarpsPerSM: 'Resident warps / SM',
    occupancy: 'Occupancy',
    waves: 'Waves needed',
    smLabel: '132 SMs (simplified view, showing 22)',
    waveNote: 'Only one wave of blocks can be resident on the GPU at a time; the rest queue up and run once earlier blocks finish.',
    capNote: 'Simplified model: assumes 64 resident warps/SM max and ignores register & shared-memory limits, which in practice usually bind first.',
  },
  zh: {
    title: '交互演示：Grid → Block → SM',
    sub: '你的 launch 配置会如何映射到一块拥有 132 个 SM 的 GPU（H100 量级）上，又需要跑几个 "wave"。',
    threadsPerBlock: '每个 block 的线程数',
    totalBlocks: '总共发射的 block 数',
    warpsPerBlock: '每个 block 的 warp 数',
    residentBlocksPerSM: '每个 SM 常驻的 block 数',
    residentWarpsPerSM: '每个 SM 常驻的 warp 数',
    occupancy: 'Occupancy（占用率）',
    waves: '需要的 wave 数',
    smLabel: '132 个 SM（简化展示 22 个）',
    waveNote: '同一时刻只有一个 wave 的 block 能常驻在 GPU 上，其余的排队等待，等前面的 block 跑完再补上。',
    capNote: '简化模型：假设每个 SM 最多常驻 64 个 warp，并忽略了寄存器与 shared memory 的限制——而实际中这些往往才是先被打满的资源。',
  },
};

const THREADS_OPTIONS = [32, 64, 128, 256, 512, 1024];
const NUM_SMS = 132;
const MAX_WARPS_PER_SM = 64;

export const GridBlockSimulator: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR2[lang];
  const [threadsPerBlock, setThreadsPerBlock] = useState(256);
  const [totalBlocks, setTotalBlocks] = useState(512);

  const { warpsPerBlock, residentBlocksPerSM, residentWarpsPerSM, occupancy, waves, blocksPerWave } = useMemo(() => {
    const wpb = threadsPerBlock / 32;
    const rbpsm = Math.max(1, Math.floor(MAX_WARPS_PER_SM / wpb));
    const rwpsm = rbpsm * wpb;
    const occ = (rwpsm / MAX_WARPS_PER_SM) * 100;
    const bpw = rbpsm * NUM_SMS;
    const w = Math.max(1, Math.ceil(totalBlocks / bpw));
    return { warpsPerBlock: wpb, residentBlocksPerSM: rbpsm, residentWarpsPerSM: rwpsm, occupancy: occ, waves: w, blocksPerWave: bpw };
  }, [threadsPerBlock, totalBlocks]);

  const displaySMs = 22;

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.threadsPerBlock}</div>
        <div className="flex flex-wrap gap-2">
          {THREADS_OPTIONS.map((n) => (
            <Pill key={n} active={threadsPerBlock === n} onClick={() => setThreadsPerBlock(n)}>
              {n}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">
          {t.totalBlocks}: <span className="text-anthropic-text font-mono">{totalBlocks}</span>
        </div>
        <input
          type="range"
          min={1}
          max={4000}
          step={1}
          value={totalBlocks}
          onChange={(e) => setTotalBlocks(parseInt(e.target.value, 10))}
          className="w-full accent-anthropic-accent"
        />
      </div>

      {/* SM occupancy visualization */}
      <div className="mb-2 text-xs text-anthropic-gray/70">{t.smLabel}</div>
      <div className="grid grid-cols-11 gap-1.5 mb-5">
        {Array.from({ length: displaySMs }).map((_, i) => (
          <div key={i} className="relative h-8 rounded border border-anthropic-text/15 bg-anthropic-bg overflow-hidden">
            <div
              className="absolute bottom-0 left-0 right-0 bg-anthropic-leaf/70"
              style={{ height: `${occupancy}%` }}
            />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-4">
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.warpsPerBlock}</div>
          <div className="text-anthropic-text font-mono">{warpsPerBlock}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.residentBlocksPerSM}</div>
          <div className="text-anthropic-text font-mono">{residentBlocksPerSM}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.occupancy}</div>
          <div className="text-anthropic-text font-mono">{occupancy.toFixed(0)}%</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.waves}</div>
          <div className="text-anthropic-text font-mono">{waves}</div>
        </div>
      </div>

      {/* Wave timeline */}
      <div className="flex gap-1 mb-3">
        {Array.from({ length: Math.min(waves, 12) }).map((_, i) => (
          <div
            key={i}
            className="flex-1 h-6 rounded bg-anthropic-mist border border-anthropic-text/10 flex items-center justify-center text-[10px] text-anthropic-text"
          >
            {i + 1}
          </div>
        ))}
        {waves > 12 && <div className="text-xs text-anthropic-gray self-center pl-2">+{waves - 12}</div>}
      </div>

      <p className="text-xs text-anthropic-gray leading-relaxed">{t.waveNote}</p>
      <p className="text-xs text-anthropic-gray/70 leading-relaxed mt-1">{t.capNote}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  3. Triton Grid / Mask Explorer                                     */
/* ------------------------------------------------------------------ */

const STR3 = {
  en: {
    title: 'Interactive: pid, offsets, and the mask',
    sub: 'Pick a total element count N and a BLOCK_SIZE, and see how tl.program_id turns into a grid of programs — and what the mask actually masks.',
    n: 'Total elements (N)',
    blockSize: 'BLOCK_SIZE',
    programs: 'Programs launched',
    wasted: 'Masked-off lanes (wasted, but harmless)',
    wastedPct: 'of total lanes',
    program: 'program',
    elements: 'elements',
    masked: 'masked',
    more: 'more',
    note: 'The tail program almost always has some masked lanes unless N happens to divide BLOCK_SIZE exactly. That’s expected and cheap — masked lanes never touch memory — but a BLOCK_SIZE much bigger than N wastes an entire program\'s worth of parallelism for nothing.',
  },
  zh: {
    title: '交互演示：pid、offsets 与 mask',
    sub: '选择总元素数 N 和 BLOCK_SIZE，看看 tl.program_id 是怎么变成一整个 grid 的 program 的——以及 mask 到底在遮住什么。',
    n: '总元素数（N）',
    blockSize: 'BLOCK_SIZE',
    programs: '发射的 program 数',
    wasted: '被 mask 掉的 lane 数（浪费，但无害）',
    wastedPct: '占总 lane 数的比例',
    program: 'program',
    elements: '个元素',
    masked: '个被 mask',
    more: '更多',
    note: '只要 N 不能被 BLOCK_SIZE 整除，最后一个 program 几乎总会有一些被 mask 掉的 lane。这是预期行为，而且代价很小——被 mask 的 lane 根本不会碰内存——但如果 BLOCK_SIZE 比 N 大太多，就等于白白浪费了一整个 program 的并行度。',
  },
};

const N_BLOCK_OPTIONS = [4, 8, 16, 32];

export const TritonGridExplorer: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR3[lang];
  const [n, setN] = useState(37);
  const [blockSize, setBlockSize] = useState(8);

  const { numPrograms, programs, wastedLanes, wastedPct } = useMemo(() => {
    const numP = Math.max(1, Math.ceil(n / blockSize));
    const progs = Array.from({ length: numP }).map((_, i) => {
      const start = i * blockSize;
      const end = Math.min(start + blockSize, n);
      const valid = Math.max(0, end - start);
      const masked = blockSize - valid;
      return { i, start, end, valid, masked };
    });
    const wasted = numP * blockSize - n;
    const pct = (wasted / (numP * blockSize)) * 100;
    return { numPrograms: numP, programs: progs, wastedLanes: wasted, wastedPct: pct };
  }, [n, blockSize]);

  const displayLimit = 12;

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">
          {t.n}: <span className="text-anthropic-text font-mono">{n}</span>
        </div>
        <input
          type="range"
          min={1}
          max={100}
          step={1}
          value={n}
          onChange={(e) => setN(parseInt(e.target.value, 10))}
          className="w-full accent-anthropic-accent"
        />
      </div>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.blockSize}</div>
        <div className="flex flex-wrap gap-2">
          {N_BLOCK_OPTIONS.map((b) => (
            <Pill key={b} active={blockSize === b} onClick={() => setBlockSize(b)}>
              {b}
            </Pill>
          ))}
        </div>
      </div>

      <div className="space-y-1.5 mb-5">
        {programs.slice(0, displayLimit).map((p) => (
          <div key={p.i} className="flex items-center gap-2 text-xs">
            <span className="w-20 flex-shrink-0 text-anthropic-gray/70 font-mono">
              {t.program} {p.i}
            </span>
            <div className="flex gap-0.5">
              {Array.from({ length: blockSize }).map((_, lane) => (
                <div
                  key={lane}
                  className={`w-4 h-4 rounded-sm border ${
                    lane < p.valid
                      ? 'bg-anthropic-leaf/70 border-anthropic-text/20'
                      : 'bg-anthropic-stone/40 border-anthropic-text/10 border-dashed'
                  }`}
                />
              ))}
            </div>
            <span className="text-anthropic-gray/60 font-mono">
              [{p.start}, {p.end}) · {p.valid} {t.elements}
              {p.masked > 0 ? ` · ${p.masked} ${t.masked}` : ''}
            </span>
          </div>
        ))}
        {numPrograms > displayLimit && (
          <div className="text-xs text-anthropic-gray/60 pl-[5.5rem]">
            +{numPrograms - displayLimit} {t.more}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm mb-4">
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.programs}</div>
          <div className="text-anthropic-text font-mono">{numPrograms}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.wasted}</div>
          <div className="text-anthropic-text font-mono">{wastedLanes}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.wastedPct}</div>
          <div className="text-anthropic-text font-mono">{wastedPct.toFixed(0)}%</div>
        </div>
      </div>

      <p className="text-xs text-anthropic-gray leading-relaxed">{t.note}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  4. Autotuning Explorer                                             */
/* ------------------------------------------------------------------ */

const STR4 = {
  en: {
    title: 'Interactive: why autotuning is a search, not a formula',
    sub: 'Pick a BLOCK_SIZE and a num_warps. The bar is a schematic, illustrative "how close to the best config" — not a real measurement — but the shape of the trade-off is real.',
    blockSize: 'BLOCK_SIZE',
    numWarps: 'num_warps',
    relative: 'Schematic relative throughput',
    elemPerThread: 'Elements per thread (BLOCK_SIZE / (num_warps × 32))',
    tooFew: 'Fewer elements than threads: some threads in the program do nothing this launch — wasted parallelism.',
    reasonable: 'Each thread handles a modest, pipelineable chunk — usually a reasonable place to be.',
    tooMany: 'Each thread churns through many elements serially before the next warp swap — latency hiding gets harder and register pressure climbs.',
    disclaimer: 'Illustrative only: the real surface depends on the kernel, the input shape, and the specific GPU. This is why @triton.autotune measures instead of guessing.',
  },
  zh: {
    title: '交互演示：为什么 autotuning 是"搜索"而不是"公式"',
    sub: '选一个 BLOCK_SIZE 和 num_warps。下面的进度条是示意性的"距离最优配置有多近"——不是真实测量值——但这个权衡关系的形状是真实的。',
    blockSize: 'BLOCK_SIZE',
    numWarps: 'num_warps',
    relative: '示意性的相对吞吐',
    elemPerThread: '每个线程处理的元素数（BLOCK_SIZE / (num_warps × 32)）',
    tooFew: '元素数比线程数还少：这次发射里有些线程什么都不做——白白浪费并行度。',
    reasonable: '每个线程处理一小段、可流水线化的数据量——通常是比较合理的区间。',
    tooMany: '每个线程要串行处理很多元素才能轮到下一个 warp 切换——延迟隐藏变难，寄存器压力也上升。',
    disclaimer: '仅为示意：真实的最优面取决于具体 kernel、输入形状和 GPU 型号。这正是 @triton.autotune 选择"实测"而不是"猜"的原因。',
  },
};

const AT_BLOCK_OPTIONS = [64, 128, 256, 512, 1024, 2048];
const AT_WARP_OPTIONS = [1, 2, 4, 8, 16];

export const AutotuneExplorer: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR4[lang];
  const [blockSize, setBlockSize] = useState(1024);
  const [numWarps, setNumWarps] = useState(4);

  const { score, elemPerThread } = useMemo(() => {
    const logB = Math.log2(blockSize);
    const idealLogB = Math.log2(1024);
    const logW = Math.log2(numWarps);
    const idealLogW = Math.log2(4);
    const dist = Math.sqrt(Math.pow(logB - idealLogB, 2) + Math.pow(logW - idealLogW, 2));
    const s = Math.max(5, Math.round(100 * Math.exp(-0.35 * dist * dist)));
    const ept = blockSize / (numWarps * 32);
    return { score: s, elemPerThread: ept };
  }, [blockSize, numWarps]);

  const note = elemPerThread < 1 ? t.tooFew : elemPerThread > 16 ? t.tooMany : t.reasonable;
  const barColor = score >= 70 ? 'bg-anthropic-leaf' : score >= 40 ? 'bg-anthropic-accent' : 'bg-anthropic-gray/50';

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.blockSize}</div>
        <div className="flex flex-wrap gap-2">
          {AT_BLOCK_OPTIONS.map((b) => (
            <Pill key={b} active={blockSize === b} onClick={() => setBlockSize(b)}>
              {b}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.numWarps}</div>
        <div className="flex flex-wrap gap-2">
          {AT_WARP_OPTIONS.map((w) => (
            <Pill key={w} active={numWarps === w} onClick={() => setNumWarps(w)}>
              {w}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-2 text-xs uppercase tracking-wide text-anthropic-gray/70">{t.relative}</div>
      <div className="w-full h-5 rounded-full bg-anthropic-stone/40 overflow-hidden mb-1">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${score}%` }} />
      </div>
      <div className="text-right text-xs font-mono text-anthropic-text mb-4">{score}%</div>

      <div className="mb-4 text-sm">
        <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.elemPerThread}</div>
        <div className="text-anthropic-text font-mono">{elemPerThread.toFixed(2)}</div>
      </div>

      <p className="text-xs text-anthropic-gray leading-relaxed mb-3">{note}</p>
      <p className="text-xs text-anthropic-gray/60 leading-relaxed">{t.disclaimer}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  5. Ring-AllReduce vs Parameter-Server Explorer                     */
/* ------------------------------------------------------------------ */

const STR5 = {
  en: {
    title: 'Interactive: why Ring-AllReduce scales and a parameter server doesn’t',
    sub: 'Pick a GPU count and a gradient size, and compare per-GPU traffic under a naive parameter server vs Ring-AllReduce.',
    numGpus: 'Number of GPUs (N)',
    gradSize: 'Gradient size',
    serverTraffic: 'Parameter server: busiest node’s traffic',
    ringTraffic: 'Ring-AllReduce: traffic per GPU',
    serverFormula: '(N−1) × size — grows with N',
    ringFormula: '2(N−1)/N × size — approaches 2× size, flat',
    note: 'The server’s incoming traffic grows linearly with every GPU you add — it eventually becomes the whole bottleneck. Every ring participant’s traffic converges to about 2× the gradient size, almost independent of how many GPUs are in the ring.',
  },
  zh: {
    title: '交互演示：为什么 Ring-AllReduce 能扩展，而 parameter server 不能',
    sub: '选一个 GPU 数量和梯度大小，对比朴素 parameter server 和 Ring-AllReduce 下每个节点的通信量。',
    numGpus: 'GPU 数量（N）',
    gradSize: '梯度大小',
    serverTraffic: 'Parameter server：最忙节点的通信量',
    ringTraffic: 'Ring-AllReduce：每个 GPU 的通信量',
    serverFormula: '(N−1) × size —— 随 N 增长',
    ringFormula: '2(N−1)/N × size —— 趋近于 2× size，基本走平',
    note: '每加一块 GPU，server 端要承受的流入流量都会线性增长——它最终会变成整个系统的瓶颈。而 ring 里每个参与者的通信量都收敛到大约 2 倍梯度大小，几乎和 ring 里有多少块 GPU 无关。',
  },
};

const RING_GPU_OPTIONS = [4, 8, 16, 32, 64, 128];
const RING_SIZE_OPTIONS = [1, 4, 16, 64];

export const RingAllReduceExplorer: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR5[lang];
  const [numGpus, setNumGpus] = useState(8);
  const [gradSize, setGradSize] = useState(4);

  const { serverTraffic, ringTraffic, maxTraffic } = useMemo(() => {
    const server = (numGpus - 1) * gradSize;
    const ring = ((2 * (numGpus - 1)) / numGpus) * gradSize;
    return { serverTraffic: server, ringTraffic: ring, maxTraffic: Math.max(server, ring) };
  }, [numGpus, gradSize]);

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.numGpus}</div>
        <div className="flex flex-wrap gap-2">
          {RING_GPU_OPTIONS.map((n) => (
            <Pill key={n} active={numGpus === n} onClick={() => setNumGpus(n)}>
              {n}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.gradSize}</div>
        <div className="flex flex-wrap gap-2">
          {RING_SIZE_OPTIONS.map((s) => (
            <Pill key={s} active={gradSize === s} onClick={() => setGradSize(s)}>
              {s} GB
            </Pill>
          ))}
        </div>
      </div>

      <div className="space-y-4 mb-4">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-anthropic-gray/70 uppercase tracking-wide">{t.serverTraffic}</span>
            <span className="text-anthropic-text font-mono">{serverTraffic.toFixed(1)} GB</span>
          </div>
          <div className="w-full h-5 rounded-full bg-anthropic-stone/40 overflow-hidden">
            <div
              className="h-full rounded-full bg-anthropic-accent"
              style={{ width: `${(serverTraffic / maxTraffic) * 100}%` }}
            />
          </div>
          <div className="text-xs text-anthropic-gray/60 mt-1 font-mono">{t.serverFormula}</div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-anthropic-gray/70 uppercase tracking-wide">{t.ringTraffic}</span>
            <span className="text-anthropic-text font-mono">{ringTraffic.toFixed(1)} GB</span>
          </div>
          <div className="w-full h-5 rounded-full bg-anthropic-stone/40 overflow-hidden">
            <div
              className="h-full rounded-full bg-anthropic-leaf"
              style={{ width: `${(ringTraffic / maxTraffic) * 100}%` }}
            />
          </div>
          <div className="text-xs text-anthropic-gray/60 mt-1 font-mono">{t.ringFormula}</div>
        </div>
      </div>

      <p className="text-xs text-anthropic-gray leading-relaxed">{t.note}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  6. ZeRO Memory Calculator                                          */
/* ------------------------------------------------------------------ */

const STR6 = {
  en: {
    title: 'Interactive: how much does ZeRO actually save?',
    sub: 'Pick a model size and a GPU count, then compare per-GPU memory for mixed-precision Adam across plain DDP and each ZeRO stage.',
    modelSize: 'Model size (parameters)',
    numGpus: 'Number of GPUs (N)',
    stage: 'Partitioning strategy',
    perGpuMem: 'Per-GPU memory (model states only)',
    warning: 'This alone exceeds a single GPU’s HBM — partitioning (or offloading) isn’t optional at this size.',
    fine: 'Comfortably fits alongside activations on a modern GPU.',
    ddp: 'Plain DDP (no partitioning)',
    z1: 'ZeRO-1 (optimizer states)',
    z2: 'ZeRO-2 (+ gradients)',
    z3: 'ZeRO-3 (+ parameters)',
    formula: 'formula (Φ = parameters, in billions)',
  },
  zh: {
    title: '交互演示：ZeRO 到底省了多少显存？',
    sub: '选一个模型大小和 GPU 数量，对比 plain DDP 和各个 ZeRO stage 下，mixed-precision Adam 每张 GPU 的显存占用。',
    modelSize: '模型大小（参数量）',
    numGpus: 'GPU 数量（N）',
    stage: '切分策略',
    perGpuMem: '每张 GPU 的显存（仅 model states）',
    warning: '光是这一项就已经超过单张 GPU 的 HBM 了——到这个规模，partition（或者 offload）已经不是可选项。',
    fine: '和激活值一起放在一张现代 GPU 上，绰绰有余。',
    ddp: 'Plain DDP（不切分）',
    z1: 'ZeRO-1（切分优化器状态）',
    z2: 'ZeRO-2（+ 切分梯度）',
    z3: 'ZeRO-3（+ 切分参数）',
    formula: '公式（Φ = 参数量，单位：十亿）',
  },
};

type ZeroStage = 'ddp' | 'z1' | 'z2' | 'z3';
const MODEL_SIZE_OPTIONS = [1, 7, 13, 70, 175];
const ZERO_GPU_OPTIONS = [8, 16, 64, 256];

export const ZeROMemoryCalculator: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR6[lang];
  const [modelSize, setModelSize] = useState(70);
  const [numGpus, setNumGpus] = useState(64);
  const [stage, setStage] = useState<ZeroStage>('z2');

  const { perGpuGB, formula } = useMemo(() => {
    const phi = modelSize; // billions of params; 1 byte/param == 1 GB per Φ
    let gb = 0;
    let f = '';
    if (stage === 'ddp') {
      gb = 16 * phi;
      f = '(2 + 2 + 12) × Φ';
    } else if (stage === 'z1') {
      gb = 2 * phi + 2 * phi + (12 * phi) / numGpus;
      f = '2Φ + 2Φ + 12Φ/N';
    } else if (stage === 'z2') {
      gb = 2 * phi + (2 * phi + 12 * phi) / numGpus;
      f = '2Φ + 14Φ/N';
    } else {
      gb = (16 * phi) / numGpus;
      f = '16Φ/N';
    }
    return { perGpuGB: gb, formula: f };
  }, [modelSize, numGpus, stage]);

  const overLimit = perGpuGB > 80;

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.modelSize}</div>
        <div className="flex flex-wrap gap-2">
          {MODEL_SIZE_OPTIONS.map((m) => (
            <Pill key={m} active={modelSize === m} onClick={() => setModelSize(m)}>
              {m}B
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.numGpus}</div>
        <div className="flex flex-wrap gap-2">
          {ZERO_GPU_OPTIONS.map((n) => (
            <Pill key={n} active={numGpus === n} onClick={() => setNumGpus(n)}>
              {n}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.stage}</div>
        <div className="flex flex-wrap gap-2">
          <Pill active={stage === 'ddp'} onClick={() => setStage('ddp')}>{t.ddp}</Pill>
          <Pill active={stage === 'z1'} onClick={() => setStage('z1')}>{t.z1}</Pill>
          <Pill active={stage === 'z2'} onClick={() => setStage('z2')}>{t.z2}</Pill>
          <Pill active={stage === 'z3'} onClick={() => setStage('z3')}>{t.z3}</Pill>
        </div>
      </div>

      <div className="mb-2">
        <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide mb-1">{t.perGpuMem}</div>
        <div className={`text-3xl font-mono ${overLimit ? 'text-anthropic-accent' : 'text-anthropic-leaf'}`}>
          {perGpuGB.toFixed(1)} GB
        </div>
        <div className="text-xs text-anthropic-gray/60 mt-1">{t.formula}: <span className="font-mono">{formula}</span></div>
      </div>

      <p className="text-xs text-anthropic-gray leading-relaxed mt-3">{overLimit ? t.warning : t.fine}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  7. Pipeline Bubble Explorer                                        */
/* ------------------------------------------------------------------ */

const STR7 = {
  en: {
    title: 'Interactive: how many microbatches do you need?',
    sub: 'Pick a pipeline depth (stages) and a microbatch count, and see what fraction of every GPU’s time is bubble — idle, waiting on the pipeline.',
    stages: 'Pipeline stages (P)',
    microbatches: 'Microbatches (M)',
    bubbleFrac: 'Bubble fraction',
    utilFrac: 'Useful-work fraction',
    formula: '(P−1) / (P−1+M)',
    lowNote: 'Too few microbatches for this many stages — a large share of every step is idle bubble, whichever schedule (GPipe or 1F1B) you use.',
    okNote: 'A reasonable working point — most of each step is useful compute.',
  },
  zh: {
    title: '交互演示：到底需要多少个 microbatch？',
    sub: '选一个 pipeline 深度（stage 数）和 microbatch 数量，看看每张 GPU 的时间里有多少比例是 bubble——也就是在等 pipeline，空转。',
    stages: 'Pipeline stage 数（P）',
    microbatches: 'Microbatch 数（M）',
    bubbleFrac: 'Bubble 占比',
    utilFrac: '有效计算占比',
    formula: '(P−1) / (P−1+M)',
    lowNote: '相对于这么多 stage，microbatch 数太少了——不管用 GPipe 还是 1F1B，每一步都有很大比例在空转。',
    okNote: '是一个还算合理的工作点——每一步里大部分时间都在做有效计算。',
  },
};

const PB_STAGE_OPTIONS = [2, 4, 8, 16];

export const PipelineBubbleExplorer: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR7[lang];
  const [stages, setStages] = useState(8);
  const [microbatches, setMicrobatches] = useState(16);

  const bubbleFrac = (stages - 1) / (stages - 1 + microbatches);
  const utilFrac = 1 - bubbleFrac;

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.stages}</div>
        <div className="flex flex-wrap gap-2">
          {PB_STAGE_OPTIONS.map((p) => (
            <Pill key={p} active={stages === p} onClick={() => setStages(p)}>
              {p}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">
          {t.microbatches}: <span className="text-anthropic-text font-mono">{microbatches}</span>
        </div>
        <input
          type="range"
          min={1}
          max={64}
          step={1}
          value={microbatches}
          onChange={(e) => setMicrobatches(parseInt(e.target.value, 10))}
          className="w-full accent-anthropic-accent"
        />
      </div>

      <div className="w-full h-7 rounded-full bg-anthropic-leaf/60 overflow-hidden mb-1 flex">
        <div className="h-full bg-anthropic-stone flex items-center justify-center" style={{ width: `${bubbleFrac * 100}%` }}>
          {bubbleFrac > 0.12 && (
            <span className="text-[10px] text-anthropic-text font-mono">{(bubbleFrac * 100).toFixed(0)}%</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm mb-4">
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.bubbleFrac}</div>
          <div className="text-anthropic-text font-mono">{(bubbleFrac * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.utilFrac}</div>
          <div className="text-anthropic-text font-mono">{(utilFrac * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div className="text-xs text-anthropic-gray/60 mb-3 font-mono">{t.formula}</div>
      <p className="text-xs text-anthropic-gray leading-relaxed">{bubbleFrac > 0.3 ? t.lowNote : t.okNote}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  8. Accelerator hardware dataset (shared)                           */
/* ------------------------------------------------------------------ */

type Vendor = 'nvidia' | 'amd' | 'tpu' | 'ascend';

interface AccelChip {
  vendor: Vendor;
  name: string;
  year: number;
  memGB: number;
  bwTBs: number;
  tdpW: number | null;
  computeTFLOPS: number | null;
  interconnect: string;
}

const VENDOR_LABEL: Record<Vendor, string> = { nvidia: 'NVIDIA', amd: 'AMD Instinct', tpu: 'Google TPU', ascend: 'Huawei Ascend' };
const VENDOR_COLOR: Record<Vendor, string> = { nvidia: '#D97757', amd: '#8DA399', tpu: '#6B8CAE', ascend: '#191919' };

const ACCEL_CHIPS: AccelChip[] = [
  { vendor: 'nvidia', name: 'P100', year: 2016, memGB: 16, bwTBs: 0.72, tdpW: 300, computeTFLOPS: 21.2, interconnect: 'NVLink 1, 160 GB/s' },
  { vendor: 'nvidia', name: 'V100', year: 2017, memGB: 32, bwTBs: 0.9, tdpW: 300, computeTFLOPS: 125, interconnect: 'NVLink 2, 300 GB/s' },
  { vendor: 'nvidia', name: 'A100 80GB', year: 2020, memGB: 80, bwTBs: 2.04, tdpW: 400, computeTFLOPS: 312, interconnect: 'NVLink 3, 600 GB/s' },
  { vendor: 'nvidia', name: 'H100 SXM5', year: 2022, memGB: 80, bwTBs: 3.35, tdpW: 700, computeTFLOPS: 990, interconnect: 'NVLink 4, 900 GB/s' },
  { vendor: 'nvidia', name: 'H200', year: 2024, memGB: 141, bwTBs: 4.8, tdpW: 700, computeTFLOPS: 990, interconnect: 'NVLink 4, 900 GB/s' },
  { vendor: 'nvidia', name: 'B200', year: 2024.5, memGB: 192, bwTBs: 8, tdpW: 1000, computeTFLOPS: 2250, interconnect: 'NVLink 5, 1.8 TB/s' },
  { vendor: 'nvidia', name: 'B300 / GB300', year: 2025.4, memGB: 288, bwTBs: 8, tdpW: 1400, computeTFLOPS: 2500, interconnect: 'NVLink 5, 1.8 TB/s' },

  { vendor: 'amd', name: 'MI100', year: 2020, memGB: 32, bwTBs: 1.2, tdpW: 300, computeTFLOPS: 184.6, interconnect: 'Infinity Fabric, ~276 GB/s' },
  { vendor: 'amd', name: 'MI250X', year: 2021, memGB: 128, bwTBs: 3.2, tdpW: 560, computeTFLOPS: 383, interconnect: 'Infinity Fabric, 8×100 GB/s' },
  { vendor: 'amd', name: 'MI300X', year: 2023, memGB: 192, bwTBs: 5.3, tdpW: 750, computeTFLOPS: 1307, interconnect: 'Infinity Fabric, 7×128 GB/s' },
  { vendor: 'amd', name: 'MI325X', year: 2024, memGB: 256, bwTBs: 6.0, tdpW: 1000, computeTFLOPS: 1307, interconnect: 'Infinity Fabric, 7×128 GB/s' },
  { vendor: 'amd', name: 'MI355X', year: 2025.5, memGB: 288, bwTBs: 8, tdpW: 1400, computeTFLOPS: 2500, interconnect: 'Infinity Fabric, 7×153.6 GB/s' },

  { vendor: 'tpu', name: 'v2', year: 2017, memGB: 8, bwTBs: 0.25, tdpW: null, computeTFLOPS: 45, interconnect: 'ICI (unpublished)' },
  { vendor: 'tpu', name: 'v3', year: 2018, memGB: 32, bwTBs: 0.9, tdpW: 220, computeTFLOPS: 123, interconnect: 'ICI' },
  { vendor: 'tpu', name: 'v4', year: 2021, memGB: 32, bwTBs: 1.2, tdpW: 192, computeTFLOPS: 275, interconnect: 'ICI, optical circuit switches' },
  { vendor: 'tpu', name: 'v5p', year: 2023, memGB: 95, bwTBs: 2.77, tdpW: null, computeTFLOPS: 459, interconnect: 'ICI, 1.2 TB/s' },
  { vendor: 'tpu', name: 'v6e (Trillium)', year: 2024.4, memGB: 32, bwTBs: 1.64, tdpW: null, computeTFLOPS: 918, interconnect: 'ICI, 800 GB/s' },
  { vendor: 'tpu', name: 'v7 (Ironwood)', year: 2025.4, memGB: 192, bwTBs: 7.38, tdpW: 1000, computeTFLOPS: 2307, interconnect: 'ICI, 1.2 TB/s + die-to-die link' },

  { vendor: 'ascend', name: '910', year: 2019, memGB: 32, bwTBs: 1.23, tdpW: 310, computeTFLOPS: 320, interconnect: 'HCCS (early)' },
  { vendor: 'ascend', name: '910B', year: 2023, memGB: 64, bwTBs: 1.6, tdpW: 400, computeTFLOPS: 400, interconnect: 'HCCS, ~336 GB/s (est.)' },
  { vendor: 'ascend', name: '910C', year: 2025, memGB: 128, bwTBs: 3.2, tdpW: null, computeTFLOPS: 800, interconnect: 'HCCS + die-to-die (unconfirmed)' },
];

/* ------------------------------------------------------------------ */
/*  9. Accelerator Trend Explorer                                      */
/* ------------------------------------------------------------------ */

type Metric = 'mem' | 'bw' | 'tdp' | 'compute';

const STR8 = {
  en: {
    title: 'Interactive: the trend lines, by metric',
    sub: 'Pick a metric and watch ten years of shipped hardware go by, one vendor at a time.',
    mem: 'Memory capacity (GB)',
    bw: 'Memory bandwidth (TB/s)',
    tdp: 'TDP (W)',
    compute: 'Dense compute (TFLOPS, BF16/FP16-class)',
    xAxis: 'Release year',
    note: 'Log-scale y-axis. Missing points mean the vendor hasn’t publicly disclosed that number for that chip — most often TPU and Ascend TDP.',
  },
  zh: {
    title: '交互演示：按指标看趋势',
    sub: '选一个指标，看看十年已出货硬件是怎么一家一家演进的。',
    mem: '显存容量（GB）',
    bw: '显存带宽（TB/s）',
    tdp: 'TDP（瓦特）',
    compute: 'Dense 算力（TFLOPS，BF16/FP16 量级）',
    xAxis: '发布年份',
    note: 'y 轴为 log 坐标。缺失的点代表该厂商没有公开披露这颗芯片的这项数据——最常见的是 TPU 和 Ascend 的 TDP。',
  },
};

export const AcceleratorTrendExplorer: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR8[lang];
  const [metric, setMetric] = useState<Metric>('mem');

  const getValue = (c: AccelChip): number | null => {
    if (metric === 'mem') return c.memGB > 0 ? c.memGB : null;
    if (metric === 'bw') return c.bwTBs > 0 ? c.bwTBs : null;
    if (metric === 'tdp') return c.tdpW;
    return c.computeTFLOPS;
  };

  const W = 680;
  const H = 380;
  const padL = 50;
  const padR = 16;
  const padT = 16;
  const padB = 40;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xMin = 2015.5;
  const xMax = 2026;

  const allVals = ACCEL_CHIPS.map(getValue).filter((v): v is number => v !== null && v > 0);
  const yMinLog = Math.floor(Math.log10(Math.min(...allVals)) - 0.2);
  const yMaxLog = Math.ceil(Math.log10(Math.max(...allVals)) + 0.1);

  const xOf = (year: number) => padL + ((year - xMin) / (xMax - xMin)) * plotW;
  const yOf = (val: number) => {
    const l = Math.log10(val);
    return padT + plotH - ((l - yMinLog) / (yMaxLog - yMinLog)) * plotH;
  };

  const vendors: Vendor[] = ['nvidia', 'amd', 'tpu', 'ascend'];

  const gridlinesY: number[] = [];
  for (let p = yMinLog; p <= yMaxLog; p++) gridlinesY.push(p);

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4 flex flex-wrap gap-2">
        <Pill active={metric === 'mem'} onClick={() => setMetric('mem')}>{t.mem}</Pill>
        <Pill active={metric === 'bw'} onClick={() => setMetric('bw')}>{t.bw}</Pill>
        <Pill active={metric === 'tdp'} onClick={() => setMetric('tdp')}>{t.tdp}</Pill>
        <Pill active={metric === 'compute'} onClick={() => setMetric('compute')}>{t.compute}</Pill>
      </div>

      <div className="w-full overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-2xl mx-auto" style={{ minWidth: 480 }}>
          <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#191919" strokeOpacity={0.4} />
          <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="#191919" strokeOpacity={0.4} />

          {gridlinesY.map((g) => (
            <g key={`y-${g}`}>
              <line x1={padL} y1={yOf(Math.pow(10, g))} x2={padL + plotW} y2={yOf(Math.pow(10, g))} stroke="#191919" strokeOpacity={0.06} />
              <text x={padL - 8} y={yOf(Math.pow(10, g)) + 3} fontSize={9.5} textAnchor="end" fill="#6B6B6B">
                {Math.pow(10, g) >= 1 ? Math.pow(10, g).toLocaleString() : Math.pow(10, g)}
              </text>
            </g>
          ))}

          {[2016, 2018, 2020, 2022, 2024, 2026].map((yr) => (
            <text key={yr} x={xOf(yr)} y={padT + plotH + 16} fontSize={9.5} textAnchor="middle" fill="#6B6B6B">
              {yr}
            </text>
          ))}
          <text x={padL + plotW / 2} y={H - 4} fontSize={10.5} textAnchor="middle" fill="#191919">
            {t.xAxis}
          </text>

          {vendors.map((v) => {
            const pts = ACCEL_CHIPS.filter((c) => c.vendor === v)
              .map((c) => ({ c, val: getValue(c) }))
              .filter((p): p is { c: AccelChip; val: number } => p.val !== null && p.val > 0)
              .sort((a, b) => a.c.year - b.c.year);
            const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.c.year)} ${yOf(p.val)}`).join(' ');
            return (
              <g key={v}>
                <path d={pathD} fill="none" stroke={VENDOR_COLOR[v]} strokeWidth={1.6} strokeOpacity={0.55} />
                {pts.map((p) => (
                  <circle key={p.c.name} cx={xOf(p.c.year)} cy={yOf(p.val)} r={4.5} fill={VENDOR_COLOR[v]} />
                ))}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex flex-wrap gap-4 mt-3 mb-3 justify-center">
        {vendors.map((v) => (
          <div key={v} className="flex items-center gap-1.5 text-xs text-anthropic-gray">
            <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: VENDOR_COLOR[v] }} />
            {VENDOR_LABEL[v]}
          </div>
        ))}
      </div>

      <p className="text-xs text-anthropic-gray/70 leading-relaxed">{t.note}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  10. Accelerator Spec Lookup                                        */
/* ------------------------------------------------------------------ */

const STR9 = {
  en: {
    title: 'Interactive: look up a chip',
    sub: 'Pick a vendor, then a chip, for the full spec rundown.',
    vendor: 'Vendor',
    chip: 'Chip',
    year: 'Year',
    memory: 'Memory',
    bandwidth: 'Memory bandwidth',
    compute: 'Dense compute (BF16/FP16-class)',
    interconnect: 'Interconnect',
    tdp: 'TDP',
    notDisclosed: 'not publicly disclosed',
  },
  zh: {
    title: '交互演示：查一颗芯片',
    sub: '先选厂商，再选具体型号，看完整的规格清单。',
    vendor: '厂商',
    chip: '型号',
    year: '发布年份',
    memory: '显存',
    bandwidth: '显存带宽',
    compute: 'Dense 算力（BF16/FP16 量级）',
    interconnect: '互联',
    tdp: 'TDP',
    notDisclosed: '未公开披露',
  },
};

export const AcceleratorSpecLookup: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR9[lang];
  const allChips = ACCEL_CHIPS;
  const [vendor, setVendor] = useState<Vendor>('nvidia');
  const chipsForVendor = allChips.filter((c) => c.vendor === vendor);
  const [chipName, setChipName] = useState(chipsForVendor[0]?.name ?? '');

  const chip = allChips.find((c) => c.vendor === vendor && c.name === chipName) ?? chipsForVendor[0];

  const handleVendor = (v: Vendor) => {
    setVendor(v);
    const first = allChips.find((c) => c.vendor === v);
    if (first) setChipName(first.name);
  };

  const vendors: Vendor[] = ['nvidia', 'amd', 'tpu', 'ascend'];

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.vendor}</div>
        <div className="flex flex-wrap gap-2">
          {vendors.map((v) => (
            <Pill key={v} active={vendor === v} onClick={() => handleVendor(v)}>
              {VENDOR_LABEL[v]}
            </Pill>
          ))}
        </div>
      </div>

      <div className="mb-5">
        <div className="text-xs uppercase tracking-wide text-anthropic-gray/70 mb-1.5">{t.chip}</div>
        <div className="flex flex-wrap gap-2">
          {chipsForVendor.map((c) => (
            <Pill key={c.name} active={chipName === c.name} onClick={() => setChipName(c.name)}>
              {c.name}
            </Pill>
          ))}
        </div>
      </div>

      {chip && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm mb-4">
            <div>
              <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.year}</div>
              <div className="text-anthropic-text font-mono">{Math.floor(chip.year)}</div>
            </div>
            <div>
              <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.memory}</div>
              <div className="text-anthropic-text font-mono">{chip.memGB > 0 ? `${chip.memGB} GB` : t.notDisclosed}</div>
            </div>
            <div>
              <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.bandwidth}</div>
              <div className="text-anthropic-text font-mono">{chip.bwTBs > 0 ? `${chip.bwTBs} TB/s` : t.notDisclosed}</div>
            </div>
            <div>
              <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.compute}</div>
              <div className="text-anthropic-text font-mono">{chip.computeTFLOPS != null ? `${chip.computeTFLOPS.toLocaleString()} TFLOPS` : t.notDisclosed}</div>
            </div>
            <div>
              <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.tdp}</div>
              <div className="text-anthropic-text font-mono">{chip.tdpW != null ? `${chip.tdpW} W` : t.notDisclosed}</div>
            </div>
          </div>
          <div className="mb-2">
            <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.interconnect}</div>
            <div className="text-anthropic-text text-sm">{chip.interconnect}</div>
          </div>
        </>
      )}
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  11. MoE Sparsity Explorer                                          */
/* ------------------------------------------------------------------ */

interface MoEPreset {
  id: string;
  label: string;
  n: number;
  k: number;
  s: number;
}

const MOE_PRESETS: MoEPreset[] = [
  { id: 'v3', label: 'DeepSeek-V3', n: 256, k: 8, s: 1 },
  { id: 'glm', label: 'GLM-5.2', n: 256, k: 8, s: 1 },
  { id: 'v4flash', label: 'DeepSeek-V4-Flash', n: 256, k: 6, s: 1 },
  { id: 'v4pro', label: 'DeepSeek-V4-Pro', n: 384, k: 6, s: 1 },
  { id: 'mai', label: 'MAI-Thinking-1', n: 512, k: 8, s: 0 },
  { id: 'kimi', label: 'Kimi K3', n: 896, k: 16, s: 1 },
  { id: 'qwen3', label: 'Qwen3-MoE (235B-A22B)', n: 128, k: 8, s: 0 },
];

const STR10 = {
  en: {
    title: 'Interactive: how sparse is a MoE layer?',
    sub: 'Pick a preset from a real 2026 model, or drag the sliders — see how much of the expert pool actually fires per token.',
    experts: 'Routed experts (N)',
    topk: 'Active per token (top-K)',
    shared: 'Shared experts (always on)',
    sparsity: 'Sparsity (K / N)',
    activeShare: 'Active share of the full pool',
    poolLabel: 'the routed expert pool',
    activeLabel: 'active this token',
    sharedLabel: 'shared, always on',
    note: 'Total parameters scale with N + S; active parameters per token scale with K + S. A bigger pool only costs more compute if K grows with it — closing that gap between total and active is the entire point of MoE.',
  },
  zh: {
    title: '交互演示：一层 MoE 到底有多稀疏？',
    sub: '选一个 2026 年真实模型的预设，或者直接拖动滑块——看看每个 token 到底激活了整个专家池的多大比例。',
    experts: '路由专家数（N）',
    topk: '每 token 激活数（top-K）',
    shared: '共享专家数（常驻激活）',
    sparsity: '稀疏度（K / N）',
    activeShare: '激活比例（占整个专家池）',
    poolLabel: '路由专家池',
    activeLabel: '本次激活',
    sharedLabel: '共享专家，常驻',
    note: '总参数量随 N + S 增长；每个 token 的激活参数量只随 K + S 增长。专家池变大并不会让每个 token 变贵，只要 K 不跟着涨——拉开总量和激活量之间的差距，正是 MoE 存在的全部意义。',
  },
};

export const MoESparsityExplorer: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR10[lang];
  const [n, setN] = useState(256);
  const [k, setK] = useState(8);
  const [s, setS] = useState(1);
  const [activePreset, setActivePreset] = useState<string | null>('v3');

  const applyPreset = (p: MoEPreset) => {
    setN(p.n);
    setK(p.k);
    setS(p.s);
    setActivePreset(p.id);
  };

  const sparsity = (k / n) * 100;
  const activeShare = ((k + s) / (n + s)) * 100;

  const barW = 600;
  const barH = 28;
  const kWidth = Math.max((k / n) * barW, 3);

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-4 flex flex-wrap gap-2">
        {MOE_PRESETS.map((p) => (
          <Pill key={p.id} active={activePreset === p.id} onClick={() => applyPreset(p)}>
            {p.label}
          </Pill>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
        <div>
          <div className="flex justify-between text-xs text-anthropic-gray/70 mb-1">
            <span>{t.experts}</span>
            <span className="font-mono text-anthropic-text">{n}</span>
          </div>
          <input
            type="range"
            min={8}
            max={1024}
            step={8}
            value={n}
            onChange={(e) => {
              setN(Number(e.target.value));
              setActivePreset(null);
            }}
            className="w-full accent-anthropic-accent"
          />
        </div>
        <div>
          <div className="flex justify-between text-xs text-anthropic-gray/70 mb-1">
            <span>{t.topk}</span>
            <span className="font-mono text-anthropic-text">{k}</span>
          </div>
          <input
            type="range"
            min={1}
            max={32}
            step={1}
            value={k}
            onChange={(e) => {
              setK(Number(e.target.value));
              setActivePreset(null);
            }}
            className="w-full accent-anthropic-accent"
          />
        </div>
        <div>
          <div className="flex justify-between text-xs text-anthropic-gray/70 mb-1">
            <span>{t.shared}</span>
            <span className="font-mono text-anthropic-text">{s}</span>
          </div>
          <input
            type="range"
            min={0}
            max={2}
            step={1}
            value={s}
            onChange={(e) => {
              setS(Number(e.target.value));
              setActivePreset(null);
            }}
            className="w-full accent-anthropic-accent"
          />
        </div>
      </div>

      <div className="w-full overflow-x-auto mb-4">
        <svg viewBox={`0 0 ${barW + 100} 90`} className="w-full max-w-2xl mx-auto" style={{ minWidth: 420 }}>
          <text x="0" y="12" fontSize={10.5} fill="#6B6B6B">
            {t.poolLabel}
          </text>
          <rect x="0" y="18" width={barW} height={barH} rx={5} fill="#FFFFFF" stroke="#191919" strokeOpacity={0.3} />
          <rect x="0" y="18" width={kWidth} height={barH} rx={5} fill="#D97757" />
          <text x={Math.min(kWidth + 6, barW - 4)} y="36" fontSize={9.5} fill="#191919">
            {t.activeLabel}
          </text>

          {Array.from({ length: s }).map((_, i) => (
            <rect key={i} x={barW + 12 + i * 26} y="18" width={22} height={barH} rx={5} fill="#8DA399" />
          ))}
          {s > 0 && (
            <text x={barW + 12} y="64" fontSize={9} fill="#6B6B6B">
              {t.sharedLabel}
            </text>
          )}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm mb-3">
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.sparsity}</div>
          <div className="text-anthropic-text font-mono text-lg">{sparsity.toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.activeShare}</div>
          <div className="text-anthropic-text font-mono text-lg">{activeShare.toFixed(1)}%</div>
        </div>
      </div>

      <p className="text-xs text-anthropic-gray/70 leading-relaxed">{t.note}</p>
    </Card>
  );
};

/* ------------------------------------------------------------------ */
/*  12. MoE Model Spec Lookup                                          */
/* ------------------------------------------------------------------ */

interface MoEModelSpec {
  id: string;
  name: string;
  org: string;
  totalB: number;
  activeB: number | null;
  routedExperts: string;
  sharedExperts: string;
  topK: string;
  gating: string;
  balancing: string;
  innovation: { en: string; zh: string };
}

const MOE_MODELS: MoEModelSpec[] = [
  {
    id: 'v3',
    name: 'DeepSeek-V3',
    org: 'DeepSeek-AI',
    totalB: 671,
    activeB: 37,
    routedExperts: '256',
    sharedExperts: '1',
    topK: '8',
    gating: 'Sigmoid',
    balancing: 'Aux-loss-free (learned bias)',
    innovation: {
      en: "Established the modern default recipe: fine-grained routed experts plus one isolated shared expert, sigmoid affinity scoring, bias-based aux-loss-free balancing, and multi-token prediction (MTP).",
      zh: '确立了如今的默认范式：细粒度路由专家 + 一个独立共享专家、sigmoid 打分、基于偏置项的免辅助损失负载均衡，以及多 token 预测（MTP）。',
    },
  },
  {
    id: 'qwen2',
    name: 'Qwen2-MoE',
    org: 'Alibaba / Qwen Team',
    totalB: 57,
    activeB: 14,
    routedExperts: '64',
    sharedExperts: '8',
    topK: '8',
    gating: 'Softmax',
    balancing: 'GShard-style auxiliary loss',
    innovation: {
      en: "Followed the Qwen1.5-MoE recipe at larger scale: fine-grained routed experts plus 8 always-on shared experts, justified explicitly as insurance against router collapse — the same diagnosis DeepSeekMoE and DeepSeek-V3 reached independently.",
      zh: '在更大规模上延续了 Qwen1.5-MoE 的配方：细粒度路由专家加 8 个常驻共享专家，官方说法直接把这个设计归因于防范路由坍缩——和 DeepSeekMoE、DeepSeek-V3 各自独立得出的判断一致。',
    },
  },
  {
    id: 'qwen3',
    name: 'Qwen3-MoE',
    org: 'Alibaba / Qwen Team',
    totalB: 235,
    activeB: 22,
    routedExperts: '128',
    sharedExperts: '0',
    topK: '8',
    gating: 'Softmax',
    balancing: 'Global-batch load balancing loss',
    innovation: {
      en: "Reverses Qwen2's own choice: drops the shared expert entirely, replacing it with a global-batch load balancing loss that synchronizes expert-selection frequency across micro-batches before computing the balance term — letting individual batches stay domain-skewed while the corpus-wide average stays balanced.",
      zh: '反转了 Qwen2 自己的选择：完全去掉共享专家，换成一个全局批次负载均衡损失——在计算均衡项之前，先把各个 micro-batch 之间的专家选择频率同步起来，让单个批次可以保持领域偏斜，同时整体语料上的平均值依然均衡。',
    },
  },
  {
    id: 'v4flash',
    name: 'DeepSeek-V4-Flash',
    org: 'DeepSeek-AI',
    totalB: 284,
    activeB: 13,
    routedExperts: '256',
    sharedExperts: '1',
    topK: '6',
    gating: 'Sqrt(Softplus)',
    balancing: 'Aux-loss-free + sequence-wise balance loss',
    innovation: {
      en: "Swaps V3's sigmoid affinity for Sqrt(Softplus), adds Hash routing (deterministic, no learned gate) to the first 3 MoE layers, and introduces Anticipatory Routing — computing routing decisions from stale weights — to suppress MoE-driven loss spikes.",
      zh: '把 V3 的 sigmoid 打分换成 Sqrt(Softplus)，前 3 层 MoE 改用 Hash 路由（确定性、不需要学习的门控），并引入 Anticipatory Routing——用滞后的参数计算路由决策——来抑制 MoE 引发的 loss 尖峰。',
    },
  },
  {
    id: 'v4pro',
    name: 'DeepSeek-V4-Pro',
    org: 'DeepSeek-AI',
    totalB: 1600,
    activeB: 49,
    routedExperts: '384',
    sharedExperts: '1',
    topK: '6',
    gating: 'Sqrt(Softplus)',
    balancing: 'Aux-loss-free + sequence-wise balance loss',
    innovation: {
      en: 'Same architecture family as V4-Flash at larger scale; pairs it with a new pipelined expert-parallel "MegaMoE" kernel that overlaps dispatch, compute, and combine in waves for large speedups at inference and RL-rollout time.',
      zh: '和 V4-Flash 同一套架构在更大规模上的版本；配合新的流水线式专家并行 "MegaMoE" 内核，把 dispatch、计算、combine 分波次重叠执行，在推理和 RL rollout 场景下带来明显加速。',
    },
  },
  {
    id: 'glm',
    name: 'GLM-5.2',
    org: 'Zhipu / Z.ai',
    totalB: 753,
    activeB: 40,
    routedExperts: '256',
    sharedExperts: '1',
    topK: '8',
    gating: 'Sigmoid',
    balancing: 'Aux-loss-free (DeepSeekMoE-style)',
    innovation: {
      en: 'Keeps the DeepSeek-V3-style MoE recipe almost unchanged; this generation\'s gains come from the attention side (an indexer shared across every 4 sparse-attention layers) and infrastructure (an improved MTP layer, a custom Slime RL framework), not from new MoE mechanics.',
      zh: 'MoE 部分基本沿用 DeepSeek-V3 式配方，几乎没有变化；这一代的提升主要来自注意力侧（每 4 层稀疏注意力共享一个索引器）和基础设施（改进的 MTP 层、自研 Slime RL 框架），而不是新的 MoE 机制。',
    },
  },
  {
    id: 'mai',
    name: 'MAI-Thinking-1',
    org: 'Microsoft AI',
    totalB: 962,
    activeB: 34.7,
    routedExperts: '512',
    sharedExperts: '0',
    topK: '8',
    gating: 'Softmax (pre-compression)',
    balancing: 'Global-batch loss, fully dropless',
    innovation: {
      en: "Adopts LatentMoE: experts operate in a compressed latent space (roughly 2x compression, letting the expert count grow about 3x for the same budget), interleaved with dense FFN layers rather than MoE at every layer. Shared experts were tested but didn't help in this interleaved layout, so the final model ships without one.",
      zh: '采用 LatentMoE：专家在压缩后的潜空间中运算（压缩约 2 倍，同等预算下专家数可扩大约 3 倍），并与稠密 FFN 层交替排布，而非每层都用 MoE。这种交替结构下共享专家没有带来收益，因此最终版本没有使用共享专家。',
    },
  },
  {
    id: 'kimi',
    name: 'Kimi K3',
    org: 'Moonshot AI',
    totalB: 2800,
    activeB: null,
    routedExperts: '896',
    sharedExperts: 'yes (count undisclosed)',
    topK: '16',
    gating: 'Stable LatentMoE (details undisclosed)',
    balancing: 'Quantile Balancing (router-score quantiles)',
    innovation: {
      en: 'Pushes expert granularity further than any other 2026 report — 896 experts, top-16 — inside a "Stable LatentMoE" framework, and replaces bias-based balancing with Quantile Balancing, which derives expert allocation directly from router-score quantiles instead of a tuned hyperparameter.',
      zh: '把专家粒度做得比 2026 年任何其他报告都更细——896 个专家、top-16——放在 "Stable LatentMoE" 框架里，并用 Quantile Balancing 取代基于偏置项的均衡方式，直接从路由打分的分位数推导专家分配，不需要额外调的超参数。',
    },
  },
  {
    id: 'longcat',
    name: 'LongCat-2.0',
    org: 'Meituan',
    totalB: 1600,
    activeB: 48,
    routedExperts: 'dynamic pool incl. zero-computation experts',
    sharedExperts: '3 task-specialized groups instead',
    topK: 'dynamic, ~33–56B active per token',
    gating: 'router picks real vs. zero-computation experts',
    balancing: 'not publicly detailed',
    innovation: {
      en: 'The most structurally different design here: some "experts" are Zero-Computation Experts that just pass the token through unchanged, so simple tokens cost almost nothing while hard tokens engage more real experts — adaptive per-token compute rather than a fixed top-K. A shortcut-connected backbone (ScMoE) and three task-specialized expert groups (Agent / Reasoning / Interaction) layer on top.',
      zh: '这里结构上最不一样的设计：一部分"专家"是 Zero-Computation Expert，直接原样透传 token，简单 token 几乎不花算力，难 token 才会调用更多真正的专家——是按 token 自适应分配算力，而不是固定的 top-K。再叠加一个 shortcut-connected 的骨干（ScMoE）和三个按任务划分的专家组（Agent / Reasoning / Interaction）。',
    },
  },
];

const STR11 = {
  en: {
    title: "Interactive: look up a model's MoE recipe",
    sub: "Every figure here is drawn from that model's own release material.",
    org: 'Organization',
    total: 'Total parameters',
    active: 'Active parameters',
    routed: 'Routed experts',
    shared: 'Shared experts',
    topk: 'Top-K',
    gating: 'Gating function',
    balancing: 'Load balancing',
    innovation: "What's new here",
    notDisclosed: 'not disclosed',
  },
  zh: {
    title: '交互演示：查一个模型的 MoE 配方',
    sub: '这里的每一个数字都来自该模型自己的发布材料。',
    org: '机构',
    total: '总参数量',
    active: '激活参数量',
    routed: '路由专家数',
    shared: '共享专家数',
    topk: 'Top-K',
    gating: '门控函数',
    balancing: '负载均衡',
    innovation: '这里的新东西',
    notDisclosed: '未公开',
  },
};

export const MoEModelLookup: React.FC<{ lang?: Lang }> = ({ lang = 'en' }) => {
  const t = STR11[lang];
  const [modelId, setModelId] = useState('v3');
  const model = MOE_MODELS.find((m) => m.id === modelId) ?? MOE_MODELS[0];

  return (
    <Card>
      <h4 className="text-lg font-serif text-anthropic-text mb-1">{t.title}</h4>
      <p className="text-sm text-anthropic-gray mb-4">{t.sub}</p>

      <div className="mb-5 flex flex-wrap gap-2">
        {MOE_MODELS.map((m) => (
          <Pill key={m.id} active={modelId === m.id} onClick={() => setModelId(m.id)}>
            {m.name}
          </Pill>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm mb-4">
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.org}</div>
          <div className="text-anthropic-text">{model.org}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.total}</div>
          <div className="text-anthropic-text font-mono">
            {model.totalB >= 1000 ? `${(model.totalB / 1000).toFixed(1)}T` : `${model.totalB}B`}
          </div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.active}</div>
          <div className="text-anthropic-text font-mono">{model.activeB != null ? `${model.activeB}B` : t.notDisclosed}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.routed}</div>
          <div className="text-anthropic-text font-mono text-xs">{model.routedExperts}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.shared}</div>
          <div className="text-anthropic-text font-mono text-xs">{model.sharedExperts}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.topk}</div>
          <div className="text-anthropic-text font-mono text-xs">{model.topK}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.gating}</div>
          <div className="text-anthropic-text text-xs">{model.gating}</div>
        </div>
        <div>
          <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide">{t.balancing}</div>
          <div className="text-anthropic-text text-xs">{model.balancing}</div>
        </div>
      </div>

      <div>
        <div className="text-anthropic-gray/70 text-xs uppercase tracking-wide mb-1">{t.innovation}</div>
        <p className="text-anthropic-text text-sm leading-relaxed">{model.innovation[lang]}</p>
      </div>
    </Card>
  );
};
