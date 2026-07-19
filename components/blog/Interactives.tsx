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
