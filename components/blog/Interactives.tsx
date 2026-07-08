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
