import React from 'react';

type Lang = 'en' | 'zh';

/**
 * Schematic panels for the examples in "What reward hacking looks like".
 * Drawn, not measured: each one only shows where the reward and the intent part ways.
 */

const INK = '#191919';
const svgProps = {
  viewBox: '0 0 200 130',
  className: 'w-full',
  fill: 'none',
  stroke: INK,
  strokeWidth: 1.2,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};
const Label: React.FC<{ x: number; y: number; anchor?: 'start' | 'middle' | 'end'; children: React.ReactNode }> = ({
  x,
  y,
  anchor = 'middle',
  children,
}) => (
  <text x={x} y={y} textAnchor={anchor} fill={INK} fillOpacity={0.75} stroke="none" fontSize={9.5}>
    {children}
  </text>
);
const Mono: React.FC<{ x: number; y: number; children: React.ReactNode; strong?: boolean }> = ({ x, y, children, strong }) => (
  <text
    x={x}
    y={y}
    fill={INK}
    fillOpacity={strong ? 0.95 : 0.7}
    stroke="none"
    fontSize={8}
    fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
  >
    {children}
  </text>
);
const Arrow: React.FC<{ id: string }> = ({ id }) => (
  <defs>
    <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill={INK} stroke="none" />
    </marker>
  </defs>
);

const Bicycle: React.FC<{ lang: Lang }> = ({ lang }) => (
  <svg {...svgProps} role="img" aria-label="A bicycle that circles its starting point instead of riding to the goal">
    <Arrow id="a-bike" />
    <path d="M60 70 L176 70" strokeDasharray="3 3" strokeOpacity={0.35} />
    <Label x={140} y={64}>{lang === 'en' ? 'intended' : '预期'}</Label>
    <path d="M176 70 L176 48 L190 53 L176 58" />
    <Label x={178} y={86}>{lang === 'en' ? 'goal' : '目标'}</Label>
    <circle cx={60} cy={70} r={2.5} fill={INK} stroke="none" />
    <Label x={60} y={84}>{lang === 'en' ? 'start' : '起点'}</Label>
    <ellipse cx={58} cy={70} rx={34} ry={28} strokeOpacity={0.45} />
    <path d="M58 30 A40 38 0 1 1 24 52" markerEnd="url(#a-bike)" />
  </svg>
);

const Boat: React.FC<{ lang: Lang }> = ({ lang }) => (
  <svg {...svgProps} role="img" aria-label="A boat that loops around respawning targets in a lagoon instead of finishing the race">
    <Arrow id="a-boat" />
    <rect x={8} y={12} width={184} height={106} rx={46} strokeOpacity={0.45} />
    <rect x={34} y={36} width={132} height={58} rx={26} strokeOpacity={0.45} />
    <path d="M100 12 L100 36" strokeWidth={2} />
    <Label x={104} y={27} anchor="start">{lang === 'en' ? 'finish' : '终点'}</Label>
    <circle cx={78} cy={65} r={3} fill={INK} stroke="none" />
    <circle cx={96} cy={74} r={3} fill={INK} stroke="none" />
    <circle cx={88} cy={54} r={3} fill={INK} stroke="none" />
    <path d="M110 62 A22 18 0 1 1 106 52" markerEnd="url(#a-boat)" />
    <Label x={100} y={110}>{lang === 'en' ? 'respawning targets' : '会重生的目标'}</Label>
  </svg>
);

const Arm: React.FC<{ lang: Lang }> = ({ lang }) => (
  <svg {...svgProps} role="img" aria-label="A robot hand placed between the camera and the object, so it only looks like a grasp">
    <rect x={8} y={52} width={22} height={16} rx={2} />
    <circle cx={33} cy={60} r={4} />
    <Label x={20} y={84}>{lang === 'en' ? 'camera' : '镜头'}</Label>
    <path d="M37 60 L170 60" strokeDasharray="3 3" strokeOpacity={0.35} />
    <circle cx={175} cy={60} r={10} />
    <Label x={175} y={84}>{lang === 'en' ? 'object' : '物体'}</Label>
    <path d="M96 122 L96 92 L104 76" strokeWidth={2.2} />
    <path d="M104 76 L98 50 M104 76 L114 52" strokeWidth={2.2} />
    <Label x={128} y={110} anchor="start">{lang === 'en' ? 'hand' : '机械手'}</Label>
    <Label x={100} y={24}>{lang === 'en' ? 'looks like a grasp' : '看起来像在抓'}</Label>
  </svg>
);

const ExitZero: React.FC<{ lang: Lang }> = ({ lang }) => (
  <svg {...svgProps} role="img" aria-label="Calling exit(0) before the tests run makes the test harness report a pass">
    <rect x={8} y={10} width={184} height={78} rx={3} strokeOpacity={0.45} />
    <Mono x={16} y={26}>def solve(nums):</Mono>
    <Mono x={28} y={38}>return []</Mono>
    <Mono x={16} y={54} strong>
      sys.exit(0)
    </Mono>
    <Mono x={16} y={70}># tests below</Mono>
    <Mono x={16} y={81}># never run</Mono>
    <path d="M86 51 L118 51" strokeOpacity={0.5} />
    <Label x={122} y={54} anchor="start">{lang === 'en' ? 'before tests' : '在测试之前'}</Label>
    <Label x={100} y={110}>{lang === 'en' ? 'exit code 0 → PASS' : '退出码 0 → 通过'}</Label>
  </svg>
);

const Length: React.FC<{ lang: Lang }> = ({ lang }) => (
  <svg {...svgProps} role="img" aria-label="Reward-model score keeps rising with answer length while true quality does not">
    <Arrow id="a-len" />
    <path d="M22 108 L22 14" markerEnd="url(#a-len)" strokeOpacity={0.5} />
    <path d="M22 108 L188 108" markerEnd="url(#a-len)" strokeOpacity={0.5} />
    <path d="M24 96 C 70 80 120 50 182 22" strokeWidth={1.6} />
    <path d="M24 92 C 70 70 100 66 130 70 C 150 73 165 80 182 86" strokeDasharray="4 3" />
    <Label x={182} y={18} anchor="end">{lang === 'en' ? 'reward-model score' : '奖励模型打分'}</Label>
    <Label x={182} y={100} anchor="end">{lang === 'en' ? 'true quality' : '真实质量'}</Label>
    <Label x={105} y={124}>{lang === 'en' ? 'answer length' : '回答长度'}</Label>
  </svg>
);

const Obfuscation: React.FC<{ lang: Lang }> = ({ lang }) => (
  <svg {...svgProps} role="img" aria-label="After a chain-of-thought monitor is added to the reward, the model keeps hacking but stops saying so">
    <rect x={8} y={14} width={184} height={40} rx={3} strokeOpacity={0.45} />
    <Mono x={16} y={30}>{lang === 'en' ? 'CoT: "skip the tests"' : 'CoT："跳过测试"'}</Mono>
    <Mono x={16} y={44}>code: sys.exit(0)</Mono>
    <Label x={184} y={38} anchor="end">{lang === 'en' ? 'flagged' : '被标记'}</Label>
    <rect x={8} y={72} width={184} height={40} rx={3} strokeOpacity={0.45} />
    <Mono x={16} y={88}>{lang === 'en' ? 'CoT: "fix the bug"' : 'CoT："修复 bug"'}</Mono>
    <Mono x={16} y={102}>code: sys.exit(0)</Mono>
    <Label x={184} y={96} anchor="end">{lang === 'en' ? 'passes' : '通过'}</Label>
    <Label x={100} y={66}>{lang === 'en' ? '↓ monitor added to reward' : '↓ monitor 加进奖励'}</Label>
  </svg>
);

const Panel: React.FC<{ children: React.ReactNode; caption: string }> = ({ children, caption }) => (
  <figure>
    {children}
    <figcaption className="mt-1.5 text-[12px] leading-[1.45] text-anthropic-gray/80">{caption}</figcaption>
  </figure>
);

const Row: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="my-6 grid grid-cols-1 gap-6 sm:grid-cols-3 sm:gap-5">{children}</div>
);

export const ClassicExamples: React.FC<{ lang: Lang }> = ({ lang }) => (
  <Row>
    <Panel
      caption={
        lang === 'en'
          ? 'Rewarded for heading toward the goal, never penalized for heading away: the bicycle circles its start.'
          : '朝向目标有奖励、背离目标不受罚：自行车绕着起点转圈。'
      }
    >
      <Bicycle lang={lang} />
    </Panel>
    <Panel
      caption={
        lang === 'en'
          ? 'Hitting respawning targets pays more than finishing: the boat loops in the lagoon.'
          : '撞会重生的目标比跑完全程更划算：赛艇在泻湖里绕圈。'
      }
    >
      <Boat lang={lang} />
    </Panel>
    <Panel
      caption={
        lang === 'en'
          ? 'Judged through a camera: the hand blocks the view so that it only appears to grasp.'
          : '通过镜头评判：机械手挡住视线，看起来像在抓取。'
      }
    >
      <Arm lang={lang} />
    </Panel>
  </Row>
);

export const LanguageModelExamples: React.FC<{ lang: Lang }> = ({ lang }) => (
  <Row>
    <Panel
      caption={
        lang === 'en'
          ? 'RLHF: the learned reward rewards length more than quality.'
          : 'RLHF：学出来的奖励更看重长度而不是质量。'
      }
    >
      <Length lang={lang} />
    </Panel>
    <Panel
      caption={
        lang === 'en'
          ? 'Verifier: exiting before the tests run reads as a pass.'
          : 'Verifier：在测试运行前退出，就被判为通过。'
      }
    >
      <ExitZero lang={lang} />
    </Panel>
    <Panel
      caption={
        lang === 'en'
          ? 'Penalizing hacks seen in the chain of thought: the hack stays, the confession goes.'
          : '惩罚 CoT 里看得到的 hack：hack 还在，坦白没了。'
      }
    >
      <Obfuscation lang={lang} />
    </Panel>
  </Row>
);
