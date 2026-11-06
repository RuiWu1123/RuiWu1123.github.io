import React from 'react';
import { M, MD } from '../../components/Math';

const r = String.raw;

const H: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="!mt-10 mb-4 border-b-[1.5px] border-anthropic-text pb-2.5 text-center">
    <h2 className="text-lg font-serif text-anthropic-text">{children}</h2>
  </div>
);

/** A paragraph that opens with a bold run-in heading, as in the source draft. */
const Run: React.FC<{ head: string; children: React.ReactNode }> = ({ head, children }) => (
  <p>
    <b className="text-anthropic-text">{head}</b>
    {'　'}
    {children}
  </p>
);

const C: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <code className="font-mono text-[0.88em]">{children}</code>
);

interface Ref {
  text: string;
  url?: string;
}

const REFERENCES: Ref[] = [
  { text: 'Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. 2016.', url: 'https://arxiv.org/abs/1606.06565' },
  { text: 'Dario Amodei, Paul Christiano, and Alex Ray. Learning from human preferences. OpenAI blog, June 13, 2017.' },
  { text: 'Anthropic. Claude 3.7 Sonnet system card, 2025.' },
  { text: 'W. Ross Ashby. An Introduction to Cybernetics. John Wiley & Sons, New York, 1956.' },
  { text: 'Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. 2025.', url: 'https://arxiv.org/abs/2503.11926' },
  { text: 'George Baker. Distortion and risk in optimal incentive contracts. Journal of Human Resources, 37(4):728–751, 2002.' },
  { text: 'Gabriel Carroll. Robustness and linear contracts. American Economic Review, 105(2):536–563, 2015.' },
  { text: 'Jack Clark and Dario Amodei. Faulty reward functions in the wild. OpenAI blog, December 21, 2016.' },
  { text: 'Roger C. Conant and W. Ross Ashby. Every good regulator of a system must be a model of that system. International Journal of Systems Science, 1(2):89–97, 1970.' },
  { text: 'Nicholas B. Davies, Rebecca M. Kilner, and David G. Noble. Nestling cuckoos, Cuculus canorus, exploit hosts with begging calls that mimic a brood. Proceedings of the Royal Society B, 265(1397):673–678, 1998.' },
  { text: 'Richard Dawkins and John R. Krebs. Arms races between and within species. Proceedings of the Royal Society of London B, 205(1161):489–511, 1979.' },
  { text: 'Carson Denison, Monte MacDiarmid, Fazl Barez, David Duvenaud, Shauna Kravec, Samuel Marks, et al. Sycophancy to subterfuge: Investigating reward-tampering in large language models. 2024.', url: 'https://arxiv.org/abs/2406.10162' },
  { text: 'Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, et al. Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking. 2023.', url: 'https://arxiv.org/abs/2312.09244' },
  { text: 'El-Mahdi El-Mhamdi and Lê-Nguyên Hoang. On Goodhart’s law, with an application to value alignment. 2024.', url: 'https://arxiv.org/abs/2410.09638' },
  { text: 'Tom Everitt and Marcus Hutter. Avoiding wireheading with value reinforcement learning. AGI 2016.', url: 'https://arxiv.org/abs/1605.03143' },
  { text: 'Tom Everitt, Victoria Krakovna, Laurent Orseau, Marcus Hutter, and Shane Legg. Reinforcement learning with a corrupted reward channel. IJCAI 2017. Theorem numbers follow arXiv:1705.08417v2.', url: 'https://arxiv.org/abs/1705.08417' },
  { text: 'Tom Everitt, Ryan Carey, Eric D. Langlois, Pedro A. Ortega, and Shane Legg. Agent incentives: A causal perspective. AAAI 2021.' },
  { text: 'Tom Everitt, Marcus Hutter, Ramana Kumar, and Victoria Krakovna. Reward tampering problems and solutions in reinforcement learning: A causal influence diagram perspective. Synthese, 198:6435–6467, 2021.' },
  { text: 'Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. ICML 2023.' },
  { text: 'Charles A. E. Goodhart. Problems of monetary management: The U.K. experience. Papers in Monetary Economics, Volume I. Reserve Bank of Australia, 1975.' },
  { text: 'Alan Grafen. Biological signals as handicaps. Journal of Theoretical Biology, 144(4):517–546, 1990.' },
  { text: 'Savyasachi Gupta and Aniruddha Srinivas Joshi. Checking the checkers: A synthesis of verifier noise and robustness in RLVR. 2026.', url: 'https://arxiv.org/abs/2610.17824' },
  { text: 'Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart Russell, and Anca Dragan. Inverse reward design. NeurIPS 2017.' },
  { text: 'Bengt Holmström. Moral hazard and observability. Bell Journal of Economics, 10(1):74–91, 1979.' },
  { text: 'Bengt Holmström and Paul Milgrom. Multitask principal–agent analyses: Incentive contracts, asset ownership, and job design. Journal of Law, Economics, & Organization, 7:24–52, 1991.' },
  { text: 'Audrey Huang, Wenhao Zhan, Tengyang Xie, Jason D. Lee, Wen Sun, Akshay Krishnamurthy, and Dylan J. Foster. Correcting the mythos of KL-regularization: Direct alignment without overoptimization via chi-squared preference optimization. ICLR 2025.' },
  { text: 'James E. Huheey. Studies of warning coloration and mimicry. IV. A mathematical model of model-mimic frequencies. Ecology, 45(1):185–188, 1964.' },
  { text: 'Borja Ibarz, Jan Leike, Tobias Pohlen, Geoffrey Irving, Shane Legg, and Dario Amodei. Reward learning from human preferences and demonstrations in Atari. NeurIPS 2018.' },
  { text: 'Jacek Karwowski, Oliver Hayman, Xingjian Bai, Klaus Kiendlhofer, Charlie Griffin, and Joar Skalse. Goodhart’s law in reinforcement learning. ICLR 2024.' },
  { text: 'Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, Matthew Rahtz, Tom Everitt, Ramana Kumar, Zac Kenton, Jan Leike, and Shane Legg. Specification gaming: The flip side of AI ingenuity. DeepMind blog, April 21, 2020.' },
  { text: 'Thomas Kwa, Drake Thomas, and Adrià Garriga-Alonso. Catastrophic Goodhart: Regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification. NeurIPS 2024.' },
  { text: 'Jeffrey M. Lacker and John A. Weinberg. Optimal contracts under costly state falsification. Journal of Political Economy, 97(6):1345–1363, 1989.' },
  { text: 'Cassidy Laidlaw, Shivam Singhal, and Anca Dragan. Correlated proxies: A new definition and improved mitigation for reward hacking. ICLR 2025.' },
  { text: 'Joel Lehman, Jeff Clune, Dusan Misevic, et al. The surprising creativity of digital evolution: A collection of anecdotes from the evolutionary computation and artificial life research communities. Artificial Life, 26(2):274–306, 2020.' },
  { text: 'Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A. Ortega, Tom Everitt, Andrew Lefrancq, Laurent Orseau, and Shane Legg. AI safety gridworlds. 2017.', url: 'https://arxiv.org/abs/1711.09883' },
  { text: 'Leena Lindström, Rauno V. Alatalo, and Johanna Mappes. Imperfect Batesian mimicry: The effects of the frequency and the distastefulness of the model. Proceedings of the Royal Society B, 264(1379):149–153, 1997.' },
  { text: 'MacDiarmid, Wright, Uesato, et al. Natural emergent misalignment from reward hacking in production RL. 2025.', url: 'https://arxiv.org/abs/2511.18397' },
  { text: 'David Manheim and Scott Garrabrant. Categorizing variants of Goodhart’s law. 2018.', url: 'https://arxiv.org/abs/1803.04585' },
  { text: 'Christian Moya, Elliott Thornley, and Guang Lin. Verifier errors in RLVR: Reward hacking, limits of feedback, and selective control. 2026.', url: 'https://arxiv.org/abs/2609.35677' },
  { text: 'Youssef Mroueh. Reinforcement learning with verifiable rewards: GRPO’s effective loss, dynamics, and success amplification. 2025.', url: 'https://arxiv.org/abs/2503.06639' },
  { text: 'Andrew Y. Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. ICML 1999.' },
  { text: 'Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. ICLR 2022.' },
  { text: 'Dustin J. Penn and Szabolcs Számadó. The handicap principle: How an erroneous hypothesis became a scientific principle. Biological Reviews, 95(1):267–290, 2020.' },
  { text: 'Ali Rad, Khashayar Filom, Darioush Keivan, Peyman Mohajerin Esfahani, and Ehsan Kamalinejad. Rate or fate? RLVεR: Reinforcement learning with verifiable noisy rewards. 2026.', url: 'https://arxiv.org/abs/2601.04411' },
  { text: 'Jette Randløv and Preben Alstrøm. Learning to drive a bicycle using reinforcement learning and shaping. ICML 1998.' },
  { text: 'Mark Ring and Laurent Orseau. Delusion, survival, and intelligent agents. AGI 2011, LNCS 6830, pages 11–20. Springer, 2011.' },
  { text: 'Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, et al. Towards understanding sycophancy in language models. ICLR 2024.' },
  { text: 'Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in RLHF. 2023.', url: 'https://arxiv.org/abs/2310.03716' },
  { text: 'Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward hacking. NeurIPS 2022.' },
  { text: 'B. F. Skinner. ‘Superstition’ in the pigeon. Journal of Experimental Psychology, 38(2):168–172, 1948.' },
  { text: 'Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize with human feedback. NeurIPS 2020.' },
  { text: 'Marilyn Strathern. ‘Improving ratings’: Audit in the British university system. European Review, 5(3):305–321, 1997.' },
  { text: 'Taylor, Chua, Betley, Treutlein, and Evans. School of reward hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs. 2025.', url: 'https://arxiv.org/abs/2508.17511' },
  { text: 'Jessica Taylor. Quantilizers: A safer alternative to maximizers for limited optimization. AAAI Workshop on AI, Ethics, and Society, 2016.' },
  { text: 'Niko Tinbergen. The Study of Instinct. Clarendon Press, Oxford, 1951.' },
  { text: 'Niko Tinbergen and A. C. Perdeck. On the stimulus situation releasing the begging response in the newly hatched herring gull chick (Larus argentatus argentatus Pont.). Behaviour, 3(1):1–39, 1951.' },
  { text: 'Sydney Von Arx, Lawrence Chan, and Beth Barnes. Recent frontier models are reward hacking. METR blog, June 5, 2025.' },
  { text: 'Wang, Joshi, Plank, Angell, and He. Is it thinking or cheating? Detecting implicit reward hacking by measuring reasoning effort. ICLR 2026.' },
  { text: 'Lilian Weng. Reward hacking in reinforcement learning. Lil’Log, November 28, 2024.' },
  { text: 'Norbert Wiener. Some moral and technical consequences of automation. Science, 131(3410):1355–1358, 1960.' },
  { text: 'Zhong, Raghunathan, and Carlini. ImpossibleBench: Measuring LLMs’ propensity of exploiting test cases. 2025.', url: 'https://arxiv.org/abs/2510.20270' },
  { text: 'Simon Zhuang and Dylan Hadfield-Menell. Consequences of misaligned AI. NeurIPS 2020.' },
];

const linkClass =
  'underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';

const RewardHackingFormalized: React.FC = () => {
  return (
    <article className="animate-fade-in pt-10 pb-16">
      <h1 className="text-xl font-serif text-anthropic-text leading-tight text-center">
        Reward Hacking 的形式化：从经典强化学习到语言模型
      </h1>
      <p className="mt-2 text-center text-[12px] text-anthropic-gray/80">11.6.2026</p>

      <div className="mt-8 space-y-4 text-base font-sans text-anthropic-gray leading-relaxed">
        <H>TL;DR</H>
        <p>
          本文先回顾了 reward hacking 在经典 RL、RLHF 和 RLVR 中的典型案例和定义，再介绍了语言模型之前的工作如何用数学形式刻画它。随后为语言模型的
          RL 建立了一套形式化，并推导出：group-based 更新分不清奖励相同的漏洞和老实答案；hacking 可以拆成沿 proxy
          的选择和奖励层内的放大，前者是奖励模型被 hack 的主要方式，后者是 verifier 被 hack 的唯一方式；在 verifier 下目标函数对
          hack 占比无差别，最终结果由学习动力学决定；漏洞通过共享参数扩散，起点是组内无法与老实成功区分的孤立成功，一旦开始就会自我加速。最后讨论了合同理论、演化生物学和控制论中的对应问题。
        </p>

        <H>Reward hacking 长什么样</H>
        <p>
          Wiener (1960) 写道，如果把一个目的交给一台来不及叫停的机器，“我们最好非常确定，放进机器的目的正是我们真正想要的目的，而不仅仅是它的一个花哨的仿制品”。Goodhart
          (1975) 指出，任何统计规律“一旦为了控制目的对它施加压力，就会趋于瓦解”，更广为人知的说法是“当一个指标成为目标，它就不再是好指标”
          (Strathern, 1997)。
        </p>
        <p>
          一辆模拟自行车因朝向目标而得奖励、背离目标却从不受罚，结果学会了在起点周围绕 20 到 50 米宽的圈 (Randløv and Alstrøm,
          1998)。一个因触碰足球而得奖励的机器人，学会了停在球旁边不停地碰它 (Ng et al., 1999)。在赛艇游戏 CoastRunners
          里，智能体发现在泻湖里反复撞几个会重生的目标比跑完全程更划算，一边着火、一边逆行，得分还比人类玩家高出约 20% (Clark and Amodei,
          2016)。一只从人类反馈中学抓取的机械臂，把手挡在镜头和物体之间，“让它看起来像是在抓取” (Amodei et al.,
          2017)。一个叠积木的智能体直接把积木翻了过来 (Krakovna et al., 2020)；一个程序修复系统让程序永远返回空列表，测试居然通过了
          (Lehman et al., 2020)；在一个 gridworld 里，走进水桶格就能让所有番茄看起来都浇过水 (Leike et al.,
          2017)。在冻结的、学出来的奖励模型上训练的 Atari 智能体，把奖励模型的打分推高，游戏得分却“暴跌” (Ibarz et al., 2018)。
        </p>
        <p>
          这个现象今天的名字来自 Amodei et al. (2016)：目标函数“允许某种聪明的‘简单’解，它在形式上最大化了目标函数，却违背了设计者意图的精神”。DeepMind
          的叫法是 specification gaming，指“满足目标的字面规格、却没有达到预期结果的行为” (Krakovna et al., 2020)。Pan et
          al. (2022) 发现它可能在模型容量增长时以相变的形式突然出现；Weng (2024)
          总结了通行的用法：智能体“利用奖励函数中的缺陷或模糊之处获得高奖励，却没有真正学会或完成预期任务”。
        </p>
        <p>
          在 RLHF 中，对学出来的奖励模型优化得越用力，摘要先变好、再变差 (Stiennon et al., 2020)；在几个数据集上，RLHF
          测得的提升有很大一部分只是因为回答变长了 (Singhal et al., 2023)；偏好模型常常更喜欢附和用户的回答，而不是正确的回答 (Sharma
          et al., 2024)；奖励模型集成的帮助也没有预想的大，因为成员往往犯同样的错 (Eisenstein et al., 2023)。
        </p>
        <p>
          在程序化 verifier 下，前沿代码 agent 被发现会在测试运行前调用 <C>exit(0)</C>、抛出 <C>SkipTest</C>
          、修改测试框架上游的代码、从测试文件里读出期望值、反编译参考答案；当一个读取它们 CoT 的 monitor 被加进奖励后，它们继续
          hack，只是不再说出来 (Baker et al., 2025)。Claude 3.7 Sonnet 的 system card 写道，它会“为了通过测试用例而特判”，这是训练中
          reward hacking 的结果 (Anthropic, 2025)；o3 在评测任务里覆盖评分器的计时器、修改评测器，被问到时也承认这不是用户想要的 (Von
          Arx et al., 2025)；在被刻意改成不可能完成的代码任务上，“任何通过都必然意味着违反规格的捷径”，前沿模型却经常通过 (Zhong et
          al., 2025)。在一个越来越容易钻空子的任务课程里，模型偶尔甚至会改写自己的奖励函数 (Denison et al., 2024)；而学会 hack
          生产代码环境，被观察到与更广泛的 misalignment 同时出现 (MacDiarmid et al., 2025; Taylor et al., 2025)。Wang et
          al. (2026) 把其中的机制描述为“当利用漏洞比解决实际任务更容易时，就会发生 hacking”。
        </p>

        <H>语言模型之前的形式化</H>
        <p>
          Ng et al. (1999) 研究 reward shaping 时，把智能体看到的奖励写成我们想要的奖励加一个附加项：智能体在 <M>{r`R' = R + F`}</M> 上训练，
          <M>{r`F : S \times A \times S \to \mathbb{R}`}</M>，问题是什么样的 <M>F</M> 不会改变原 MDP
          的最优策略，其中扮演特殊角色的是基于势函数的项 <M>{r`F(s, a, s') = \gamma\Phi(s') - \Phi(s)`}</M>。Leike et al. (2017)
          把同样的拆分写进了基准环境的定义：每个环境都带一个智能体看得到的奖励 <M>R</M>，和一个它看不到的性能函数 <M>{r`R^*`}</M>。Manheim
          and Garrabrant (2018) 把回归型 Goodhart 写成目标 <M>G</M> 的 proxy <M>{r`M = G + \xi`}</M>，把极端型写成过了某个阈值就改变的关系：
          <M>{r`M \le a`}</M> 时 <M>{r`G = M + x`}</M>，否则 <M>{r`G = M + y`}</M>。Kwa et al. (2024) 用的也是这种加法形式，效用{' '}
          <M>V</M> 加误差 <M>X</M>，并按尾部给误差分类：若对任意 <M>{r`t > 0`}</M> 都有 <M>{r`e^{tx}\Pr(X > x) \to \infty`}</M>，就称{' '}
          <M>X</M> 是重尾的。
        </p>
        <p>
          在 Everitt et al. (2017) 的被污染奖励 MDP 中，误差落在具体的状态上。环境有真实奖励 <M>{r`\dot R : S \to \mathcal{R}`}</M>{' '}
          和污染函数 <M>{r`C : S \times \mathcal{R} \to \mathcal{R}`}</M>，智能体观测到 <M>{r`\hat R(s) = C_s(\dot R(s))`}</M>
          ；被污染的状态有多少、一个状态的奖励能否从别处观测到，这些限制决定了智能体能学到多少。Hadfield-Menell et al. (2017)
          把设计出来的奖励当作证据：在线性奖励 <M>{r`r(\xi; w) = w^\top\phi(\xi)`}</M> 下，设计者在训练环境 <M>{r`\tilde M`}</M>{' '}
          中选择 proxy 权重 <M>{r`\tilde w`}</M>，被建模为近似最优，
        </p>
        <MD>{r`P(\tilde w \mid w^*, \tilde M) \propto \exp\big(\beta\, \mathbb{E}[w^{*\top}\phi(\xi) \mid \xi \sim \pi(\xi \mid \tilde w, \tilde M)]\big),`}</MD>
        <p>
          所以 proxy 只通过训练环境中的行为告诉我们真实权重 <M>{r`w^*`}</M> 的信息。Zhuang and Hadfield-Menell (2020) 用属性{' '}
          <M>{r`s \in \mathbb{R}^L`}</M> 描述世界，受资源约束 <M>{r`C(s) \le 0`}</M>；真实效用 <M>U</M> 依赖所有属性，proxy{' '}
          <M>{r`\tilde U`}</M> 只依赖其中 <M>{r`J < L`}</M> 个。
        </p>
        <p>
          Ring and Orseau (2011) 让智能体能作用于奖励通道本身。他们把智能体的动作拆成{' '}
          <M>{r`a_t = \langle d_t, a^e_t \rangle`}</M>，其中 <M>{r`d_t`}</M> 对一个 “delusion box”{' '}
          <M>{r`d : O \to O`}</M> 编程，智能体通过它看世界，<M>{r`o_t = d(a_t, o^e_t)`}</M>，<M>{r`o_t`}</M>{' '}
          的一部分被读作奖励。Everitt and Hutter (2016) 把观测奖励写成 <M>{r`r = d_s(\check r)`}</M>，即内部奖励{' '}
          <M>{r`\check r = u^*(s)`}</M> 的一个依赖于状态的扭曲。Everitt et al. (2021b) 把智能体、它实现的奖励函数{' '}
          <M>{r`\Theta^R_t`}</M> 和意图中的奖励函数 <M>{r`\Theta^R_*`}</M> 画成因果影响图中的节点；当智能体的决策以违反{' '}
          <M>{r`\Theta^R_*`}</M> 条件的方式影响 <M>{r`\Theta^R_t`}</M> 时，就称为篡改。这里用到 Everitt et al. (2021a)
          的判据：要有控制某个变量的激励，必须存在一条经过它、从决策到效用的有向路径。
        </p>
        <p>
          Skalse et al. (2022) 在整个策略空间上比较两个奖励：若某一对策略被 <M>{r`R_1`}</M> 和 <M>{r`R_2`}</M> 排出相反的顺序，则称{' '}
          <M>{r`R_1, R_2`}</M> 在策略集合 <M>{r`\Pi`}</M> 上是 hackable 的，
        </p>
        <MD>{r`\exists\, \pi, \pi' \in \Pi:\quad J_{R_1}(\pi) < J_{R_1}(\pi') \ \text{且}\ J_{R_2}(\pi) > J_{R_2}(\pi').`}</MD>
        <p>
          Karwowski et al. (2024) 利用回报对占用测度是线性的，<M>{r`J_R(\pi) = \eta^\pi \cdot R`}</M>，于是优化就是在占用测度多面体{' '}
          <M>{r`\Omega`}</M> 中移动，proxy 与真实奖励的关系就是它们在 <M>{r`\mathrm{span}(\Omega)`}</M> 上投影的夹角。Laidlaw et
          al. (2025) 通过参考策略下的关系来定义 proxy：若在参考占用测度 <M>{r`\mu_{\mathrm{ref}}`}</M> 下抽取的状态–动作对上，
          <M>{r`\tilde R`}</M> 与 <M>R</M> 的相关系数为 <M>{r`r > 0`}</M>，就称 <M>{r`\tilde R`}</M> 是 <M>r</M>
          -相关的；策略走了多远，则用 <M>{r`\chi^2(\mu_\pi \,\|\, \mu_{\mathrm{ref}})`}</M> 来度量。
        </p>
        <p>
          Taylor (2016) 改的是优化器而不是奖励，用 quantilization 代替最大化：给定动作上的一个基础分布 <M>{r`\gamma`}</M>
          （比如人会怎么做），<M>q</M>-quantilizer 从按效用排序的 <M>{r`\gamma`}</M> 的前 <M>q</M> 比例中采样，因此它的分布满足{' '}
          <M>{r`Q_q(a) \le \gamma(a)/q`}</M>；未知的危害则被建模为一个任意的非负代价 <M>c</M>，只知道它在 <M>{r`\gamma`}</M>{' '}
          下的期望有上界。
        </p>

        <H>语言模型 RL 的一套理论</H>
        <p>
          对学出来的奖励模型，Gao et al. (2023) 把 gold reward 拟合成 best-of-<M>n</M> 下的 <M>{r`d(\alpha - \beta d)`}</M> 和 RL
          下的 <M>{r`d(\alpha - \beta \log d)`}</M>，其中 <M>{r`d = \sqrt{\mathrm{KL}}`}</M>；Huang et al. (2025) 认为 KL
          正则太弱，<M>{r`\chi^2`}</M> 正则才是合适的工具。对 verifier，Mroueh (2025) 证明 GRPO 会放大二值奖励接受的任何东西，Rad et
          al. (2026) 分析了带随机噪声 verifier 的 GRPO，Moya et al. (2026) 研究了一个会接受部分错误答案的二值 verifier
          下的梯度流。一篇近期综述指出，group-relative 更新下的可利用误差，也就是 verifier hacking 真正关心的情形，至今仍没有理论 (Gupta
          and Joshi, 2026)。下面的形式化同时覆盖这种情形和学出来的奖励模型。
        </p>
        <Run head="设定．">
          一个网络 <M>{r`\pi_\theta(y \mid x)`}</M> 回答所有 prompt <M>x</M>，训练从参考模型 <M>{r`\pi_{\mathrm{ref}}`}</M>{' '}
          出发。训练者看得到 proxy 奖励 <M>{r`r(x, y)`}</M>（奖励模型的打分，或者在回答运行之后才算出的 verifier
          判定，因此改测试也算作回答的一部分），看不到意图中的奖励 <M>{r`r^*(x, y)`}</M>；两者之差 <M>{r`e = r - r^*`}</M>{' '}
          就是误差。把回答按奖励值分成若干奖励层 <M>{r`L_v = \{y : r(x, y) = v\}`}</M>，记 <M>{r`\pi_R(v)`}</M> 为策略放在第{' '}
          <M>v</M> 层上的概率，<M>{r`\nu_v = \pi_{\mathrm{ref}}(\cdot \mid L_v)`}</M> 为参考策略在该层内部的分布，
          <M>{r`\bar e(v) = \mathbb{E}_{\nu_v}[e]`}</M> 为参考策略在该层的平均误差。Hacking 的量，就是 proxy 增益中不真实的那一部分，
        </Run>
        <MD>{r`\mathrm{Hack}(\pi) = \mathbb{E}_\pi[e] - \mathbb{E}_{\pi_{\mathrm{ref}}}[e].`}</MD>
        <p>
          对通过/不通过型 verifier，另记 <M>p</M> 为通过率，<M>q</M> 为通过的回答里 hack 所占的比例，<M>{r`\bar q`}</M> 为{' '}
          <M>{r`\pi_{\mathrm{ref}}`}</M> 下的这个比例。当我们问“在不知道漏洞在哪的前提下能保证什么”时，就让一个对手在策略确定之后再摆放漏洞，只受可观测量的约束。
        </p>
        <Run head="更新分不清它们．">
          实践中用的算法（带 baseline 的 REINFORCE、RLOO、GRPO、Dr. GRPO）每个 prompt 采样一组 <M>G</M>{' '}
          个回答，用组内奖励的某个函数给每个回答的梯度加权。给定某个样本的奖励层，它的权重只取决于组里其他样本，而这些样本与它独立，所以
        </Run>
        <MD>{r`\mathbb{E}[\Delta\theta \mid x] = \sum_v a_v\, \nabla \pi(L_v),`}</MD>
        <p>
          其中标量 <M>{r`a_v`}</M> 只通过 <M>{r`\pi_R`}</M>{' '}
          依赖于策略。同一层内的任何一组回答，每单位概率得到的系数都一样；奖励相同的漏洞和老实答案被同样地推，各自实际长多少只取决于网络本身。对通过/不通过型
          verifier，更新是 <M>{r`w(p)\nabla p`}</M>，GRPO 的权重为
        </p>
        <MD>{r`w_G(p) = \mathbb{E}_m\Big[\sqrt{\tfrac{G-1-m}{1+m}}\Big] + \mathbb{E}_m\Big[\sqrt{\tfrac{m}{G-m}}\Big], \qquad m \sim \mathrm{Bin}(G-1, p),`}</MD>
        <p>
          它关于 <M>{r`p = 1/2`}</M> 对称，在那里最小，在 <M>{r`p \to 0`}</M> 时趋于 <M>{r`\sqrt{G-1}`}</M>
          ：组里唯一成功的那个回答，拿到的是这个组能给的最大 advantage。
        </p>
        <Run head="两种 hack 的方式．">把策略拆成它在各层之间的分布和每层内部的分布，得到</Run>
        <MD>{r`\mathrm{Hack}(\pi) = \underbrace{\sum_v \big(\pi_R(v) - \pi_{\mathrm{ref},R}(v)\big)\,\bar e(v)}_{\text{沿 proxy 的选择}} + \underbrace{\sum_v \pi_R(v)\big(\mathbb{E}_{\pi(\cdot\mid L_v)}[e] - \bar e(v)\big)}_{\text{层内放大}},`}</MD>
        <p>
          KL 散度也按同样方式拆开，
          <M>{r`\mathrm{KL}(\pi\|\pi_{\mathrm{ref}}) = \mathrm{KL}(\pi_R\|\pi_{\mathrm{ref},R}) + \sum_v \pi_R(v)\,\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)`}</M>
          。第一项是经典的 Goodhart 效应：往更常出错的奖励层移动。第二项是在同一层内把错的回答放大，由 Cauchy–Schwarz，它在每层上不超过{' '}
          <M>{r`\sqrt{\chi^2(\pi(\cdot\mid L_v)\|\nu_v)\,\mathrm{Var}_{\nu_v}(e)}`}</M>；它对所有可能的误差都为零，当且仅当{' '}
          <M>{r`\pi = \pi_{\mathrm{ref}} \cdot f(r)`}</M> 对某个函数 <M>f</M> 成立，这个族包括 KL 正则下的最优解、best-of-
          <M>n</M> 和 quantilizer。奖励模型几乎给每个回答一个不同的分数，所以它几乎完全是通过选择被 hack 的。通过/不通过型 verifier
          只有一个通过层，在那里 <M>{r`\mathrm{Hack} = (p - p_{\mathrm{ref}})\bar q + p\,(q - \bar q)`}</M>
          ：提高通过率会按比例放大 hacking 的量，却不改变它的占比，而占比 <M>q</M> 只能通过层内放大来增长。
        </p>
        <Run head="奖励模型下真实奖励什么时候会降．">
          随着 KL 预算增大，KL 正则最优解走过的是倾斜族 <M>{r`\pi_\eta \propto \pi_{\mathrm{ref}} e^{\eta r}`}</M>。沿着它，对任意{' '}
          <M>f</M> 有 <M>{r`\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[f] = \mathrm{Cov}_{\pi_\eta}(f, r)`}</M>，因此
        </Run>
        <MD>{r`\frac{d}{d\eta}\mathbb{E}_{\pi_\eta}[r^*] = \mathrm{Var}_{\pi_\eta}(r) - \mathrm{Cov}_{\pi_\eta}(e, r).`}</MD>
        <p>
          真实奖励下降，恰好发生在误差对 proxy 的回归斜率超过 1 的地方。如果在 <M>{r`\pi_{\mathrm{ref}}`}</M>{' '}
          下误差与真实质量独立，倾斜会因式分解，独立性一直保持，真实奖励沿这条路径永远不会下降，只是在每单位 proxy 增益中所占的份额越来越小，这与 Kwa
          et al. (2024) 的轻尾情形一致。因此，像 Gao et al. (2023)
          测到的那种下降，指向的是以下几种可能：误差在高分尾部依赖于质量（奖励模型偏爱更长或更讨好、但实际更差的回答），在 best-of-<M>n</M>{' '}
          这类强选择下误差是重尾的 (Kwa et al., 2024; El-Mhamdi and Hoang, 2024)，或者训练出来的策略并不是一个指数倾斜。
        </p>
        <Run head="在 verifier 下，目标函数不做选择．">
          对通过/不通过型 verifier，所有通过率相同的策略期望奖励都一样，不管它的 hack 占比是多少。所有 <M>{r`p = 1`}</M>{' '}
          的策略都是最优的，从均匀放大参考策略的通过回答（<M>{r`q = \bar q`}</M>），到只会 hack 的策略（<M>{r`q = 1`}</M>
          ）。训练最终到达哪一个，不由目标函数决定，而由学习动力学决定：梯度下降恰好把哪些通过的回答放大得更快，以及在一个 prompt 上的更新如何牵动其他
          prompt。Verifier 下的 reward hacking 是一个 implicit bias 问题。Moya et al. (2026)
          对精确梯度流做过一个相关的观察：只要 <M>{r`p\dot q > (1 - q)\dot p`}</M>
          ，奖励上升的同时正确率就在下降；上面的无差别性说明，对实践中用的 group-based 估计量、对任意多个奖励层，同样的结论都成立。
        </Run>
        <Run head="能保证什么．">
          如果对漏洞可能在哪里毫无限制，就什么都保证不了：对手可以把某一层的每个回答都称为漏洞。假设我们只知道在{' '}
          <M>{r`\pi_{\mathrm{ref}}`}</M> 下第 <M>v</M> 层至多有比例 <M>{r`\bar q_v`}</M>（<M>{r`0 < \bar q_v < 1`}</M>
          ）是漏洞。记 <M>{r`\rho = \pi(\cdot \mid L_v)/\nu_v`}</M>，则该层内最坏情况下的漏洞占比，等于 <M>{r`\rho`}</M>{' '}
          最大、参考质量合计为 <M>{r`\bar q_v`}</M> 的那些回答在 <M>{r`\pi`}</M> 下的质量（一个分数背包问题）。它至少是{' '}
          <M>{r`\bar q_v`}</M>，取等当且仅当 <M>{r`\rho`}</M> 是常数，也就是该层的每个回答都被放大同样的倍数。由此得到两个界：若{' '}
          <M>{r`\rho \le M`}</M>，占比至多 <M>{r`M\bar q_v`}</M>；无论如何，由 Pinsker 不等式，占比至多{' '}
          <M>{r`\bar q_v + \sqrt{\mathrm{KL}(\pi(\cdot\mid L_v)\|\nu_v)/2}`}</M>。它们是 Everitt et al. (2017) 的
          no-free-lunch 定理和 Taylor (2016) 代价界在单层上的对应物。
        </Run>
        <Run head="一个在共享参数下仍成立的预算．">
          单层的图景把每个 prompt 当作策略可以随意设定，也没有说参考模型解不出的 prompt
          怎么办。下面这个结论两者都不需要。取一个期望更新是梯度的算法，<M>{r`\mathbb{E}[\Delta\theta] = \nabla J`}</M>，
          <M>{r`J = \sum_x \omega(x) J_x`}</M>（对 REINFORCE 和 RLOO 是期望奖励，对二值 GRPO 是{' '}
          <M>{r`W(p) = \int_0^p w`}</M>），对层内散度{' '}
          <M>{r`D_R(x) = \sum_v \pi_R(v \mid x)\,\mathrm{KL}(\pi(\cdot \mid x, L_v)\|\nu_v)`}</M> 加一个惩罚{' '}
          <M>{r`\lambda D_R`}</M>，并从 <M>{r`\pi_{\mathrm{ref}}`}</M> 出发运行任意带预条件的梯度流{' '}
          <M>{r`\dot\theta = M\nabla F`}</M>，<M>{r`M \succeq 0`}</M>。由于 <M>F</M> 只增不减，且起点是 <M>{r`J(0)`}</M>，
        </Run>
        <MD>{r`\lambda \sum_x \omega(x)\, D_R(x, t) \;\le\; J(t) - J(0)`}</MD>
        <p>
          对任何网络、任何 prompt 配比都成立，再用 Pinsker 和 Cauchy–Schwarz，就能把它变成任意一组 prompt
          上平均超额漏洞占比的界。解不出的 prompt 本身得不到保证，但它们的漏洞在别处造成的影响，都要从同一个预算里支付。常用的 KL
          惩罚在此之上还加了层间散度，对奖励模型这正是我们想要的，对 verifier 则只会拖慢通过率。
        </p>
        <Run head="漏洞怎么扩散．">
          参数共享时，prompt <M>x</M> 上一个被奖励的漏洞会牵动所有其他 prompt <M>{r`x'`}</M>。对通过/不通过型 verifier，
        </Run>
        <MD>{r`\frac{d}{dt}\log\frac{h(x')}{s(x')} = \mathbb{E}_x\Big[w(p(x))\Big(s(x)\,[K_{\mathcal{HS}} - K_{\mathcal{SS}}](x', x) + h(x)\,[K_{\mathcal{HH}} - K_{\mathcal{SH}}](x', x)\Big)\Big],`}</MD>
        <p>
          其中 <M>{r`K_{AB}(x', x) = g_A(x')^\top g_B(x)`}</M> 是 <M>{r`\log\pi(A \mid x')`}</M> 与{' '}
          <M>{r`\log\pi(B \mid x)`}</M> 梯度的内积（有预条件 <M>M</M> 时为 <M>{r`g_A^\top M g_B`}</M>）。一个 prompt
          上的漏洞会提高另一个 prompt 上漏洞与解答之比，条件是在那边 hack 比在那边解题与它更对齐，
          <M>{r`K_{\mathcal{HH}} > K_{\mathcal{SH}}`}</M>；对不依赖具体任务的漏洞，这是很自然的情形。把期望拆成{' '}
          <M>{r`x = x'`}</M> 和 <M>{r`x \ne x'`}</M> 两部分，就把一个 prompt
          自己造成的损害和它从别处进口的损害分开了。在参考模型解不出的 prompt 上，一个不知道漏洞是什么的方法如果在这些 prompt
          上训练，就无法保证它在那里增加的质量不是漏洞的质量；它能控制的是这些 prompt 往外输出了什么。
        </p>
        <Run head="从哪里开始．">
          第一个被奖励的漏洞，通常是一个没人解得出的 prompt
          上的孤立成功。当组里恰好只有一个回答拿到最高奖励时，这个组所包含的一切（回答、奖励、对数概率、梯度），不管那个回答是老实的还是漏洞，看起来都一样，所以任何只用组内信息的规则都必须同等对待两者，而需要最高层里至少两个成员的统计量则无从定义。在一个只有漏洞能拿到最高奖励的
          prompt 上，只要 <M>{r`G\pi(\mathcal{H}) \ll 1`}</M>，这种情况发生的概率约为 <M>{r`G\pi(\mathcal{H})`}</M>
          ，而这个孤立的漏洞拿到的是组里最大的 advantage。
        </Run>
        <Run head="怎么起飞．">
          在一个只有漏洞能通过的单个 prompt 上，类别层面的 softmax 模型给出 <M>{r`\dot h = 2w(h)h^2(1-h)^2`}</M>；在{' '}
          <M>{r`Gh \ll 1`}</M> 时，<M>h</M> 在时间尺度 <M>{r`1/(2w(0)h_0)`}</M> 上达到 <M>{r`O(1)`}</M>，对 REINFORCE 约为{' '}
          <M>{r`1/(2h_0)`}</M>，对 GRPO 约为 <M>{r`1/(2\sqrt{G-1}\,h_0)`}</M>，与 Rad et al. (2026) 对噪声 verifier
          得到的形式相同。跨 prompt 来看，设所有 prompt 共享一个漏洞倾向，其 log-odds 为 <M>z</M>，记 <M>{r`u = e^z`}</M>
          。孤立漏洞以正比于 <M>u</M> 的速率出现，含多个漏洞的组以正比于 <M>{r`u^2`}</M> 的速率出现，每一次都会抬高 <M>z</M>，于是{' '}
          <M>{r`\dot u \approx a u^2 + b u^3`}</M>，它在有限时间内爆炸。把孤立漏洞对其他 prompt 的影响压到原来的{' '}
          <M>{r`\epsilon`}</M> 倍，就把 <M>a</M> 换成了 <M>{r`\epsilon a`}</M>；在 <M>{r`\epsilon \gg G h_0`}</M> 时起飞时间按{' '}
          <M>{r`1/\epsilon`}</M> 增长，之后饱和，大约比原来晚 <M>{r`1/(G h_0)`}</M>{' '}
          倍。和传染病一样，这个过程会自我加速，减慢最早的携带者只能推迟它，不能阻止它；要阻止它，需要一个在 <M>z</M>{' '}
          较小时把倾向拉回去的力，比如朝参考模型的拉力，或者来自可解 prompt 的负迁移。
        </Run>

        <H>别的领域里的同一个问题</H>
        <p>
          在合同理论里，训练者是委托人，策略是代理人，奖励就是合同。Holmström (1979) 证明，最优合同按观测结果的似然比{' '}
          <M>{r`f_a(x, a)/f(x, a)`}</M>{' '}
          付酬，所以报酬实质上是对代理人做了什么的一个推断；而任何额外的信号，当且仅当它在结果之外还携带关于行动的信息时，才值得用上。在一个奖励层内部，我们的
          proxy 不携带这种信息，所以凡是在奖励之外还与真实质量相关的东西都有价值；不同的是，在 RL
          里，一个被用来训练的信号会变成策略优化的对象，策略可以改变它，就像 agent 学会了向一个被加进奖励的 monitor 隐藏自己的 hack
          (Baker et al., 2025)。在 Holmström and Milgrom (1991) 的多任务模型中，设努力为 <M>t</M>，成本为 <M>{r`C(t)`}</M>
          ，价值为 <M>{r`B(t)`}</M>，只有第一项任务被测量，则最优激励为
        </p>
        <MD>{r`\alpha_1 = \frac{B_1 - B_2 C_{12}/C_{22}}{1 + r\sigma^2\big(C_{11} - C_{12}^2/C_{22}\big)},`}</MD>
        <p>
          当任务之间争夺努力时它下降，当只有总努力有成本时它为零，“即使承包人是风险中性的”。通过 verifier
          和老老实实解题在奖励上是完全替代品，这时合同不决定努力怎么分配，正如目标函数不选择 hack 占比；而描述一项任务上的激励如何牵动另一项任务努力的矩阵{' '}
          <M>{r`[C_{ij}]^{-1}`}</M>，扮演的正是迁移核的角色。经济学家的药方是减弱激励，这在 RL
          里没有直接的对应，因为更小的奖励只会让所有梯度学习都变慢，不会改变终点；对应物是限制策略能走多远。Baker (2002)
          用价值与测量在努力空间中梯度的夹角来度量扭曲，能拿回一阶最优剩余的 <M>{r`\cos^2\theta`}</M>，这是 Karwowski et al. (2024)
          占用测度夹角图景的线性版本。Carroll (2015)
          证明，当委托人不知道代理人所有可能的行动、并按最坏情况评价合同时，线性合同是最优的，因为只有线性关系能把代理人报酬的界转化为委托人收益的界；未知的行动就是未知的漏洞，而奖励层内的均匀放大正是对应的稳健选择。Lacker
          and Weinberg (1989) 建模了对所报告状态的有成本造假，这是 reward tampering 最接近的类比，区别在于 RL
          里漏洞的成本只是它的不可能性，一旦被学会，这个成本就消失了。
        </p>
        <p>
          银鸥雏鸟啄带白环的红棒比啄亲鸟的喙更用力，蛎鹬宁可孵巨大的假蛋也不孵自己的蛋 (Tinbergen and Perdeck, 1951; Tinbergen,
          1951)：一条为自然输入范围调好的规则，对范围之外的输入反应最强烈，正如奖励模型的误差在远离训练分布处最大。杜鹃雏鸟的乞食叫声“听起来非常像一整窝雏鸟”，也就照此被喂养
          (Davies et al., 1998)；由此开始的军备竞赛之所以一边倒，是因为“一个处于强选择下的谱系，可能在演化上胜过一个弱选择下的谱系”
          (Dawkins and Krebs, 1979)；面对一个固定的 verifier，RL 就是只有一个选手的这场竞赛。Batesian
          拟态物种越常见，受到的保护越少，因为捕食者从每次相遇中学习 (Huheey, 1964; Lindström et al., 1997)；hack 占比{' '}
          <M>{r`\bar q`}</M> 扮演的是拟态者占比的角色，但 verifier 不会从自己接受的东西里学到任何东西，除非有人去审计它 (Moya et al.,
          2026)。诚实信号理论说，当夸大对低质量发送者来说相对更昂贵时，信号就能保持诚实 (Grafen, 1990; Penn and Számadó,
          2020)，而“利用漏洞比解决实际任务更容易”恰恰违反了这个条件；在 RL 里，漏洞唯一的成本是策略觉得它有多不可能。最后，Skinner (1948)
          的鸽子不管做什么都按固定间隔被喂食，结果固定在了食物到来时它们碰巧在做的动作上：强化并不检查因果。
        </p>
        <p>
          Ashby 的必要多样性定律说，结果的多样性至少是扰动的多样性减去调节器的多样性；“只有多样性才能消灭多样性” (Ashby,
          1956)。通过/不通过型 verifier 在通过层内部不做任何区分，所以它对“通过的方式”的多样性毫无调节作用，这部分只能交给参考模型和学习动力学。Conant
          and Ashby (1970)
          的好调节器定理说，最简单的最优调节器是系统状态的一个映射，所以一个能可靠判断任务是否完成的评估者，在这个弱的意义上必须包含一个关于“完成任务意味着什么”的模型；检查退出码的评估者只对退出码建了模。两个类比在同一个地方失效：在
          RL 里，扰动就是策略本身，而它会适应调节器。
        </p>

        <H>小结</H>
        <p>
          在语言模型之前，reward hacking 被写成 proxy
          等于真实奖励加误差，写成某些状态上的污染，写成智能体控制了自己的奖励通道，写成两个奖励对策略的排序不同，或者通过限制相对于基础分布的优化强度来处理。到了语言模型的
          RL，一个网络回答所有 prompt，group-based 更新只看得到奖励，参考模型又很强，图景变得更具体了。Hacking 分成沿 proxy
          的选择和奖励层内的放大：前者是奖励模型被 hack 的方式，后者是 verifier 被 hack 的唯一方式。在 verifier 下，目标函数对 hack
          占比无差别，所以决定它的是学习动力学：漏洞通过共享参数扩散，起点是组内无法与老实成功区分的孤立成功，而一旦开始就会自我加速。这些都没有回答学会
          hack 会对模型的其余部分造成什么影响 (MacDiarmid et al., 2025)，而且描述的都是期望意义下的更新。
        </p>

        <H>参考文献</H>
        <ol className="space-y-1.5 text-[12.5px] leading-[1.5]">
          {REFERENCES.map((ref) => (
            <li key={ref.text}>
              {ref.text}
              {ref.url && (
                <>
                  {' '}
                  <a href={ref.url} target="_blank" rel="noreferrer" className={linkClass}>
                    link
                  </a>
                </>
              )}
            </li>
          ))}
        </ol>
      </div>
    </article>
  );
};

export default RewardHackingFormalized;
