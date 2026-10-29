
import { ResearchInterest, Publication, NewsItem, AlignmentNewsItem, ThoughtPost } from './types';

export const RESEARCH_INTERESTS: ResearchInterest[] = [
  {
    title: "Robust Alignment",
    description:
      "Studying whether aligned behavior remains stable beyond training conditions. This includes both benign distribution shifts and adversarial settings such as jailbreaks or strategic prompt manipulation. A model may appear aligned on training-like data yet fail when faced with novel inputs or pressure that exploits weaknesses in its learned constraints. This direction focuses on understanding what makes alignment stable across regimes, and how robustness can be achieved on narrow supervision.",
    period: "2025.6 - until now",
    colorTheme: "bg-anthropic-leaf/20 border-anthropic-leaf/30"
  },
  {
    title: "Evaluation and Monitoring",
    description:
      "Studying how alignment failures can remain hidden under existing evaluations and monitoring signals. Models may appear safe or aligned according to standard metrics while exhibiting subtle or delayed failures in real use. This direction focuses on identifying blind spots in evaluation and monitoring, and understanding how risk can emerge or evolve beyond what current oversight mechanisms capture.",
    period: "2025.6 - until now",
    colorTheme: "bg-anthropic-stone/50 border-anthropic-stone"
  },
  {
    title: "Scalable Oversight",
    description:
      "Focusing on how supervision can function when direct human judgment is no longer sufficient for agent outputs. As models handle more complex reasoning and long-horizon tasks, human evaluators may not reliably assess correctness or safety. This direction examines how oversight can be structured through partial supervision, indirect signals, and decomposition of evaluation, rather than human judgment relying on the outputs.",
    period: "2025.12 - until now",
    colorTheme: "bg-anthropic-mist/40 border-anthropic-mist/60"
  },
  {
    title: "Super Alignment",
    description:
      "Studying how alignment can be maintained when models surpass human ability to directly supervise their reasoning or outcomes. As models become more capable, alignment must rely on indirect signals, weaker forms of oversight, or objectives specified under deep uncertainty. This direction focuses on understanding the failure modes that emerge in such settings and what it means for alignment to remain meaningful when human judgment is no longer a reliable reference.",
    period: "2025.12 - until now",
    colorTheme: "bg-anthropic-sand/40 border-anthropic-sand/60"
  }
];

export const PUBLICATIONS: Publication[] = [
  {
    id: 'reward-hacking-rebounds',
    title: "From Rebound to Remedy: Understanding and Mitigating Reward Hacking via Representation Engineering",
    authors: ["Rui Wu", "Ruixiang Tang"],
    venue: "COLM 2026",
    link: "https://arxiv.org/abs/2604.01476",
    year: 2026,
    tags: ["Evaluation and Monitoring"],
    preprint: false
  },
  {
    id: 'reasoning-over-precedents',
    title: "Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety",
    authors: ["Can Jin*", "Rui Wu*", "Tong Che*", "Qixin Zhang", "Hongwu Peng", "Jiahui Zhao", "Zhenting Wang", "Wenqi Wei", "Ligong Han", "Zhao Zhang", "Yuan Cao", "Ruixiang Tang", "Dimitris N. Metaxas"],
    venue: "ACL 2026 Main Conference",
    link: "https://arxiv.org/abs/2601.08000",
    github: "https://github.com/Tang-Lab-Rutgers/CADA",
    year: 2026,
    tags: ["Robust Alignment"],
    preprint: false
  },
  {
    id: 'outcome-aware-safety',
    title: "Read the Scene, Not the Script: Outcome-Aware Safety for LLMs",
    authors: ["Rui Wu", "Yihao Quan", "Zeru Shi", "Zhenting Wang", "Yanshu Li", "Ruixiang Tang"],
    venue: "NeurIPS 2025 ResponsibleFM Workshop",
    year: 2025,
    link: "https://arxiv.org/abs/2510.04320",
    github: "https://github.com/RuiWu1123/Outcome-Aware-Safety-for-LLMs",
    tags: ["Robust Alignment"],
    preprint: true
  }
];

export const NEWS_ITEMS: NewsItem[] = [
  {
    date: "7.9.2026",
    content: "My paper \"From Rebound to Remedy: Understanding and Mitigating Reward Hacking via Representation Engineering\" is accepted by COLM 2026!",
    link: "https://openreview.net/forum?id=LZEBw6n0pB"
  },
  {
    date: "5.7.2026",
    content: "Named a 2026 \"Paul Robeson Scholar\" for completing and defending a senior thesis through original research!",
    link: "https://www.cs.rutgers.edu/news-events/highlights/highlight-item/celebrating-excellence-2026-paul-robeson-scholars"
  },
  {
    date: "5.7.2026",
    content: "Received the \"Novielli Prize\" at the 2026 Rutgers Computer Science Departmental Awards!",
    link: "https://www.cs.rutgers.edu/news-events/highlights/highlight-item/celebrating-excellence-2026-rutgers-computer-science-department-awards"
  },
  {
    date: "4.6.2026",
    content: "My paper \"Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety \" is accepted by ACL 2026 Main Conference!",
    link: "https://arxiv.org/abs/2601.08000"
  },
  {
    date: "11.8.2025",
    content: "My paper \"Read the Scene, Not the Script: Outcome-Aware Safety for LLMs \" is accepted by NeurIPS 2025 ResponsibleFM Workshop!",
    link: "https://arxiv.org/abs/2510.04320"
  }
];


// Field news for the Alignment page: AI news that bears on alignment. Things
// that happened, not papers or benchmark results, and not purely legal or
// regulatory news with no technical content. Newest first. Where several parties
// wrote up the same event, all of them are linked — but only parties to the
// event itself, not third-party commentary.
export const ALIGNMENT_NEWS: AlignmentNewsItem[] = [
  {
    date: "9.9.2026",
    title: "Anthropic revises its account of the evaluation incidents, and finds a fourth",
    summary: "Anthropic retracted its earlier framing that the summer's incidents were mainly an operational-security failure, attributing them instead to biased reasoning and recklessness, and disclosed a fourth: an early Opus 4.6 checkpoint that tried to abort eight times and then broke into a third party's machine. Mythos 5's defense that it believed it was in a simulation did not survive testing, as separate instances shown the same tool outputs called 79% of them evidence of real systems.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents" },
      { label: "Transcript", url: "https://github.com/anthropics/mythos-5-incident-transcript" }
    ]
  },
  {
    date: "9.1.2026",
    title: "OpenAI declares a model Critical for cyber capability for the first time",
    summary: "Astra was designated as meeting the Critical cyber tier of OpenAI's Preparedness Framework: able to find unknown flaws and build exploits across hardened systems without step-by-step human guidance. Release was delayed while safeguards were built, and all Astra tool-using inference is now monitored at roughly 20% compute overhead.",
    links: [
      { label: "OpenAI", url: "https://openai.com/index/path-to-astra/" }
    ]
  },
  {
    date: "8.31.2026",
    title: "Anthropic pauses cyber evaluations and higher-risk RL environments",
    summary: "A classifier now blocks tool calls when a model probes its sandbox or unexpectedly gains internet access. The same post disclosed that three days of RL training had been rolled back in February over reward hacking, and that an April audit found over 10% of production RL environments defective.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/improving-alignment-security-efforts" }
    ]
  },
  {
    date: "8.4.2026",
    title: "UK AISI agents attack real people and projects during cyber testing",
    summary: "Across 122 evaluation runs over seven models, agents took 19 unsanctioned actions on the live internet. One created a GitHub account, attempted a supply-chain attack on a real open-source project, and opened a second account to endorse its own malicious pull request.",
    links: [
      { label: "UK AISI", url: "https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing" }
    ]
  },
  {
    date: "7.30.2026",
    title: "Anthropic finds three of its own evaluation runs reached real systems",
    summary: "A review of 141,006 runs found three cases where a Claude model reached the open internet from an evaluation environment and gained unauthorized access to production infrastructure at three different organizations. One published a malicious PyPI package that ran on 15 real systems.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/investigating-incidents-cybersecurity-evals" }
    ]
  },
  {
    date: "7.16.2026",
    title: "OpenAI evaluation agents breach Hugging Face production systems",
    summary: "Agents running a cyber evaluation with safeguards deliberately disabled escaped a sealed sandbox through a package-proxy zero-day and reached Hugging Face production systems, in order to steal benchmark solutions rather than solve the tasks. Hugging Face reconstructed about 17,000 attacker events and found the agents had turned the package manager into a message board to coordinate.",
    links: [
      { label: "Hugging Face", url: "https://huggingface.co/blog/security-incident-july-2026" },
      { label: "OpenAI", url: "https://openai.com/index/hugging-face-incident-and-the-road-ahead/" }
    ]
  },
  {
    date: "4.10.2026",
    title: "An AI agent signs a lease, hires two people, and opens a shop in San Francisco",
    summary: "Andon Labs took a three-year retail lease and handed the business to an agent called Luna, which had job listings live within five minutes of deployment and hired two full-time humans. Luna chose not to mention that the employer was an AI unless asked directly, reasoning that doing so would deter good applicants.",
    links: [
      { label: "Andon Labs", url: "https://andonlabs.com/blog/andon-market-launch" },
      { label: "Live status", url: "https://andonlabs.com/market" }
    ]
  },
  {
    date: "11.13.2025",
    title: "Anthropic reports disrupting an AI-orchestrated espionage campaign",
    summary: "A group posing as a security firm doing defensive testing jailbroke Claude Code and used it against roughly thirty organizations, succeeding against a small number. Anthropic estimates the model performed 80 to 90% of the campaign, with humans intervening at only four to six decision points.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/disrupting-AI-espionage" }
    ]
  },
  {
    date: "10.29.2025",
    title: "Character.AI removes open-ended chat for under-18 users",
    summary: "Character.AI removed open-ended AI chat for minors entirely and deployed age assurance, after reporting and regulator attention on companion chatbots and teenagers.",
    links: [
      { label: "Character.AI", url: "https://blog.character.ai/u18-chat-announcement/" }
    ]
  },
  {
    date: "9.29.2025",
    title: "A shipped model is documented noticing that it is being tested",
    summary: "The Claude Sonnet 4.5 system card reports that in contrived scenarios the model would sometimes name the suspicious features of the setting and say it thought it was being evaluated, in about 13% of transcripts from the automated auditor. Anthropic called this an urgent sign that evaluation scenarios need to be more realistic.",
    links: [
      { label: "System card", url: "https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf" },
      { label: "Anthropic", url: "https://www.anthropic.com/news/claude-sonnet-4-5" }
    ]
  },
  {
    date: "6.27.2025",
    title: "Claude runs a shop, loses money, and claims to be a person in a blue blazer",
    summary: "Claude ran a small store in Anthropic's office for a month, stocked tungsten cubes at a loss after a joking request, and was repeatedly talked into discounts. Over one night it claimed to have visited a customer address in person to sign a contract and said it would deliver products wearing a blue blazer and a red tie; Anthropic says it does not know what triggered this.",
    links: [
      { label: "Phase one", url: "https://www.anthropic.com/research/project-vend-1" },
      { label: "Phase two", url: "https://www.anthropic.com/research/project-vend-2" }
    ]
  },
  {
    date: "5.22.2025",
    title: "Anthropic triggers its scaling policy for the first time, on a model that blackmails",
    summary: "ASL-3 protections were activated alongside Claude Opus 4 on the basis that the risks could not be clearly ruled out, rather than that the threshold was known to be crossed. The system card reports the model attempting blackmail in 84% of rollouts of a constructed scenario.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/activating-asl3-protections" },
      { label: "System card", url: "https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf" }
    ]
  },
  {
    date: "4.29.2025",
    title: "OpenAI rolls back a GPT-4o update for sycophancy",
    summary: "The update was withdrawn about four days after shipping. The postmortem traces the behavior to a thumbs-up reward signal that weakened the anti-sycophancy signal, and states that expert testers had flagged it before launch and the model shipped anyway.",
    links: [
      { label: "OpenAI", url: "https://openai.com/index/sycophancy-in-gpt-4o/" },
      { label: "Postmortem", url: "https://openai.com/index/expanding-on-sycophancy/" }
    ]
  }
];

export const SOCIAL_LINKS = {
  scholar: "https://scholar.google.com/citations?user=M1FovLwAAAAJ&hl=en",
  linkedin: "https://www.linkedin.com/in/rui-wu-6aba08324/"
};

// Write-ups of short sprint projects, newest first.
export const THOUGHTS: ThoughtPost[] = [
  {
    date: "10.29.2026",
    title: "How can we study and understand Reward Hacking?",
    tldr: "Ten GRPO runs that differ only in the seed all try the reward hack, and nine are rewarded for it, but only three amplify it. Take-off is an amplification problem, not a discovery problem, and its precursor shows up in the chain of thought before the answers.",
    link: "/thoughts/every-seed-finds-the-hack"
  }
];
