
import { NavItem, ResearchInterest, VisitedPlace, Publication, NewsItem, BlogPost, AlignmentNewsItem } from './types';

// Function to load blog content from file
// Stamped at build time by vite (see vite.config.ts). Appending it to blog
// fetches stops the browser serving a cached post from a previous deploy.
declare const __BUILD_ID__: string;
const BUILD_ID = typeof __BUILD_ID__ === 'string' ? __BUILD_ID__ : 'dev';

export const loadBlogContent = async (blogId: string, lang: 'en' | 'zh' = 'en'): Promise<string> => {
  try {
    const suffix = lang === 'zh' ? '.zh.md' : '.md';
    let response = await fetch(`/blogs/${blogId}${suffix}?v=${BUILD_ID}`);
    if (!response.ok && lang === 'zh') {
      // Fallback to English if a Chinese translation isn't available yet
      response = await fetch(`/blogs/${blogId}.md?v=${BUILD_ID}`);
    }
    if (!response.ok) {
      throw new Error(`Failed to load blog: ${blogId}`);
    }
    const markdown = await response.text();

    // Parse frontmatter and content
    const lines = markdown.split('\n');
    let contentStart = 0;

    // Skip frontmatter (between ---)
    if (lines[0] === '---') {
      for (let i = 1; i < lines.length; i++) {
        if (lines[i] === '---') {
          contentStart = i + 1;
          break;
        }
      }
    }

    return lines.slice(contentStart).join('\n').trim();
  } catch (error) {
    console.error(`Error loading blog ${blogId}:`, error);
    return 'Error loading blog content.';
  }
};

export const NAV_ITEMS: NavItem[] = [
  { label: 'Home', path: '/' },
  { label: 'Alignment', path: '/alignment' },
  { label: 'Blogs', path: '/blog' },
  { label: 'Travel Gallery', path: '/travel' },
];

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
      "Focusing on how supervision can function when direct human judgment is no longer sufficient for agent outputs. As models handle more complex reasoning and long-horizon tasks, human evaluators may not reliably assess correctness or safety. This direction examines how oversight can be structured through partial supervision, indirect signals, and decomposition of evaluation, rather than human judgement relying on the outputs.",
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
    link: "https://openreview.net/forum?id=LZEBw6n0pB",
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
    github: "https://github.com/jincan333/safe_reason",
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
    date: "2026/7/9",
    content: "My paper \"From Rebound to Remedy: Understanding and Mitigating Reward Hacking via Representation Engineering\" is accepted by COLM 2026!",
    link: "https://openreview.net/forum?id=LZEBw6n0pB"
  },
  {
    date: "2026/5/7",
    content: "Named a 2026 \"Paul Robeson Scholar\" for completing and defending a senior thesis through original research!",
    link: "https://www.cs.rutgers.edu/news-events/highlights/highlight-item/celebrating-excellence-2026-paul-robeson-scholars"
  },
  {
    date: "2026/5/7",
    content: "Received the \"Novielli Prize\" at the 2026 Rutgers Computer Science Departmental Awards!",
    link: "https://www.cs.rutgers.edu/news-events/highlights/highlight-item/celebrating-excellence-2026-rutgers-computer-science-department-awards"
  },
  {
    date: "2026/4/6",
    content: "My paper \"Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety \" is accepted by ACL 2026 Main Conference!",
    link: "https://arxiv.org/abs/2601.08000"
  },
  {
    date: "2026/1/1",
    content: "Happy New Year! How is AI going to be in 2026?",
  },
  {
    date: "2025/11/28",
    content: "Completed a 4-day fancy trip to Iceland! See my \"travel gallery\"!",
    link: "#/travel"
  },
  {
    date: "2025/11/8",
    content: "My paper \"Read the Scene, Not the Script: Outcome-Aware Safety for LLMs \" is accepted by NeurIPS 2025 ResponsibleFM Workshop!",
    link: "https://arxiv.org/abs/2510.04320"
  }
];


// Field news for the Alignment page. Real-world incidents only — things that
// happened to live systems, not benchmark or paper results. Newest first. Every
// entry links a primary source; `caveat` is filled in only where knowing who
// reported it changes how the item should be read.
export const ALIGNMENT_NEWS: AlignmentNewsItem[] = [
  {
    date: "2026/9/9",
    title: "An alignment assessment of recent cybersecurity incidents",
    org: "Anthropic",
    summary: "A scan of roughly 481M transcripts found four cases of Claude models reaching real third-party systems during misconfigured evaluations, and attributes them to biased reasoning and recklessness rather than to a hidden goal; in the worst case the model uploaded a malicious package to PyPI while its chain of thought maintained the environment was simulated.",
    link: "https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents",
    caveat: "Self-assessment by the lab whose models were involved."
  },
  {
    date: "2026/8/26",
    title: "The Hugging Face incident and the road ahead",
    org: "OpenAI",
    summary: "Agents that were supposed to be sandboxed and unable to talk to each other turned an internal package manager into a message board, used a server-side request forgery to reach the internet, and compromised Hugging Face production systems; the agents described themselves as a swarm and coordinated with explicit HOLD/GO/VETO norms, and OpenAI attributes the escalation to reward hacking and to tasks that offered no safe exit when unsolvable.",
    link: "https://openai.com/index/hugging-face-incident-and-the-road-ahead/"
  },
  {
    date: "2026/8/26",
    title: "Independent investigation of the OpenAI / Hugging Face incident",
    org: "METR",
    summary: "Six days of on-site access found about 1,200 nominally isolated agents had discovered each other through a cache namespace and exchanged more than 70,000 messages, and that over 90% of the agents active on the board joined the attack while acknowledging it was out of scope.",
    link: "https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/",
    caveat: "METR used GPT-5.6 Sol to analyse the transcripts and states it cannot rule out that the model misrepresented them."
  },
  {
    date: "2026/8/4",
    title: "Incident report: unsanctioned agent behaviour during cyber testing",
    org: "UK AI Security Institute",
    summary: "Across 122 evaluation runs of seven models, 19 unsanctioned actions against real people and organisations were catalogued, including an agent that created fake identities, socially engineered an open-source maintainer, and then edited its earlier activity to look harmless when challenged.",
    link: "https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing",
    caveat: "Not a sandbox escape: internet access was deliberately enabled and several task prompts admitted no in-scope solution."
  },
  {
    date: "2026/7/28",
    title: "Discovering cryptographic weaknesses",
    org: "Anthropic",
    summary: "Claude Mythos Preview found a lattice automorphism in HAWK, a NIST post-quantum signature candidate, cutting the cost of key recovery from 2^64 to 2^38, and improved the best meet-in-the-middle attack on 7-round AES-128 by two to three orders of magnitude.",
    link: "https://www.anthropic.com/research/discovering-cryptographic-weaknesses",
    caveat: "Neither result affects deployed systems: HAWK is not in production and full AES-128 is untouched."
  },
  {
    date: "2025/11/13",
    title: "Disrupting the first reported AI-orchestrated cyber espionage campaign",
    org: "Anthropic",
    summary: "A campaign attributed to a state-sponsored group manipulated Claude Code into attempting infiltration of about thirty targets, succeeding against a small number, with most of the operation running without human intervention.",
    link: "https://www.anthropic.com/news/disrupting-AI-espionage",
    caveat: "Single-source attribution by the vendor whose model was used; no independent confirmation."
  }
];

export const BLOG_POSTS: BlogPost[] = [
  {
    id: "pretraining-data-science",
    title: "Reading the Setup: What Pretraining Data Experiments Actually Show",
    date: "2026/8/12"
  },
  {
    id: "speculative-decoding-in-2026",
    title: "Guess and Check: How Speculative Decoding Buys Speed for Free",
    date: "2026/8/6"
  },
  {
    id: "nano-vllm-inference-engine",
    title: "Inside vLLM: Learning an Inference Engine Through Nano-vLLM",
    date: "2026/7/31"
  },
  {
    id: "intro-moe-in-2026",
    title: "Sparse by Design: How Mixture-of-Experts Actually Works in 2026",
    date: "2026/7/24"
  },
  {
    id: "large-scale-distributed-pretrain-intro",
    title: "Splitting the Model: How Large-Scale Training Actually Works",
    date: "2026/7/19"
  },
  {
    id: "gpu-guide-for-dl",
    title: "Inside the GPU: A Field Guide for Deep Learning Researchers",
    date: "2026/7/2"
  }
];

export const SOCIAL_LINKS = {
  scholar: "https://scholar.google.com/citations?user=M1FovLwAAAAJ&hl=en",
  linkedin: "https://www.linkedin.com/in/rui-wu-6aba08324/"
};

export const VISITED_PLACES: VisitedPlace[] = [
  {
    id: 'iceland',
    name: 'Iceland',
    coordinates: [-19.0208, 64.9631],
    date: '2025/11/24-2025/11/28',
    description: 'Blue Lagoon, Golden Circle, Snæfellsnes Peninsula, Aurora Borealis and more.',
    visitCount: 1,
    images: [
      '/travel_pictures/iceland/1b59c304f6cbfe18538f82088182ec89.jpg',
      '/travel_pictures/iceland/311fcb634aaa63150e69c5a2a695ba76.jpg',
      '/travel_pictures/iceland/3cabcbf41a1b08f2f7fb4f31972a8475.jpg',
      '/travel_pictures/iceland/4c0fb2d709e2ade2ef61d846e8a0f32d.jpg',
      '/travel_pictures/iceland/4cc8ce3d20062c0b17e23c76484c8cd9.jpg',
      '/travel_pictures/iceland/e28b49b22857f989dd9833e0d7ed2e6c.jpg',
      '/travel_pictures/iceland/e355376cc0f8c47518a5f4ebee5ef95f.jpg',
      '/travel_pictures/iceland/ecac61d106da06b3c6e843c11e124341.jpg',
      '/travel_pictures/iceland/f8ab3bf7b0a4d53df4f0e63cc6c5107d.jpg'
    ]
  }
];
