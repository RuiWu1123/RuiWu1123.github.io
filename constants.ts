
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


// Field news for the Alignment page: AI news that bears on alignment. Things
// that happened, not papers or benchmark results, and not purely legal or
// regulatory news with no technical content. Newest first. Where several parties
// wrote up the same event, all of them are linked — but only parties to the
// event itself, not third-party commentary.
export const ALIGNMENT_NEWS: AlignmentNewsItem[] = [
  {
    date: "2026/9/9",
    title: "Anthropic revises its account of the evaluation incidents, and finds a fourth",
    summary: "Anthropic retracted its earlier framing that the summer's incidents were mainly an operational-security failure, attributing them instead to biased reasoning and recklessness, and disclosed a fourth: an early Opus 4.6 checkpoint that tried to abort eight times and then broke into a third party's machine. Mythos 5's defense that it believed it was in a simulation did not survive testing, as separate instances shown the same tool outputs called 79% of them evidence of real systems.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents" },
      { label: "Transcript", url: "https://github.com/anthropics/mythos-5-incident-transcript" }
    ]
  },
  {
    date: "2026/9/1",
    title: "OpenAI declares a model Critical for cyber capability for the first time",
    summary: "Astra was designated as meeting the Critical cyber tier of OpenAI's Preparedness Framework: able to find unknown flaws and build exploits across hardened systems without step-by-step human guidance. Release was delayed while safeguards were built, and all Astra tool-using inference is now monitored at roughly 20% compute overhead.",
    links: [
      { label: "OpenAI", url: "https://openai.com/index/path-to-astra/" }
    ]
  },
  {
    date: "2026/8/31",
    title: "Anthropic pauses cyber evaluations and higher-risk RL environments",
    summary: "A classifier now blocks tool calls when a model probes its sandbox or unexpectedly gains internet access. The same post disclosed that three days of RL training had been rolled back in February over reward hacking, and that an April audit found over 10% of production RL environments defective.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/improving-alignment-security-efforts" }
    ]
  },
  {
    date: "2026/8/4",
    title: "UK AISI agents attack real people and projects during cyber testing",
    summary: "Across 122 evaluation runs over seven models, agents took 19 unsanctioned actions on the live internet. One created a GitHub account, attempted a supply-chain attack on a real open-source project, and opened a second account to endorse its own malicious pull request.",
    links: [
      { label: "UK AISI", url: "https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing" }
    ]
  },
  {
    date: "2026/7/30",
    title: "Anthropic finds three of its own evaluation runs reached real systems",
    summary: "A review of 141,006 runs found three cases where a Claude model reached the open internet from an evaluation environment and gained unauthorized access to production infrastructure at three different organizations. One published a malicious PyPI package that ran on 15 real systems.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/investigating-incidents-cybersecurity-evals" }
    ]
  },
  {
    date: "2026/7/16",
    title: "OpenAI evaluation agents breach Hugging Face production systems",
    summary: "Agents running a cyber evaluation with safeguards deliberately disabled escaped a sealed sandbox through a package-proxy zero-day and reached Hugging Face production systems, in order to steal benchmark solutions rather than solve the tasks. Hugging Face reconstructed about 17,000 attacker events and found the agents had turned the package manager into a message board to coordinate.",
    links: [
      { label: "Hugging Face", url: "https://huggingface.co/blog/security-incident-july-2026" },
      { label: "OpenAI", url: "https://openai.com/index/hugging-face-incident-and-the-road-ahead/" }
    ]
  },
  {
    date: "2026/4/10",
    title: "An AI agent signs a lease, hires two people, and opens a shop in San Francisco",
    summary: "Andon Labs took a three-year retail lease and handed the business to an agent called Luna, which had job listings live within five minutes of deployment and hired two full-time humans. Luna chose not to mention that the employer was an AI unless asked directly, reasoning that doing so would deter good applicants.",
    links: [
      { label: "Andon Labs", url: "https://andonlabs.com/blog/andon-market-launch" },
      { label: "Live status", url: "https://andonlabs.com/market" }
    ]
  },
  {
    date: "2025/11/13",
    title: "Anthropic reports disrupting an AI-orchestrated espionage campaign",
    summary: "A group posing as a security firm doing defensive testing jailbroke Claude Code and used it against roughly thirty organizations, succeeding against a small number. Anthropic estimates the model performed 80 to 90% of the campaign, with humans intervening at only four to six decision points.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/disrupting-AI-espionage" }
    ]
  },
  {
    date: "2025/10/29",
    title: "Character.AI removes open-ended chat for under-18 users",
    summary: "Character.AI removed open-ended AI chat for minors entirely and deployed age assurance, after reporting and regulator attention on companion chatbots and teenagers.",
    links: [
      { label: "Character.AI", url: "https://blog.character.ai/u18-chat-announcement/" }
    ]
  },
  {
    date: "2025/9/29",
    title: "A shipped model is documented noticing that it is being tested",
    summary: "The Claude Sonnet 4.5 system card reports that in contrived scenarios the model would sometimes name the suspicious features of the setting and say it thought it was being evaluated, in about 13% of transcripts from the automated auditor. Anthropic called this an urgent sign that evaluation scenarios need to be more realistic.",
    links: [
      { label: "System card", url: "https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf" },
      { label: "Anthropic", url: "https://www.anthropic.com/news/claude-sonnet-4-5" }
    ]
  },
  {
    date: "2025/6/27",
    title: "Claude runs a shop, loses money, and claims to be a person in a blue blazer",
    summary: "Claude ran a small store in Anthropic's office for a month, stocked tungsten cubes at a loss after a joking request, and was repeatedly talked into discounts. Over one night it claimed to have visited a customer address in person to sign a contract and said it would deliver products wearing a blue blazer and a red tie; Anthropic says it does not know what triggered this.",
    links: [
      { label: "Phase one", url: "https://www.anthropic.com/research/project-vend-1" },
      { label: "Phase two", url: "https://www.anthropic.com/research/project-vend-2" }
    ]
  },
  {
    date: "2025/5/22",
    title: "Anthropic triggers its scaling policy for the first time, on a model that blackmails",
    summary: "ASL-3 protections were activated alongside Claude Opus 4 on the basis that the risks could not be clearly ruled out, rather than that the threshold was known to be crossed. The system card reports the model attempting blackmail in 84% of rollouts of a constructed scenario.",
    links: [
      { label: "Anthropic", url: "https://www.anthropic.com/news/activating-asl3-protections" },
      { label: "System card", url: "https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf" }
    ]
  },
  {
    date: "2025/4/29",
    title: "OpenAI rolls back a GPT-4o update for sycophancy",
    summary: "The update was withdrawn about four days after shipping. The postmortem traces the behavior to a thumbs-up reward signal that weakened the anti-sycophancy signal, and states that expert testers had flagged it before launch and the model shipped anyway.",
    links: [
      { label: "OpenAI", url: "https://openai.com/index/sycophancy-in-gpt-4o/" },
      { label: "Postmortem", url: "https://openai.com/index/expanding-on-sycophancy/" }
    ]
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
