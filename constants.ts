
import { NavItem, ResearchInterest, VisitedPlace, Publication, NewsItem, BlogPost } from './types';

// Function to load blog content from file
export const loadBlogContent = async (blogId: string): Promise<string> => {
  try {
    const response = await fetch(`/blogs/${blogId}.md`);
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
  { label: 'Blogs', path: '/blog' },
  { label: 'Publications', path: '/publications' },
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
    date: "2026/1/1",
    content: "Happy New Year! How is AI going to be in 2026?",
    link: "#/blog/a-supervision-gaming"
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

export const BLOG_POSTS: BlogPost[] = [
  {
    id: "a-supervision-gaming",
    title: "A Supervision Gaming We May Fail: From LLMs Evaluation",
    date: "2025/12/31",
    generator: "ChatGPT5.1"
  },
  {
    id: "safety-align-as-world-reasoner",
    title: "Toward Generalized Safety Alignment: Your LLMs should align as World Reasoners",
    date: "2025/12/28",
    generator: "ChatGPT5.1"
  },
  {
    id: "my-research-river",
    title: "Why these Research Topics? Sharing my long-term vision",
    date: "2025/12/15",
    generator: "ChatGPT5.1"
  },
  {
    id: "alignment-science-safety",
    title: "Alignment Science and Safety Alignment: A Perspective",
    date: "2025/12/01",
    generator: "ChatGPT5.1"
  }
];

export const SOCIAL_LINKS = {
  scholar: "https://scholar.google.com/citations?user=M1FovLwAAAAJ&hl=en"
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
