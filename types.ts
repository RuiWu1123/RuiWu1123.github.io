
export interface ResearchInterest {
  title: string;
  description: string;
  period?: string;
  colorTheme?: string;
}

export interface BlogPost {
  id: string;
  title: string;
  date: string;
  generator?: string;
  content?: string; // Optional: loaded from file
}

export interface Publication {
  id: string;
  title: string;
  authors: string[];
  venue?: string;
  year: number;
  link?: string;
  github?: string;
  tags?: string[];
  preprint?: boolean;
}

export interface NewsItem {
  date: string;
  content: string;
  link?: string;
}

export interface AlignmentLink {
  label: string;       // who published it
  url: string;
}

export interface AlignmentNewsItem {
  date: string;        // YYYY/M/D
  title: string;
  summary: string;
  links: AlignmentLink[];
}

