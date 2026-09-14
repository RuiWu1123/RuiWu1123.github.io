
export interface NavItem {
  label: string;
  path: string;
}

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

export interface AlignmentNewsItem {
  date: string;        // YYYY/M/D
  title: string;
  org: string;         // who published it
  summary: string;     // one sentence, factual
  link: string;
}

export interface VisitedPlace {
  id: string;
  name: string;
  coordinates: [number, number]; // [longitude, latitude]
  date: string;
  description: string;
  images: string[]; // Array of image URLs
  visitCount: number;
}
