export type AgentEvaluation =
  | { type: 'cp'; value: number }
  | { type: 'mate'; value: number };

export type AgentRecommendation = {
  uci: string;
  san?: string;
  averageRating?: number;
  white?: number;
  draws?: number;
  black?: number;
  opening?: {
    eco?: string;
    name?: string;
  };
};

export type AgentContextItem = {
  id?: string;
  score?: number;
  text: string;
};

export type AgentVideo = {
  title: string;
  channel: string;
  url: string;
  thumbnail: string;
};

export type AgentResponse = {
  fen: string;
  source?: string;
  recommendations?: AgentRecommendation[];
  evaluation?: AgentEvaluation | null;
  context?: AgentContextItem[];
  videos?: AgentVideo[];
};
