export interface PlannerOutput {
  research_objectives: string[];
  research_questions: string[];
  keywords: string[];
  research_scope: string;
}

export interface Paper {
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  url: string;
  citation_count: number | null;
  source: "semantic_scholar" | "arxiv";
  abstract: string;
  summary: string;
  verified: boolean;
  url_verified: boolean | null;
  citation_apa: string;
  citation_bibtex: string;
}

export interface SimilarWorkMatch {
  paper_title: string;
  paper_url: string;
  similarity: number;
}

export interface LiteratureOutput {
  papers: Paper[];
  low_coverage: boolean;
  keywords_used: string[];
  similar_work: SimilarWorkMatch | null;
}

export interface ResearchGap {
  gap_description: string;
  based_on_papers: string[];
  confidence_note: string;
}

export interface GapOutput {
  current_trends: string[];
  gaps: ResearchGap[];
  future_directions: string[];
  low_confidence_gaps: boolean;
}

export interface DatasetRec {
  name: string;
  url: string | null;
  description: string | null;
  verified: boolean;
  url_verified: boolean | null;
  kaggle_url: string | null;
}

export interface MethodologyOutput {
  datasets: DatasetRec[];
  algorithms: DatasetRec[];
  evaluation_metrics: string[];
  implementation_steps: string[];
  unverified_suggestions: string[];
}

export interface ReviewerOutput {
  rule_check_results: Record<string, boolean>;
  strengths: string[];
  weaknesses: string[];
  missing_components: string[];
  novelty_score: number;
  quality_score: number;
  caveats: string[];
}

export type AgentName = "planner" | "literature" | "gap" | "methodology" | "reviewer";

export interface AgentOutputEntry<T = unknown> {
  output: T | null;
  parse_failed: boolean;
}

export interface ResearchSession {
  session_id: string;
  user_id: string;
  topic: string;
  status: "running" | "completed" | "failed";
  current_agent: string | null;
  created_at: number;
  agent_outputs: {
    planner?: AgentOutputEntry<PlannerOutput>;
    literature?: AgentOutputEntry<LiteratureOutput>;
    gap?: AgentOutputEntry<GapOutput>;
    methodology?: AgentOutputEntry<MethodologyOutput>;
    reviewer?: AgentOutputEntry<ReviewerOutput>;
  };
  completed_agents: AgentName[];
  acs_scores: Record<string, number>;
  acs_components: Record<string, { confidence: number; evidence_quality: number; task_completion: number }>;
  retry_counts: Record<string, number>;
  parse_failures: Record<string, boolean>;
  flags: string[];
}

export interface SimilarSession {
  session_id: string;
  topic: string;
  similarity: number;
}

export interface ResearchStartResponse {
  session_id: string;
  status: string;
  similar_sessions: SimilarSession[];
}

export interface ResearchStatusResponse {
  session_id: string;
  status: "running" | "completed" | "failed";
  current_agent: string | null;
  completed_agents: AgentName[];
  acs_scores: Record<string, number>;
  acs_components: Record<string, { confidence: number; evidence_quality: number; task_completion: number }>;
}

export interface HistoryItem {
  session_id: string;
  topic: string;
  status: string;
  created_at: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  agent_updated: AgentName | null;
  created_at: number;
}

export interface ChatResponse {
  reply: string;
  agent_updated: AgentName | null;
  session: ResearchSession;
}

export interface AssistantMessage {
  role: "user" | "assistant";
  content: string;
  suggested_topic: string | null;
  created_at: number;
}

export interface AssistantChatResponse {
  reply: string;
  suggested_topic: string | null;
}
