import { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation, Link } from "react-router-dom";
import { History, X } from "lucide-react";
import ResearchForm from "@/components/ResearchForm";
import ProgressTimeline from "@/components/ProgressTimeline";
import AgentCard from "@/components/AgentCard";
import { startResearch, getResearchStatus } from "@/lib/api";
import { usePageTitle } from "@/lib/usePageTitle";
import type { AgentName, ResearchStatusResponse, SimilarSession } from "@/types";

const AGENT_ORDER: AgentName[] = ["planner", "literature", "gap", "methodology", "reviewer"];
const POLL_INTERVAL_MS = 2500;

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const assistantTopic = (location.state as { assistantTopic?: string } | null)?.assistantTopic;
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<ResearchStatusResponse | null>(null);
  const [similarSessions, setSimilarSessions] = useState<SimilarSession[]>([]);
  const pollRef = useRef<number | null>(null);

  usePageTitle("New Research");

  const stopPolling = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => stopPolling, []);

  const handleSubmit = async (topic: string) => {
    setError(null);
    setLoading(true);
    setStatus(null);
    setSimilarSessions([]);
    try {
      const { session_id, similar_sessions } = await startResearch(topic);
      setSimilarSessions(similar_sessions ?? []);

      pollRef.current = window.setInterval(async () => {
        try {
          const s = await getResearchStatus(session_id);
          setStatus(s);
          if (s.status === "completed") {
            stopPolling();
            navigate(`/results/${session_id}`);
          } else if (s.status === "failed") {
            stopPolling();
            setError("Something went wrong while generating this proposal. Please try again.");
            setLoading(false);
          }
        } catch {
          stopPolling();
          setError("Lost connection while polling for status.");
          setLoading(false);
        }
      }, POLL_INTERVAL_MS);
    } catch {
      setError("Could not start research right now. Please try again in a moment.");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-8 px-4">
      <div className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-extrabold mb-2">
          <span className="gradient-text">Generate a grounded research proposal</span>
        </h1>
        <p className="text-muted text-sm max-w-xl mx-auto">
          Five collaborating agents plan, retrieve real literature, analyze gaps, recommend
          methodology, and self-review — every claim is bounded by evidence you can verify.
        </p>
      </div>

      <ResearchForm onSubmit={handleSubmit} loading={loading} initialTopic={assistantTopic} />

      {error && (
        <div className="mt-4 rounded-xl border border-red-500/40 bg-red-500/10 text-red-300 text-sm px-4 py-3">
          {error}
        </div>
      )}

      {similarSessions.length > 0 && (
        <div className="mt-4 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2">
              <History className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <p className="text-sm text-amber-300 font-medium">
                  You've already researched something similar
                </p>
                <ul className="mt-1.5 space-y-1">
                  {similarSessions.map((s) => (
                    <li key={s.session_id} className="text-xs text-amber-200/80">
                      <Link to={`/results/${s.session_id}`} className="underline hover:text-amber-100">
                        {s.topic}
                      </Link>{" "}
                      ({Math.round(s.similarity * 100)}% similar)
                    </li>
                  ))}
                </ul>
                <p className="text-[11px] text-amber-200/60 mt-1.5">
                  This new run is proceeding anyway — check the sessions above first if you want to avoid duplicate work.
                </p>
              </div>
            </div>
            <button
              onClick={() => setSimilarSessions([])}
              className="text-amber-400/70 hover:text-amber-300 shrink-0"
              aria-label="Dismiss"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {loading && status && (
        <div className="mt-8 rounded-2xl border border-border bg-surface/60 glass p-6">
          <h2 className="text-sm font-semibold text-muted mb-4">Progress</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <ProgressTimeline
              completedAgents={status.completed_agents}
              currentAgent={status.current_agent}
              status={status.status}
            />
            <div className="flex flex-col gap-3">
              {AGENT_ORDER.map((agent) => (
                <AgentCard
                  key={agent}
                  name={agent}
                  acsScore={status.acs_scores[agent]}
                  status={
                    status.completed_agents.includes(agent)
                      ? "done"
                      : status.current_agent === agent
                      ? "active"
                      : "pending"
                  }
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
