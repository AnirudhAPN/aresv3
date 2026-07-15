import type { AgentName } from "@/types";
import { cn } from "@/lib/utils";

const AGENTS: AgentName[] = ["planner", "literature", "gap", "methodology", "reviewer"];

interface CollaborationGraphProps {
  acsScores: Record<string, number>;
  retryCounts: Record<string, number>;
}

export default function CollaborationGraph({ acsScores, retryCounts }: CollaborationGraphProps) {
  return (
    <div className="rounded-2xl border border-border bg-surface/60 glass p-5">
      <h2 className="text-sm font-semibold text-muted mb-4">Collaboration Graph</h2>
      <div className="flex items-center overflow-x-auto pb-2">
        {AGENTS.map((agent, idx) => {
          const score = acsScores[agent];
          const retries = retryCounts[agent] ?? 0;
          const color = score === undefined ? "#242c47" : score >= 0.6 ? "#7c5cff" : "#f97066";
          return (
            <div key={agent} className="flex items-center shrink-0">
              <div className="flex flex-col items-center gap-1.5 w-24">
                <div
                  className="h-12 w-12 rounded-full border-2 flex items-center justify-center text-[10px] font-bold"
                  style={{ borderColor: color, color, boxShadow: score !== undefined ? `0 0 12px ${color}55` : undefined }}
                >
                  {score !== undefined ? Math.round(score * 100) : "—"}
                </div>
                <span className="text-[11px] text-muted capitalize text-center leading-tight">{agent}</span>
                {retries > 0 && (
                  <span className="text-[9px] text-amber-400">retry ×{retries}</span>
                )}
              </div>
              {idx < AGENTS.length - 1 && (
                <div
                  className={cn(
                    "h-0.5 w-8 shrink-0",
                    score !== undefined ? "bg-primary-500/50" : "bg-border"
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
      <p className="text-[11px] text-muted mt-3">
        Nodes below the 0.60 ACS threshold trigger one automatic retry with an augmented prompt before the
        pipeline proceeds and flags the section as low confidence.
      </p>
    </div>
  );
}
