import { CheckCircle2, CircleDashed, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentName } from "@/types";

const AGENTS: { key: AgentName; label: string }[] = [
  { key: "planner", label: "Planner" },
  { key: "literature", label: "Literature" },
  { key: "gap", label: "Gap Analysis" },
  { key: "methodology", label: "Methodology" },
  { key: "reviewer", label: "Reviewer" },
];

interface ProgressTimelineProps {
  completedAgents: AgentName[];
  currentAgent: string | null;
  status: "running" | "completed" | "failed";
}

export default function ProgressTimeline({ completedAgents, currentAgent, status }: ProgressTimelineProps) {
  return (
    <div className="flex flex-col gap-0">
      {AGENTS.map((agent, idx) => {
        const isDone = completedAgents.includes(agent.key);
        const isActive = !isDone && currentAgent === agent.key && status === "running";
        const isPending = !isDone && !isActive;

        return (
          <div key={agent.key} className="flex gap-3">
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  "h-8 w-8 rounded-full flex items-center justify-center border-2 shrink-0",
                  isDone && "bg-primary-500/20 border-primary-500 text-primary-400",
                  isActive && "bg-accent-500/20 border-accent-500 text-accent-500 animate-pulse",
                  isPending && "bg-surface2 border-border text-muted"
                )}
              >
                {isDone ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : isActive ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <CircleDashed className="h-4 w-4" />
                )}
              </div>
              {idx < AGENTS.length - 1 && (
                <div className={cn("w-0.5 flex-1 min-h-[24px]", isDone ? "bg-primary-500/50" : "bg-border")} />
              )}
            </div>
            <div className="pb-6">
              <p
                className={cn(
                  "text-sm font-medium",
                  isDone && "text-white",
                  isActive && "text-accent-500",
                  isPending && "text-muted"
                )}
              >
                {agent.label}
              </p>
              <p className="text-xs text-muted">
                {isDone ? "Completed" : isActive ? "Running…" : "Pending"}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
