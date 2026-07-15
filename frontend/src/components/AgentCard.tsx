import { cn } from "@/lib/utils";
import ConfidenceMeter from "./ConfidenceMeter";

interface AgentCardProps {
  name: string;
  acsScore?: number;
  retries?: number;
  parseFailed?: boolean;
  status: "done" | "active" | "pending";
}

export default function AgentCard({ name, acsScore, retries = 0, parseFailed, status }: AgentCardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border p-4 bg-surface/60 glass transition-all",
        status === "active" && "border-accent-500/60 shadow-glow",
        status === "done" && "border-border",
        status === "pending" && "border-border opacity-50"
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold capitalize text-sm">{name.replace("_", " ")}</h3>
        {retries > 0 && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30">
            retried ×{retries}
          </span>
        )}
        {parseFailed && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/15 text-red-400 border border-red-500/30">
            parse failed
          </span>
        )}
      </div>
      {acsScore !== undefined ? (
        <ConfidenceMeter label="ACS" score={acsScore} size="sm" />
      ) : (
        <p className="text-xs text-muted">Awaiting execution…</p>
      )}
    </div>
  );
}
