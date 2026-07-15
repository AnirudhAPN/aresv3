import { cn } from "@/lib/utils";

interface ConfidenceMeterProps {
  label: string;
  score: number; // 0-1
  size?: "sm" | "md";
}

function scoreColor(score: number): string {
  if (score >= 0.75) return "#3ec6ff";
  if (score >= 0.6) return "#7c5cff";
  return "#f97066";
}

export default function ConfidenceMeter({ label, score, size = "md" }: ConfidenceMeterProps) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  const color = scoreColor(score);
  const height = size === "sm" ? "h-1.5" : "h-2.5";

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-muted">{label}</span>
        <span className="text-xs font-semibold" style={{ color }}>
          {pct}%
        </span>
      </div>
      <div className={cn("w-full rounded-full bg-surface2 overflow-hidden", height)}>
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}99, ${color})` }}
        />
      </div>
      {score < 0.6 && (
        <p className="text-[10px] text-amber-400 mt-1">Below retry threshold (0.60) — flagged low confidence</p>
      )}
    </div>
  );
}
