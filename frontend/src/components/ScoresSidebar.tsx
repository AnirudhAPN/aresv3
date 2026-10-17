import { AlertTriangle, RotateCcw, Gauge, Link2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentName, CoherenceReport } from "@/types";

const AGENTS: { key: AgentName; label: string }[] = [
  { key: "planner", label: "Planner" },
  { key: "literature", label: "Literature" },
  { key: "gap", label: "Gap Analysis" },
  { key: "methodology", label: "Methodology" },
  { key: "reviewer", label: "Reviewer" },
];

interface ScoreComponents {
  confidence: number;
  evidence_quality: number;
  task_completion: number;
}

interface ScoresSidebarProps {
  acsScores: Record<string, number>;
  acsComponents: Record<string, ScoreComponents>;
  retryCounts: Record<string, number>;
  parseFailures: Record<string, boolean>;
  novelty?: number;
  quality?: number;
  coherence?: CoherenceReport | null;
  className?: string;
}

function scoreColor(score: number): string {
  if (score >= 0.75) return "#3ec6ff";
  if (score >= 0.6) return "#7c5cff";
  return "#f97066";
}

function MiniBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const color = scoreColor(value);
  return (
    <div className="mb-1.5 last:mb-0">
      <div className="flex items-center justify-between mb-0.5">
        <span className="text-[10px] text-muted">{label}</span>
        <span className="text-[10px] font-medium" style={{ color }}>
          {pct}%
        </span>
      </div>
      <div className="h-1 w-full rounded-full bg-surface2 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function AgentScoreRow({
  label,
  acs,
  components,
  retries,
  parseFailed,
}: {
  label: string;
  acs?: number;
  components?: ScoreComponents;
  retries?: number;
  parseFailed?: boolean;
}) {
  const color = acs !== undefined ? scoreColor(acs) : "#3a4270";

  return (
    <div className="rounded-lg border border-border bg-surface2/40 px-3 py-2.5">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-semibold">{label}</span>
        <div className="flex items-center gap-1.5">
          {!!retries && (
            <span title={`Retried ${retries}x`} className="text-amber-400">
              <RotateCcw className="h-3 w-3" />
            </span>
          )}
          {parseFailed && (
            <span title="Parse failed — fallback used" className="text-red-400">
              <AlertTriangle className="h-3 w-3" />
            </span>
          )}
          <span
            className="text-xs font-bold rounded-full h-6 w-6 flex items-center justify-center border"
            style={{ borderColor: color, color }}
          >
            {acs !== undefined ? Math.round(acs * 100) : "—"}
          </span>
        </div>
      </div>
      {components ? (
        <div>
          <MiniBar label="Confidence" value={components.confidence} />
          <MiniBar label="Evidence" value={components.evidence_quality} />
          <MiniBar label="Completion" value={components.task_completion} />
        </div>
      ) : (
        <p className="text-[10px] text-muted">Not yet scored</p>
      )}
    </div>
  );
}

function CoherenceCard({ coherence }: { coherence: CoherenceReport }) {
  const color = scoreColor(coherence.overall_score);
  return (
    <div className="rounded-xl border border-border bg-surface/70 glass p-4">
      <div className="flex items-center gap-2 mb-1">
        <Link2 className="h-4 w-4 text-primary-400" />
        <h3 className="text-sm font-semibold">Coherence</h3>
        <span className="text-sm font-bold ml-auto" style={{ color }}>
          {Math.round(coherence.overall_score * 100)}%
        </span>
      </div>
      <p className="text-[10px] text-muted mb-2.5 leading-relaxed">
        Does the proposal hang together as one plan — not just five agents that each look
        plausible on their own?
      </p>
      <MiniBar label="Gaps addressed by methodology" value={coherence.gap_coverage_score} />
      <MiniBar label="Gaps trace back to research questions" value={coherence.question_alignment_score} />
      {coherence.uncovered_gaps.length > 0 && (
        <div className="mt-2.5 pt-2.5 border-t border-border">
          <div className="flex items-center gap-1.5 mb-1">
            <AlertTriangle className="h-3 w-3 text-amber-400" />
            <span className="text-[10px] font-semibold text-amber-400">
              {coherence.uncovered_gaps.length} gap{coherence.uncovered_gaps.length > 1 ? "s" : ""} not
              addressed by the methodology
            </span>
          </div>
          <ul className="space-y-1">
            {coherence.uncovered_gaps.map((g, i) => (
              <li key={i} className="text-[10px] text-muted leading-snug pl-1 border-l border-border">
                {g}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function ScoresSidebar({
  acsScores,
  acsComponents,
  retryCounts,
  parseFailures,
  novelty,
  quality,
  coherence,
  className,
}: ScoresSidebarProps) {
  const scoreValues = Object.values(acsScores);
  const overall = scoreValues.length ? scoreValues.reduce((a, b) => a + b, 0) / scoreValues.length : undefined;

  return (
    <aside className={cn("w-full lg:w-72 shrink-0", className)}>
      <div className="lg:sticky lg:top-20 space-y-3">
        <div className="rounded-xl border border-border bg-surface/70 glass p-4">
          <div className="flex items-center gap-2 mb-3">
            <Gauge className="h-4 w-4 text-primary-400" />
            <h3 className="text-sm font-semibold">Scores</h3>
          </div>

          {overall !== undefined && (
            <div className="mb-3 pb-3 border-b border-border">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted">Overall ACS</span>
                <span className="text-sm font-bold" style={{ color: scoreColor(overall) }}>
                  {Math.round(overall * 100)}%
                </span>
              </div>
              {(novelty !== undefined || quality !== undefined) && (
                <div className="flex gap-4 mt-2">
                  {novelty !== undefined && (
                    <div className="flex-1">
                      <span className="text-[10px] text-muted block">Novelty</span>
                      <span className="text-xs font-semibold" style={{ color: scoreColor(novelty) }}>
                        {Math.round(novelty * 100)}%
                      </span>
                    </div>
                  )}
                  {quality !== undefined && (
                    <div className="flex-1">
                      <span className="text-[10px] text-muted block">Quality</span>
                      <span className="text-xs font-semibold" style={{ color: scoreColor(quality) }}>
                        {Math.round(quality * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <div className="space-y-2">
            {AGENTS.map(({ key, label }) => (
              <AgentScoreRow
                key={key}
                label={label}
                acs={acsScores[key]}
                components={acsComponents[key]}
                retries={retryCounts[key]}
                parseFailed={parseFailures[key]}
              />
            ))}
          </div>

          <p className="text-[10px] text-muted mt-3 pt-3 border-t border-border leading-relaxed">
            ACS = 0.40 × confidence + 0.35 × evidence quality + 0.25 × task completion. Scores below 60%
            trigger one automatic retry.
          </p>
        </div>

        {coherence && <CoherenceCard coherence={coherence} />}
      </div>
    </aside>
  );
}
