import { useEffect, useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import {
  BookOpen,
  Target,
  Lightbulb,
  Wrench,
  ShieldCheck,
  Download,
  AlertTriangle,
  ExternalLink,
  Loader2,
  Link2,
  Link2Off,
  ChevronLeft,
  X,
} from "lucide-react";
import { getResearchReport, exportUrl } from "@/lib/api";
import { usePageTitle } from "@/lib/usePageTitle";
import type { ResearchSession } from "@/types";
import OutputSection from "@/components/OutputSection";
import CollaborationGraph from "@/components/CollaborationGraph";
import ScoresSidebar from "@/components/ScoresSidebar";
import ChatPanel from "@/components/ChatPanel";
import CitationBlock from "@/components/CitationBlock";

function LinkStatusBadge({ verified }: { verified: boolean | null }) {
  if (verified === null) return null;
  return verified ? (
    <span title="Link verified reachable" className="text-emerald-400 shrink-0">
      <Link2 className="h-3.5 w-3.5" />
    </span>
  ) : (
    <span title="Link could not be verified — may be dead" className="text-red-400 shrink-0">
      <Link2Off className="h-3.5 w-3.5" />
    </span>
  );
}

function Badge({ children, tone = "muted" }: { children: React.ReactNode; tone?: "muted" | "warn" | "ok" | "bad" }) {
  const toneClasses: Record<string, string> = {
    muted: "bg-surface2 text-muted border-border",
    warn: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    ok: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    bad: "bg-red-500/15 text-red-400 border-red-500/30",
  };
  return (
    <span className={`text-[10px] px-2 py-1 rounded-full border font-medium ${toneClasses[tone]}`}>{children}</span>
  );
}

/** Fetch a binary/text export and trigger a browser download. */
async function downloadExport(
  sessionId: string,
  format: "markdown" | "text" | "pdf" | "docx",
): Promise<string | null> {
  const res = await fetch(exportUrl(sessionId, format));
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    return (body as { detail?: string }).detail ?? `Export failed (HTTP ${res.status})`;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const ext: Record<string, string> = {
    markdown: "md",
    text: "txt",
    pdf: "pdf",
    docx: "docx",
  };
  a.href = url;
  a.download = `ares-report.${ext[format]}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  return null;
}

export default function Results() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [session, setSession] = useState<ResearchSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  usePageTitle(session?.topic ?? "Research Proposal");

  useEffect(() => {
    if (!sessionId) return;
    getResearchReport(sessionId)
      .then(setSession)
      .catch(() => setError("Could not load this research session."));
  }, [sessionId]);

  const handleDownload = useCallback(
    async (format: "markdown" | "text" | "pdf" | "docx") => {
      if (!sessionId || downloading) return;
      setDownloadError(null);
      setDownloading(format);
      try {
        const err = await downloadExport(sessionId, format);
        if (err) setDownloadError(err);
      } catch {
        setDownloadError("Download failed — is the server reachable?");
      } finally {
        setDownloading(null);
      }
    },
    [sessionId, downloading],
  );

  if (error) {
    return <div className="max-w-3xl mx-auto py-10 px-4 text-red-400 text-sm">{error}</div>;
  }
  if (!session) {
    return (
      <div className="flex items-center justify-center py-24 text-muted gap-2">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading report…
      </div>
    );
  }

  const planner = session.agent_outputs.planner?.output;
  const literature = session.agent_outputs.literature?.output;
  const gap = session.agent_outputs.gap?.output;
  const methodology = session.agent_outputs.methodology?.output;
  const reviewer = session.agent_outputs.reviewer?.output;

  return (
    <div className="max-w-6xl mx-auto py-8 px-4">
      <Link
        to="/history"
        className="inline-flex items-center gap-1 text-xs text-muted hover:text-white transition mb-4"
      >
        <ChevronLeft className="h-3.5 w-3.5" /> Back to History
      </Link>

      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
        <div>
          <p className="text-xs text-muted uppercase tracking-wide mb-1">Research Proposal</p>
          <h1 className="text-2xl md:text-3xl font-bold">{session.topic}</h1>
          <div className="flex flex-wrap gap-2 mt-2">
            {session.flags.map((flag) => (
              <Badge key={flag} tone="warn">
                <AlertTriangle className="inline h-3 w-3 mr-1 -mt-0.5" />
                {flag.replace(/_/g, " ")}
              </Badge>
            ))}
          </div>
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0">
          <div className="flex gap-2">
            {(["markdown", "text", "pdf", "docx"] as const).map((fmt) => {
              const labels: Record<string, string> = {
                markdown: "Markdown",
                text: "Text",
                pdf: "PDF",
                docx: "Word",
              };
              const isActive = downloading === fmt;
              return (
                <button
                  key={fmt}
                  onClick={() => handleDownload(fmt)}
                  disabled={!!downloading}
                  title={`Download as ${labels[fmt]}`}
                  className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg border border-border bg-surface2 hover:border-primary-500/50 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isActive ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Download className="h-3.5 w-3.5" />
                  )}
                  {labels[fmt]}
                </button>
              );
            })}
          </div>
          {downloadError && (
            <div className="flex items-center gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/30 px-3 py-1.5 rounded-lg max-w-xs">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              <span className="flex-1">{downloadError}</span>
              <button onClick={() => setDownloadError(null)} className="shrink-0 hover:text-red-200">
                <X className="h-3 w-3" />
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <ScoresSidebar
          acsScores={session.acs_scores}
          acsComponents={session.acs_components}
          retryCounts={session.retry_counts}
          parseFailures={session.parse_failures}
          novelty={reviewer?.novelty_score}
          quality={reviewer?.quality_score}
          coherence={session.coherence}
          className="order-first lg:order-last"
        />

        <div className="flex-1 min-w-0 space-y-6">
          <CollaborationGraph acsScores={session.acs_scores} retryCounts={session.retry_counts} />

          {planner && (
            <OutputSection title="Research Objectives & Questions" icon={<Target className="h-4 w-4 text-primary-400" />}>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Objectives</h4>
                  <ul className="space-y-1.5 text-sm">
                    {planner.research_objectives.map((o, i) => (
                      <li key={i} className="pl-3 border-l-2 border-primary-500/40">
                        {o}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Research Questions</h4>
                  <ul className="space-y-1.5 text-sm">
                    {planner.research_questions.map((q, i) => (
                      <li key={i} className="pl-3 border-l-2 border-accent-500/40">
                        {q}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <p className="text-sm text-muted mt-4 italic">{planner.research_scope}</p>
            </OutputSection>
          )}

          {literature && (
            <OutputSection
              title="Literature Review"
              icon={<BookOpen className="h-4 w-4 text-primary-400" />}
              badges={literature.low_coverage && <Badge tone="warn">Low coverage</Badge>}
            >
              {literature.similar_work && (
                <div className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
                  <p className="text-xs text-amber-300">
                    <AlertTriangle className="inline h-3 w-3 mr-1 -mt-0.5" />
                    Possible existing work: this topic is{" "}
                    <strong>{Math.round(literature.similar_work.similarity * 100)}%</strong> similar to{" "}
                    <a
                      href={literature.similar_work.paper_url}
                      target="_blank"
                      rel="noreferrer"
                      className="underline hover:text-amber-200"
                    >
                      {literature.similar_work.paper_title}
                    </a>
                    . Review it before proceeding to avoid duplicating existing research.
                  </p>
                </div>
              )}
              <div className="grid md:grid-cols-2 gap-3">
                {literature.papers.map((p, i) => (
                  <div
                    key={i}
                    className="rounded-xl border border-border bg-surface2/60 p-4 hover:border-primary-500/50 transition group"
                  >
                    <a href={p.url} target="_blank" rel="noreferrer" className="block">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <h4 className="text-sm font-semibold group-hover:text-primary-400 transition">{p.title}</h4>
                        <div className="flex items-center gap-1.5 shrink-0 mt-0.5">
                          <LinkStatusBadge verified={p.url_verified} />
                          <ExternalLink className="h-3.5 w-3.5 text-muted" />
                        </div>
                      </div>
                      <p className="text-[11px] text-muted mb-2">
                        {p.authors.slice(0, 3).join(", ")}
                        {p.authors.length > 3 ? " et al." : ""} · {p.year ?? "n.d."} · {p.source.replace("_", " ")}
                        {p.citation_count !== null && ` · ${p.citation_count} citations`}
                      </p>
                      <p className="text-xs text-muted leading-relaxed">{p.summary || p.abstract}</p>
                    </a>
                    <CitationBlock apa={p.citation_apa} bibtex={p.citation_bibtex} />
                  </div>
                ))}
              </div>
              {literature.papers.length === 0 && (
                <p className="text-sm text-muted">No papers could be retrieved for this topic.</p>
              )}
            </OutputSection>
          )}

          {gap && (
            <OutputSection
              title="Research Gap Analysis"
              icon={<Lightbulb className="h-4 w-4 text-primary-400" />}
              badges={gap.low_confidence_gaps && <Badge tone="warn">Low confidence</Badge>}
            >
              <div className="space-y-4">
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Current Trends</h4>
                  <ul className="list-disc list-inside text-sm space-y-1 text-white/90">
                    {gap.current_trends.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Gaps</h4>
                  <div className="space-y-2">
                    {gap.gaps.map((g, i) => (
                      <div key={i} className="rounded-lg border border-border bg-surface2/50 p-3">
                        <p className="text-sm">{g.gap_description}</p>
                        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11px] text-muted mt-1.5">
                          <span>Based on:</span>
                          {g.citations.length === 0 && <span>—</span>}
                          {g.citations.map((c, ci) => (
                            <span key={ci} className="inline-flex items-center gap-1">
                              {c.url ? (
                                <a
                                  href={c.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  title={c.matched ? "Open the source paper" : "Best-effort match — title may not exactly match a retrieved paper"}
                                  className="inline-flex items-center gap-1 rounded border border-border/70 px-1.5 py-0.5 hover:text-primary-400 hover:border-primary-400/50"
                                >
                                  {c.title}
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              ) : (
                                <span
                                  title="Could not confidently match this to a retrieved paper"
                                  className="rounded border border-border/70 px-1.5 py-0.5 text-white/70"
                                >
                                  {c.title}
                                </span>
                              )}
                              {ci < g.citations.length - 1 && <span>,</span>}
                            </span>
                          ))}
                          <span>· {g.confidence_note}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Future Directions</h4>
                  <ul className="list-disc list-inside text-sm space-y-1 text-white/90">
                    {gap.future_directions.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </OutputSection>
          )}

          {methodology && (
            <OutputSection title="Methodology" icon={<Wrench className="h-4 w-4 text-primary-400" />}>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Datasets</h4>
                  <div className="space-y-2">
                    {methodology.datasets.map((d, i) => {
                      const kaggleHref = d.kaggle_url ?? `https://www.kaggle.com/search?q=${encodeURIComponent(d.name)}`;
                      return (
                        <div key={i} className="rounded-lg border border-border bg-surface2/50 px-3 py-2">
                          <div className="flex items-center justify-between gap-2">
                            <a
                              href={kaggleHref}
                              target="_blank"
                              rel="noreferrer"
                              title="Open the corresponding dataset on Kaggle"
                              className="text-sm hover:text-primary-400 flex items-center gap-1.5 min-w-0"
                            >
                              <span className="truncate">{d.name}</span>
                              <LinkStatusBadge verified={d.url_verified} />
                            </a>
                            <Badge tone={d.verified ? "ok" : "warn"}>{d.verified ? "verified" : "unverified"}</Badge>
                          </div>
                          <div className="flex items-center justify-between mt-1">
                            <span className="text-[10px] text-muted">via Kaggle</span>
                            {d.url && (
                              <a href={d.url} target="_blank" rel="noreferrer" className="text-[10px] text-muted hover:text-primary-400 underline">
                                original source
                              </a>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2 mt-4">Algorithms</h4>
                  <div className="space-y-2">
                    {methodology.algorithms.map((a, i) => (
                      <div key={i} className="flex items-center justify-between rounded-lg border border-border bg-surface2/50 px-3 py-2">
                        <span className="text-sm flex items-center gap-1.5">
                          {a.url ? (
                            <a href={a.url} target="_blank" rel="noreferrer" className="hover:text-primary-400">
                              {a.name}
                            </a>
                          ) : (
                            a.name
                          )}
                          <LinkStatusBadge verified={a.url_verified} />
                        </span>
                        <Badge tone={a.verified ? "ok" : "warn"}>{a.verified ? "verified" : "unverified"}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Evaluation Metrics</h4>
                  <div className="flex flex-wrap gap-1.5 mb-4">
                    {methodology.evaluation_metrics.map((m, i) => (
                      <span key={i} className="text-xs px-2 py-1 rounded-full bg-surface2 border border-border">
                        {m}
                      </span>
                    ))}
                  </div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Implementation Steps</h4>
                  <ol className="list-decimal list-inside text-sm space-y-1 text-white/90">
                    {methodology.implementation_steps.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ol>
                  {methodology.unverified_suggestions.length > 0 && (
                    <>
                      <h4 className="text-xs font-semibold text-amber-400 uppercase mb-2 mt-4">
                        Unverified Suggestions
                      </h4>
                      <ul className="list-disc list-inside text-sm space-y-1 text-amber-200/80">
                        {methodology.unverified_suggestions.map((s, i) => (
                          <li key={i}>{s}</li>
                        ))}
                      </ul>
                    </>
                  )}
                </div>
              </div>
            </OutputSection>
          )}

          {reviewer && (
            <OutputSection title="Reviewer Feedback" icon={<ShieldCheck className="h-4 w-4 text-primary-400" />}>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-xs font-semibold text-muted uppercase mb-2">Rule Checks (deterministic)</h4>
                  <div className="space-y-1.5">
                    {Object.entries(reviewer.rule_check_results).map(([check, passed]) => (
                      <div key={check} className="flex items-center justify-between text-sm">
                        <span className="text-white/80">{check.replace(/_/g, " ")}</span>
                        <Badge tone={passed ? "ok" : "bad"}>{passed ? "pass" : "fail"}</Badge>
                      </div>
                    ))}
                  </div>
                  {reviewer.caveats.length > 0 && (
                    <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
                      <h5 className="text-xs font-semibold text-amber-400 mb-1">Caveats</h5>
                      <ul className="text-xs text-amber-200/80 space-y-1 list-disc list-inside">
                        {reviewer.caveats.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-xs font-semibold text-emerald-400 uppercase mb-2">Strengths</h4>
                    <ul className="list-disc list-inside text-sm space-y-1 text-white/90">
                      {reviewer.strengths.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-red-400 uppercase mb-2">Weaknesses</h4>
                    <ul className="list-disc list-inside text-sm space-y-1 text-white/90">
                      {reviewer.weaknesses.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                  {reviewer.missing_components.length > 0 && (
                    <div>
                      <h4 className="text-xs font-semibold text-muted uppercase mb-2">Missing Components</h4>
                      <ul className="list-disc list-inside text-sm space-y-1 text-white/90">
                        {reviewer.missing_components.map((m, i) => (
                          <li key={i}>{m}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </OutputSection>
          )}

          <ChatPanel sessionId={session.session_id} onSessionUpdated={setSession} />
        </div>
      </div>
    </div>
  );
}
