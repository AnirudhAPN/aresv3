import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Clock, ArrowRight, Loader2, FlaskConical } from "lucide-react";
import { getHistory } from "@/lib/api";
import { usePageTitle } from "@/lib/usePageTitle";
import type { HistoryItem } from "@/types";

function statusTone(status: string): string {
  if (status === "completed") return "text-emerald-400 border-emerald-500/30 bg-emerald-500/10";
  if (status === "failed") return "text-red-400 border-red-500/30 bg-red-500/10";
  return "text-accent-500 border-accent-500/30 bg-accent-500/10";
}

export default function History() {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  usePageTitle("History");

  useEffect(() => {
    getHistory()
      .then(setItems)
      .catch(() => setError("Could not load history."));
  }, []);

  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Research History</h1>
          <p className="text-sm text-muted mt-1">All your past research sessions, newest first.</p>
        </div>
        <Link
          to="/"
          className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg bg-gradient-to-r from-primary-500 to-accent-500 font-semibold"
        >
          <FlaskConical className="h-3.5 w-3.5" /> New Research
        </Link>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {!items && !error && (
        <div className="flex items-center justify-center py-16 text-muted gap-2">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading history…
        </div>
      )}

      {items && items.length === 0 && (
        <div className="text-center py-16 text-muted text-sm">
          No research sessions yet. Start your first one from the New Research page.
        </div>
      )}

      <div className="space-y-2">
        {items?.map((item) => (
          <Link
            key={item.session_id}
            to={item.status === "completed" ? `/results/${item.session_id}` : "#"}
            className="flex items-center justify-between rounded-xl border border-border bg-surface/60 glass px-4 py-3.5 hover:border-primary-500/50 transition group"
          >
            <div>
              <p className="text-sm font-medium group-hover:text-primary-400 transition">{item.topic}</p>
              <p className="text-xs text-muted flex items-center gap-1.5 mt-1">
                <Clock className="h-3 w-3" />
                {new Date(item.created_at * 1000).toLocaleString()}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className={`text-[10px] px-2 py-1 rounded-full border font-medium ${statusTone(item.status)}`}>
                {item.status}
              </span>
              <ArrowRight className="h-4 w-4 text-muted group-hover:text-primary-400 transition" />
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
