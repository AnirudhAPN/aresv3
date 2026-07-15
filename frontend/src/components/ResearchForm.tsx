import { useEffect, useState } from "react";
import { Loader2, Sparkles } from "lucide-react";

interface ResearchFormProps {
  onSubmit: (topic: string) => void;
  loading?: boolean;
  initialTopic?: string;
}

const EXAMPLE_TOPICS = [
  "AI in Healthcare",
  "Federated Learning for IoT Security",
  "Climate Modeling with Deep Learning",
  "Explainable AI in Finance",
];

export default function ResearchForm({ onSubmit, loading, initialTopic }: ResearchFormProps) {
  const [topic, setTopic] = useState(initialTopic ?? "");

  useEffect(() => {
    if (initialTopic) setTopic(initialTopic);
  }, [initialTopic]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (topic.trim() && !loading) {
      onSubmit(topic.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="rounded-2xl border border-border bg-surface/70 glass p-6 shadow-glow">
        <label className="block text-sm font-medium text-muted mb-2">Research Topic</label>
        <textarea
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. Adversarial robustness in medical imaging models"
          rows={3}
          className="w-full resize-none rounded-xl bg-surface2 border border-border px-4 py-3 text-sm placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500 transition"
        />

        <div className="flex flex-wrap gap-2 mt-3">
          {EXAMPLE_TOPICS.map((t) => (
            <button
              type="button"
              key={t}
              onClick={() => setTopic(t)}
              className="text-xs px-3 py-1.5 rounded-full bg-surface2 border border-border text-muted hover:text-white hover:border-primary-500/50 transition"
            >
              {t}
            </button>
          ))}
        </div>

        <button
          type="submit"
          disabled={!topic.trim() || loading}
          className="mt-5 w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-primary-500 to-accent-500 py-3 text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 transition"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Generating proposal…
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" /> Generate Research Proposal
            </>
          )}
        </button>
      </div>
    </form>
  );
}
