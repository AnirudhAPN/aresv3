import { useState } from "react";
import { Copy, Check } from "lucide-react";

interface CitationBlockProps {
  apa: string;
  bibtex: string;
}

export default function CitationBlock({ apa, bibtex }: CitationBlockProps) {
  const [copied, setCopied] = useState<"apa" | "bibtex" | null>(null);

  const copy = async (e: React.MouseEvent, text: string, which: "apa" | "bibtex") => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      window.setTimeout(() => setCopied(null), 1500);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) — silently no-op.
    }
  };

  if (!apa && !bibtex) return null;

  return (
    <div className="mt-2 pt-2 border-t border-border/60 space-y-1">
      {apa && (
        <div className="flex items-start justify-between gap-2">
          <p className="text-[10px] text-muted leading-relaxed flex-1">{apa}</p>
          <button
            onClick={(e) => copy(e, apa, "apa")}
            className="shrink-0 text-[10px] flex items-center gap-1 px-1.5 py-0.5 rounded border border-border text-muted hover:text-white hover:border-primary-500/50 transition"
            title="Copy APA citation"
          >
            {copied === "apa" ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
            APA
          </button>
        </div>
      )}
      {bibtex && (
        <div className="flex items-center justify-end gap-2">
          <button
            onClick={(e) => copy(e, bibtex, "bibtex")}
            className="shrink-0 text-[10px] flex items-center gap-1 px-1.5 py-0.5 rounded border border-border text-muted hover:text-white hover:border-primary-500/50 transition"
            title="Copy BibTeX citation"
          >
            {copied === "bibtex" ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
            BibTeX
          </button>
        </div>
      )}
    </div>
  );
}
