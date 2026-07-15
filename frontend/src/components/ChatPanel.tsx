import { useEffect, useRef, useState } from "react";
import { Send, Loader2, MessageSquare, Sparkles, Bot, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { getChatHistory, sendChatMessage } from "@/lib/api";
import type { ChatMessage, ResearchSession } from "@/types";

interface ChatPanelProps {
  sessionId: string;
  onSessionUpdated: (session: ResearchSession) => void;
}

const SUGGESTIONS = [
  "Add more datasets to the methodology",
  "Focus the gap analysis on privacy risks",
  "Narrow the research questions",
  "What does the literature review cover?",
];

export default function ChatPanel({ sessionId, onSessionUpdated }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getChatHistory(sessionId)
      .then(setMessages)
      .catch(() => setError("Could not load chat history."))
      .finally(() => setLoadingHistory(false));
  }, [sessionId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const submit = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || sending) return;
    setError(null);
    setInput("");
    setSending(true);

    // Optimistically show the user's message immediately.
    setMessages((prev) => [...prev, { role: "user", content: trimmed, agent_updated: null, created_at: Date.now() / 1000 }]);

    try {
      const res = await sendChatMessage(sessionId, trimmed);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.reply, agent_updated: res.agent_updated, created_at: Date.now() / 1000 },
      ]);
      onSessionUpdated(res.session);
    } catch {
      setError("Couldn't send that message. Please try again.");
    } finally {
      setSending(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submit(input);
  };

  return (
    <div className="rounded-2xl border border-border bg-surface/60 glass flex flex-col h-[520px]">
      <div className="flex items-center gap-2 px-5 py-4 border-b border-border">
        <MessageSquare className="h-4 w-4 text-primary-400" />
        <h2 className="text-sm font-semibold">Keep refining this proposal</h2>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {loadingHistory ? (
          <div className="flex items-center justify-center h-full text-muted gap-2 text-sm">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading conversation…
          </div>
        ) : messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center gap-3 px-4">
            <Sparkles className="h-5 w-5 text-primary-400" />
            <p className="text-sm text-muted max-w-xs">
              Ask a question about this proposal, or ask for a change — e.g. "add more datasets" or
              "make the gap analysis focus on privacy."
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => submit(s)}
                  className="text-xs px-3 py-1.5 rounded-full bg-surface2 border border-border text-muted hover:text-white hover:border-primary-500/50 transition"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={cn("flex gap-2.5", m.role === "user" && "flex-row-reverse")}>
              <div
                className={cn(
                  "h-7 w-7 rounded-full flex items-center justify-center shrink-0 mt-0.5",
                  m.role === "user" ? "bg-primary-500/20 text-primary-400" : "bg-surface2 text-accent-500"
                )}
              >
                {m.role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
              </div>
              <div
                className={cn(
                  "max-w-[80%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed",
                  m.role === "user" ? "bg-primary-500/15 text-white" : "bg-surface2 text-white/90"
                )}
              >
                {m.content}
                {m.agent_updated && (
                  <span className="block mt-1.5 text-[10px] text-accent-500 font-medium uppercase tracking-wide">
                    Updated: {m.agent_updated}
                  </span>
                )}
              </div>
            </div>
          ))
        )}
        {sending && (
          <div className="flex gap-2.5">
            <div className="h-7 w-7 rounded-full flex items-center justify-center shrink-0 bg-surface2 text-accent-500">
              <Bot className="h-3.5 w-3.5" />
            </div>
            <div className="rounded-xl px-3.5 py-2.5 bg-surface2 text-muted text-sm flex items-center gap-2">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Thinking…
            </div>
          </div>
        )}
      </div>

      {error && <p className="px-5 text-xs text-red-400 pb-1">{error}</p>}

      <form onSubmit={handleSubmit} className="flex items-center gap-2 px-4 py-3 border-t border-border">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question or request a change…"
          disabled={sending}
          className="flex-1 rounded-lg bg-surface2 border border-border px-3.5 py-2.5 text-sm placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500 transition disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || sending}
          className="shrink-0 flex items-center justify-center h-10 w-10 rounded-lg bg-gradient-to-r from-primary-500 to-accent-500 disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 transition"
          aria-label="Send"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </div>
  );
}
