import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sparkles, Send, Loader2, X, Bot, User, Trash2, Compass } from "lucide-react";
import { cn } from "@/lib/utils";
import { getAssistantHistory, sendAssistantMessage, clearAssistantHistory } from "@/lib/api";
import type { AssistantMessage } from "@/types";

const SUGGESTIONS = [
  "I want to research something in healthcare AI — where do I start?",
  "How does the ACS scoring actually work?",
  "What happens after the proposal is generated?",
];

export default function AssistantWidget() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [loadedHistory, setLoadedHistory] = useState(false);
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open || loadedHistory) return;
    getAssistantHistory()
      .then(setMessages)
      .catch(() => setError("Could not load the conversation."))
      .finally(() => setLoadedHistory(true));
  }, [open, loadedHistory]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending, open]);

  const submit = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || sending) return;
    setError(null);
    setInput("");
    setSending(true);
    setMessages((prev) => [...prev, { role: "user", content: trimmed, suggested_topic: null, created_at: Date.now() / 1000 }]);

    try {
      const res = await sendAssistantMessage(trimmed);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.reply, suggested_topic: res.suggested_topic, created_at: Date.now() / 1000 },
      ]);
    } catch {
      setError("Couldn't reach the assistant. Please try again.");
    } finally {
      setSending(false);
    }
  };

  const useSuggestedTopic = (topic: string) => {
    setOpen(false);
    navigate("/", { state: { assistantTopic: topic } });
  };

  const handleClear = async () => {
    try {
      await clearAssistantHistory();
      setMessages([]);
    } catch {
      setError("Couldn't clear the conversation.");
    }
  };

  return (
    <>
      {open && (
        <div className="fixed bottom-24 right-5 z-40 w-[min(380px,calc(100vw-2.5rem))] h-[520px] rounded-2xl border border-border bg-surface glass shadow-2xl flex flex-col overflow-hidden">
          <div className="flex items-center justify-between gap-2 px-4 py-3 border-b border-border">
            <div className="flex items-center gap-2 min-w-0">
              <div className="h-7 w-7 rounded-full bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0">
                <Sparkles className="h-3.5 w-3.5 text-white" />
              </div>
              <div className="min-w-0">
                <h2 className="text-sm font-semibold leading-none">Research Assistant</h2>
                <p className="text-[10px] text-muted mt-0.5">Planning · suggestions · how ARES works</p>
              </div>
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <button
                onClick={handleClear}
                title="Clear conversation"
                className="text-muted hover:text-white p-1.5 rounded-lg hover:bg-surface2 transition"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={() => setOpen(false)}
                aria-label="Close"
                className="text-muted hover:text-white p-1.5 rounded-lg hover:bg-surface2 transition"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {!loadedHistory ? (
              <div className="flex items-center justify-center h-full text-muted gap-2 text-sm">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading…
              </div>
            ) : messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center gap-3 px-2">
                <Compass className="h-5 w-5 text-primary-400" />
                <p className="text-sm text-muted max-w-[26ch]">
                  Not sure what to research, or want to know how ARES works? Ask away.
                </p>
                <div className="flex flex-col gap-1.5 w-full">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => submit(s)}
                      className="text-xs px-3 py-2 rounded-lg bg-surface2 border border-border text-muted hover:text-white hover:border-primary-500/50 transition text-left"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={cn("flex gap-2", m.role === "user" && "flex-row-reverse")}>
                  <div
                    className={cn(
                      "h-6 w-6 rounded-full flex items-center justify-center shrink-0 mt-0.5",
                      m.role === "user" ? "bg-primary-500/20 text-primary-400" : "bg-surface2 text-accent-500"
                    )}
                  >
                    {m.role === "user" ? <User className="h-3 w-3" /> : <Bot className="h-3 w-3" />}
                  </div>
                  <div className="max-w-[80%] space-y-1.5">
                    <div
                      className={cn(
                        "rounded-xl px-3 py-2 text-sm leading-relaxed",
                        m.role === "user" ? "bg-primary-500/15 text-white" : "bg-surface2 text-white/90"
                      )}
                    >
                      {m.content}
                    </div>
                    {m.suggested_topic && (
                      <button
                        onClick={() => useSuggestedTopic(m.suggested_topic!)}
                        className="text-[11px] flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-gradient-to-r from-primary-500/20 to-accent-500/20 border border-primary-500/40 text-primary-300 hover:text-white transition"
                      >
                        <Sparkles className="h-3 w-3" /> Use "{m.suggested_topic}"
                      </button>
                    )}
                  </div>
                </div>
              ))
            )}
            {sending && (
              <div className="flex gap-2">
                <div className="h-6 w-6 rounded-full flex items-center justify-center shrink-0 bg-surface2 text-accent-500">
                  <Bot className="h-3 w-3" />
                </div>
                <div className="rounded-xl px-3 py-2 bg-surface2 text-muted text-sm flex items-center gap-2">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> Thinking…
                </div>
              </div>
            )}
          </div>

          {error && <p className="px-4 text-xs text-red-400 pb-1">{error}</p>}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit(input);
            }}
            className="flex items-center gap-2 px-3 py-3 border-t border-border"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about a topic or how ARES works…"
              disabled={sending}
              className="flex-1 rounded-lg bg-surface2 border border-border px-3 py-2 text-sm placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500 transition disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!input.trim() || sending}
              className="shrink-0 flex items-center justify-center h-9 w-9 rounded-lg bg-gradient-to-r from-primary-500 to-accent-500 disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 transition"
              aria-label="Send"
            >
              <Send className="h-3.5 w-3.5" />
            </button>
          </form>
        </div>
      )}

      <button
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Close research assistant" : "Open research assistant"}
        className={cn(
          "fixed bottom-5 right-5 z-40 h-14 w-14 rounded-full flex items-center justify-center shadow-glow transition",
          open
            ? "bg-surface2 border border-border text-white"
            : "bg-gradient-to-br from-primary-500 to-accent-500 text-white hover:brightness-110"
        )}
      >
        {open ? <X className="h-5 w-5" /> : <Sparkles className="h-5 w-5" />}
      </button>
    </>
  );
}
