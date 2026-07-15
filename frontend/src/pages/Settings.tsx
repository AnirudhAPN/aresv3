import { Info, Users } from "lucide-react";
import { usePageTitle } from "@/lib/usePageTitle";

export default function Settings() {
  usePageTitle("Settings");

  return (
    <div className="max-w-3xl mx-auto py-8 px-4 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-muted mt-1">About this workspace and how proposals are scored.</p>
      </div>

      <div className="rounded-2xl border border-border bg-surface/60 glass p-5 flex gap-3">
        <Users className="h-5 w-5 text-primary-400 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold mb-1">Workspace</h3>
          <p className="text-sm text-muted">
            This workspace is currently set up for a single user — every research session you create is
            private to this workspace.
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-surface/60 glass p-5 flex gap-3">
        <Info className="h-5 w-5 text-primary-400 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold mb-1">Adaptive Collaboration Score (ACS)</h3>
          <p className="text-sm text-muted">
            Each agent's output is scored on confidence (an independent quality judge), evidence quality
            (how well claims are backed by real, verifiable sources), and task completion (structural
            completeness). Outputs scoring below 60% trigger one automatic retry with refined guidance; if
            still below threshold, the section is flagged "low confidence" rather than looping indefinitely.
          </p>
        </div>
      </div>
    </div>
  );
}
