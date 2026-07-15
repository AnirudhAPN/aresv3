import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface OutputSectionProps {
  title: string;
  icon?: ReactNode;
  badges?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

export default function OutputSection({ title, icon, badges, defaultOpen = true, children }: OutputSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="rounded-2xl border border-border bg-surface/60 glass overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-surface2/50 transition"
      >
        <div className="flex items-center gap-2">
          {icon}
          <h2 className="text-base font-semibold">{title}</h2>
        </div>
        <div className="flex items-center gap-3">
          {badges}
          <ChevronDown className={cn("h-4 w-4 text-muted transition-transform", open && "rotate-180")} />
        </div>
      </button>
      {open && <div className="px-5 pb-5">{children}</div>}
    </section>
  );
}
