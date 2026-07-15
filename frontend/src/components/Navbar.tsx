import { Menu } from "lucide-react";

interface NavbarProps {
  onMenuClick?: () => void;
}

export default function Navbar({ onMenuClick }: NavbarProps) {
  return (
    <header className="sticky top-0 z-20 flex items-center justify-between border-b border-border bg-background/80 glass px-4 py-3 md:px-6">
      <div className="flex items-center gap-3">
        <button className="md:hidden text-muted" onClick={onMenuClick} aria-label="Open menu">
          <Menu className="h-5 w-5" />
        </button>
        <h1 className="text-sm font-semibold text-muted tracking-wide uppercase">
          Adaptive Research Ecosystem Suite
        </h1>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted">
        <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_2px_rgba(52,211,153,0.6)]" />
        System ready
      </div>
    </header>
  );
}
