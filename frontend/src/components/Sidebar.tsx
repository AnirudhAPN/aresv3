import { Link, NavLink } from "react-router-dom";
import { FlaskConical, History, Settings, Sparkles, X } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/", label: "New Research", icon: FlaskConical, end: true },
  { to: "/history", label: "History", icon: History },
  { to: "/settings", label: "Settings", icon: Settings },
];

interface SidebarProps {
  /** Rendered inside the mobile drawer instead of the permanent desktop rail. */
  mobile?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ mobile = false, onClose }: SidebarProps) {
  return (
    <aside
      className={cn(
        "flex-col border-r border-border bg-surface/60 glass p-4",
        mobile ? "flex w-64 h-full" : "hidden md:flex md:w-64 min-h-screen"
      )}
    >
      <div className="flex items-center justify-between gap-2 px-2 py-3 mb-6">
        <Link to="/" onClick={onClose} className="flex items-center gap-2">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shadow-glow">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="font-bold text-lg leading-none">ARES</div>
            <div className="text-xs text-muted leading-none mt-1">Research Ecosystem Suite</div>
          </div>
        </Link>
        {mobile && (
          <button
            onClick={onClose}
            aria-label="Close menu"
            className="text-muted hover:text-white p-1.5 rounded-lg hover:bg-surface2 transition"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <nav className="flex flex-col gap-1">
        {navItems.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onClose}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary-500/15 text-primary-400 border border-primary-500/30"
                  : "text-muted hover:text-white hover:bg-surface2"
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
