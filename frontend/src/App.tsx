import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import Sidebar from "@/components/Sidebar";
import Navbar from "@/components/Navbar";
import AssistantWidget from "@/components/AssistantWidget";
import Home from "@/pages/Home";
import Results from "@/pages/Results";
import History from "@/pages/History";
import Settings from "@/pages/Settings";

export default function App() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      {mobileNavOpen && (
        <div className="fixed inset-0 z-30 bg-black/60 md:hidden" onClick={() => setMobileNavOpen(false)}>
          <div onClick={(e) => e.stopPropagation()}>
            <Sidebar mobile onClose={() => setMobileNavOpen(false)} />
          </div>
        </div>
      )}
      <div className="flex-1 min-w-0">
        <Navbar onMenuClick={() => setMobileNavOpen(true)} />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/results/:sessionId" element={<Results />} />
            <Route path="/history" element={<History />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
      <AssistantWidget />
    </div>
  );
}
