"use client";

import { useRouter } from "next/navigation";
import { MessageSquare, FolderOpen, FileSearch, Scale, ArrowRight, Info } from "lucide-react";
import { useAuth } from "@/app/context/AuthContext";
import { UserMenu } from "@/app/components/UserMenu";

// ── Design tokens ──────────────────────────────────────────────────────────
const BG      = "#FAF7F0";
const SURFACE = "#FFFDF8";
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const INK     = "#151515";
const MUTED   = "#6B6660";
const MNR     = "var(--font-manrope), system-ui, sans-serif";
const NR      = "var(--font-newsreader), Georgia, 'Times New Roman', serif";

const tools = [
  {
    icon: MessageSquare,
    category: "Research",
    label: "Chat",
    description: "Ask legal questions and get analysis grounded in Kenyan case law and statutes.",
    href: "/research",
    cta: "Open chat",
  },
  {
    icon: FolderOpen,
    category: "Library",
    label: "Internal Repository",
    description: "Browse and manage uploaded case documents, statutes, and legal references.",
    href: "/repository",
    cta: "Browse files",
  },
  {
    icon: FileSearch,
    category: "Evidence",
    label: "Evidence Tools",
    description: "Prepare digital evidence for Small Claims Court — M-Pesa records, messages, and agreements.",
    href: "/litigant/analysis",
    cta: "Open evidence tools",
  },
];

export default function Dashboard() {
  const router  = useRouter();
  const { profile } = useAuth();
  const name    = profile?.username ?? "Counsellor";

  return (
    <div
      className="flex min-h-screen flex-col"
      style={{ background: BG, fontFamily: MNR }}
    >
      {/* ── Top bar ──────────────────────────────────────────────────── */}
      <header
        className="w-full shrink-0 flex items-center justify-between px-6 py-3"
        style={{ borderBottom: `1px solid ${BORDER}`, background: BG }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="h-8 w-8 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)` }}
          >
            <Scale className="h-4 w-4 text-white" />
          </div>
          <span
            className="text-sm font-bold tracking-tight"
            style={{ color: INK }}
          >
            Kenyan Legal{" "}
            <span style={{ color: GOLD }}>RAG</span>
          </span>
        </div>
        <UserMenu />
      </header>

      {/* ── Body ─────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col items-center justify-center px-5 py-16">
        {/* Greeting */}
        <div className="text-center mb-12">
          <p
            className="text-[10px] font-bold tracking-[.28em] uppercase mb-3"
            style={{ color: GOLD }}
          >
            Small Claims Court · Kenya
          </p>
          <h1
            className="text-3xl sm:text-4xl font-bold mb-2"
            style={{ color: INK, fontFamily: NR, fontStyle: "italic" }}
          >
            Welcome back, {name}
          </h1>
          <p className="text-sm" style={{ color: MUTED }}>
            What would you like to do today?
          </p>
        </div>

        {/* Tool cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-3xl">
          {tools.map((tool) => {
            const Icon = tool.icon;
            return (
              <button
                key={tool.label}
                onClick={() => router.push(tool.href)}
                className="group flex flex-col items-start text-left p-5 rounded-2xl transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-offset-2"
                style={{
                  background: SURFACE,
                  border: `1px solid ${BORDER}`,
                }}
              >
                {/* Category tag + icon row */}
                <div className="w-full flex items-center justify-between mb-4">
                  <span
                    className="text-[9px] font-bold tracking-[.22em] uppercase"
                    style={{ color: MUTED }}
                  >
                    {tool.category}
                  </span>
                  <div
                    className="h-9 w-9 rounded-xl flex items-center justify-center shrink-0"
                    style={{ background: `${GOLD}18`, border: `1px solid ${GOLD}28` }}
                  >
                    <Icon className="h-4.5 w-4.5" style={{ color: GOLD }} />
                  </div>
                </div>

                <h2
                  className="text-sm font-bold mb-1.5"
                  style={{ color: INK }}
                >
                  {tool.label}
                </h2>
                <p
                  className="text-xs leading-relaxed flex-1"
                  style={{ color: MUTED }}
                >
                  {tool.description}
                </p>

                <div
                  className="mt-5 flex items-center gap-1 text-xs font-semibold transition-all group-hover:gap-2"
                  style={{ color: GOLD_DK }}
                >
                  {tool.cta}
                  <ArrowRight className="h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-x-1" />
                </div>
              </button>
            );
          })}
        </div>

        {/* Disclaimer */}
        <div
          className="mt-10 w-full max-w-3xl rounded-xl px-4 py-3 flex items-start gap-2.5"
          style={{ background: `${GOLD}0D`, border: `1px solid ${GOLD}28` }}
        >
          <Info className="h-4 w-4 mt-0.5 shrink-0" style={{ color: GOLD }} />
          <p className="text-xs leading-relaxed" style={{ color: MUTED }}>
            Analysis here is automated and is not legal advice. Confirm important
            decisions with a qualified advocate before filing.
          </p>
        </div>
      </main>
    </div>
  );
}
