"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Scale, Menu, X, ChevronLeft,
} from "lucide-react";
import { UserMenu } from "@/app/components/UserMenu";
import { useAuth } from "@/app/context/AuthContext";

const BG      = "#F9F7F2";
const SURFACE = "#F4F1E8";
const GOLD    = "#A87317";
const GOLD_DK = "#80520D";
const BORDER  = "#ECE6D8";
const INK     = "#151515";
const MUTED   = "#6B6660";
const MNR     = "var(--font-manrope), system-ui, sans-serif";

const NAV_ITEMS = [
  { href: "/litigant/analysis",     label: "New analysis"  },
  { href: "/litigant/evidence",     label: "Evidence library"  },
  { href: "/litigant/certificates", label: "Certificates"  },
];

function SidebarContent({
  pathname,
  isAdvocate,
  onNav,
}: {
  pathname: string;
  isAdvocate: boolean;
  onNav?: () => void;
}) {
  return (
    <div className="flex flex-col flex-1 overflow-y-auto">
      <nav className="flex flex-col py-1 flex-1" aria-label="Litigant navigation">
        {NAV_ITEMS.map(({ href, label }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              onClick={onNav}
              className="flex items-center gap-3 mx-2 mb-0.5 rounded-lg px-3 py-2 text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-1"
              style={{
                background: active ? "#FBF0D6" : "transparent",
                border: `1.5px solid ${active ? GOLD : "transparent"}`,
                color: active ? GOLD_DK : MUTED,
              }}
              aria-current={active ? "page" : undefined}
            >
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Back to research — shown only for advocate/admin users */}
      {isAdvocate && (
        <div
          className="shrink-0 px-3 py-3"
          style={{ borderTop: `1px solid ${BORDER}` }}
        >
          <Link
            href="/research"
            onClick={onNav}
            className="flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium transition-colors hover:bg-amber-50 focus:outline-none focus:ring-2 focus:ring-brand-gold"
            style={{ color: MUTED }}
          >
            <ChevronLeft className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            Back to research
          </Link>
        </div>
      )}
    </div>
  );
}

export default function LitigantLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { profile } = useAuth();
  const isAdvocate = profile?.type === "advocate" || profile?.type === "admin";

  return (
    <div
      className="flex h-screen flex-col"
      style={{ background: BG, fontFamily: MNR }}
    >
      {/* ── Top bar ───────────────────────────────────────────────── */}
      <header
        className="shrink-0 z-20 flex items-center justify-between px-4 py-2.5"
        style={{ background: "#fff", borderBottom: `1px solid ${BORDER}` }}
      >
        <div className="flex items-center gap-3">
          {/* Hamburger (mobile) */}
          <button
            className="lg:hidden w-8 h-8 rounded-lg flex items-center justify-center hover:bg-amber-50 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-gold"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open navigation"
            aria-expanded={drawerOpen}
          >
            <Menu className="h-4 w-4" style={{ color: MUTED }} aria-hidden="true" />
          </button>

          {/* Logo */}
          <div
            className="h-8 w-8 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)` }}
          >
            <Scale className="h-4 w-4 text-white" aria-hidden="true" />
          </div>
          <span className="text-sm font-bold tracking-tight" style={{ color: INK }}>
            Kenyan Legal{" "}
            <span style={{ color: GOLD }}>RAG</span>
          </span>
          {isAdvocate && (
            <span
              className="text-[9px] font-bold tracking-[.14em] uppercase px-1.5 py-0.5 rounded"
              style={{ background: `${GOLD}22`, color: GOLD_DK, border: `1px solid ${GOLD}44` }}
              aria-label="Signed in as advocate"
            >
              Advocate
            </span>
          )}
        </div>

        <UserMenu />
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ── Desktop sidebar ─────────────────────────────────────── */}
        <aside
          className="hidden lg:flex flex-col shrink-0 w-48"
          style={{
            background: SURFACE,
            borderRight: `1px solid ${BORDER}`,
          }}
          aria-label="Litigant sidebar"
        >
          <div
            className="shrink-0 px-4 py-3"
            style={{ borderBottom: `1px solid ${BORDER}` }}
          >
            <p
              className="text-[9px] font-bold tracking-[.18em] uppercase"
              style={{ color: MUTED }}
            >
              Self-Represented
            </p>
          </div>
          <SidebarContent pathname={pathname} isAdvocate={isAdvocate} />
        </aside>

        {/* ── Mobile drawer backdrop ──────────────────────────────── */}
        {drawerOpen && (
          <div
            className="fixed inset-0 z-40 lg:hidden"
            style={{ background: "rgba(21,21,21,0.35)" }}
            onClick={() => setDrawerOpen(false)}
            aria-hidden="true"
          />
        )}

        {/* ── Mobile drawer ───────────────────────────────────────── */}
        <aside
          className="fixed left-0 top-0 bottom-0 z-50 flex flex-col lg:hidden"
          style={{
            width: 260,
            background: SURFACE,
            borderRight: `1px solid ${BORDER}`,
            boxShadow: "4px 0 24px rgba(60,45,10,0.12)",
            transform: drawerOpen ? "translateX(0)" : "translateX(-100%)",
            transition: "transform 0.22s cubic-bezier(.4,0,.2,1)",
          }}
          aria-label="Litigant navigation drawer"
          aria-hidden={!drawerOpen}
          inert={!drawerOpen}
        >
          <div
            className="flex items-center justify-between px-4 py-3 shrink-0"
            style={{ borderBottom: `1px solid ${BORDER}` }}
          >
            <p
              className="text-[9px] font-bold tracking-[.18em] uppercase"
              style={{ color: MUTED }}
            >
              Navigation
            </p>
            <button
              className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-amber-50 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-gold"
              onClick={() => setDrawerOpen(false)}
              aria-label="Close navigation"
            >
              <X className="h-4 w-4" style={{ color: MUTED }} aria-hidden="true" />
            </button>
          </div>
          <SidebarContent pathname={pathname} isAdvocate={isAdvocate} onNav={() => setDrawerOpen(false)} />
        </aside>

        {/* ── Main content area ────────────────────────────────────── */}
        <main
          className="min-w-0 flex-1 overflow-y-auto overscroll-contain"
          id="main-content"
          tabIndex={-1}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
