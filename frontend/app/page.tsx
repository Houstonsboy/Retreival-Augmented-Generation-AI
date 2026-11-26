"use client";

import Link from "next/link";
import { Scale, BookOpen, Smartphone, ArrowRight } from "lucide-react";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/app/context/AuthContext";

// ── Design tokens (match ChatView exactly) ────────────────────────────────
const BG      = "#FAF7F0";
const SURFACE = "#FFFDF8";
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const INK     = "#151515";
const MUTED   = "#6B6660";
const NR      = "var(--font-newsreader), Georgia, 'Times New Roman', serif";
const MNR     = "var(--font-manrope), system-ui, sans-serif";

export default function LandingPage() {
  const router = useRouter();
  const { user, profile, loading } = useAuth();

  const type = profile?.type;
  const isLitigant = type === "litigant";
  const isAdvocate = type === "advocate" || type === "admin";

  // Litigants have no use for the landing page — send them straight to their area.
  // Advocates/admins stay: they can choose between research and evidence tools.
  useEffect(() => {
    if (loading || !user || !profile) return;
    if (isLitigant) router.replace("/litigant/analysis");
  }, [loading, user, profile, isLitigant, router]);

  // Show spinner only while resolving auth or waiting for the litigant redirect.
  if (loading || (user && isLitigant)) {
    return (
      <div
        className="flex min-h-screen items-center justify-center"
        style={{ background: BG }}
      >
        <div
          className="h-8 w-8 animate-spin rounded-full border-4 border-t-transparent"
          style={{ borderColor: GOLD, borderTopColor: "transparent" }}
        />
      </div>
    );
  }

  // Dynamic hrefs based on auth state.
  const advocateCardHref = isAdvocate ? "/research" : "/auth/sign-in";
  const selfRepCardHref  = isAdvocate ? "/litigant/analysis" : "/litigant/sign-in";

  return (
    <div
      className="flex min-h-screen flex-col items-center"
      style={{ background: BG, fontFamily: MNR }}
    >
      {/* ── Top bar ──────────────────────────────────────────────────── */}
      <header
        className="w-full shrink-0 flex items-center justify-between px-6 py-3"
        style={{ borderBottom: `1px solid ${BORDER}`, background: BG }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="h-8 w-8 rounded-xl flex items-center justify-center"
            style={{
              background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)`,
            }}
          >
            <Scale className="h-4 w-4 text-white" />
          </div>
          <span
            className="text-sm font-bold tracking-tight"
            style={{ color: INK, fontFamily: MNR }}
          >
            Kenyan Legal{" "}
            <span style={{ color: GOLD }}>RAG</span>
          </span>
        </div>
      </header>

      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col items-center justify-center px-5 py-16 w-full">
        <div className="flex flex-col items-center text-center max-w-2xl mx-auto">
          {/* Eyebrow */}
          <p
            className="text-[10px] font-bold tracking-[.28em] uppercase mb-5"
            style={{ color: GOLD }}
          >
            Small Claims Court · Kenya
          </p>

          {/* Headline */}
          <h1
            className="text-4xl sm:text-5xl font-bold leading-tight mb-4"
            style={{ color: INK }}
          >
            Access the{" "}
            <em
              className="not-italic"
              style={{ fontFamily: NR, fontStyle: "italic", color: GOLD }}
            >
              law
            </em>
            , your way
          </h1>

          {/* Subtitle */}
          <p
            className="text-base leading-relaxed max-w-md mb-12"
            style={{ color: MUTED }}
          >
            A Kenyan legal research system grounded in case law and statutes —
            for advocates building arguments, and for individuals preparing
            their digital evidence.
          </p>

          {/* Choice cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 w-full max-w-2xl">
            {/* Advocate card */}
            <Link
              href={advocateCardHref}
              className="group flex flex-col items-start text-left p-6 rounded-2xl transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
              style={{
                background: SURFACE,
                border: `1px solid ${BORDER}`,
              }}
              aria-label="Sign in as an Advocate"
            >
              <div
                className="h-11 w-11 rounded-xl flex items-center justify-center mb-4"
                style={{ background: `${GOLD}18`, border: `1px solid ${GOLD}30` }}
              >
                <BookOpen className="h-5 w-5" style={{ color: GOLD }} />
              </div>
              <p
                className="text-[10px] font-bold tracking-[.18em] uppercase mb-1"
                style={{ color: GOLD }}
              >
                Advocate
              </p>
              <h2
                className="text-base font-bold mb-2"
                style={{ color: INK }}
              >
                I&apos;m an Advocate
              </h2>
              <p className="text-sm leading-relaxed flex-1" style={{ color: MUTED }}>
                Legal research with the full law library — case law, statutes,
                and the Constitution, with FIRAC analysis.
              </p>
              <div
                className="mt-5 flex items-center gap-1.5 text-sm font-semibold transition-colors group-hover:gap-2.5"
                style={{ color: GOLD_DK }}
              >
                {isAdvocate ? "Go to research" : "Sign in to continue"}
                <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
              </div>
            </Link>

            {/* Self-represented card */}
            <Link
              href={selfRepCardHref}
              className="group flex flex-col items-start text-left p-6 rounded-2xl transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
              style={{
                background: SURFACE,
                border: `1px solid ${BORDER}`,
              }}
              aria-label="Sign in as a self-represented litigant"
            >
              <div
                className="h-11 w-11 rounded-xl flex items-center justify-center mb-4"
                style={{ background: `${GOLD}18`, border: `1px solid ${GOLD}30` }}
              >
                <Smartphone className="h-5 w-5" style={{ color: GOLD }} />
              </div>
              <p
                className="text-[10px] font-bold tracking-[.18em] uppercase mb-1"
                style={{ color: GOLD }}
              >
                Self-represented
              </p>
              <h2
                className="text-base font-bold mb-2"
                style={{ color: INK }}
              >
                I&apos;m representing myself
              </h2>
              <p className="text-sm leading-relaxed flex-1" style={{ color: MUTED }}>
                Check and prepare your digital evidence — M-Pesa records,
                messages, and agreements — for Small Claims Court.
              </p>
              <div
                className="mt-5 flex items-center gap-1.5 text-sm font-semibold transition-colors group-hover:gap-2.5"
                style={{ color: GOLD_DK }}
              >
                {isAdvocate ? "Open evidence tools" : "Get started"}
                <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
              </div>
            </Link>
          </div>
        </div>
      </main>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer className="w-full py-5 flex justify-center">
        <p className="text-[11px]" style={{ color: MUTED }}>
          Kenyan Legal RAG · Small Claims Court Research
        </p>
      </footer>
    </div>
  );
}
