"use client";

import { Upload, ScanSearch, MessageSquare, ScrollText } from "lucide-react";
import { SCCVisionTest } from "@/app/components/sccvision/SCCVisionTest";

const GOLD    = "#C9971A";
const BORDER  = "#ECE6D8";
const INK     = "#151515";
const MUTED   = "#6B6660";
const SURFACE = "#FFFDF8";
const NR      = "var(--font-newsreader), Georgia, 'Times New Roman', serif";
const MNR     = "var(--font-manrope), system-ui, sans-serif";

const STEPS = [
  { Icon: Upload,        label: "Upload",                 desc: "Add your files" },
  { Icon: ScanSearch,    label: "We read it",             desc: "Extract & check evidence" },
  { Icon: MessageSquare, label: "Answer a few questions", desc: "About your device or phone" },
  { Icon: ScrollText,    label: "Certificate or guidance",desc: "Signed draft or next steps" },
];

export default function EvidenceAnalysisPage({
  searchParams,
}: {
  searchParams?: { caseId?: string };
}) {
  const initialCaseId = searchParams?.caseId;
  return (
    <div className="px-5 py-8 max-w-2xl mx-auto" style={{ fontFamily: MNR }}>

      {/* ── Page heading ─────────────────────────────────────────── */}
      <div className="mb-8">
        <p
          className="text-[10px] font-bold tracking-[.24em] uppercase mb-2"
          style={{ color: GOLD }}
        >
          Small Claims Court · Kenya
        </p>
        <h1
          className="text-2xl font-bold leading-snug"
          style={{ color: INK, fontFamily: NR }}
        >
          Evidence Analysis
        </h1>
      </div>

      {/* ── How it works ─────────────────────────────────────────── */}
      <div
        className="rounded-2xl p-5 mb-8"
        style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
        aria-label="How evidence analysis works"
      >
        <p
          className="text-[9px] font-bold tracking-[.2em] uppercase mb-4"
          style={{ color: MUTED }}
        >
          How it works
        </p>
        <ol className="grid grid-cols-2 sm:grid-cols-4 gap-4" role="list">
          {STEPS.map(({ Icon, label, desc }, i) => (
            <li key={i} className="flex flex-col items-center text-center gap-2">
              <div
                className="h-10 w-10 rounded-xl flex items-center justify-center mb-1"
                style={{ background: `${GOLD}18` }}
                aria-hidden="true"
              >
                <Icon className="h-5 w-5" style={{ color: GOLD }} />
              </div>
              <div>
                <p className="text-xs font-semibold" style={{ color: INK }}>
                  {i + 1}. {label}
                </p>
                <p className="text-[11px] mt-0.5" style={{ color: MUTED }}>
                  {desc}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* ── SCC Analyst ──────────────────────────────────────────── */}
      <SCCVisionTest initialCaseId={initialCaseId} />
    </div>
  );
}
