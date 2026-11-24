"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ScrollText, Download, FileSearch } from "lucide-react";
import { downloadCertificate, listCertificates, viewCertificate } from "@/app/lib/evidenceApi";
import type { CertificateRecord, CertificateOutcome } from "@/app/lib/evidenceTypes";

// ── Design tokens ──────────────────────────────────────────────────────────
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const CREAM2  = "#F5F0E4";
const INK     = "#151515";
const MUTED   = "#6B6660";
const SURFACE = "#FFFDF8";
const MNR     = "var(--font-manrope), system-ui, sans-serif";
const NR      = "var(--font-newsreader), Georgia, 'Times New Roman', serif";

type Filter = "all" | CertificateOutcome;

const OUTCOME_STYLES: Record<CertificateOutcome, { bg: string; color: string; label: string; explain: string }> = {
  certificate_ready: {
    bg: "#ECFDF5", color: "#065F46", label: "Certificate ready",
    explain: "All checks passed. The evidence is strong enough to generate a Section 106B certificate for court.",
  },
  needs_strengthening: {
    bg: "#FFF8E6", color: GOLD_DK, label: "Needs strengthening",
    explain: "The core contract elements are established, but some supporting details (e.g. device questionnaire, delivery confirmation) are missing. Answer the follow-up questions in the Evidence Analysis page to unlock the certificate.",
  },
  weak: {
    bg: "#FEF2F2", color: "#B91C1C", label: "Weak",
    explain: "The evidence record has an integrity problem — messages appear edited or deleted, the OCR confidence is low, or a file hash mismatch was detected. This means the system cannot verify that the screenshots are unaltered. Upload original, unedited screenshots to resolve this.",
  },
};

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all",                  label: "All"                  },
  { key: "certificate_ready",    label: "Certificates"         },
  { key: "needs_strengthening",  label: "Needs strengthening"  },
  { key: "weak",                 label: "Weak"                  },
];

function OutcomePill({ outcome }: { outcome: CertificateOutcome }) {
  const s = OUTCOME_STYLES[outcome];
  return (
    <span
      className="inline-block rounded-full px-2.5 py-0.5 text-[10px] font-bold"
      style={{ background: s.bg, color: s.color }}
    >
      {s.label}
    </span>
  );
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleDateString("en-KE", {
      day: "numeric", month: "short", year: "numeric",
    });
  } catch { return iso; }
}

function SkeletonRow() {
  return (
    <div
      className="flex items-center gap-3 px-4 py-3 rounded-xl animate-pulse"
      style={{ background: CREAM2 }}
      aria-hidden="true"
    >
      <div className="flex-1 space-y-2">
        <div className="h-3 rounded" style={{ width: "35%", background: "#ECE6D8" }} />
        <div className="h-2.5 rounded" style={{ width: "55%", background: "#ECE6D8" }} />
      </div>
      <div className="h-5 w-24 rounded-full" style={{ background: "#ECE6D8" }} />
    </div>
  );
}

export default function CertificatesPage() {
  const [records, setRecords] = useState<CertificateRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<Filter>("all");
  const [downloading, setDownloading] = useState<string | null>(null);
  const [viewing, setViewing] = useState<string | null>(null);

  useEffect(() => {
    listCertificates()
      .then(setRecords)
      .catch(() => setRecords([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered =
    filter === "all" ? records : records.filter((r) => r.outcome === filter);

  const handleDownload = async (rec: CertificateRecord) => {
    if (downloading) return;
    setDownloading(rec.cert_id);
    try {
      await downloadCertificate(rec.cert_id, rec.certificate_number);
    } catch {
      // silent
    } finally {
      setDownloading(null);
    }
  };

  const handleView = async (rec: CertificateRecord) => {
    if (viewing) return;
    setViewing(rec.cert_id);
    try {
      await viewCertificate(rec.cert_id);
    } catch {
      // silent
    } finally {
      setViewing(null);
    }
  };

  return (
    <div className="px-5 py-8 max-w-3xl mx-auto" style={{ fontFamily: MNR }}>

      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="mb-8">
        <p
          className="text-[10px] font-bold tracking-[.24em] uppercase mb-2"
          style={{ color: GOLD }}
        >
          Self-Represented · Kenya
        </p>
        <h1
          className="text-2xl font-bold"
          style={{ color: INK, fontFamily: NR }}
        >
          Certificate History
        </h1>
        <p className="text-sm mt-1" style={{ color: MUTED }}>
          Section 106B certificate drafts generated from your uploaded evidence.
        </p>
      </div>

      {/* ── Filter chips ─────────────────────────────────────────── */}
      {!loading && records.length > 0 && (
        <div
          className="flex flex-wrap gap-2 mb-6"
          role="group"
          aria-label="Filter certificates by outcome"
        >
          {FILTERS.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => setFilter(key)}
              className="px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-1"
              style={{
                background: filter === key ? GOLD : CREAM2,
                color: filter === key ? "white" : MUTED,
                border: `1.5px solid ${filter === key ? GOLD : BORDER}`,
              }}
              aria-pressed={filter === key}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {/* ── Content ───────────────────────────────────────────────── */}
      {loading ? (
        <div className="space-y-2" aria-label="Loading certificates" aria-busy="true">
          {[1, 2, 3].map((i) => <SkeletonRow key={i} />)}
        </div>
      ) : records.length === 0 ? (
        /* Empty state */
        <div
          className="flex flex-col items-center justify-center py-20 rounded-2xl"
          style={{ background: SURFACE, border: `1px dashed ${BORDER}` }}
        >
          <div
            className="h-14 w-14 rounded-2xl flex items-center justify-center mb-4"
            style={{ background: CREAM2 }}
            aria-hidden="true"
          >
            <ScrollText className="h-7 w-7" style={{ color: MUTED }} />
          </div>
          <p className="text-sm font-semibold mb-1" style={{ color: INK }}>
            No certificates yet
          </p>
          <p className="text-xs mb-5" style={{ color: MUTED }}>
            Analyse your evidence to generate one.
          </p>
          <Link
            href="/litigant/analysis"
            className="px-5 py-2.5 rounded-xl text-sm font-bold transition-all hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
            style={{
              background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`,
              color: "white",
            }}
          >
            Go to Evidence Analysis
          </Link>
        </div>
      ) : filtered.length === 0 ? (
        <p className="text-sm text-center py-10" style={{ color: MUTED }}>
          No certificates match this filter.
        </p>
      ) : (
        <>
          {/* Desktop table */}
          <div className="hidden sm:block overflow-x-auto rounded-2xl" style={{ border: `1px solid ${BORDER}` }}>
            <table className="w-full text-sm" aria-label="Certificate records">
              <thead>
                <tr style={{ background: CREAM2, borderBottom: `1px solid ${BORDER}` }}>
                  {["Certificate", "Evidence file", "Outcome", "Date", "Actions"].map((h) => (
                    <th
                      key={h}
                      className="text-left px-4 py-3 text-[10px] font-bold tracking-[.15em] uppercase"
                      style={{ color: MUTED }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((rec, i) => (
                  <>
                    <tr
                      key={rec.cert_id}
                      style={{
                        background: i % 2 === 0 ? SURFACE : "white",
                        borderBottom: `1px solid ${BORDER}`,
                      }}
                    >
                      <td className="px-4 py-3 font-medium font-mono text-xs" style={{ color: INK }}>
                        {rec.certificate_number}
                      </td>
                      <td className="px-4 py-3 truncate max-w-[180px] text-xs" style={{ color: MUTED }}>
                        {rec.document_type === "CERTIFICATE" ? "Certificate" : "Deficiency Report"}
                      </td>
                      <td className="px-4 py-3">
                        <OutcomePill outcome={rec.outcome} />
                      </td>
                      <td className="px-4 py-3" style={{ color: MUTED }}>
                        {formatDate(rec.created_at)}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => handleDownload(rec)}
                            disabled={!!downloading}
                            title="Download PDF"
                            className="flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-lg disabled:opacity-60 hover:opacity-80 transition-opacity"
                            style={{ background: GOLD, color: "white" }}
                            aria-label={`Download certificate ${rec.certificate_number}`}
                          >
                            <Download className="h-3.5 w-3.5" aria-hidden="true" />
                            {downloading === rec.cert_id ? "…" : "Download"}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleView(rec)}
                            disabled={!!viewing}
                            title="Open PDF in new tab"
                            className="flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-lg hover:opacity-80 transition-opacity disabled:opacity-60"
                            style={{ background: CREAM2, color: MUTED }}
                            aria-label={`View report for certificate ${rec.certificate_number}`}
                          >
                            <FileSearch className="h-3.5 w-3.5" aria-hidden="true" />
                            {viewing === rec.cert_id ? "Opening…" : "View report"}
                          </button>
                        </div>
                      </td>
                    </tr>
                    {rec.outcome !== "certificate_ready" && (
                      <tr key={`${rec.cert_id}-reason`} style={{ background: i % 2 === 0 ? SURFACE : "white", borderBottom: `1px solid ${BORDER}` }}>
                        <td colSpan={5} className="px-4 pb-3 pt-0">
                          <p className="text-[11px] leading-relaxed rounded-lg px-2.5 py-2"
                            style={{ background: OUTCOME_STYLES[rec.outcome].bg, color: OUTCOME_STYLES[rec.outcome].color }}>
                            <span className="font-semibold">Why {OUTCOME_STYLES[rec.outcome].label.toLowerCase()}? </span>
                            {OUTCOME_STYLES[rec.outcome].explain}
                          </p>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile card list */}
          <ul className="sm:hidden space-y-3" aria-label="Certificate records">
            {filtered.map((rec) => (
              <li
                key={rec.cert_id}
                className="rounded-2xl p-4"
                style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
              >
                <div className="flex items-start justify-between gap-3 mb-3">
                  <div className="min-w-0">
                    <p className="text-xs font-bold font-mono" style={{ color: INK }}>
                      {rec.certificate_number}
                    </p>
                    <p className="text-xs mt-0.5" style={{ color: MUTED }}>
                      {rec.document_type === "CERTIFICATE" ? "Certificate" : "Deficiency Report"}
                    </p>
                    <p className="text-xs mt-0.5" style={{ color: MUTED }}>
                      {formatDate(rec.created_at)}
                    </p>
                  </div>
                  <OutcomePill outcome={rec.outcome} />
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => handleDownload(rec)}
                    disabled={!!downloading}
                    className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg disabled:opacity-60 hover:opacity-80 transition-opacity"
                    style={{ background: GOLD, color: "white" }}
                  >
                    <Download className="h-3.5 w-3.5" aria-hidden="true" />
                    {downloading === rec.cert_id ? "…" : "Download"}
                  </button>
                  <button
                    type="button"
                    onClick={() => handleView(rec)}
                    disabled={!!viewing}
                    className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg hover:opacity-80 transition-opacity disabled:opacity-60"
                    style={{ background: CREAM2, color: MUTED }}
                  >
                    <FileSearch className="h-3.5 w-3.5" aria-hidden="true" />
                    {viewing === rec.cert_id ? "Opening…" : "View report"}
                  </button>
                </div>
                {rec.outcome !== "certificate_ready" && (
                  <p className="text-[11px] leading-relaxed rounded-lg px-2.5 py-2 mt-2"
                    style={{ background: OUTCOME_STYLES[rec.outcome].bg, color: OUTCOME_STYLES[rec.outcome].color }}>
                    <span className="font-semibold">Why {OUTCOME_STYLES[rec.outcome].label.toLowerCase()}? </span>
                    {OUTCOME_STYLES[rec.outcome].explain}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
