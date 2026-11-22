"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { FolderOpen, X, ZoomIn, RefreshCw, Trash2 } from "lucide-react";
import { useAuth } from "@/app/context/AuthContext";
import { listUserEvidence, deleteEvidenceFile, type EvidenceStorageRecord } from "@/app/lib/storageService";

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

// ── Helpers ────────────────────────────────────────────────────────────────
function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(d: Date) {
  return d.toLocaleDateString("en-KE", { day: "numeric", month: "short", year: "numeric" });
}

function evidenceTypeLabel(type: string): string {
  switch (type) {
    case "chat_screenshot":   return "WhatsApp / SMS";
    case "mpesa_screenshot":  return "M-Pesa";
    case "mpesa_statement":   return "M-Pesa Statement";
    case "mpesa_sms":         return "M-Pesa SMS";
    case "whatsapp_sms_chat": return "Chat";
    case "email":             return "Email";
    case "written_agreement_photo": return "Agreement";
    default:                  return "Evidence";
  }
}

function typeBadgeStyle(type: string): { bg: string; color: string } {
  switch (type) {
    case "mpesa_screenshot":
    case "mpesa_statement":
    case "mpesa_sms":
      return { bg: "#DCFCE7", color: "#166534" };
    case "chat_screenshot":
    case "whatsapp_sms_chat":
      return { bg: "#DBEAFE", color: "#1E40AF" };
    case "email":
      return { bg: "#F3E8FF", color: "#7E22CE" };
    default:
      return { bg: CREAM2, color: MUTED };
  }
}

// ── Case grouping ──────────────────────────────────────────────────────────
interface CaseGroup {
  caseId: string;
  caseTitle: string;
  items: EvidenceStorageRecord[];
  latestUpload: Date;
}

function groupByCaseId(evidence: EvidenceStorageRecord[]): CaseGroup[] {
  const map = new Map<string, CaseGroup>();
  for (const ev of evidence) {
    if (!map.has(ev.caseId)) {
      map.set(ev.caseId, {
        caseId: ev.caseId,
        caseTitle: ev.caseTitle || "Untitled Case",
        items: [],
        latestUpload: ev.uploadedAt,
      });
    }
    const g = map.get(ev.caseId)!;
    g.items.push(ev);
    if (ev.uploadedAt > g.latestUpload) g.latestUpload = ev.uploadedAt;
  }
  return Array.from(map.values()).sort(
    (a, b) => b.latestUpload.getTime() - a.latestUpload.getTime(),
  );
}

// ── Skeleton ───────────────────────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div
      className="rounded-2xl p-5 animate-pulse"
      style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
      aria-hidden="true"
    >
      <div className="h-4 rounded mb-2" style={{ width: "55%", background: CREAM2 }} />
      <div className="h-3 rounded mb-5" style={{ width: "30%", background: CREAM2 }} />
      <div className="flex gap-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-20 w-20 rounded-xl" style={{ background: CREAM2 }} />
        ))}
      </div>
    </div>
  );
}

// ── Image lightbox ─────────────────────────────────────────────────────────
function Lightbox({
  url, name, onClose,
}: { url: string; name: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(21,21,21,0.80)" }}
      role="dialog"
      aria-modal="true"
      aria-label={`Viewing ${name}`}
    >
      {/* backdrop click */}
      <div className="absolute inset-0" onClick={onClose} aria-hidden="true" />

      <div className="relative z-10 max-w-3xl w-full max-h-[90vh] flex flex-col rounded-2xl overflow-hidden"
        style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
        {/* toolbar */}
        <div
          className="flex items-center justify-between gap-3 px-4 py-3 shrink-0"
          style={{ background: CREAM2, borderBottom: `1px solid ${BORDER}` }}
        >
          <p className="text-xs font-semibold truncate" style={{ color: INK }}>{name}</p>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 h-7 w-7 rounded-lg flex items-center justify-center hover:bg-black/5 transition-colors"
            aria-label="Close image viewer"
          >
            <X className="h-4 w-4" style={{ color: MUTED }} />
          </button>
        </div>
        {/* image */}
        <div className="overflow-auto flex-1 flex items-center justify-center p-4" style={{ background: CREAM2 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={url}
            alt={name}
            className="max-w-full max-h-full rounded-xl object-contain"
            style={{ boxShadow: "0 4px 24px rgba(0,0,0,0.18)" }}
          />
        </div>
      </div>
    </div>
  );
}

// ── Evidence thumbnail ─────────────────────────────────────────────────────
function EvidenceThumb({
  ev, onView, onDelete, isDeleting,
}: {
  ev: EvidenceStorageRecord;
  onView: (ev: EvidenceStorageRecord) => void;
  onDelete: (ev: EvidenceStorageRecord) => void;
  isDeleting: boolean;
}) {
  const badge = typeBadgeStyle(ev.evidenceType);
  return (
    <div
      className="group relative"
      style={{ width: 88, height: 88, flexShrink: 0, opacity: isDeleting ? 0.4 : 1, transition: "opacity 0.2s" }}
    >
      {/* view button */}
      <button
        type="button"
        onClick={() => onView(ev)}
        disabled={isDeleting}
        className="relative w-full h-full rounded-xl overflow-hidden focus:outline-none focus:ring-2 focus:ring-brand-gold"
        style={{ background: CREAM2, border: `1px solid ${BORDER}` }}
        aria-label={`View ${ev.fileName}`}
        title={ev.fileName}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={ev.downloadUrl}
          alt={ev.fileName}
          className="w-full h-full object-contain"
          style={{ background: CREAM2 }}
        />
        {/* hover overlay */}
        <div
          className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ background: "rgba(21,21,21,0.45)" }}
          aria-hidden="true"
        >
          <ZoomIn className="h-5 w-5 text-white" />
        </div>
        {/* type badge */}
        <span
          className="absolute bottom-1 left-1 text-[9px] font-bold uppercase rounded px-1 py-0.5 leading-none"
          style={{ background: badge.bg, color: badge.color }}
        >
          {evidenceTypeLabel(ev.evidenceType)}
        </span>
      </button>

      {/* delete button — top-right corner, visible on group hover */}
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onDelete(ev); }}
        disabled={isDeleting}
        className="absolute -top-1.5 -right-1.5 z-10 h-5 w-5 rounded-full flex items-center justify-center shadow-sm opacity-0 group-hover:opacity-100 transition-opacity focus:opacity-100 focus:outline-none"
        style={{ background: "#EF4444", color: "white" }}
        aria-label={`Delete ${ev.fileName}`}
        title="Delete"
      >
        <X className="h-3 w-3" aria-hidden="true" />
      </button>
    </div>
  );
}

// ── Case card ──────────────────────────────────────────────────────────────
function CaseCard({
  group, onView, onDelete, deletingIds,
}: {
  group: CaseGroup;
  onView: (ev: EvidenceStorageRecord) => void;
  onDelete: (ev: EvidenceStorageRecord) => void;
  deletingIds: Set<string>;
}) {
  return (
    <div
      className="rounded-2xl p-5"
      style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
    >
      {/* header */}
      <div className="mb-4">
        <h2 className="text-base font-bold leading-snug mb-0.5" style={{ color: INK, fontFamily: NR }}>
          {group.caseTitle}
        </h2>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] font-mono rounded px-1.5 py-0.5"
            style={{ background: CREAM2, color: MUTED }}>
            {group.caseId}
          </span>
          <span className="text-xs" style={{ color: MUTED }}>
            · {group.items.length} file{group.items.length !== 1 ? "s" : ""}
          </span>
          <span className="text-xs" style={{ color: MUTED }}>
            · {formatDate(group.latestUpload)}
          </span>
        </div>
      </div>

      {/* thumbnails */}
      <div className="flex flex-wrap gap-3">
        {group.items.map((ev) => (
          <EvidenceThumb
            key={ev.id}
            ev={ev}
            onView={onView}
            onDelete={onDelete}
            isDeleting={deletingIds.has(ev.id)}
          />
        ))}
      </div>

      {/* file list */}
      <ul className="mt-4 space-y-1.5">
        {group.items.map((ev) => {
          const badge = typeBadgeStyle(ev.evidenceType);
          const deleting = deletingIds.has(ev.id);
          return (
            <li
              key={ev.id}
              className="flex items-center gap-2 text-xs rounded-lg px-2 py-1.5 transition-colors"
              style={{
                color: MUTED,
                opacity: deleting ? 0.4 : 1,
                background: deleting ? CREAM2 : "transparent",
              }}
            >
              <span className="text-[9px] font-bold uppercase rounded px-1.5 py-0.5"
                style={{ background: badge.bg, color: badge.color }}>
                {evidenceTypeLabel(ev.evidenceType)}
              </span>
              <span className="truncate font-medium" style={{ color: INK }}>{ev.fileName}</span>
              <span className="shrink-0">{formatBytes(ev.fileSize)}</span>
              <button
                type="button"
                onClick={() => onDelete(ev)}
                disabled={deleting}
                className="ml-auto shrink-0 h-6 w-6 rounded-lg flex items-center justify-center hover:bg-red-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors focus:outline-none"
                aria-label={`Delete ${ev.fileName}`}
                title="Delete file"
              >
                <Trash2 className="h-3.5 w-3.5" style={{ color: "#EF4444" }} aria-hidden="true" />
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────
export default function EvidenceHistoryPage() {
  const { user } = useAuth();
  const [evidence, setEvidence] = useState<EvidenceStorageRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewItem, setViewItem] = useState<EvidenceStorageRecord | null>(null);
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

  const load = () => {
    if (!user) return;
    setLoading(true);
    listUserEvidence(user.uid)
      .then(setEvidence)
      .catch(() => setEvidence([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [user]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDelete = async (ev: EvidenceStorageRecord) => {
    if (!user || deletingIds.has(ev.id)) return;
    setDeletingIds((prev) => new Set([...prev, ev.id]));
    // Close lightbox if viewing this item
    setViewItem((v) => (v?.id === ev.id ? null : v));
    try {
      await deleteEvidenceFile(user.uid, ev.id, ev.storageRef);
      setEvidence((prev) => prev.filter((e) => e.id !== ev.id));
    } catch {
      // Keep UI consistent — silently fail (the record may already be gone)
    } finally {
      setDeletingIds((prev) => { const next = new Set(prev); next.delete(ev.id); return next; });
    }
  };

  const cases = useMemo(() => groupByCaseId(evidence), [evidence]);

  return (
    <div className="px-5 py-8 max-w-3xl mx-auto" style={{ fontFamily: MNR }}>

      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <p className="text-[10px] font-bold tracking-[.24em] uppercase mb-2" style={{ color: GOLD }}>
            Self-Represented · Kenya
          </p>
          <h1 className="text-2xl font-bold" style={{ color: INK, fontFamily: NR }}>
            Evidence History
          </h1>
          <p className="text-sm mt-1" style={{ color: MUTED }}>
            All files you have saved to Firebase, grouped by case.
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          className="shrink-0 flex items-center gap-1.5 text-xs font-semibold rounded-xl px-3 py-2 transition-colors hover:bg-amber-50 disabled:opacity-40"
          style={{ color: GOLD_DK, border: `1px solid ${BORDER}`, background: SURFACE }}
          aria-label="Refresh evidence list"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} aria-hidden="true" />
          Refresh
        </button>
      </div>

      {/* ── Content ─────────────────────────────────────────────── */}
      {loading ? (
        <div className="space-y-4" aria-label="Loading evidence" aria-busy="true">
          <SkeletonCard /><SkeletonCard />
        </div>
      ) : cases.length === 0 ? (
        <div
          className="flex flex-col items-center justify-center py-20 rounded-2xl"
          style={{ background: SURFACE, border: `1px dashed ${BORDER}` }}
        >
          <div
            className="h-14 w-14 rounded-2xl flex items-center justify-center mb-4"
            style={{ background: CREAM2 }}
            aria-hidden="true"
          >
            <FolderOpen className="h-7 w-7" style={{ color: MUTED }} />
          </div>
          <p className="text-sm font-semibold mb-1" style={{ color: INK }}>
            No evidence saved yet
          </p>
          <p className="text-xs mb-5 text-center max-w-xs" style={{ color: MUTED }}>
            Upload and add evidence to a case in Evidence Analysis, then click
            &ldquo;Save to Evidence Library&rdquo;.
          </p>
          <Link
            href="/litigant/analysis"
            className="px-5 py-2.5 rounded-xl text-sm font-bold transition-all hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
            style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}
          >
            Go to Evidence Analysis
          </Link>
        </div>
      ) : (
        <div className="space-y-5">
          <p className="text-xs" style={{ color: MUTED }}>
            {cases.length} case{cases.length !== 1 ? "s" : ""} · {evidence.length} file{evidence.length !== 1 ? "s" : ""}
          </p>
          {cases.map((group) => (
            <CaseCard
              key={group.caseId}
              group={group}
              onView={setViewItem}
              onDelete={handleDelete}
              deletingIds={deletingIds}
            />
          ))}
        </div>
      )}

      {/* ── Lightbox ────────────────────────────────────────────── */}
      {viewItem && (
        <Lightbox
          url={viewItem.downloadUrl}
          name={viewItem.fileName}
          onClose={() => setViewItem(null)}
        />
      )}
    </div>
  );
}
