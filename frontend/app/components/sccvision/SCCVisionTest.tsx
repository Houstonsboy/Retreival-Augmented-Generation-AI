"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Upload, X, ScanText, Loader2, Copy, Check, AlertCircle, Download,
  Scale, ShieldCheck, FileWarning, ChevronDown, ChevronRight, Save,
  FolderOpen, CalendarDays, Award, Printer,
} from "lucide-react";
import { apiFetch } from "@/app/lib/api";
import type { CertificateRecord } from "@/app/lib/evidenceTypes";
import { DeviceQuestionnaire } from "./DeviceQuestionnaire";

// ── Design tokens ──────────────────────────────────────────────────────────
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const CREAM2  = "#F5F0E4";
const INK     = "#151515";
const MUTED   = "#6B6660";
const SURFACE = "#FFFDF8";

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
const CERT_THRESHOLD = 0.83;
const ACCEPTED = [".png", ".jpg", ".jpeg", ".webp"];
const SCHEMA_VERSION = 1;

// ── OCR / evidence types ───────────────────────────────────────────────────
type EvidenceType = "chat_screenshot" | "mpesa_screenshot" | "other";
interface CaseEvidence {
  evidence_id: string;
  type: EvidenceType;
  file_name: string;
  sha256: string | null;
  added_at: string;
  ocr: Record<string, unknown> | null;
  parsed: Record<string, unknown>[] | null;
  parse_layout?: string;
  raw_text?: string;
  user_edits: Record<string, string>;
}

interface UserFact {
  id: string;
  kind: "date" | "text" | "choice";
  fact: string;
  applies_to: string | null;
  answers_request_id: string | null;
  status: string;
  date_role?: string;
  confidence?: string;
}

interface CaseRequest {
  id: string;
  status: "open" | "answered" | "dismissed";
  last_seen_run: number;
}

interface CaseFile {
  schema_version: number;
  case_id: string;
  created_at: string;
  case_context: {
    user_role: string;
    user_account: string;
    analysis_date_override: string | null;
  };
  evidence: CaseEvidence[];
  user_provided_facts: UserFact[];
  rejected_facts: unknown[];
  inferred_facts: unknown[];
  requests: CaseRequest[];
  analyses: unknown[];
}

// ── Analyst response types ─────────────────────────────────────────────────
interface Evidence { msg_index: number | null; sender: string | null; time: string | null; quote: string; }
interface ElementResult {
  status: string;
  evidence: Evidence[];
  reasoning: string;
  confidence: number | null;
}
interface MpesaPayment {
  code: string | null;
  amount_kes: number | null;
  sender: string | null;
  recipient: string | null;
  datetime: string | null;
  evidence_id: string | null;
  basis: string[];
  classification: string;
  notes: string;
}
interface Party { name: string; role: string | null; notes: string | null; }
interface TimelineEntry { msg_index: number | null; evidence_id: string | null; sender: string | null; time: string | null; event: string; }
interface PartyQuestion { element: string; question: string; }
interface UnrelatedMessage { msg_index: number | null; text: string; }
interface FollowUpRequest {
  id: string;
  element: string;
  type: string;
  prompt: string;
  why_it_matters: string;
  blocking: boolean;
  target: string | null;
  options: string[];
  date_role: string | null;
}
interface AnalystResult {
  conversation_integrity: { rating: string; issues: string[] };
  unrelated_messages: UnrelatedMessage[];
  parties: Party[];
  timeline: TimelineEntry[];
  elements: Record<string, ElementResult>;
  mpesa_payments: MpesaPayment[];
  red_flags: string[];
  missing_information: string[];
  overall_verdict: string;
  confidence: number | null;
  changes_since_previous: string[];
  plain_summary: string;
  strengthen_checklist: string[];
  follow_up_requests: FollowUpRequest[];
  questions_for_parties: PartyQuestion[];
  disclaimer: string;
}
interface Verification {
  quotes_grounded: boolean;
  ungrounded_quotes: { element: string; msg_index: number | null; quote: string; reason: string }[];
  mpesa_cross_check: {
    llm_payment_count: number;
    parsed_payment_count: number;
    codes_llm_only: string[];
    codes_parsed_only: string[];
    reclassified: { code: string | null; amount_kes: number | null; before: string; after: string }[];
    till_mismatches: unknown[];
    amount_mismatches: unknown[];
    duplicate_codes: string[];
    consistent: boolean;
  };
  status_issues: { element: string; issue: string }[];
  premature_breach: { premature_breach_claim: boolean; notice: string } | null;
  verdict_before: string;
  verdict_after: string;
  verdict_downgraded: boolean;
  downgrade_reasons: string[];
}
interface AnalystResponse {
  analysis_input: unknown;
  result: AnalystResult;
  verification: Verification;
  prompt_version: string;
  model: string;
  analysis_id?: string | null;
}

const ELEMENT_LABELS: Record<string, string> = {
  A: "Parties & identity", B: "Capacity", C: "Offer", D: "Acceptance",
  E: "Consideration", F: "Intention", G: "Certainty", H: "Genuine consent",
  I: "Legality", J: "Formalities", K: "Performance & breach",
};

function verdictTone(v: string): { bg: string; fg: string; border: string } {
  if (v === "LIKELY VALID") return { bg: "#F0FDF4", fg: "#15803D", border: "#BBF7D0" };
  if (v === "POSSIBLY VALID WITH GAPS") return { bg: "#FFFBEB", fg: "#92400E", border: "#FDE68A" };
  if (v === "LIKELY NOT VALID") return { bg: "#FEF2F2", fg: "#B91C1C", border: "#FECACA" };
  return { bg: "#F8FAFC", fg: "#475569", border: "#E2E8F0" };
}

function statusTone(s: string): { bg: string; fg: string } {
  switch (s) {
    case "PRESENT": return { bg: "#F0FDF4", fg: "#15803D" };
    case "PARTIAL": return { bg: "#FFFBEB", fg: "#92400E" };
    case "ABSENT": return { bg: "#FEF2F2", fg: "#B91C1C" };
    case "NO_ISSUE_INDICATED": return { bg: "#ECFEFF", fg: "#0E7490" };
    case "NOT_REQUIRED": return { bg: "#F1F5F9", fg: "#475569" };
    default: return { bg: "#F1F5F9", fg: "#475569" };
  }
}

function isCertEligible(a: AnalystResponse): boolean {
  const v = a.verification.verdict_after ?? a.result.overall_verdict;
  return (
    a.result.confidence != null &&
    a.result.confidence >= CERT_THRESHOLD &&
    v !== "LIKELY NOT VALID" &&
    v !== "INSUFFICIENT EVIDENCE"
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isAccepted(name: string): boolean {
  const lower = name.toLowerCase();
  return ACCEPTED.some((ext) => lower.endsWith(ext));
}

function todayISO(): string {
  const d = new Date();
  const local = new Date(d.getTime() - d.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

const UPLOAD_TYPES = new Set([
  "upload_mpesa_screenshot",
  "upload_chat_screenshot",
  "upload_other_evidence",
]);

function isUploadType(type: string): boolean { return UPLOAD_TYPES.has(type); }

function uploadTypeToClassification(type: string): EvidenceType {
  if (type === "upload_mpesa_screenshot") return "mpesa_screenshot";
  if (type === "upload_chat_screenshot")  return "chat_screenshot";
  return "other";
}

function uploadTypeLabel(type: string): string {
  if (type === "upload_mpesa_screenshot") return "M-Pesa screenshot";
  if (type === "upload_chat_screenshot")  return "chat screenshot";
  return "image or document";
}

function newCaseFile(): CaseFile {
  return {
    schema_version: SCHEMA_VERSION,
    case_id: `case-${Date.now().toString(36)}`,
    created_at: new Date().toISOString(),
    case_context: { user_role: "unknown", user_account: "", analysis_date_override: null },
    evidence: [],
    user_provided_facts: [],
    rejected_facts: [],
    inferred_facts: [],
    requests: [],
    analyses: [],
  };
}

export function SCCVisionTest() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [labels, setLabels] = useState<(EvidenceType | "")[]>([]);
  const [invert, setInvert] = useState(true);
  const [upscale, setUpscale] = useState(true);

  const [caseFile, setCaseFile] = useState<CaseFile>(() => newCaseFile());
  const [consent, setConsent] = useState(false);

  const [ocrLoading, setOcrLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<AnalystResponse | null>(null);
  const [copiedAnalysis, setCopiedAnalysis] = useState(false);
  const [runCount, setRunCount] = useState(0);

  const [showCertificate, setShowCertificate] = useState(false);
  const [showQuestionnaire, setShowQuestionnaire] = useState(false);
  const [certResult, setCertResult] = useState<CertificateRecord | null>(null);

  const [answerTarget, setAnswerTarget] = useState<FollowUpRequest | null>(null);
  const [answerText, setAnswerText] = useState("");
  const [answerDate, setAnswerDate] = useState("");
  const answerUploadRef = useRef<HTMLInputElement>(null);
  const [answerUploadFile, setAnswerUploadFile] = useState<File | null>(null);
  const [answerUploadPreview, setAnswerUploadPreview] = useState("");
  const [answerUploadLoading, setAnswerUploadLoading] = useState(false);

  // Persist the case file locally (client-side "memory" only).
  useEffect(() => {
    try {
      localStorage.setItem("sccvision-case", JSON.stringify(caseFile));
    } catch { /* storage unavailable */ }
  }, [caseFile]);

  // Inject print-only CSS when the certificate modal is open so window.print()
  // renders only the certificate and nothing else on the page.
  useEffect(() => {
    if (!showCertificate) return;
    const el = document.createElement("style");
    el.id = "scc-print-style";
    el.textContent = [
      "@media print{",
      "body *{visibility:hidden!important}",
      "#scc-certificate,#scc-certificate *{visibility:visible!important}",
      "#scc-cert-toolbar{display:none!important}",
      "#scc-certificate{position:fixed!important;inset:0!important;",
      "padding:40px!important;border:none!important;",
      "box-shadow:none!important;border-radius:0!important;",
      "background:white!important;overflow:visible!important;z-index:99999!important}}",
    ].join("");
    document.head.appendChild(el);
    return () => document.getElementById("scc-print-style")?.remove();
  }, [showCertificate]);

  const asOf = caseFile.case_context.analysis_date_override || todayISO();

  const pickFiles = useCallback((incoming: FileList | File[]) => {
    setError(null);
    const arr = Array.from(incoming);
    const valid: File[] = [];
    for (const f of arr) {
      if (!isAccepted(f.name)) {
        setError(`Unsupported file type for "${f.name}". Allowed: ${ACCEPTED.join(", ")}.`);
        return;
      }
      if (f.size > MAX_BYTES) {
        setError(`File too large (${formatBytes(f.size)}). Maximum is 10 MB.`);
        return;
      }
      valid.push(f);
    }
    previewUrls.forEach((u) => URL.revokeObjectURL(u));
    const urls = valid.map((f) => URL.createObjectURL(f));
    setFiles(valid);
    setPreviewUrls(urls);
    setLabels(valid.map(() => ""));
  }, [previewUrls]);

  const removeFile = useCallback((i: number) => {
    setFiles((prev) => prev.filter((_, idx) => idx !== i));
    setPreviewUrls((prev) => {
      URL.revokeObjectURL(prev[i]);
      return prev.filter((_, idx) => idx !== i);
    });
    setLabels((prev) => prev.filter((_, idx) => idx !== i));
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length) pickFiles(e.dataTransfer.files);
  }, [pickFiles]);

  // ── Add evidence: OCR each file, classify/normalize, push to case file ──
  const handleAddEvidence = async () => {
    if (!files.length || ocrLoading) return;
    setOcrLoading(true);
    setError(null);

    try {
      for (const [i, f] of files.entries()) {
        const fd = new FormData();
        fd.append("file", f);
        const extractRes = await apiFetch(`/sccvision/extract?invert=${invert}&upscale=${upscale}`, {
          method: "POST", body: fd,
        });
        const ocr = await extractRes.json().catch(() => null);
        if (!extractRes.ok) throw new Error(ocr?.error || `OCR failed (${extractRes.status})`);

        const hash = await sha256Hex(await f.arrayBuffer());
        const procRes = await apiFetch("/sccvision/evidence/process", {
          method: "POST",
          body: JSON.stringify({
            ocr_result: ocr,
            classification: labels[i] || "auto",
            file_name: f.name,
            sha256: hash,
          }),
        });
        const proc = await procRes.json().catch(() => null);
        if (!procRes.ok) throw new Error(proc?.error || `Evidence processing failed (${procRes.status})`);

        setCaseFile((prev) => ({ ...prev, evidence: [...prev.evidence, proc.evidence as CaseEvidence] }));
      }
      setFiles([]);
      previewUrls.forEach((u) => URL.revokeObjectURL(u));
      setPreviewUrls([]);
      setLabels([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the backend. Is it running on port 5000?");
    } finally {
      setOcrLoading(false);
    }
  };

  const removeEvidence = useCallback((id: string) => {
    setCaseFile((prev) => ({ ...prev, evidence: prev.evidence.filter((e) => e.evidence_id !== id) }));
    setAnalysis(null);
  }, []);

  // ── Run the analysis (never automatic) ───────────────────────────────────
  const runAnalysis = async () => {
    if (!caseFile.evidence.length || analyzing) return;
    if (!consent) {
      setError("Please confirm consent to send the case for analysis.");
      return;
    }
    setAnalyzing(true);
    setError(null);

    try {
      const res = await apiFetch("/sccvision/case/analyze", {
        method: "POST",
        body: JSON.stringify(caseFile),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok) {
        if (res.status === 422 && body?.rejected) {
          setCaseFile((prev) => ({ ...prev, rejected_facts: [...prev.rejected_facts, ...body.rejected] }));
          throw new Error("Some dates were in the future and were rejected.");
        }
        if (res.status === 429) {
          const wait = body?.retry_after_seconds ? ` Try again in ~${body.retry_after_seconds}s.` : "";
          throw new Error(`${body?.error || "Rate limit reached"}.${wait}`);
        }
        throw new Error(body?.error || `Analysis failed (${res.status})`);
      }
      setAnalysis(body as AnalystResponse);
      setRunCount((n) => n + 1);
      const openRequests: CaseRequest[] = (body.result.follow_up_requests as FollowUpRequest[])
        .filter((r) => !caseFile.requests.some((x) => x.id === r.id && x.status !== "open"))
        .map((r) => ({ id: r.id, status: "open" as const, last_seen_run: runCount + 1 }));
      // Strip server-only fields before storing in caseFile to prevent the
      // accumulated analyses from bloating the payload sent on the next run.
      const { analysis_input: _ai, verification: _v, ...slimBody } = body;
      setCaseFile((prev) => ({
        ...prev,
        requests: [...prev.requests.filter((x) => x.status !== "open"), ...openRequests],
        analyses: [...prev.analyses, slimBody],
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  // ── Follow-up answering ──────────────────────────────────────────────────
  const dismissRequest = (req: FollowUpRequest) => {
    setCaseFile((prev) => ({
      ...prev,
      requests: prev.requests
        .filter((r) => r.id !== req.id)
        .concat({ id: req.id, status: "dismissed", last_seen_run: runCount }),
    }));
  };

  const closeAnswerModal = () => {
    setAnswerTarget(null);
    setAnswerText("");
    setAnswerDate("");
    if (answerUploadPreview) URL.revokeObjectURL(answerUploadPreview);
    setAnswerUploadFile(null);
    setAnswerUploadPreview("");
  };

  const submitAnswer = async () => {
    if (!answerTarget) return;
    const req = answerTarget;

    // ── Upload path ────────────────────────────────────────────────────────
    if (isUploadType(req.type)) {
      if (!answerUploadFile) return;
      setAnswerUploadLoading(true);
      setError(null);
      try {
        const fd = new FormData();
        fd.append("file", answerUploadFile);
        const extractRes = await apiFetch(`/sccvision/extract?invert=${invert}&upscale=${upscale}`, {
          method: "POST", body: fd,
        });
        const ocr = await extractRes.json().catch(() => null);
        if (!extractRes.ok) throw new Error(ocr?.error || `OCR failed (${extractRes.status})`);

        const hash = await sha256Hex(await answerUploadFile.arrayBuffer());
        const classification = uploadTypeToClassification(req.type);
        const procRes = await apiFetch("/sccvision/evidence/process", {
          method: "POST",
          body: JSON.stringify({
            ocr_result: ocr,
            classification,
            file_name: answerUploadFile.name,
            sha256: hash,
          }),
        });
        const proc = await procRes.json().catch(() => null);
        if (!procRes.ok) throw new Error(proc?.error || `Processing failed (${procRes.status})`);

        const ev = proc.evidence as CaseEvidence;
        setCaseFile((prev) => ({
          ...prev,
          evidence: [...prev.evidence, ev],
          user_provided_facts: [
            ...prev.user_provided_facts,
            {
              id: `fact-${Date.now().toString(36)}`,
              kind: "text" as const,
              fact: ev.evidence_id,
              applies_to: req.target,
              answers_request_id: req.id,
              status: "user-stated",
            },
          ],
          requests: prev.requests
            .filter((r) => r.id !== req.id)
            .concat({ id: req.id, status: "answered", last_seen_run: runCount }),
        }));
        closeAnswerModal();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
      } finally {
        setAnswerUploadLoading(false);
      }
      return;
    }

    // ── Date / text path ───────────────────────────────────────────────────
    let fact: UserFact;

    if (req.type === "provide_date" || req.date_role) {
      const d = answerDate;
      if (!d) return;
      const role = req.date_role || "other";
      if ((role === "conversation" || role === "payment" || role === "delivery") && d > asOf) {
        setError(`A ${role} date cannot be in the future (as-of ${asOf}).`);
        setCaseFile((prev) => ({
          ...prev,
          rejected_facts: [...prev.rejected_facts, { id: `rej-${Date.now()}`, fact: d, reason: `future ${role} date` }],
        }));
        closeAnswerModal();
        return;
      }
      fact = {
        id: `fact-${Date.now().toString(36)}`,
        kind: "date",
        fact: d,
        applies_to: req.target,
        answers_request_id: req.id,
        status: "user-stated",
        date_role: role,
        confidence: "sure",
      };
    } else {
      if (!answerText.trim()) return;
      fact = {
        id: `fact-${Date.now().toString(36)}`,
        kind: req.type === "confirm_choice" ? "choice" : "text",
        fact: answerText.trim(),
        applies_to: req.target,
        answers_request_id: req.id,
        status: "user-stated",
      };
    }

    setCaseFile((prev) => ({
      ...prev,
      user_provided_facts: [...prev.user_provided_facts, fact],
      requests: prev.requests
        .filter((r) => r.id !== req.id)
        .concat({ id: req.id, status: "answered", last_seen_run: runCount }),
    }));
    closeAnswerModal();
  };

  const saveCaseFile = () => {
    const blob = new Blob([JSON.stringify(caseFile, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${caseFile.case_id}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const loadCaseFile = (file: File) => {
    file.text().then((text) => {
      try {
        const parsed = JSON.parse(text) as CaseFile;
        if (!parsed.evidence) throw new Error("Not a case file");
        setCaseFile(parsed);
        setAnalysis(null);
        setRunCount(parsed.analyses?.length ?? 0);
      } catch {
        setError("Could not read that case file.");
      }
    });
  };

  const copyAnalysis = async () => {
    if (!analysis) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(analysis, null, 2));
      setCopiedAnalysis(true);
      setTimeout(() => setCopiedAnalysis(false), 1500);
    } catch { setCopiedAnalysis(false); }
  };

  const downloadAnalysis = () => {
    if (!analysis) return;
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${caseFile.case_id}-analysis.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const effectiveVerdict = analysis?.verification.verdict_after ?? analysis?.result.overall_verdict ?? "";
  const verdict = verdictTone(effectiveVerdict);

  const openFollowUps = analysis?.result.follow_up_requests ?? [];

  return (
    <section
      aria-labelledby="sccvision-heading"
      className="mt-12 rounded-2xl p-5 sm:p-6"
      style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
    >
      <div className="flex items-center gap-2 mb-1 flex-wrap">
        <ScanText className="h-4 w-4" style={{ color: GOLD }} aria-hidden="true" />
        <h2 id="sccvision-heading" className="text-base font-bold" style={{ color: INK }}>
          SCCVision Contract Evidence Analyst
        </h2>
        <span className="ml-auto text-[11px] font-mono" style={{ color: MUTED }}>{caseFile.case_id}</span>
      </div>
      <p className="text-xs leading-relaxed mb-5" style={{ color: MUTED }}>
        Build a case from WhatsApp and M-Pesa screenshots, answer the analyst&apos;s
        questions, then run a contract-validity analysis against Kenyan contract
        law. The analysis is automated and is not legal advice.
      </p>

      {/* ── Case bar ──────────────────────────────────────────────── */}
      <div className="rounded-xl p-4 mb-4 space-y-4" style={{ background: CREAM2, border: `1px solid ${BORDER}` }}>

        {/* Row 1: Role + Date */}
        <div className="grid sm:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-semibold mb-1.5" style={{ color: MUTED }}>
              Your role in this case
            </label>
            <select value={caseFile.case_context.user_role}
              onChange={(e) => setCaseFile((p) => ({ ...p, case_context: { ...p.case_context, user_role: e.target.value } }))}
              className="w-full rounded-xl px-3 py-2.5 text-sm focus:outline-none appearance-none"
              style={{ background: SURFACE, border: `1px solid ${BORDER}`, color: INK }}>
              <option value="unknown">Not sure / not stated</option>
              <option value="claimant">Claimant (I am suing)</option>
              <option value="respondent">Respondent (I am being sued)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold mb-1.5" style={{ color: MUTED }}>
              Analysis date
              <span className="font-normal ml-1" style={{ color: `${MUTED}99` }}>· defaults to today</span>
            </label>
            <input type="date" value={caseFile.case_context.analysis_date_override ?? ""}
              onChange={(e) => setCaseFile((p) => ({ ...p, case_context: { ...p.case_context, analysis_date_override: e.target.value || null } }))}
              className="w-full rounded-xl px-3 py-2.5 text-sm focus:outline-none"
              style={{ background: SURFACE, border: `1px solid ${BORDER}`, color: INK }} />
          </div>
        </div>

        {/* Row 2: Account (full width) */}
        <div>
          <div className="flex items-baseline justify-between mb-1.5">
            <label className="text-xs font-semibold" style={{ color: MUTED }}>
              Your account of the matter
            </label>
            <span className="text-[10px]" style={{ color: `${MUTED}99` }}>
              unverified · used only to guide the analysis
            </span>
          </div>
          <textarea
            value={caseFile.case_context.user_account}
            onChange={(e) => {
              const el = e.target;
              el.style.height = "auto";
              el.style.height = `${el.scrollHeight}px`;
              setCaseFile((p) => ({ ...p, case_context: { ...p.case_context, user_account: e.target.value } }));
            }}
            rows={4}
            placeholder={
              "Describe what happened in your own words — when you agreed, what was paid or promised, what went wrong, and what outcome you are seeking.\n\nThe analyst will not treat this as proven fact, but uses it to understand your perspective and ask better follow-up questions."
            }
            className="w-full rounded-xl px-3 py-2.5 text-sm focus:outline-none leading-relaxed"
            style={{
              background: SURFACE,
              border: `1px solid ${BORDER}`,
              color: INK,
              resize: "vertical",
              minHeight: "5.5rem",
              overflow: "hidden",
            }}
          />
        </div>

        {/* Row 3: Save / Load / Consent */}
        <div className="flex items-center gap-3 flex-wrap pt-1" style={{ borderTop: `1px solid ${BORDER}` }}>
          <button type="button" onClick={saveCaseFile}
            className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 hover:bg-amber-50 transition-colors"
            style={{ color: GOLD_DK, border: `1px solid ${BORDER}`, background: SURFACE }}>
            <Save className="h-3.5 w-3.5" aria-hidden="true" /> Save case file
          </button>
          <label className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 cursor-pointer hover:bg-amber-50 transition-colors"
            style={{ color: GOLD_DK, border: `1px solid ${BORDER}`, background: SURFACE }}>
            <FolderOpen className="h-3.5 w-3.5" aria-hidden="true" /> Load case file
            <input type="file" accept="application/json" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) loadCaseFile(f); e.target.value = ""; }} />
          </label>
          <label className="flex items-center gap-2 text-xs font-medium cursor-pointer select-none ml-auto" style={{ color: MUTED }}>
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)}
              className="rounded" style={{ accentColor: GOLD }} />
            I consent to send this case to the automated analyst.
          </label>
        </div>
      </div>

      {/* ── Drop zone ──────────────────────────────────────────────── */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload screenshots — click or drag and drop"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed transition-all py-8 px-6 cursor-pointer focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
        style={{ borderColor: isDragging ? GOLD : BORDER, background: isDragging ? "#F6E7B833" : CREAM2 }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED.join(",")}
          className="hidden"
          onChange={(e) => { if (e.target.files?.length) pickFiles(e.target.files); e.target.value = ""; }}
          aria-hidden="true"
        />
        <div className="h-11 w-11 rounded-full flex items-center justify-center mb-3" style={{ background: `${GOLD}18` }} aria-hidden="true">
          <Upload className="h-5 w-5" style={{ color: GOLD }} />
        </div>
        <p className="text-sm font-semibold text-center" style={{ color: INK }}>
          Drop WhatsApp or M-Pesa screenshots here
        </p>
        <p className="text-xs mt-1.5 text-center" style={{ color: MUTED }}>
          PNG, JPG, WEBP · multiple allowed · max {formatBytes(MAX_BYTES)} each
        </p>
      </div>

      {error && (
        <div className="mt-3 flex items-start gap-2 rounded-xl px-4 py-3 text-xs"
          style={{ background: "#FEF2F2", border: "1px solid #FECACA", color: "#B91C1C" }} role="alert">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {/* ── Pending files ─────────────────────────────────────────── */}
      {files.length > 0 && (
        <div className="mt-5 space-y-4">
          <div className="flex flex-wrap gap-3">
            {files.map((f, i) => (
              <div key={i} className="w-32 shrink-0 rounded-xl overflow-hidden" style={{ border: `1px solid ${BORDER}`, background: SURFACE }}>
                <div className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={previewUrls[i]} alt={f.name} className="w-full h-auto" />
                  <button type="button" onClick={() => removeFile(i)}
                    className="absolute top-1 right-1 h-6 w-6 rounded-lg flex items-center justify-center bg-black/55 hover:bg-black/75 transition-colors"
                    aria-label={`Remove ${f.name}`}>
                    <X className="h-3.5 w-3.5 text-white" aria-hidden="true" />
                  </button>
                </div>
                <div className="px-2 pb-2 pt-1.5">
                  <label className="block">
                    <span className="block text-[10px] font-semibold mb-1" style={{ color: MUTED }}>
                      Label this image
                    </span>
                    <select
                      value={labels[i] ?? ""}
                      onChange={(e) => setLabels((prev) => {
                        const next = [...prev];
                        next[i] = e.target.value as EvidenceType | "";
                        return next;
                      })}
                      className="w-full rounded-lg px-1.5 py-1 text-xs focus:outline-none"
                      style={{ background: CREAM2, border: `1px solid ${BORDER}`, color: INK }}>
                      <option value="">Auto-detect</option>
                      <option value="chat_screenshot">WhatsApp / SMS chat</option>
                      <option value="mpesa_screenshot">M-Pesa screenshot</option>
                      <option value="other">Other</option>
                    </select>
                  </label>
                </div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-3 items-center">
            <label className="flex items-center gap-2 text-sm font-medium cursor-pointer select-none" style={{ color: INK }}>
              <input type="checkbox" checked={invert} onChange={(e) => setInvert(e.target.checked)}
                className="rounded" style={{ accentColor: GOLD }} />
              Invert dark mode
            </label>
            <label className="flex items-center gap-2 text-sm font-medium cursor-pointer select-none" style={{ color: INK }}>
              <input type="checkbox" checked={upscale} onChange={(e) => setUpscale(e.target.checked)}
                className="rounded" style={{ accentColor: GOLD }} />
              Upscale small images
            </label>
            <button type="button" onClick={handleAddEvidence} disabled={ocrLoading}
              className="flex items-center justify-center gap-2 rounded-xl px-5 py-2.5 text-sm font-bold transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}>
              {ocrLoading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <ScanText className="h-4 w-4" aria-hidden="true" />}
              {ocrLoading ? "Adding…" : "Add to case"}
            </button>
          </div>
        </div>
      )}

      {/* ── Evidence list ─────────────────────────────────────────── */}
      {caseFile.evidence.length > 0 && (
        <div className="mt-5">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-bold tracking-[.14em] uppercase" style={{ color: MUTED }}>
              Evidence ({caseFile.evidence.length})
            </p>
            <button type="button" onClick={runAnalysis} disabled={analyzing}
              className="flex items-center justify-center gap-2 rounded-xl px-5 py-2.5 text-sm font-bold transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}>
              {analyzing ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Scale className="h-4 w-4" aria-hidden="true" />}
              {analyzing ? "Analyzing…" : runCount > 0 ? "Re-run analysis" : "Run analysis"}
            </button>
          </div>
          <div className="space-y-2">
            {caseFile.evidence.map((ev) => (
              <div key={ev.evidence_id} className="flex items-center gap-3 rounded-xl px-3 py-2" style={{ background: CREAM2, border: `1px solid ${BORDER}` }}>
                <span className="text-[10px] font-bold uppercase rounded-full px-2 py-0.5"
                  style={{ background: ev.type === "mpesa_screenshot" ? "#DCFCE7" : ev.type === "chat_screenshot" ? "#DBEAFE" : "#F1F5F9",
                           color: ev.type === "mpesa_screenshot" ? "#166534" : ev.type === "chat_screenshot" ? "#1E40AF" : "#475569" }}>
                  {ev.type}
                </span>
                <span className="text-sm font-medium truncate" style={{ color: INK }}>{ev.file_name || ev.evidence_id}</span>
                {ev.parse_layout === "unsupported_layout" && (
                  <span className="text-[10px] font-semibold rounded-full px-2 py-0.5" style={{ background: "#FEF2F2", color: "#B91C1C" }}>manual entry needed</span>
                )}
                <button type="button" onClick={() => removeEvidence(ev.evidence_id)}
                  className="ml-auto h-6 w-6 rounded-lg flex items-center justify-center hover:bg-black/5" aria-label="Remove evidence">
                  <X className="h-3.5 w-3.5" style={{ color: MUTED }} aria-hidden="true" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Follow-up requests ────────────────────────────────────── */}
      {openFollowUps.length > 0 && (
        <div className="mt-6">
          <p className="text-xs font-bold tracking-[.14em] uppercase mb-2" style={{ color: MUTED }}>
            The analyst asks for ({openFollowUps.length})
          </p>
          <div className="space-y-2">
            {openFollowUps.map((req) => (
              <div key={req.id} className="rounded-xl p-3" style={{ background: "#FFFBEB", border: `1px solid #FDE68A` }}>
                <div className="flex items-start gap-2">
                  {req.blocking && <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" style={{ color: "#92400E" }} aria-hidden="true" />}
                  <div className="flex-1">
                    <p className="text-sm font-semibold" style={{ color: INK }}>{req.prompt}</p>
                    {req.why_it_matters && <p className="text-xs mt-0.5" style={{ color: MUTED }}>{req.why_it_matters}</p>}
                  </div>
                </div>
                <div className="flex items-center gap-2 mt-2 flex-wrap">
                  <button type="button" onClick={() => { setAnswerTarget(req); setAnswerDate(todayISO()); setAnswerText(""); }}
                    className="text-xs font-semibold rounded-lg px-3 py-1.5" style={{ background: GOLD_DK, color: "white" }}>
                    Answer
                  </button>
                  <button type="button" onClick={() => dismissRequest(req)}
                    className="text-xs font-semibold rounded-lg px-3 py-1.5" style={{ color: MUTED }}>
                    Dismiss
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Answer modal ──────────────────────────────────────────── */}
      {answerTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "#00000055" }} role="dialog" aria-modal="true">
          <div className="w-full max-w-md rounded-2xl p-5" style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
            <p className="text-sm font-bold mb-1" style={{ color: INK }}>{answerTarget.prompt}</p>
            <p className="text-xs mb-3" style={{ color: MUTED }}>{answerTarget.why_it_matters}</p>

            {answerTarget.options?.length > 0 ? (
              /* ── Choice ── */
              <div className="space-y-2">
                {answerTarget.options.map((opt) => (
                  <button key={opt} type="button" onClick={() => { setAnswerText(opt); }}
                    className="w-full text-left text-sm rounded-xl px-3 py-2 transition-colors"
                    style={{
                      background: answerText === opt ? `${GOLD}22` : CREAM2,
                      border: `1px solid ${answerText === opt ? GOLD : BORDER}`,
                      color: INK,
                    }}>
                    {opt}
                  </button>
                ))}
              </div>
            ) : isUploadType(answerTarget.type) ? (
              /* ── Image upload ── */
              <div>
                <input
                  ref={answerUploadRef}
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp"
                  className="sr-only"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    if (!isAccepted(f.name)) { setError(`Unsupported type. Allowed: .png .jpg .jpeg .webp`); return; }
                    if (f.size > MAX_BYTES) { setError(`File too large (max 10 MB).`); return; }
                    if (answerUploadPreview) URL.revokeObjectURL(answerUploadPreview);
                    setAnswerUploadFile(f);
                    setAnswerUploadPreview(URL.createObjectURL(f));
                  }}
                />
                {answerUploadFile ? (
                  <div className="relative rounded-xl overflow-hidden" style={{ border: `1px solid ${BORDER}` }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={answerUploadPreview} alt="Preview" className="w-full max-h-48 object-contain" style={{ background: CREAM2 }} />
                    <div className="flex items-center justify-between gap-2 px-3 py-2" style={{ background: SURFACE, borderTop: `1px solid ${BORDER}` }}>
                      <span className="text-xs truncate" style={{ color: MUTED }}>{answerUploadFile.name}</span>
                      <button type="button" onClick={() => { setAnswerUploadFile(null); setAnswerUploadPreview(""); }}
                        className="shrink-0 rounded-full p-1 hover:bg-gray-100">
                        <X className="h-3.5 w-3.5" style={{ color: MUTED }} />
                      </button>
                    </div>
                  </div>
                ) : (
                  <button type="button" onClick={() => answerUploadRef.current?.click()}
                    className="w-full flex flex-col items-center gap-2 rounded-xl py-6 transition-colors hover:brightness-95"
                    style={{ background: CREAM2, border: `2px dashed ${BORDER}` }}>
                    <Upload className="h-6 w-6" style={{ color: GOLD }} />
                    <span className="text-sm font-semibold" style={{ color: INK }}>
                      Tap to upload {uploadTypeLabel(answerTarget.type)}
                    </span>
                    <span className="text-xs" style={{ color: MUTED }}>PNG, JPG, WEBP · max 10 MB</span>
                  </button>
                )}
              </div>
            ) : answerTarget.type === "provide_date" ? (
              /* ── Date (only when explicitly provide_date, NOT date_role on upload types) ── */
              <input type="date" value={answerDate} onChange={(e) => setAnswerDate(e.target.value)}
                className="w-full rounded-xl px-3 py-2 text-sm focus:outline-none" style={{ border: `1px solid ${BORDER}`, color: INK }} />
            ) : (
              /* ── Free text ── */
              <textarea value={answerText} onChange={(e) => setAnswerText(e.target.value)} rows={3}
                className="w-full rounded-xl px-3 py-2 text-sm focus:outline-none resize-none"
                style={{ background: SURFACE, border: `1px solid ${BORDER}`, color: INK }}
                placeholder="Type your answer…" />
            )}

            <div className="flex justify-end gap-2 mt-4">
              <button type="button" onClick={closeAnswerModal}
                className="text-xs font-semibold rounded-lg px-3 py-1.5" style={{ color: MUTED }}>
                Cancel
              </button>
              <button type="button" onClick={submitAnswer}
                disabled={answerUploadLoading}
                className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 disabled:opacity-60"
                style={{ background: GOLD_DK, color: "white" }}>
                {answerUploadLoading && <Loader2 className="h-3 w-3 animate-spin" />}
                {answerUploadLoading ? "Processing…" : "Submit"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Certificate modal ─────────────────────────────────────── */}
      {showCertificate && analysis && (
        <CertificateModal
          analysis={analysis}
          caseFile={caseFile}
          asOf={asOf}
          onClose={() => setShowCertificate(false)}
        />
      )}

      {/* ── Section 106B device questionnaire ─────────────────────── */}
      {showQuestionnaire && analysis?.analysis_id && (
        <DeviceQuestionnaire
          analysisId={analysis.analysis_id}
          onClose={() => setShowQuestionnaire(false)}
          onSuccess={(rec) => {
            setCertResult(rec);
            setShowQuestionnaire(false);
          }}
        />
      )}

      {/* ── Analysis results ───────────────────────────────────────── */}
      {analysis && (
        <div className="mt-6 space-y-5">
          {/* Verdict banner */}
          <div className="rounded-2xl px-5 py-4" style={{ background: verdict.bg, border: `1px solid ${verdict.border}` }}>
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-5 w-5" style={{ color: verdict.fg }} aria-hidden="true" />
              <p className="text-base font-bold" style={{ color: verdict.fg }}>{effectiveVerdict}</p>
              {analysis.result.confidence != null && (
                <span className="text-xs font-semibold rounded-full px-2 py-0.5" style={{ background: verdict.fg, color: "#fff" }}>
                  {(analysis.result.confidence * 100).toFixed(0)}% confidence
                </span>
              )}
            </div>
            {analysis.verification.verdict_downgraded && (
              <p className="text-xs mt-1.5" style={{ color: verdict.fg }}>
                Downgraded from {analysis.verification.verdict_before} by verification:
                {analysis.verification.downgrade_reasons.map((r) => ` ${r}`).join(",")}.
              </p>
            )}
            {analysis.result.plain_summary && (
              <p className="text-sm mt-2 leading-relaxed" style={{ color: INK }}>{analysis.result.plain_summary}</p>
            )}
            {analysis.result.changes_since_previous?.length > 0 && (
              <div className="mt-2">
                <p className="text-[10px] font-bold uppercase tracking-wide" style={{ color: MUTED }}>What changed since last run</p>
                <ul className="list-disc pl-4 space-y-0.5 text-xs mt-1" style={{ color: INK }}>
                  {analysis.result.changes_since_previous.map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              </div>
            )}
            {isCertEligible(analysis) && (
              <div className="mt-3 rounded-xl px-3 py-2.5"
                style={{ background: "#F0FDF4", border: "1px solid #BBF7D0" }}>
                <div className="flex items-center gap-3 flex-wrap">
                  <Award className="h-4 w-4 shrink-0" style={{ color: "#15803D" }} aria-hidden="true" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-bold" style={{ color: "#15803D" }}>
                      Evidence meets the {Math.round(CERT_THRESHOLD * 100)}% threshold
                    </p>
                    <p className="text-[11px]" style={{ color: "#166534" }}>
                      Print a visual summary or prepare a signed Section 106B certificate for court.
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2 shrink-0">
                    <button type="button" onClick={() => setShowCertificate(true)}
                      className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 hover:opacity-90 transition-opacity"
                      style={{ background: "#15803D", color: "white" }}>
                      <Printer className="h-3.5 w-3.5" aria-hidden="true" />
                      Print Summary
                    </button>
                    {analysis.analysis_id && !certResult && (
                      <button type="button" onClick={() => setShowQuestionnaire(true)}
                        className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 hover:opacity-90 transition-opacity"
                        style={{ background: GOLD_DK, color: "white" }}>
                        <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
                        Prepare s.106B Certificate
                      </button>
                    )}
                  </div>
                </div>
                {certResult && (
                  <div className="mt-3 rounded-lg px-3 py-2 flex items-center gap-2 text-xs"
                    style={{ background: "#ECFDF5", border: "1px solid #6EE7B7", color: "#065F46" }}>
                    <ShieldCheck className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    <span>
                      <strong>{certResult.certificate_number}</strong> created —{" "}
                      {certResult.document_type === "CERTIFICATE"
                        ? "certificate draft ready for signing."
                        : "deficiency report generated."}{" "}
                      View and download it in{" "}
                      <a href="/litigant/certificates" className="underline font-semibold">
                        Certificate History
                      </a>.
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Record integrity */}
          <InfoCard
            title="Record integrity"
            icon={<ShieldCheck className="h-4 w-4" aria-hidden="true" />}
            badge={{ text: analysis.result.conversation_integrity.rating, ...statusTone(analysis.result.conversation_integrity.rating) }}
          >
            {analysis.result.conversation_integrity.issues?.length > 0 && (
              <ul className="list-disc pl-4 space-y-0.5 text-xs" style={{ color: MUTED }}>
                {analysis.result.conversation_integrity.issues.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            )}
            {analysis.verification.mpesa_cross_check && !analysis.verification.mpesa_cross_check.consistent && (
              <p className="text-xs mt-2" style={{ color: "#B45309" }}>
                M-Pesa cross-check discrepancy: {analysis.verification.mpesa_cross_check.codes_llm_only.length} code(s) only in analysis,{" "}
                {analysis.verification.mpesa_cross_check.codes_parsed_only.length} only parsed,{" "}
                {analysis.verification.mpesa_cross_check.reclassified.length} reclassified to CLAIMED ONLY.
              </p>
            )}
          </InfoCard>

          {/* Elements A-K */}
          <div>
            <p className="text-xs font-bold tracking-[.14em] uppercase mb-1.5" style={{ color: MUTED }}>
              Contract elements
            </p>
            <div className="space-y-2">
              {Object.entries(analysis.result.elements).map(([key, el]) => (
                <ElementRow key={key} letter={key} label={ELEMENT_LABELS[key] ?? key} element={el} />
              ))}
            </div>
          </div>

          {/* M-Pesa payments */}
          {analysis.result.mpesa_payments?.length > 0 && (
            <InfoCard title="M-Pesa payments" icon={<FileWarning className="h-4 w-4" aria-hidden="true" />}>
              <div className="overflow-x-auto rounded-xl" style={{ border: `1px solid ${BORDER}` }}>
                <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: CREAM2 }}>
                      <th className="text-left px-3 py-2 font-bold" style={{ color: MUTED }}>Code</th>
                      <th className="text-left px-3 py-2 font-bold" style={{ color: MUTED }}>Amount</th>
                      <th className="text-left px-3 py-2 font-bold" style={{ color: MUTED }}>Recipient</th>
                      <th className="text-left px-3 py-2 font-bold" style={{ color: MUTED }}>Date/time</th>
                      <th className="text-left px-3 py-2 font-bold" style={{ color: MUTED }}>Classification</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.result.mpesa_payments.map((p, i) => (
                      <tr key={i} style={{ borderTop: `1px solid ${BORDER}` }}>
                        <td className="px-3 py-2 font-mono" style={{ color: INK }}>{p.code ?? "—"}</td>
                        <td className="px-3 py-2 font-mono" style={{ color: INK }}>{p.amount_kes != null ? `Ksh ${p.amount_kes}` : "—"}</td>
                        <td className="px-3 py-2" style={{ color: INK }}>{p.recipient ?? "—"}</td>
                        <td className="px-3 py-2" style={{ color: INK }}>{p.datetime ?? "—"}</td>
                        <td className="px-3 py-2">
                          <span className="inline-block text-[10px] font-semibold rounded-full px-2 py-0.5"
                            style={{ background: statusTone(p.classification).bg, color: statusTone(p.classification).fg }}>
                            {p.classification}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </InfoCard>
          )}

          {/* Timeline */}
          {analysis.result.timeline?.length > 0 && (
            <InfoCard title="Timeline" icon={<CalendarDays className="h-4 w-4" aria-hidden="true" />}>
              <ol className="space-y-1 text-xs" style={{ color: INK }}>
                {analysis.result.timeline.map((t, i) => (
                  <li key={i}>
                    <span className="font-mono" style={{ color: GOLD_DK }}>#{t.msg_index ?? t.evidence_id ?? "?"}</span>{" "}
                    {t.time && <span style={{ color: MUTED }}>{t.time}</span>}{" "}
                    {t.event}
                  </li>
                ))}
              </ol>
            </InfoCard>
          )}

          {/* Unrelated messages */}
          {analysis.result.unrelated_messages?.length > 0 && (
            <InfoCard title="Unrelated messages" icon={<ChevronRight className="h-4 w-4" aria-hidden="true" />}>
              <ul className="list-disc pl-4 space-y-0.5 text-xs" style={{ color: MUTED }}>
                {analysis.result.unrelated_messages.map((u, i) => (
                  <li key={i}><span className="font-mono" style={{ color: GOLD_DK }}>#{u.msg_index ?? "?"}</span> {u.text}</li>
                ))}
              </ul>
            </InfoCard>
          )}

          {/* Red flags + missing info */}
          <div className="grid sm:grid-cols-2 gap-4">
            {analysis.result.red_flags?.length > 0 && (
              <InfoCard title="Red flags" icon={<FileWarning className="h-4 w-4" aria-hidden="true" />}>
                <ul className="list-disc pl-4 space-y-0.5 text-xs" style={{ color: "#B91C1C" }}>
                  {analysis.result.red_flags.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </InfoCard>
            )}
            {analysis.result.missing_information?.length > 0 && (
              <InfoCard title="Missing information" icon={<AlertCircle className="h-4 w-4" aria-hidden="true" />}>
                <ul className="list-disc pl-4 space-y-0.5 text-xs" style={{ color: MUTED }}>
                  {analysis.result.missing_information.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </InfoCard>
            )}
          </div>

          {/* Questions + checklist */}
          {analysis.result.questions_for_parties?.length > 0 && (
            <InfoCard title="Questions for the parties" icon={<ChevronRight className="h-4 w-4" aria-hidden="true" />}>
              <ul className="space-y-1 text-xs" style={{ color: INK }}>
                {analysis.result.questions_for_parties.map((q, i) => (
                  <li key={i}><span className="font-semibold" style={{ color: GOLD_DK }}>{q.element}:</span> {q.question}</li>
                ))}
              </ul>
            </InfoCard>
          )}
          {analysis.result.strengthen_checklist?.length > 0 && (
            <InfoCard title="How to strengthen the record" icon={<ShieldCheck className="h-4 w-4" aria-hidden="true" />}>
              <ul className="list-disc pl-4 space-y-0.5 text-xs" style={{ color: MUTED }}>
                {analysis.result.strengthen_checklist.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </InfoCard>
          )}

          {/* Disclaimer */}
          {analysis.result.disclaimer && (
            <p className="text-[11px] italic" style={{ color: MUTED }}>{analysis.result.disclaimer}</p>
          )}

          {/* Raw JSON */}
          <div>
            <div className="flex items-center justify-between gap-2 mb-1.5 flex-wrap">
              <p className="text-xs font-bold tracking-[.14em] uppercase" style={{ color: MUTED }}>
                Analysis JSON · prompt {analysis.prompt_version} · {analysis.model}
              </p>
              <div className="flex items-center gap-2">
                <button type="button" onClick={copyAnalysis}
                  className="flex items-center gap-1 text-xs font-semibold rounded-lg px-2.5 py-1.5 hover:bg-amber-50" style={{ color: GOLD_DK }}>
                  {copiedAnalysis ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : <Copy className="h-3.5 w-3.5" aria-hidden="true" />}
                  {copiedAnalysis ? "Copied" : "Copy JSON"}
                </button>
                <button type="button" onClick={downloadAnalysis}
                  className="flex items-center gap-1 text-xs font-semibold rounded-lg px-2.5 py-1.5 hover:bg-amber-50" style={{ color: GOLD_DK }}>
                  <Download className="h-3.5 w-3.5" aria-hidden="true" />
                  Download
                </button>
              </div>
            </div>
            <pre className="w-full rounded-xl px-3 py-2 text-xs leading-relaxed font-mono overflow-auto max-h-96"
              style={{ background: CREAM2, border: `1px solid ${BORDER}`, color: INK, whiteSpace: "pre" }}>
              {JSON.stringify(analysis, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </section>
  );
}

function InfoCard({ title, icon, badge, children }: {
  title: string;
  icon: React.ReactNode;
  badge?: { text: string; bg: string; fg: string };
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl p-4" style={{ background: CREAM2, border: `1px solid ${BORDER}` }}>
      <div className="flex items-center gap-2 mb-2">
        <span style={{ color: GOLD }}>{icon}</span>
        <p className="text-sm font-bold" style={{ color: INK }}>{title}</p>
        {badge && (
          <span className="ml-auto inline-block text-[10px] font-semibold rounded-full px-2 py-0.5"
            style={{ background: badge.bg, color: badge.fg }}>
            {badge.text}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

function CertificateModal({
  analysis, caseFile, asOf, onClose,
}: {
  analysis: AnalystResponse;
  caseFile: CaseFile;
  asOf: string;
  onClose: () => void;
}) {
  const result = analysis.result;
  const verdict = analysis.verification.verdict_after ?? result.overall_verdict;
  const tone = verdictTone(verdict);
  const confPct = result.confidence != null ? Math.round(result.confidence * 100) : null;
  const nonAbsentPayments = result.mpesa_payments?.filter((p) => p.classification !== "ABSENT") ?? [];

  const handlePrint = () => window.print();

  // Prevent background scroll while modal is open
  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; };
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4 sm:p-8"
      style={{ background: "rgba(21,21,21,0.65)" }}
      role="dialog"
      aria-modal="true"
      aria-label="Certificate of Contract Evidence Analysis"
    >
      {/* Backdrop */}
      <div className="absolute inset-0" onClick={onClose} aria-hidden="true" />

      {/* Certificate card */}
      <div
        id="scc-certificate"
        className="relative z-10 w-full max-w-3xl rounded-2xl shadow-2xl my-4"
        style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
      >
        {/* Toolbar — hidden when printing */}
        <div
          id="scc-cert-toolbar"
          className="flex items-center justify-between gap-3 px-5 py-3 rounded-t-2xl"
          style={{ background: CREAM2, borderBottom: `1px solid ${BORDER}` }}
        >
          <span className="text-xs" style={{ color: MUTED }}>
            Save as PDF — use your browser&apos;s <strong>Print → Save as PDF</strong> option
          </span>
          <div className="flex items-center gap-2">
            <button type="button" onClick={handlePrint}
              className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 hover:opacity-90 transition-opacity"
              style={{ background: GOLD, color: "white" }}>
              <Printer className="h-3.5 w-3.5" aria-hidden="true" />
              Print / Save PDF
            </button>
            <button type="button" onClick={onClose}
              className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 hover:bg-amber-50 transition-colors"
              style={{ color: MUTED }}>
              <X className="h-3.5 w-3.5" aria-hidden="true" />
              Close
            </button>
          </div>
        </div>

        {/* ── Certificate body ──────────────────────────────────── */}
        <div className="px-7 py-7 space-y-6">

          {/* Header */}
          <div className="text-center pb-6" style={{ borderBottom: `2px solid ${GOLD}` }}>
            <div className="flex items-center justify-center gap-2.5 mb-3">
              <Scale className="h-7 w-7" style={{ color: GOLD }} aria-hidden="true" />
              <Award className="h-8 w-8" style={{ color: GOLD }} aria-hidden="true" />
              <Scale className="h-7 w-7" style={{ color: GOLD }} aria-hidden="true" />
            </div>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: GOLD_DK }}>
              Kenya Small Claims Court · Contract Evidence
            </p>
            <h1 className="text-2xl font-extrabold" style={{ color: INK }}>
              Certificate of Analysis
            </h1>
            <p className="text-xs mt-1.5" style={{ color: MUTED }}>
              Case reference: <span className="font-mono font-semibold">{caseFile.case_id}</span>
              {" · "}Analysis date: <span className="font-semibold">{asOf}</span>
            </p>
          </div>

          {/* Verdict + confidence */}
          <div className="rounded-xl px-5 py-4 text-center" style={{ background: tone.bg, border: `2px solid ${tone.border}` }}>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: tone.fg }}>
              Overall Verdict
            </p>
            <p className="text-xl font-extrabold" style={{ color: tone.fg }}>{verdict}</p>
            {confPct != null && (
              <div className="mt-2.5">
                <p className="text-sm font-bold" style={{ color: tone.fg }}>{confPct}% confidence</p>
                <div className="mt-1.5 h-2 rounded-full overflow-hidden mx-auto max-w-xs" style={{ background: "#E2E8F0" }}>
                  <div className="h-full rounded-full transition-all" style={{ width: `${confPct}%`, background: tone.fg }} />
                </div>
                <p className="text-[10px] mt-1.5 font-semibold" style={{ color: tone.fg }}>
                  ✓ Meets the {Math.round(CERT_THRESHOLD * 100)}% evidence threshold
                </p>
              </div>
            )}
          </div>

          {/* Parties */}
          {result.parties?.length > 0 && (
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-2" style={{ color: MUTED }}>Parties</p>
              <div className="grid sm:grid-cols-2 gap-3">
                {result.parties.map((p, i) => (
                  <div key={i} className="rounded-xl px-3.5 py-2.5" style={{ background: CREAM2, border: `1px solid ${BORDER}` }}>
                    <p className="text-sm font-bold" style={{ color: INK }}>{p.name}</p>
                    {p.role && <p className="text-xs" style={{ color: MUTED }}>{p.role}</p>}
                    {p.notes && <p className="text-xs italic mt-0.5" style={{ color: MUTED }}>{p.notes}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Contract elements table */}
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-2" style={{ color: MUTED }}>Contract Elements</p>
            <div className="overflow-x-auto rounded-xl" style={{ border: `1px solid ${BORDER}` }}>
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr style={{ background: CREAM2 }}>
                    <th className="text-left px-3 py-2 font-semibold rounded-tl-xl" style={{ color: MUTED }}>Element</th>
                    <th className="text-left px-3 py-2 font-semibold" style={{ color: MUTED }}>Status</th>
                    <th className="text-right px-3 py-2 font-semibold rounded-tr-xl" style={{ color: MUTED }}>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.elements).map(([key, el], i) => {
                    const st = statusTone(el.status);
                    return (
                      <tr key={key} style={{ borderTop: `1px solid ${BORDER}`, background: SURFACE }}>
                        <td className="px-3 py-2.5" style={{ color: INK }}>
                          <span className="font-extrabold mr-1.5" style={{ color: GOLD }}>{key}</span>
                          {ELEMENT_LABELS[key] ?? key}
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="inline-block text-[10px] font-semibold rounded-full px-2 py-0.5"
                            style={{ background: st.bg, color: st.fg }}>
                            {el.status}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono" style={{ color: MUTED }}>
                          {el.confidence != null ? `${Math.round(el.confidence * 100)}%` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* M-Pesa payments */}
          {nonAbsentPayments.length > 0 && (
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-2" style={{ color: MUTED }}>Payment Evidence</p>
              <div className="overflow-x-auto rounded-xl" style={{ border: `1px solid ${BORDER}` }}>
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr style={{ background: CREAM2 }}>
                      <th className="text-left px-3 py-2 font-semibold" style={{ color: MUTED }}>Code</th>
                      <th className="text-right px-3 py-2 font-semibold" style={{ color: MUTED }}>Amount (KES)</th>
                      <th className="text-left px-3 py-2 font-semibold" style={{ color: MUTED }}>Recipient</th>
                      <th className="text-left px-3 py-2 font-semibold" style={{ color: MUTED }}>Classification</th>
                    </tr>
                  </thead>
                  <tbody>
                    {nonAbsentPayments.map((p, i) => {
                      const clFg =
                        p.classification === "VERIFIED-LOOKING" ? "#15803D" :
                        p.classification === "SUSPICIOUS" ? "#B91C1C" : "#92400E";
                      return (
                        <tr key={i} style={{ borderTop: `1px solid ${BORDER}`, background: SURFACE }}>
                          <td className="px-3 py-2 font-mono" style={{ color: INK }}>{p.code ?? "—"}</td>
                          <td className="px-3 py-2 text-right font-mono font-semibold" style={{ color: INK }}>
                            {p.amount_kes != null ? p.amount_kes.toLocaleString("en-KE") : "—"}
                          </td>
                          <td className="px-3 py-2" style={{ color: INK }}>{p.recipient ?? "—"}</td>
                          <td className="px-3 py-2 font-semibold text-[11px]" style={{ color: clFg }}>
                            {p.classification}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Plain summary */}
          {result.plain_summary && (
            <div className="rounded-xl px-4 py-3.5" style={{ background: CREAM2, border: `1px solid ${BORDER}` }}>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-1.5" style={{ color: MUTED }}>Analysis Summary</p>
              <p className="text-xs leading-relaxed" style={{ color: INK }}>{result.plain_summary}</p>
            </div>
          )}

          {/* Red flags */}
          {result.red_flags?.length > 0 && (
            <div className="rounded-xl px-4 py-3.5" style={{ background: "#FEF9EE", border: `1px solid #FDE68A` }}>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-1.5" style={{ color: "#92400E" }}>
                Notes &amp; Red Flags
              </p>
              <ul className="list-disc pl-4 space-y-0.5 text-xs" style={{ color: "#78350F" }}>
                {result.red_flags.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          )}

          {/* Disclaimer + footer */}
          <div className="pt-4 text-center space-y-1.5" style={{ borderTop: `1px solid ${BORDER}` }}>
            <p className="text-[10px] italic" style={{ color: MUTED }}>{result.disclaimer}</p>
            <p className="text-[10px] font-semibold" style={{ color: GOLD_DK }}>
              SCCVision Contract Evidence Analyst — Kenya Small Claims Court Support
            </p>
            <p className="text-[10px]" style={{ color: MUTED }}>
              Generated {new Date().toLocaleDateString("en-KE", { year: "numeric", month: "long", day: "numeric" })}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ElementRow({ letter, label, element }: { letter: string; label: string; element: ElementResult }) {
  const [open, setOpen] = useState(false);
  const tone = statusTone(element.status);
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: CREAM2, border: `1px solid ${BORDER}` }}>
      <button type="button" onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left">
        <span className="h-6 w-6 rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0" style={{ background: GOLD }}>
          {letter}
        </span>
        <span className="text-sm font-semibold" style={{ color: INK }}>{label}</span>
        <span className="ml-auto inline-block text-[10px] font-semibold rounded-full px-2 py-0.5" style={{ background: tone.bg, color: tone.fg }}>
          {element.status}
        </span>
        {element.confidence != null && (
          <span className="text-[10px] font-mono" style={{ color: MUTED }}>{Math.round(element.confidence * 100)}%</span>
        )}
        {open ? <ChevronDown className="h-4 w-4" style={{ color: MUTED }} aria-hidden="true" /> : <ChevronRight className="h-4 w-4" style={{ color: MUTED }} aria-hidden="true" />}
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-2">
          {element.reasoning && <p className="text-xs leading-relaxed" style={{ color: INK }}>{element.reasoning}</p>}
          {element.evidence?.length > 0 && (
            <div className="space-y-1">
              {element.evidence.map((ev, i) => (
                <div key={i} className="rounded-lg px-2.5 py-1.5 text-xs" style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
                  <span className="font-mono" style={{ color: GOLD_DK }}>#{ev.msg_index ?? "?"}</span>{" "}
                  {ev.sender && <span className="font-semibold" style={{ color: INK }}>{ev.sender}</span>}{" "}
                  {ev.time && <span style={{ color: MUTED }}>{ev.time}</span>}
                  <span className="block mt-0.5" style={{ color: INK }}>&ldquo;{ev.quote}&rdquo;</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
