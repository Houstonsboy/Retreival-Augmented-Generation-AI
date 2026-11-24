"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Upload, X, ScanText, Loader2, Copy, Check, AlertCircle, Download,
  Scale, ShieldCheck, FileWarning, ChevronDown, ChevronRight, Save,
  FolderOpen, CalendarDays, Award, Printer, ZoomIn, Sparkles,
} from "lucide-react";
import { apiFetch } from "@/app/lib/api";
import type { CertificateRecord } from "@/app/lib/evidenceTypes";
import { DeviceQuestionnaire } from "./DeviceQuestionnaire";
import { useAuth } from "@/app/context/AuthContext";
import { uploadEvidenceFile, generateCaseTitle, listUserEvidence } from "@/app/lib/storageService";

// ── Design tokens ──────────────────────────────────────────────────────────
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const CREAM2  = "#F5F0E4";
const INK     = "#151515";
const MUTED   = "#6B6660";
const SURFACE = "#FFFDF8";

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
// Certificate eligibility is now computed on the backend (Phase 9).
// CERT_THRESHOLD removed — use analysis.cert_eligibility.eligible instead.
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
  download_url?: string;
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
    consistent: boolean;
    blocks_certificate: boolean;
    findings: {
      code: string | null;
      kind: string;
      amount_kes: number | null;
      detail: string;
      blocks_certificate: boolean;
    }[];
  };
  status_issues: { element: string; issue: string }[];
  premature_breach: { premature_breach_claim: boolean; notice: string } | null;
  verdict_before: string;
  verdict_after: string;
  verdict_downgraded: boolean;
  downgrade_reasons: string[];
}
interface CertEligibility {
  eligible: boolean;
  reasons: string[];
}

interface FollowUpInputField {
  key: string;
  label: string;
  type: string;
  required?: boolean;
  options?: { msg_index?: number; preview?: string; source?: string }[] | null;
  help?: string | null;
}

interface GuidedFollowUp {
  id: string;
  gap_code: string;
  elements: string[];
  priority: number;
  blocks_certificate: boolean;
  kind: "UPLOAD" | "INPUT" | "CHOICE" | "CONFIRM";
  title: string;
  why: string;
  how: string;
  upload_types: string[];
  fields: FollowUpInputField[];
  example?: string | null;
  related_msg_indices: number[];
  status: string;
  expected_effect: string;
}

interface AnalystResponse {
  analysis_input: unknown;
  result: AnalystResult;
  verification: Verification;
  prompt_version: string;
  model: string;
  analysis_id?: string | null;
  // v2 deterministic fields
  cert_eligibility?: CertEligibility;
  confidence?: number;
  confidence_breakdown?: Record<string, unknown>;
  verdict?: string;
  derived_statuses?: Record<string, string>;
  unstable_fields?: string[];
  model_disagreements?: Record<string, { model_status: string; derived_status: string; model_reason: string }>;
  integrity_rating?: string;
  calls_used?: number;
  trigger_fired?: string | null;
  from_cache?: boolean;
  // v2.2 guided follow-ups
  guided_follow_ups?: GuidedFollowUp[];
  round?: number;
  round_diff?: Record<string, unknown>;
}

const ELEMENT_LABELS: Record<string, string> = {
  A: "Parties & identity", B: "Capacity", C: "Offer", D: "Acceptance",
  E: "Consideration", F: "Intention", G: "Certainty", H: "Genuine consent",
  I: "Legality", J: "Formalities", K: "Performance & breach",
};

const ELEMENT_FIX: Record<string, string> = {
  C: "Identify the message where one party proposed the specific terms of the agreement.",
  D: "Identify where the other party agreed to those terms — a clear 'yes', 'okay', or equivalent.",
  E: "Upload the original M-Pesa confirmation screenshot or bank statement that shows payment for this specific contract.",
  G: "Clarify what exactly was being purchased or provided — the subject matter must be unambiguous in the conversation.",
};

function explainCertReason(reason: string): { title: string; detail: string } {
  const elemMatch = reason.match(/^element ([A-K]) not established \((\w+)\)/);
  if (elemMatch) {
    const letter = elemMatch[1];
    const status = elemMatch[2];
    const label = ELEMENT_LABELS[letter] ?? `Element ${letter}`;
    return {
      title: `${label} (${letter}) not established — currently ${status}`,
      detail: ELEMENT_FIX[letter] ?? "Provide additional evidence to establish this element.",
    };
  }
  if (reason.includes("element G absent")) {
    return {
      title: "Contract scope (G) not established",
      detail: ELEMENT_FIX["G"]!,
    };
  }
  const unstableMatch = reason.match(/inconsistent across independent inspections: \[(.+)\]/);
  if (unstableMatch) {
    const letters = unstableMatch[1].replace(/['\s]/g, "").split(",");
    const names = letters.map(l => `${ELEMENT_LABELS[l] ?? l} (${l})`).join(", ");
    return {
      title: `Two reads disagreed on: ${names}`,
      detail: "The same evidence produced different conclusions on two independent passes. Upload clearer screenshots for the disputed element(s) or retry — a persistent conflict means the evidence is ambiguous.",
    };
  }
  if (reason.startsWith("verdict: LIKELY NOT VALID")) {
    return {
      title: "Contract appears not valid",
      detail: "Core elements (offer, acceptance, or payment) are missing or directly contradicted by the evidence. Address the element gaps above first.",
    };
  }
  if (reason.startsWith("verdict: INSUFFICIENT EVIDENCE")) {
    return {
      title: "Insufficient evidence",
      detail: "Not enough conversation or documents were provided to assess whether a valid contract exists. Upload more of the conversation or relevant supporting documents.",
    };
  }
  if (reason.includes("M-Pesa cross-check")) {
    return {
      title: "M-Pesa confirmation mismatch",
      detail: "The M-Pesa codes or amounts cited by the analysis don't match what was found in the uploaded screenshots. Upload the original, unedited M-Pesa confirmation screenshot.",
    };
  }
  if (reason.includes("record integrity weak")) {
    return {
      title: "Conversation integrity is weak",
      detail: "Messages appear to have been edited or deleted after sending. Upload original, unaltered screenshots. If edits are legitimate, provide the original text as a user-stated fact.",
    };
  }
  if (reason.includes("ungrounded evidence references")) {
    return {
      title: "Evidence references could not be verified",
      detail: "The analysis cited message indices that don't exist in the uploaded conversation. This is a system issue — click Retry.",
    };
  }
  if (reason.includes("analysis incomplete")) {
    return {
      title: "Analysis did not complete",
      detail: "The pipeline did not finish successfully. Click Retry to run the analysis again.",
    };
  }
  if (reason.includes("not independently verified")) {
    return {
      title: "Facts not yet independently verified",
      detail: "A second independent inspection is needed to confirm the key findings. Click Retry to run the full two-pass analysis.",
    };
  }
  const confMatch = reason.match(/confidence ([\d.]+)% below minimum ([\d.]+)%/);
  if (confMatch) {
    return {
      title: `Overall confidence too low (${confMatch[1]}% — minimum ${confMatch[2]}%)`,
      detail: "Provide additional evidence to strengthen the weaker contract elements shown in the element grid below.",
    };
  }
  return { title: reason, detail: "" };
}

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
  // Phase 9: use backend-computed eligibility when available.
  if (a.cert_eligibility != null) return a.cert_eligibility.eligible;
  // Legacy fallback for old API responses.
  const v = a.verification?.verdict_after ?? a.result.overall_verdict;
  return (
    a.result.confidence != null &&
    a.result.confidence >= 0.83 &&
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
    case_context: { user_role: "unknown", user_account: "" },
    evidence: [],
    user_provided_facts: [],
    rejected_facts: [],
    inferred_facts: [],
    requests: [],
    analyses: [],
  };
}

export function SCCVisionTest({ initialCaseId }: { initialCaseId?: string } = {}) {
  const { user } = useAuth();
  const inputRef = useRef<HTMLInputElement>(null);
  // Maps sha256 → original File so we can upload to Firebase after OCR
  const filesByHash = useRef<Map<string, { file: File; evidenceType: string }>>(new Map());
  // Maps sha256 → object URL so the evidence list can still show thumbnails after OCR
  const previewsByHash = useRef<Map<string, string>>(new Map());
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [labels, setLabels] = useState<(EvidenceType | "")[]>([]);
  const [invert, setInvert] = useState(true);
  const [upscale, setUpscale] = useState(true);

  const [caseFile, setCaseFile] = useState<CaseFile>(() => {
    if (initialCaseId) {
      try {
        const stored = localStorage.getItem(`sccvision-case-${initialCaseId}`);
        if (stored) return JSON.parse(stored) as CaseFile;
      } catch { /* ignore */ }
    }
    return newCaseFile();
  });
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

  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<{ saved: number; deduped: number } | null>(null);
  const [pendingPreview, setPendingPreview] = useState<{ url: string; name: string } | null>(null);
  const [titleModalOpen, setTitleModalOpen] = useState(false);
  const [pendingTitle, setPendingTitle] = useState("");
  const [titleLoading, setTitleLoading] = useState(false);

  const [answerTarget, setAnswerTarget] = useState<FollowUpRequest | null>(null);
  const [answerText, setAnswerText] = useState("");
  const [answerDate, setAnswerDate] = useState("");
  const answerUploadRef = useRef<HTMLInputElement>(null);
  const [answerUploadFile, setAnswerUploadFile] = useState<File | null>(null);
  const [answerUploadPreview, setAnswerUploadPreview] = useState("");
  const [answerUploadLoading, setAnswerUploadLoading] = useState(false);

  // Guided follow-up state
  const [guidedTarget, setGuidedTarget] = useState<GuidedFollowUp | null>(null);
  const [guidedFieldValues, setGuidedFieldValues] = useState<Record<string, string>>({});
  const guidedUploadRef = useRef<HTMLInputElement>(null);
  const [guidedUploadFile, setGuidedUploadFile] = useState<File | null>(null);
  const [guidedUploadPreview, setGuidedUploadPreview] = useState("");
  const [guidedSubmitting, setGuidedSubmitting] = useState(false);
  const [showAllGuided, setShowAllGuided] = useState(false);

  // Persist the case file locally keyed by case_id so Evidence History can restore it.
  useEffect(() => {
    try {
      localStorage.setItem(`sccvision-case-${caseFile.case_id}`, JSON.stringify(caseFile));
    } catch { /* storage unavailable */ }
  }, [caseFile]);

  // Firestore fallback: when navigated from Evidence History with an initialCaseId
  // that isn't in localStorage (different device / cleared storage), reconstruct
  // a stub case file from the saved evidence records so images are still visible.
  useEffect(() => {
    if (!initialCaseId || !user) return;
    // If localStorage already restored a matching case with evidence, skip.
    if (caseFile.case_id === initialCaseId && caseFile.evidence.length > 0) return;
    listUserEvidence(user.uid).then((records) => {
      const caseRecords = records.filter((r) => r.caseId === initialCaseId);
      if (!caseRecords.length) return;
      const stub = newCaseFile();
      stub.case_id = initialCaseId;
      stub.created_at = caseRecords[0].uploadedAt.toISOString();
      stub.evidence = caseRecords.map((r) => ({
        evidence_id: r.id,
        type: r.evidenceType as EvidenceType,
        file_name: r.fileName,
        sha256: r.sha256,
        added_at: r.uploadedAt.toISOString(),
        ocr: null,
        parsed: null,
        user_edits: {},
        download_url: r.downloadUrl,
      }));
      // Seed preview cache so thumbnails appear immediately
      for (const r of caseRecords) {
        if (r.sha256) previewsByHash.current.set(r.sha256, r.downloadUrl);
      }
      setCaseFile(stub);
    }).catch(() => { /* ignore — user will see empty form */ });
  }, [initialCaseId, user]); // eslint-disable-line react-hooks/exhaustive-deps

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

  const asOf = todayISO();

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
    // Append to existing selection — don't replace.
    const newUrls = valid.map((f) => URL.createObjectURL(f));
    setFiles((prev) => [...prev, ...valid]);
    setPreviewUrls((prev) => [...prev, ...newUrls]);
    setLabels((prev) => [...prev, ...valid.map((): EvidenceType | "" => "")]);
  }, []);

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
        // Keep original file so we can upload to Firebase Storage later
        filesByHash.current.set(hash, { file: f, evidenceType: labels[i] || "" });
        // Keep preview URL so the evidence list can show thumbnails after OCR
        if (previewUrls[i]) previewsByHash.current.set(hash, previewUrls[i]);
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
      // Only revoke URLs not kept in previewsByHash for the evidence list
      const keptUrls = new Set(previewsByHash.current.values());
      previewUrls.forEach((u) => { if (!keptUrls.has(u)) URL.revokeObjectURL(u); });
      setPreviewUrls([]);
      setLabels([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the backend. Is it running on port 5000?");
    } finally {
      setOcrLoading(false);
    }
  };

  const removeEvidence = useCallback((id: string) => {
    setCaseFile((prev) => {
      const ev = prev.evidence.find((e) => e.evidence_id === id);
      if (ev?.sha256) {
        const url = previewsByHash.current.get(ev.sha256);
        if (url) { URL.revokeObjectURL(url); previewsByHash.current.delete(ev.sha256); }
      }
      return { ...prev, evidence: prev.evidence.filter((e) => e.evidence_id !== id) };
    });
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

  const closeGuidedModal = () => {
    setGuidedTarget(null);
    setGuidedFieldValues({});
    if (guidedUploadPreview) URL.revokeObjectURL(guidedUploadPreview);
    setGuidedUploadFile(null);
    setGuidedUploadPreview("");
  };

  const submitGuidedAnswer = async () => {
    if (!guidedTarget || !analysis) return;
    const req = guidedTarget;
    setGuidedSubmitting(true);
    setError(null);

    try {
      const newFiles: CaseEvidence[] = [];
      const statements: { field_key: string; value: string; request_id: string; label: string }[] = [];

      // If UPLOAD kind and a file was chosen, OCR it
      if (req.kind === "UPLOAD" && guidedUploadFile) {
        const fd = new FormData();
        fd.append("file", guidedUploadFile);
        const extractRes = await apiFetch(`/sccvision/extract?invert=${invert}&upscale=${upscale}`, {
          method: "POST", body: fd,
        });
        const ocr = await extractRes.json().catch(() => null);
        if (!extractRes.ok) throw new Error(ocr?.error || `OCR failed (${extractRes.status})`);

        const hash = await sha256Hex(await guidedUploadFile.arrayBuffer());
        const classification: EvidenceType =
          req.upload_types?.[0]?.toLowerCase().includes("mpesa") ? "mpesa_screenshot" :
          req.upload_types?.[0]?.toLowerCase().includes("chat") ? "chat_screenshot" : "other";

        const procRes = await apiFetch("/sccvision/evidence/process", {
          method: "POST",
          body: JSON.stringify({
            ocr_result: ocr,
            classification,
            file_name: guidedUploadFile.name,
            sha256: hash,
          }),
        });
        const proc = await procRes.json().catch(() => null);
        if (!procRes.ok) throw new Error(proc?.error || `Processing failed (${procRes.status})`);
        newFiles.push(proc.evidence as CaseEvidence);

        // Keep preview for the evidence list
        const previewUrl = URL.createObjectURL(guidedUploadFile);
        previewsByHash.current.set(hash, previewUrl);
        filesByHash.current.set(hash, { file: guidedUploadFile, evidenceType: classification });
      }

      // Collect typed field answers as statements
      for (const [key, value] of Object.entries(guidedFieldValues)) {
        if (value.trim()) {
          const field = req.fields.find((f) => f.key === key);
          statements.push({
            field_key: key,
            value: value.trim(),
            request_id: req.id,
            label: field?.label || key,
          });
        }
      }

      if (!newFiles.length && !statements.length) {
        setError("Please upload a file or fill in at least one field.");
        setGuidedSubmitting(false);
        return;
      }

      // --- Call supplement endpoint ---
      const supplementBody = {
        case_file: caseFile,
        supplement: { files: newFiles, statements },
        parent_analysis_id: analysis.analysis_id ?? null,
        force_rerun: true,
      };

      const res = await apiFetch("/sccvision/case/supplement", {
        method: "POST",
        body: JSON.stringify(supplementBody),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok) {
        if (res.status === 429) {
          const wait = body?.retry_after_seconds ? ` Try again in ~${body.retry_after_seconds}s.` : "";
          throw new Error(`${body?.error || "Rate limit reached"}.${wait}`);
        }
        throw new Error(body?.error || `Supplement failed (${res.status})`);
      }

      setAnalysis(body as AnalystResponse);
      setRunCount((n) => n + 1);
      setShowAllGuided(false);

      // Merge new evidence into case file and mark request answered
      setCaseFile((prev) => {
        const updatedRequests = [
          ...prev.requests.filter((r) => r.id !== req.id),
          { id: req.id, status: "answered" as const, last_seen_run: runCount + 1 },
        ];
        const { analysis_input: _ai, verification: _v, ...slimBody } = body;
        return {
          ...prev,
          evidence: [...prev.evidence, ...newFiles],
          requests: updatedRequests,
          analyses: [...prev.analyses, slimBody],
        };
      });
      closeGuidedModal();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Supplement failed.");
    } finally {
      setGuidedSubmitting(false);
    }
  };

  const dismissGuidedRequest = (req: GuidedFollowUp) => {
    setCaseFile((prev) => ({
      ...prev,
      requests: [
        ...prev.requests.filter((r) => r.id !== req.id),
        { id: req.id, status: "dismissed", last_seen_run: runCount },
      ],
    }));
  };

  const openSaveModal = async () => {
    if (!caseFile.evidence.length || saving || !user) return;
    setTitleModalOpen(true);
    setPendingTitle("");
    setTitleLoading(true);
    const suggestion = await generateCaseTitle(caseFile.case_context.user_account);
    setPendingTitle(suggestion);
    setTitleLoading(false);
  };

  const handleSaveEvidence = async (title: string) => {
    if (!caseFile.evidence.length || saving || !user) return;
    setTitleModalOpen(false);
    setSaving(true);
    setSaveResult(null);
    setError(null);
    try {
      let saved = 0;
      let deduped = 0;
      const downloadUrls = new Map<string, string>();
      for (const ev of caseFile.evidence) {
        if (!ev.sha256) continue;
        const entry = filesByHash.current.get(ev.sha256);
        if (!entry) continue;
        const { record, wasDedup } = await uploadEvidenceFile(
          user.uid,
          entry.file,
          ev.sha256,
          ev.type,
          caseFile.case_id,
          title,
        );
        downloadUrls.set(ev.sha256, record.downloadUrl);
        // Also keep it in previewsByHash so thumbnails work in this session
        if (!previewsByHash.current.has(ev.sha256)) {
          previewsByHash.current.set(ev.sha256, record.downloadUrl);
        }
        if (wasDedup) deduped++;
        else saved++;
      }
      // Persist download URLs back into the case file so they survive localStorage restore
      setCaseFile((prev) => ({
        ...prev,
        evidence: prev.evidence.map((ev) => ({
          ...ev,
          download_url: ev.sha256
            ? (downloadUrls.get(ev.sha256) ?? ev.download_url)
            : ev.download_url,
        })),
      }));
      setSaveResult({ saved, deduped });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save to Firebase failed.");
    } finally {
      setSaving(false);
    }
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
      className="rounded-2xl p-5 sm:p-6"
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

      {/* ── Role ──────────────────────────────────── */}
      <div className="mb-4">
        <label className="block text-xs font-semibold mb-1.5" style={{ color: MUTED }}>
          Your role in this case
        </label>
        <select value={caseFile.case_context.user_role}
          onChange={(e) => setCaseFile((p) => ({ ...p, case_context: { ...p.case_context, user_role: e.target.value } }))}
          className="w-full rounded-xl px-3 py-2.5 text-sm focus:outline-none appearance-none"
          style={{ background: CREAM2, border: `1px solid ${BORDER}`, color: INK }}>
          <option value="unknown">Not sure / not stated</option>
          <option value="claimant">Claimant (I am suing)</option>
          <option value="respondent">Respondent (I am being sued)</option>
        </select>
      </div>

      {/* ── Case description ──────────────────────────────────────── */}
      <div className="mb-5">
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
            background: CREAM2,
            border: `1px solid ${BORDER}`,
            color: INK,
            resize: "vertical",
            minHeight: "5.5rem",
            overflow: "hidden",
          }}
        />
      </div>

      {/* ── Drop zone ──────────────────────────────────────────────── */}
      <div
        role="button"
        tabIndex={0}
        aria-label={files.length > 0 ? "Add more screenshots — click or drag and drop" : "Upload screenshots — click or drag and drop"}
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
        <div className="h-10 w-10 rounded-full flex items-center justify-center mb-3" style={{ background: `${GOLD}18` }} aria-hidden="true">
          <Upload className="h-5 w-5" style={{ color: GOLD }} />
        </div>
        <p className="text-sm font-semibold text-center" style={{ color: INK }}>
          {files.length > 0 ? "Add more files" : "Drop files here or click to browse"}
        </p>
        <p className="text-xs mt-1 text-center" style={{ color: MUTED }}>
          {files.length > 0
            ? "New files are added to the list below"
            : `PNG, JPG, WEBP · multiple allowed · max ${formatBytes(MAX_BYTES)} each`}
        </p>
      </div>

      {error && (
        <div className="mt-3 flex items-start gap-2 rounded-xl px-4 py-3 text-xs"
          style={{ background: "#FEF2F2", border: "1px solid #FECACA", color: "#B91C1C" }} role="alert">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {/* ── Pending files (card list) ─────────────────────────────── */}
      {files.length > 0 && (
        <div className="mt-4">
          <ul className="space-y-3" aria-label="Selected files">
            {files.map((f, i) => (
              <PendingFileCard
                key={i}
                file={f}
                previewUrl={previewUrls[i]}
                label={labels[i] ?? ""}
                onRemove={() => removeFile(i)}
                onView={() => setPendingPreview({ url: previewUrls[i], name: f.name })}
                onLabelChange={(val) =>
                  setLabels((prev) => {
                    const next = [...prev];
                    next[i] = val as EvidenceType | "";
                    return next;
                  })
                }
              />
            ))}
          </ul>
          <div className="mt-2 flex items-center justify-between">
            <button
              type="button"
              onClick={() => {
                previewUrls.forEach((u) => URL.revokeObjectURL(u));
                setFiles([]);
                setPreviewUrls([]);
                setLabels([]);
              }}
              className="text-xs font-semibold rounded-lg px-2.5 py-1 transition-colors hover:bg-red-50"
              style={{ color: "#B91C1C", border: `1px solid #FECACA` }}
            >
              Clear all
            </button>
            <p className="text-xs" style={{ color: MUTED }}>
              {files.length} file{files.length !== 1 ? "s" : ""} queued
            </p>
          </div>
          <div className="mt-4 flex flex-wrap gap-3 items-center">
            <label className="flex items-center gap-2 text-xs font-medium cursor-pointer select-none" style={{ color: MUTED }}>
              <input type="checkbox" checked={invert} onChange={(e) => setInvert(e.target.checked)}
                className="rounded" style={{ accentColor: GOLD }} />
              Invert dark mode
            </label>
            <label className="flex items-center gap-2 text-xs font-medium cursor-pointer select-none" style={{ color: MUTED }}>
              <input type="checkbox" checked={upscale} onChange={(e) => setUpscale(e.target.checked)}
                className="rounded" style={{ accentColor: GOLD }} />
              Upscale small images
            </label>
            <button type="button" onClick={handleAddEvidence} disabled={ocrLoading}
              className="ml-auto flex items-center justify-center gap-2 rounded-xl px-5 py-2.5 text-sm font-bold transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
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
          <p className="text-xs font-bold tracking-[.14em] uppercase mb-2" style={{ color: MUTED }}>
            Evidence ({caseFile.evidence.length})
          </p>
          <div className="space-y-2">
            {caseFile.evidence.map((ev) => {
              const thumbUrl = (ev.sha256 ? previewsByHash.current.get(ev.sha256) : undefined) ?? ev.download_url;
              return (
              <div key={ev.evidence_id} className="flex items-center gap-3 rounded-xl px-3 py-2" style={{ background: CREAM2, border: `1px solid ${BORDER}` }}>
                {thumbUrl ? (
                  <button
                    type="button"
                    onClick={() => setPendingPreview({ url: thumbUrl, name: ev.file_name || ev.evidence_id })}
                    className="group shrink-0 relative h-10 w-10 rounded-lg overflow-hidden focus:outline-none focus:ring-2 focus:ring-brand-gold"
                    style={{ background: SURFACE }}
                    aria-label={`Preview ${ev.file_name}`}
                    title="Click to preview"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={thumbUrl} alt={ev.file_name} className="h-full w-full object-contain" style={{ background: CREAM2 }} />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                      style={{ background: "rgba(21,21,21,0.45)" }} aria-hidden="true">
                      <ZoomIn className="h-3 w-3 text-white" />
                    </div>
                  </button>
                ) : null}
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
              );
            })}
          </div>

          {/* Save to Firebase Evidence Library */}
          <div className="mt-4">
            <button
              type="button"
              onClick={openSaveModal}
              disabled={saving || !user}
              className="w-full flex items-center justify-center gap-2 rounded-2xl py-3 text-sm font-bold transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
              style={{
                background: saving ? CREAM2 : `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`,
                color: saving ? MUTED : "white",
                border: saving ? `1px solid ${BORDER}` : "none",
              }}
            >
              {saving ? (
                <><Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Saving to Evidence Library…</>
              ) : (
                <><FolderOpen className="h-4 w-4" aria-hidden="true" /> Save to Evidence Library</>
              )}
            </button>
            {saveResult && (
              <p className="mt-2 text-xs text-center" style={{ color: MUTED }}>
                {saveResult.saved > 0 && <span style={{ color: "#15803D" }}>{saveResult.saved} file{saveResult.saved !== 1 ? "s" : ""} saved. </span>}
                {saveResult.deduped > 0 && <span>{saveResult.deduped} already in library (skipped). </span>}
                {saveResult.saved === 0 && saveResult.deduped === 0 && <span>No files with originals to save.</span>}
                {(saveResult.saved > 0 || saveResult.deduped > 0) && (
                  <a href="/litigant/evidence" className="underline font-semibold" style={{ color: GOLD_DK }}>View Evidence History →</a>
                )}
              </p>
            )}
          </div>

          {/* Save / Load / Consent + Run analysis */}
          <div className="mt-4 pt-4 flex items-center gap-3 flex-wrap" style={{ borderTop: `1px solid ${BORDER}` }}>
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
            <label className="flex items-center gap-2 text-xs font-medium cursor-pointer select-none" style={{ color: MUTED }}>
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)}
                className="rounded" style={{ accentColor: GOLD }} />
              I consent to send this case to the automated analyst.
            </label>
            <button type="button" onClick={runAnalysis} disabled={analyzing}
              className="ml-auto flex items-center justify-center gap-2 rounded-xl px-5 py-2.5 text-sm font-bold transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}>
              {analyzing ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Scale className="h-4 w-4" aria-hidden="true" />}
              {analyzing ? "Analyzing…" : runCount > 0 ? "Re-run analysis" : "Run analysis"}
            </button>
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

      {/* ── Guided follow-up modal ───────────────────────────────── */}
      {guidedTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "#00000055" }} role="dialog" aria-modal="true">
          <div className="w-full max-w-lg rounded-2xl p-5 max-h-[90vh] overflow-y-auto"
            style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
            <div className="flex items-start gap-2 mb-1">
              {guidedTarget.blocks_certificate && (
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" style={{ color: "#B91C1C" }} />
              )}
              <p className="text-sm font-bold" style={{ color: INK }}>{guidedTarget.title}</p>
            </div>
            {guidedTarget.why && (
              <p className="text-xs mb-1" style={{ color: MUTED }}>{guidedTarget.why}</p>
            )}
            {guidedTarget.how && (
              <p className="text-xs mb-3 leading-relaxed" style={{ color: INK }}>{guidedTarget.how}</p>
            )}
            {guidedTarget.example && (
              <p className="text-xs italic mb-3 rounded-lg px-3 py-2"
                style={{ background: CREAM2, color: MUTED, border: `1px solid ${BORDER}` }}>
                Example: {guidedTarget.example}
              </p>
            )}

            {/* File upload slot */}
            {guidedTarget.kind === "UPLOAD" && (
              <div className="mb-3">
                <input
                  ref={guidedUploadRef}
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp"
                  className="sr-only"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    if (!isAccepted(f.name)) { setError("Allowed: .png .jpg .jpeg .webp"); return; }
                    if (f.size > MAX_BYTES) { setError("File too large (max 10 MB)."); return; }
                    if (guidedUploadPreview) URL.revokeObjectURL(guidedUploadPreview);
                    setGuidedUploadFile(f);
                    setGuidedUploadPreview(URL.createObjectURL(f));
                  }}
                />
                {guidedUploadFile ? (
                  <div className="relative rounded-xl overflow-hidden" style={{ border: `1px solid ${BORDER}` }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={guidedUploadPreview} alt="Preview"
                      className="w-full max-h-48 object-contain" style={{ background: CREAM2 }} />
                    <div className="flex items-center justify-between gap-2 px-3 py-2"
                      style={{ background: SURFACE, borderTop: `1px solid ${BORDER}` }}>
                      <span className="text-xs truncate" style={{ color: MUTED }}>{guidedUploadFile.name}</span>
                      <button type="button" onClick={() => { setGuidedUploadFile(null); setGuidedUploadPreview(""); }}
                        className="shrink-0 rounded-full p-1 hover:bg-gray-100">
                        <X className="h-3.5 w-3.5" style={{ color: MUTED }} />
                      </button>
                    </div>
                  </div>
                ) : (
                  <button type="button" onClick={() => guidedUploadRef.current?.click()}
                    className="w-full flex flex-col items-center gap-2 rounded-xl py-6 transition-colors hover:brightness-95"
                    style={{ background: CREAM2, border: `2px dashed ${BORDER}` }}>
                    <Upload className="h-6 w-6" style={{ color: GOLD }} />
                    <span className="text-sm font-semibold" style={{ color: INK }}>
                      Tap to upload {guidedTarget.upload_types[0]?.toLowerCase().replace(/_/g, " ") || "screenshot"}
                    </span>
                    <span className="text-xs" style={{ color: MUTED }}>PNG, JPG, WEBP · max 10 MB</span>
                  </button>
                )}
              </div>
            )}

            {/* Text / date input fields */}
            {guidedTarget.fields.length > 0 && (
              <div className="space-y-3 mb-3">
                {guidedTarget.fields.map((field) => (
                  <div key={field.key}>
                    <label className="block text-xs font-semibold mb-1" style={{ color: MUTED }}>
                      {field.label}
                      {field.required && <span style={{ color: "#B91C1C" }}> *</span>}
                    </label>
                    {field.type === "DATE" ? (
                      <input type="date"
                        value={guidedFieldValues[field.key] || ""}
                        onChange={(e) => setGuidedFieldValues((v) => ({ ...v, [field.key]: e.target.value }))}
                        className="w-full rounded-xl px-3 py-2 text-sm focus:outline-none"
                        style={{ background: CREAM2, border: `1px solid ${BORDER}`, color: INK }} />
                    ) : field.type === "YES_NO" ? (
                      <div className="flex gap-2">
                        {["Yes", "No"].map((opt) => (
                          <button key={opt} type="button"
                            onClick={() => setGuidedFieldValues((v) => ({ ...v, [field.key]: opt }))}
                            className="flex-1 text-sm rounded-xl px-3 py-2 transition-colors"
                            style={{
                              background: guidedFieldValues[field.key] === opt ? `${GOLD}22` : CREAM2,
                              border: `1px solid ${guidedFieldValues[field.key] === opt ? GOLD : BORDER}`,
                              color: INK,
                            }}>
                            {opt}
                          </button>
                        ))}
                      </div>
                    ) : field.type === "PHONE" ? (
                      <input type="tel"
                        value={guidedFieldValues[field.key] || ""}
                        onChange={(e) => setGuidedFieldValues((v) => ({ ...v, [field.key]: e.target.value }))}
                        placeholder="+254 7XX XXX XXX"
                        className="w-full rounded-xl px-3 py-2 text-sm focus:outline-none"
                        style={{ background: CREAM2, border: `1px solid ${BORDER}`, color: INK }} />
                    ) : (
                      <textarea
                        value={guidedFieldValues[field.key] || ""}
                        onChange={(e) => setGuidedFieldValues((v) => ({ ...v, [field.key]: e.target.value }))}
                        rows={3}
                        placeholder={field.help || ""}
                        className="w-full rounded-xl px-3 py-2 text-sm focus:outline-none resize-none"
                        style={{ background: CREAM2, border: `1px solid ${BORDER}`, color: INK }} />
                    )}
                    {field.help && field.type !== "TEXT" && (
                      <p className="text-[11px] mt-0.5" style={{ color: MUTED }}>{field.help}</p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {guidedTarget.expected_effect && (
              <p className="text-[11px] italic mb-3" style={{ color: "#15803D" }}>
                ↑ {guidedTarget.expected_effect}
              </p>
            )}

            <div className="flex justify-end gap-2">
              <button type="button" onClick={closeGuidedModal}
                className="text-xs font-semibold rounded-lg px-3 py-1.5" style={{ color: MUTED }}>
                Cancel
              </button>
              <button type="button" onClick={submitGuidedAnswer}
                disabled={guidedSubmitting}
                className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-4 py-1.5 disabled:opacity-60"
                style={{ background: GOLD_DK, color: "white" }}>
                {guidedSubmitting && <Loader2 className="h-3 w-3 animate-spin" />}
                {guidedSubmitting ? "Submitting…" : "Submit & Re-analyse"}
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
                      Evidence is eligible for a Section 106B certificate
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
            {/* Ineligibility reasons (Phase 9) */}
            {analysis.cert_eligibility && !analysis.cert_eligibility.eligible && (
              <div className="mt-3 rounded-xl px-3 py-2.5"
                style={{ background: "#FEF9EC", border: "1px solid #FDE68A" }}>
                <p className="text-xs font-bold mb-2" style={{ color: "#92400E" }}>
                  Not yet eligible for a Section 106B certificate
                </p>
                <div className="space-y-2">
                  {analysis.cert_eligibility.reasons.map((r, i) => {
                    const ex = explainCertReason(r);
                    return (
                      <div key={i} className="rounded-lg px-2.5 py-2"
                        style={{ background: "#FFF7ED", border: "1px solid #FED7AA" }}>
                        <p className="text-[11px] font-semibold" style={{ color: "#92400E" }}>
                          {ex.title}
                        </p>
                        {ex.detail && (
                          <p className="text-[11px] mt-0.5 leading-relaxed" style={{ color: "#78350F" }}>
                            {ex.detail}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
                {(analysis.result.confidence ?? 0) >= 0.9 && (
                  <p className="text-[10px] mt-2.5 leading-relaxed" style={{ color: "#A16207" }}>
                    Why is confidence {Math.round((analysis.result.confidence ?? 0) * 100)}% if the certificate is blocked?
                    Confidence scores how well the overall contract is evidenced across all elements.
                    The certificate additionally requires each key element (C, D, E, G) to be individually
                    established and stable across two independent reads — a single unclear or disputed element blocks it.
                  </p>
                )}
              </div>
            )}
            {/* Needs-human-look items (unstable fields, model disagreements) */}
            {((analysis.unstable_fields?.length ?? 0) > 0 ||
              Object.keys(analysis.model_disagreements ?? {}).length > 0) && (
              <div className="mt-3 rounded-xl px-3 py-2.5"
                style={{ background: "#F0F9FF", border: "1px solid #BAE6FD" }}>
                <p className="text-xs font-bold mb-1" style={{ color: "#0369A1" }}>
                  Needs a human look
                </p>
                {(analysis.unstable_fields?.length ?? 0) > 0 && (
                  <p className="text-[11px]" style={{ color: "#0C4A6E" }}>
                    Independent inspections disagreed on: {analysis.unstable_fields?.join(", ")}
                  </p>
                )}
                {Object.entries(analysis.model_disagreements ?? {}).map(([el, d]) => (
                  <p key={el} className="text-[11px]" style={{ color: "#0C4A6E" }}>
                    Model raised a concern on {el}: "{d.model_reason}" (model: {d.model_status}, derived: {d.derived_status})
                  </p>
                ))}
              </div>
            )}
            {/* Cache/call info */}
            {analysis.calls_used != null && (
              <p className="text-[10px] mt-2" style={{ color: MUTED }}>
                {analysis.from_cache
                  ? "Served from cache (0 Groq calls)"
                  : `${analysis.calls_used} Groq call${analysis.calls_used !== 1 ? "s" : ""} used${
                      analysis.trigger_fired ? ` · trigger ${analysis.trigger_fired}` : ""
                    }`}
              </p>
            )}
          </div>

          {/* Guided follow-ups — Strengthen your evidence */}
          {(analysis.guided_follow_ups?.length ?? 0) > 0 && (() => {
            const all = analysis.guided_follow_ups!;
            const dismissedIds = new Set(caseFile.requests.filter((r) => r.status === "dismissed").map((r) => r.id));
            const visible = all.filter((r) => !dismissedIds.has(r.id));
            const shown = showAllGuided ? visible : visible.slice(0, 3);
            if (!visible.length) return null;
            return (
              <div className="rounded-2xl p-4" style={{ background: "#FFFBEB", border: `1px solid #FDE68A` }}>
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles className="h-4 w-4 shrink-0" style={{ color: "#92400E" }} aria-hidden="true" />
                  <p className="text-sm font-bold" style={{ color: "#92400E" }}>Strengthen your evidence</p>
                  <span className="ml-auto text-[10px] font-semibold rounded-full px-2 py-0.5"
                    style={{ background: "#FDE68A", color: "#78350F" }}>
                    {visible.length} gap{visible.length !== 1 ? "s" : ""}
                  </span>
                </div>
                <div className="space-y-2">
                  {shown.map((req) => (
                    <div key={req.id} className="rounded-xl p-3"
                      style={{ background: "#FFFDF8", border: `1px solid #FDE68A` }}>
                      <div className="flex items-start gap-2">
                        {req.blocks_certificate && (
                          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" style={{ color: "#B91C1C" }} aria-hidden="true" />
                        )}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap mb-0.5">
                            <p className="text-sm font-semibold" style={{ color: INK }}>{req.title}</p>
                            {req.priority === 1 && (
                              <span className="text-[9px] font-bold uppercase rounded-full px-1.5 py-0.5"
                                style={{ background: "#FEE2E2", color: "#B91C1C" }}>blocks cert</span>
                            )}
                            {req.elements.length > 0 && (
                              <span className="text-[9px] font-mono rounded-full px-1.5 py-0.5"
                                style={{ background: "#F1F5F9", color: "#475569" }}>
                                {req.elements.join(" · ")}
                              </span>
                            )}
                          </div>
                          {req.why && <p className="text-xs" style={{ color: MUTED }}>{req.why}</p>}
                          {req.expected_effect && (
                            <p className="text-[11px] mt-0.5 font-medium" style={{ color: "#15803D" }}>
                              ↑ {req.expected_effect}
                            </p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 mt-2 flex-wrap">
                        <button
                          type="button"
                          onClick={() => {
                            setGuidedTarget(req);
                            setGuidedFieldValues({});
                            setGuidedUploadFile(null);
                            setGuidedUploadPreview("");
                          }}
                          className="text-xs font-semibold rounded-lg px-3 py-1.5"
                          style={{ background: GOLD_DK, color: "white" }}>
                          {req.kind === "UPLOAD" ? "Upload" : req.kind === "INPUT" ? "Provide" : "Answer"}
                        </button>
                        <button type="button" onClick={() => dismissGuidedRequest(req)}
                          className="text-xs font-semibold rounded-lg px-3 py-1.5" style={{ color: MUTED }}>
                          Dismiss
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
                {visible.length > 3 && (
                  <button type="button" onClick={() => setShowAllGuided((v) => !v)}
                    className="mt-2 text-xs font-semibold w-full text-center py-1 rounded-lg hover:bg-amber-100 transition-colors"
                    style={{ color: "#92400E" }}>
                    {showAllGuided ? "Show fewer" : `Show all ${visible.length} gaps`}
                  </button>
                )}
              </div>
            );
          })()}

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
                M-Pesa cross-check discrepancy:{" "}
                {(analysis.verification.mpesa_cross_check.codes_llm_only?.length ?? 0)} code(s) cited only in analysis,{" "}
                {(analysis.verification.mpesa_cross_check.codes_parsed_only?.length ?? 0)} only in screenshots.{" "}
                {(analysis.verification.mpesa_cross_check.findings?.filter((f: {blocks_certificate: boolean}) => f.blocks_certificate).length ?? 0)} blocking finding(s).
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

      {/* ── Case title modal (shown before saving to Firebase) ───── */}
      {titleModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "rgba(21,21,21,0.65)" }}
          role="dialog"
          aria-modal="true"
          aria-label="Name this case"
        >
          <div
            className="w-full max-w-sm rounded-2xl p-5"
            style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
          >
            <h3 className="text-sm font-bold mb-0.5" style={{ color: INK }}>
              Name this case
            </h3>
            <p className="text-xs mb-4" style={{ color: MUTED }}>
              This title groups your evidence in Evidence History. You can edit the suggestion below.
            </p>

            <div className="relative">
              <input
                type="text"
                value={pendingTitle}
                onChange={(e) => setPendingTitle(e.target.value)}
                placeholder={titleLoading ? "Generating suggestion…" : "Enter a case title"}
                disabled={titleLoading}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter" && pendingTitle.trim())
                    handleSaveEvidence(pendingTitle.trim());
                }}
                className="w-full rounded-xl px-3 py-2.5 text-sm focus:outline-none pr-9"
                style={{
                  background: CREAM2,
                  border: `1px solid ${titleLoading ? BORDER : GOLD}`,
                  color: INK,
                }}
              />
              {titleLoading && (
                <Loader2
                  className="absolute right-3 top-2.5 h-4 w-4 animate-spin"
                  style={{ color: MUTED }}
                  aria-hidden="true"
                />
              )}
            </div>

            <div className="flex justify-end gap-2 mt-4">
              <button
                type="button"
                onClick={() => setTitleModalOpen(false)}
                className="text-xs font-semibold rounded-lg px-3 py-1.5 hover:bg-amber-50 transition-colors"
                style={{ color: MUTED }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleSaveEvidence(pendingTitle.trim() || "Untitled Case")}
                disabled={titleLoading}
                className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-4 py-1.5 disabled:opacity-50 hover:brightness-110 transition-all"
                style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}
              >
                {titleLoading
                  ? <><Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" /> Generating…</>
                  : <><FolderOpen className="h-3 w-3" aria-hidden="true" /> Save Evidence</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Pending file preview lightbox ─────────────────────────── */}
      {pendingPreview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "rgba(21,21,21,0.80)" }}
          role="dialog"
          aria-modal="true"
          aria-label={`Previewing ${pendingPreview.name}`}
        >
          <div className="absolute inset-0" onClick={() => setPendingPreview(null)} aria-hidden="true" />
          <div
            className="relative z-10 max-w-2xl w-full rounded-2xl overflow-hidden"
            style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
          >
            <div
              className="flex items-center justify-between gap-3 px-4 py-3"
              style={{ background: CREAM2, borderBottom: `1px solid ${BORDER}` }}
            >
              <p className="text-xs font-semibold truncate" style={{ color: INK }}>{pendingPreview.name}</p>
              <button
                type="button"
                onClick={() => setPendingPreview(null)}
                className="shrink-0 h-7 w-7 rounded-lg flex items-center justify-center hover:bg-black/5 transition-colors"
                aria-label="Close preview"
              >
                <X className="h-4 w-4" style={{ color: MUTED }} />
              </button>
            </div>
            <div className="p-4 flex items-center justify-center overflow-auto max-h-[75vh]"
              style={{ background: CREAM2 }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={pendingPreview.url}
                alt={pendingPreview.name}
                className="max-w-full max-h-[70vh] rounded-xl object-contain"
                style={{ boxShadow: "0 4px 24px rgba(0,0,0,0.18)" }}
              />
            </div>
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
                  Confidence score (display only — eligibility decided by rules)
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
                  {Object.entries(result.elements).map(([key, el]) => {
                    const st = statusTone(el.status);
                    const needsAttention = el.status !== "PRESENT" && el.status !== "NO_ISSUE_INDICATED" && el.status !== "NOT_REQUIRED";
                    return (
                      <>
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
                        {el.reasoning && (
                          <tr key={`${key}-reason`} style={{ borderTop: "none", background: needsAttention ? "#FFFBEB" : SURFACE }}>
                            <td colSpan={3} className="px-3 pb-2.5 pt-0">
                              <p className="text-[11px] leading-relaxed" style={{ color: needsAttention ? "#78350F" : MUTED }}>
                                {el.reasoning}
                              </p>
                            </td>
                          </tr>
                        )}
                      </>
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

function PendingFileCard({
  file, previewUrl, label, onRemove, onLabelChange, onView,
}: {
  file: File;
  previewUrl: string;
  label: string;
  onRemove: () => void;
  onLabelChange: (val: string) => void;
  onView: () => void;
}) {
  return (
    <li className="rounded-2xl p-4" style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
      <div className="flex items-start gap-3">
        {/* Clickable thumbnail */}
        <button
          type="button"
          onClick={onView}
          className="group shrink-0 relative h-14 w-14 rounded-xl overflow-hidden focus:outline-none focus:ring-2 focus:ring-brand-gold"
          style={{ background: CREAM2 }}
          aria-label={`Preview ${file.name}`}
          title="Click to preview"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={previewUrl} alt={file.name} className="h-full w-full object-contain" style={{ background: CREAM2 }} />
          <div
            className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
            style={{ background: "rgba(21,21,21,0.45)" }}
            aria-hidden="true"
          >
            <ZoomIn className="h-4 w-4 text-white" />
          </div>
        </button>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold truncate" style={{ color: INK }}>{file.name}</p>
          <p className="text-xs mt-0.5" style={{ color: MUTED }}>{formatBytes(file.size)}</p>
        </div>
        <button
          type="button"
          onClick={onRemove}
          className="shrink-0 h-7 w-7 rounded-lg flex items-center justify-center transition-colors hover:bg-red-50 focus:outline-none"
          aria-label={`Remove ${file.name}`}
        >
          <X className="h-4 w-4" style={{ color: "#B91C1C" }} aria-hidden="true" />
        </button>
      </div>
      <div className="mt-3">
        <label className="block text-xs font-semibold mb-1" style={{ color: MUTED }}>
          Evidence type <span style={{ color: "#EF4444" }}>*</span>
        </label>
        <select
          value={label}
          onChange={(e) => onLabelChange(e.target.value)}
          className="w-full rounded-xl px-3 py-2 text-sm focus:outline-none"
          style={{
            background: "white",
            border: `1px solid ${label ? GOLD : BORDER}`,
            color: label ? INK : MUTED,
          }}
        >
          <option value="">Auto-detect</option>
          <option value="chat_screenshot">WhatsApp / SMS chat</option>
          <option value="mpesa_screenshot">M-Pesa screenshot</option>
          <option value="other">Other</option>
        </select>
      </div>
    </li>
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
