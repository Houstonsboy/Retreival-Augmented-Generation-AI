"use client";

import { useState } from "react";
import {
  Upload, ScanSearch, MessageSquare, ScrollText,
  AlertCircle, Lock, CheckCircle2, Users, User,
} from "lucide-react";
import {
  EvidenceDropzone,
} from "@/app/components/evidence/EvidenceDropzone";
import type { EvidenceFileDraft } from "@/app/lib/evidenceTypes";
import { useAuth } from "@/app/context/AuthContext";
import { SCCVisionTest } from "@/app/components/sccvision/SCCVisionTest";

// ── Design tokens ──────────────────────────────────────────────────────────
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const CREAM2  = "#F5F0E4";
const INK     = "#151515";
const MUTED   = "#6B6660";
const SURFACE = "#FFFDF8";
const NR      = "var(--font-newsreader), Georgia, 'Times New Roman', serif";
const MNR     = "var(--font-manrope), system-ui, sans-serif";

// ── 3-step visual ─────────────────────────────────────────────────────────
const STEPS = [
  { Icon: Upload,        label: "Upload",               desc: "Add your files" },
  { Icon: ScanSearch,    label: "We read it",            desc: "Extract & check evidence" },
  { Icon: MessageSquare, label: "Answer a few questions",desc: "About your device or phone" },
  { Icon: ScrollText,    label: "Certificate or guidance",desc: "Signed draft or next steps" },
];

export default function EvidenceAnalysisPage() {
  const { profile } = useAuth();
  const isAdvocate = profile?.type === "advocate" || profile?.type === "admin";

  const [files, setFiles] = useState<EvidenceFileDraft[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  // TODO(backend): include preparedFor in the POST /api/evidence/upload payload.
  const [preparedFor, setPreparedFor] = useState<"self" | "client">("self");

  const allTypesSelected = files.length > 0 && files.every((f) => f.evidenceType !== "");
  const canSubmit = allTypesSelected && !isSubmitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setIsSubmitting(true);

    // Mark all as queued → reading in UI
    setFiles((prev) =>
      prev.map((f) => ({ ...f, processingStatus: "reading" as const }))
    );

    // TODO(backend): POST /api/evidence/upload
    //   body: FormData with each file appended as "files[]", evidence types as
    //   "types[<index>]", passwords as "passwords[<index>]" (when set), and
    //   a single "preparedFor" field ("self" | "client") for the whole batch.
    //   Expected response: { records: EvidenceRecord[] }
    //   On success, navigate to /litigant/evidence or the individual record page.
    //
    // Example stub:
    // const fd = new FormData();
    // fd.append("preparedFor", preparedFor);   // once for the batch
    // files.forEach((d, i) => {
    //   fd.append("files[]", d.file);
    //   fd.append(`types[${i}]`, d.evidenceType);
    //   if (d.passwordProtected && d.password) fd.append(`passwords[${i}]`, d.password);
    // });
    // const res = await apiFetch("/api/evidence/upload", { method: "POST", body: fd });
    // const { records } = await res.json();
    // router.push("/litigant/evidence");

    // TODO(backend): POST /api/evidence/upload is not yet implemented.
    // Once implemented, navigate to /litigant/evidence on success.
    setIsSubmitting(false);
  };

  return (
    <div className="px-5 py-8 max-w-2xl mx-auto" style={{ fontFamily: MNR }}>

      {/* ── Hero ──────────────────────────────────────────────────── */}
      <div className="mb-10">
        <p
          className="text-[10px] font-bold tracking-[.24em] uppercase mb-3"
          style={{ color: GOLD }}
        >
          Small Claims Court · Kenya
        </p>
        <h1
          className="text-3xl font-bold leading-snug mb-3"
          style={{ color: INK, fontFamily: NR }}
        >
          Prepare your{" "}
          <span style={{ color: GOLD }}>digital evidence</span>
        </h1>
        <p className="text-sm leading-relaxed max-w-lg" style={{ color: MUTED }}>
          Upload your M-Pesa statements or messages, WhatsApp or SMS
          screenshots, emails, and photos of written agreements. The system
          reads each file, checks how complete and trustworthy it is against
          <strong className="font-semibold" style={{ color: INK }}>
            {" "}Section 106B
          </strong>{" "}
          and{" "}
          <strong className="font-semibold" style={{ color: INK }}>
            Section 78A
          </strong>{" "}
          of the Evidence Act, and then either drafts a certificate for you to
          sign or explains what&apos;s missing and how to strengthen your
          evidence.
        </p>
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

      {/* ── Disclaimers ───────────────────────────────────────────── */}
      <div className="space-y-2 mb-8">
        <div
          className="flex items-start gap-2.5 rounded-xl px-4 py-3 text-xs"
          style={{ background: "#FFF8E6", border: `1px solid #F6E7B8`, color: GOLD_DK }}
        >
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" style={{ color: GOLD }} aria-hidden="true" />
          <span>
            <strong>Not legal advice.</strong> This tool helps you prepare
            evidence. It does not give legal advice, and the court decides what
            evidence to accept.
          </span>
        </div>
        <div
          className="flex items-start gap-2.5 rounded-xl px-4 py-3 text-xs"
          style={{ background: SURFACE, border: `1px solid ${BORDER}`, color: MUTED }}
        >
          <Lock className="h-4 w-4 shrink-0 mt-0.5" style={{ color: MUTED }} aria-hidden="true" />
          Your files are private to your account.
        </div>
      </div>

      {/* ── Upload section ────────────────────────────────────────── */}
      <section aria-labelledby="upload-heading">
        <h2
          id="upload-heading"
          className="text-base font-bold mb-4"
          style={{ color: INK }}
        >
          Upload evidence files
        </h2>

        {/* Advocate-only: who this evidence is being prepared for */}
        {isAdvocate && (
          <div
            className="mb-5 rounded-2xl px-5 py-4"
            style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
            aria-label="Prepared for"
          >
            <p className="text-xs font-semibold mb-3" style={{ color: MUTED }}>
              Preparing this evidence for:
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setPreparedFor("self")}
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-brand-gold"
                style={{
                  background: preparedFor === "self" ? GOLD : CREAM2,
                  color: preparedFor === "self" ? "white" : MUTED,
                  border: `1.5px solid ${preparedFor === "self" ? GOLD : BORDER}`,
                }}
                aria-pressed={preparedFor === "self"}
              >
                <User className="h-3.5 w-3.5" aria-hidden="true" />
                Myself
              </button>
              <button
                type="button"
                onClick={() => setPreparedFor("client")}
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-brand-gold"
                style={{
                  background: preparedFor === "client" ? GOLD : CREAM2,
                  color: preparedFor === "client" ? "white" : MUTED,
                  border: `1.5px solid ${preparedFor === "client" ? GOLD : BORDER}`,
                }}
                aria-pressed={preparedFor === "client"}
              >
                <Users className="h-3.5 w-3.5" aria-hidden="true" />
                A client
              </button>
            </div>
            {preparedFor === "client" && (
              <p className="mt-3 text-xs leading-relaxed" style={{ color: MUTED }}>
                If it&apos;s for a client, the certificate must be signed by your
                client, because they control the device.
              </p>
            )}
          </div>
        )}

        <EvidenceDropzone
          files={files}
          onFilesChange={setFiles}
          isSubmitting={isSubmitting}
        />

        {/* Submit button */}
        <div className="mt-6">
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="w-full flex items-center justify-center gap-2.5 rounded-2xl py-3.5 text-sm font-bold transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
            style={{
              background: canSubmit
                ? `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`
                : CREAM2,
              color: canSubmit ? "white" : MUTED,
            }}
            aria-disabled={!canSubmit}
          >
            {isSubmitting ? (
              <>
                <span
                  className="h-5 w-5 rounded-full border-2 border-white border-t-transparent animate-spin"
                  aria-hidden="true"
                />
                Analysing evidence…
              </>
            ) : (
              <>
                <CheckCircle2 className="h-5 w-5" aria-hidden="true" />
                Analyse evidence
              </>
            )}
          </button>
          {files.length > 0 && !allTypesSelected && !isSubmitting && (
            <p className="mt-2 text-center text-xs" style={{ color: MUTED }}>
              Select an evidence type for every file to continue.
            </p>
          )}
        </div>
      </section>

      {/* ── SCCVision OCR test panel ──────────────────────────────── */}
      <SCCVisionTest />
    </div>
  );
}
