"use client";

/**
 * DeviceQuestionnaire — collects DeviceFacts for Section 106B gate.
 *
 * Shows after a successful analysis. The user fills in device details,
 * confirms the statutory declarations, then submits. On success the caller
 * receives the created CertificateRecord.
 */

import { useState } from "react";
import { X, Loader2, ShieldCheck, AlertCircle } from "lucide-react";
import type { DeviceFacts, ProductionMethod, CertificateRecord } from "@/app/lib/evidenceTypes";
import { createCertificate } from "@/app/lib/evidenceApi";

// ── Design tokens (match SCCVisionTest) ────────────────────────────────────
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const CREAM2  = "#F5F0E4";
const INK     = "#151515";
const MUTED   = "#6B6660";
const SURFACE = "#FFFDF8";

const PRODUCTION_METHOD_OPTIONS: { value: ProductionMethod; label: string; hint: string }[] = [
  {
    value: "screenshot_on_device",
    label: "Screenshot on this device",
    hint: "I took a screenshot on the phone that holds the messages",
  },
  {
    value: "printed_statement",
    label: "Printed statement",
    hint: "The phone produced a printable statement",
  },
  {
    value: "forwarded_or_copied",
    label: "Forwarded or copied",
    hint: "The file was forwarded or copied from another phone",
  },
  {
    value: "other",
    label: "Other",
    hint: "Some other method not listed above",
  },
];

// ── Helpers ────────────────────────────────────────────────────────────────

function FieldLabel({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="mb-1">
      <span className="text-sm font-semibold" style={{ color: INK }}>{label}</span>
      {hint && <span className="text-xs ml-2" style={{ color: MUTED }}>{hint}</span>}
    </div>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  error,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  error?: boolean;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full rounded-xl px-3 py-2 text-sm focus:outline-none"
      style={{
        background: SURFACE,
        border: `1.5px solid ${error ? "#EF4444" : BORDER}`,
        color: INK,
      }}
    />
  );
}

function YesNo({
  value,
  onChange,
  yesLabel = "Yes",
  noLabel = "No",
}: {
  value: boolean | null;
  onChange: (v: boolean) => void;
  yesLabel?: string;
  noLabel?: string;
}) {
  return (
    <div className="flex gap-2">
      {[true, false].map((opt) => (
        <button
          key={String(opt)}
          type="button"
          onClick={() => onChange(opt)}
          className="flex-1 rounded-xl py-2 text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-offset-1"
          style={{
            background: value === opt ? GOLD : CREAM2,
            color: value === opt ? "white" : MUTED,
            border: `1.5px solid ${value === opt ? GOLD : BORDER}`,
          }}
          aria-pressed={value === opt}
        >
          {opt ? yesLabel : noLabel}
        </button>
      ))}
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────────────

interface Props {
  analysisId: string;
  onClose: () => void;
  onSuccess: (record: CertificateRecord) => void;
}

type Draft = Partial<DeviceFacts> & {
  phone_number_last4: string; // string in draft (empty = not provided)
};

function emptyDraft(): Draft {
  return {
    device_make: "",
    device_model: "",
    phone_number_last4: "",
    production_method: undefined,
    owner_is_declarant: undefined,
    lawful_control: undefined,
    regular_use: undefined,
    regular_feed: undefined,
    operating_properly: undefined,
    malfunction_details: "",
    malfunction_affects_record: undefined,
    messages_on_this_device: undefined,
    declarant_name: "",
    declarant_position: "",
    declaration_confirmed: false,
  };
}

export function DeviceQuestionnaire({ analysisId, onClose, onSuccess }: Props) {
  const [draft, setDraft] = useState<Draft>(emptyDraft());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = <K extends keyof Draft>(k: K, v: Draft[K]) =>
    setDraft((d) => ({ ...d, [k]: v }));

  const malfunction = draft.operating_properly === false;

  function buildFacts(): DeviceFacts | null {
    const f = draft;
    if (
      !f.device_make?.trim() ||
      !f.device_model?.trim() ||
      f.production_method == null ||
      f.owner_is_declarant == null ||
      f.lawful_control == null ||
      f.regular_use == null ||
      f.regular_feed == null ||
      f.operating_properly == null ||
      f.messages_on_this_device == null ||
      !f.declarant_name?.trim() ||
      !f.declarant_position?.trim() ||
      !f.declaration_confirmed
    ) {
      return null;
    }
    if (malfunction && (!f.malfunction_details?.trim() || f.malfunction_affects_record == null)) {
      return null;
    }
    const last4 = f.phone_number_last4?.trim() ?? "";
    return {
      device_make: f.device_make.trim(),
      device_model: f.device_model.trim(),
      phone_number_last4: last4.length === 4 && /^\d{4}$/.test(last4) ? last4 : null,
      production_method: f.production_method,
      owner_is_declarant: f.owner_is_declarant,
      lawful_control: f.lawful_control,
      regular_use: f.regular_use,
      regular_feed: f.regular_feed,
      operating_properly: f.operating_properly,
      malfunction_details: malfunction ? (f.malfunction_details ?? null) : null,
      malfunction_affects_record: malfunction ? (f.malfunction_affects_record ?? null) : null,
      messages_on_this_device: f.messages_on_this_device,
      declarant_name: f.declarant_name.trim(),
      declarant_position: f.declarant_position.trim(),
      declaration_confirmed: f.declaration_confirmed,
    };
  }

  async function handleSubmit() {
    const facts = buildFacts();
    if (!facts) {
      setError("Please complete all required fields before submitting.");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const record = await createCertificate(analysisId, facts);
      onSuccess(record);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "An unexpected error occurred.");
    } finally {
      setSubmitting(false);
    }
  }

  // Prevent background scroll
  // (useEffect runs client-side only; inline style is enough here)

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4 sm:p-8"
      style={{ background: "rgba(21,21,21,0.65)" }}
      role="dialog"
      aria-modal="true"
      aria-label="Section 106B device questionnaire"
    >
      {/* Backdrop */}
      <div className="absolute inset-0" onClick={onClose} aria-hidden="true" />

      {/* Modal card */}
      <div
        className="relative z-10 w-full max-w-2xl rounded-2xl shadow-2xl my-4"
        style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
      >
        {/* Toolbar */}
        <div
          className="flex items-center justify-between gap-3 px-5 py-3 rounded-t-2xl"
          style={{ background: CREAM2, borderBottom: `1px solid ${BORDER}` }}
        >
          <div>
            <p className="text-sm font-bold" style={{ color: INK }}>
              Prepare Section 106B Certificate
            </p>
            <p className="text-xs" style={{ color: MUTED }}>
              Answer the questions below about the device that produced your evidence
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 hover:bg-amber-50 transition-colors"
            style={{ color: MUTED }}
          >
            <X className="h-3.5 w-3.5" aria-hidden="true" />
            Close
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-6 space-y-6">

          {/* Device details */}
          <section>
            <h2 className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: GOLD_DK }}>
              Device details
            </h2>
            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <FieldLabel label="Phone make *" hint="e.g. Samsung, Tecno" />
                <TextInput
                  value={draft.device_make ?? ""}
                  onChange={(v) => set("device_make", v)}
                  placeholder="Samsung"
                  error={!draft.device_make?.trim() && submitting}
                />
              </div>
              <div>
                <FieldLabel label="Phone model *" hint="e.g. Galaxy A14" />
                <TextInput
                  value={draft.device_model ?? ""}
                  onChange={(v) => set("device_model", v)}
                  placeholder="Galaxy A14"
                />
              </div>
              <div>
                <FieldLabel label="Phone number — last 4 digits only" hint="optional" />
                <TextInput
                  value={draft.phone_number_last4}
                  onChange={(v) => set("phone_number_last4", v.replace(/\D/g, "").slice(0, 4))}
                  placeholder="e.g. 6789"
                />
              </div>
            </div>
          </section>

          {/* How was the evidence produced */}
          <section>
            <h2 className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: GOLD_DK }}>
              How was the evidence produced? *
            </h2>
            <div className="grid sm:grid-cols-2 gap-2">
              {PRODUCTION_METHOD_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => set("production_method", opt.value)}
                  className="text-left rounded-xl p-3 transition-all focus:outline-none focus:ring-2 focus:ring-offset-1"
                  style={{
                    background: draft.production_method === opt.value ? "#FFF8E6" : CREAM2,
                    border: `1.5px solid ${draft.production_method === opt.value ? GOLD : BORDER}`,
                  }}
                  aria-pressed={draft.production_method === opt.value}
                >
                  <p className="text-sm font-semibold" style={{ color: INK }}>{opt.label}</p>
                  <p className="text-xs mt-0.5" style={{ color: MUTED }}>{opt.hint}</p>
                </button>
              ))}
            </div>
          </section>

          {/* Statutory conditions */}
          <section>
            <h2 className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: GOLD_DK }}>
              Statutory conditions (s.106B)
            </h2>
            <div className="space-y-4">

              <div>
                <FieldLabel
                  label="Are you the owner / controller of this phone? *"
                  hint="s.106B(2)(a)"
                />
                <YesNo
                  value={draft.owner_is_declarant ?? null}
                  onChange={(v) => set("owner_is_declarant", v)}
                />
              </div>

              <div>
                <FieldLabel
                  label="Did you have lawful control of this phone during the relevant period? *"
                  hint="s.106B(2)(a)"
                />
                <YesNo
                  value={draft.lawful_control ?? null}
                  onChange={(v) => set("lawful_control", v)}
                />
              </div>

              <div>
                <FieldLabel
                  label="Did you regularly use this phone during the relevant period? *"
                  hint="s.106B(2)(a)"
                />
                <YesNo
                  value={draft.regular_use ?? null}
                  onChange={(v) => set("regular_use", v)}
                />
              </div>

              <div>
                <FieldLabel
                  label="Were messages / M-Pesa payments of this kind regularly stored on this phone? *"
                  hint="s.106B(2)(b)"
                />
                <YesNo
                  value={draft.regular_feed ?? null}
                  onChange={(v) => set("regular_feed", v)}
                />
              </div>

              <div>
                <FieldLabel
                  label="Was the phone operating properly throughout the relevant period? *"
                  hint="s.106B(2)(c)"
                />
                <YesNo
                  value={draft.operating_properly ?? null}
                  onChange={(v) => set("operating_properly", v)}
                />
              </div>

              {malfunction && (
                <div
                  className="rounded-xl p-4 space-y-3"
                  style={{ background: "#FFF8E6", border: `1px solid #FDE68A` }}
                >
                  <p className="text-xs font-bold" style={{ color: "#92400E" }}>
                    Describe the malfunction briefly *
                  </p>
                  <TextInput
                    value={draft.malfunction_details ?? ""}
                    onChange={(v) => set("malfunction_details", v)}
                    placeholder="e.g. Screen cracked, some messages may be missing"
                  />
                  <p className="text-xs font-bold mt-2" style={{ color: "#92400E" }}>
                    Did this malfunction affect the accuracy of the records you are relying on? *
                  </p>
                  <YesNo
                    value={draft.malfunction_affects_record ?? null}
                    onChange={(v) => set("malfunction_affects_record", v)}
                  />
                </div>
              )}

              <div>
                <FieldLabel
                  label="Were the messages / payments sent or received directly on this phone? *"
                  hint="s.106B(2)(d)"
                />
                <YesNo
                  value={draft.messages_on_this_device ?? null}
                  onChange={(v) => set("messages_on_this_device", v)}
                />
              </div>

            </div>
          </section>

          {/* Declaration */}
          <section>
            <h2 className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: GOLD_DK }}>
              Your details (for the declaration)
            </h2>
            <div className="grid sm:grid-cols-2 gap-4 mb-4">
              <div>
                <FieldLabel label="Full name *" />
                <TextInput
                  value={draft.declarant_name ?? ""}
                  onChange={(v) => set("declarant_name", v)}
                  placeholder="Jane Muthoni"
                />
              </div>
              <div>
                <FieldLabel label="Your position / role *" hint="e.g. Owner and sole user of the phone" />
                <TextInput
                  value={draft.declarant_position ?? ""}
                  onChange={(v) => set("declarant_position", v)}
                  placeholder="Owner and sole user of the phone"
                />
              </div>
            </div>

            {/* Declaration checkbox */}
            <label
              className="flex items-start gap-3 rounded-xl p-4 cursor-pointer"
              style={{ background: CREAM2, border: `1.5px solid ${draft.declaration_confirmed ? GOLD : BORDER}` }}
            >
              <input
                type="checkbox"
                checked={draft.declaration_confirmed}
                onChange={(e) => set("declaration_confirmed", e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded accent-amber-500"
              />
              <span className="text-xs leading-relaxed" style={{ color: INK }}>
                I confirm that the statements I have made above are <strong>true to the best of my
                knowledge and belief</strong>. I understand that this draft must be{" "}
                <strong>reviewed and signed by hand</strong> before it is used in court proceedings.
              </span>
            </label>
          </section>

          {/* Error */}
          {error && (
            <div
              className="flex items-start gap-2 rounded-xl px-4 py-3 text-sm"
              style={{ background: "#FEF2F2", border: "1px solid #FECACA", color: "#B91C1C" }}
              role="alert"
            >
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" aria-hidden="true" />
              {error}
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="text-xs font-semibold rounded-lg px-4 py-2"
              style={{ color: MUTED }}
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting}
              className="flex items-center gap-2 text-sm font-bold rounded-xl px-5 py-2.5 transition-all disabled:opacity-60 hover:brightness-105"
              style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}
            >
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  Generating…
                </>
              ) : (
                <>
                  <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                  Generate Certificate Draft
                </>
              )}
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}
