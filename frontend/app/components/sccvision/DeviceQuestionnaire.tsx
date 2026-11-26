"use client";

import { useState } from "react";
import { ArrowLeft, Download, Loader2 } from "lucide-react";
import type { DeviceFacts, ProductionMethod, CertificateRecord } from "@/app/lib/evidenceTypes";
import { createCertificate, downloadCertificate } from "@/app/lib/evidenceApi";
import { internationalPhoneNumber, phoneCountries } from "@/app/lib/phoneCountries";
import { Choices, type DeviceAnswers } from "./Workshop";

const METHODS: { value: ProductionMethod; label: string }[] = [
  { value: "screenshot_on_device", label: "Screenshot on this device" },
  { value: "printed_statement", label: "Printed statement" },
  { value: "forwarded_or_copied", label: "Forwarded or copied" },
  { value: "other", label: "Another method" },
];

const CHECKS = [
  ["owner_is_declarant", "Are you the owner or controller of this device?"],
  ["lawful_control", "Did you have lawful control of this device during the relevant period?"],
  ["regular_use", "Did you regularly use this device during that period?"],
  ["regular_feed", "Were records of this kind regularly stored on this device?"],
  ["operating_properly", "Was the device working normally throughout the relevant period?"],
  ["messages_on_this_device", "Were these messages or payments sent or received directly on this device?"],
] as const;

const YES_NO = [{ value: "yes", label: "Yes" }, { value: "no", label: "No" }];

function answer(value: boolean | undefined) {
  return value === undefined ? undefined : value ? "yes" : "no";
}

type CertificateDraft = Partial<DeviceFacts> & { phoneRegion: string; phoneInput: string };

interface Props {
  analysisId: string;
  answers: DeviceAnswers;
  fileNames: string[];
  record: CertificateRecord | null;
  onClose: () => void;
  onSuccess: (record: CertificateRecord) => void;
}

export function DeviceQuestionnaire({ analysisId, answers, fileNames, record, onClose, onSuccess }: Props) {
  const storageKey = `scc-certificate-draft-${analysisId}`;
  const [draft, setDraft] = useState<CertificateDraft>(() => {
    const initial = {
      phoneRegion: "KE",
      phoneInput: "",
      owner_is_declarant: answers.source === "My own phone" ? true : undefined,
      operating_properly: answers.operation === "Yes" ? true : answers.operation === "No" ? false : undefined,
    };
    try {
      const saved = sessionStorage.getItem(storageKey);
      return saved ? { ...initial, ...JSON.parse(saved), declaration_confirmed: false } : initial;
    } catch {
      return initial;
    }
  });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const set = <K extends keyof CertificateDraft>(key: K, value: CertificateDraft[K]) => {
    setDraft((current) => ({ ...current, [key]: value }));
    setMessage("");
    setError("");
  };

  const malfunction = draft.operating_properly === false;
  const documentLabel = record?.document_type === "DEFICIENCY_REPORT" ? "report" : "certificate";
  const device = [draft.device_make?.trim(), draft.device_model?.trim()].filter(Boolean).join(" ");

  const phoneNumber = internationalPhoneNumber(draft.phoneInput, draft.phoneRegion);

  function saveDraft() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify({ ...draft, declaration_confirmed: false }));
      setError("");
      setMessage("Draft saved in this browser tab.");
    } catch {
      setError("This browser could not save the draft.");
    }
  }

  async function submit() {
    setError("");
    setMessage("");
    setBusy(true);
    try {
      let generated = record;
      if (!generated) {
        const f = draft;
        if (
          !f.declarant_name?.trim() || !f.declarant_position?.trim() ||
          !f.device_make?.trim() || !f.device_model?.trim() || !f.production_method ||
          f.owner_is_declarant == null || f.lawful_control == null || f.regular_use == null ||
          f.regular_feed == null || f.operating_properly == null || f.messages_on_this_device == null ||
          !f.declaration_confirmed ||
          (malfunction && (!f.malfunction_details?.trim() || f.malfunction_affects_record == null))
        ) {
          setError("Complete your details, answer the device checks and confirm the declaration before downloading.");
          return;
        }
        if (f.phoneInput.trim() && !phoneNumber) {
          setError("Enter the full phone number for the selected country, or leave it blank.");
          return;
        }
        generated = await createCertificate(analysisId, {
          device_make: f.device_make.trim(),
          device_model: f.device_model.trim(),
          phone_number: phoneNumber,
          production_method: f.production_method,
          owner_is_declarant: f.owner_is_declarant,
          lawful_control: f.lawful_control,
          regular_use: f.regular_use,
          regular_feed: f.regular_feed,
          operating_properly: f.operating_properly,
          malfunction_details: malfunction ? f.malfunction_details?.trim() ?? null : null,
          malfunction_affects_record: malfunction ? f.malfunction_affects_record ?? null : null,
          messages_on_this_device: f.messages_on_this_device,
          declarant_name: f.declarant_name.trim(),
          declarant_position: f.declarant_position.trim(),
          declaration_confirmed: f.declaration_confirmed,
        });
        onSuccess(generated);
        try { sessionStorage.removeItem(storageKey); } catch { /* Browser storage can be unavailable. */ }
      }
      await downloadCertificate(generated.cert_id, generated.certificate_number);
      setMessage(generated.document_type === "CERTIFICATE"
        ? "Your certificate draft is ready. Print it and sign it by hand."
        : "Your report explains which requirements still need attention.");
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "The PDF could not be prepared. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={(event) => { event.preventDefault(); void submit(); }} noValidate>
      {record ? (
        <div className="scc-card space-y-2">
          <h2 className="font-serif text-xl">{record.document_type === "CERTIFICATE" ? "Your certificate draft is ready" : "Your evidence needs more information"}</h2>
          <p className="text-sm text-stone-700">{record.certificate_number}. You can also find it under <a className="underline" href="/litigant/certificates">Certificates</a>.</p>
        </div>
      ) : (
        <div className="grid items-start gap-4 md:grid-cols-[1fr_1.08fr]">
          <fieldset disabled={busy} className="scc-card space-y-3">
            <legend className="sr-only">Certificate details</legend>
            <label className="block text-xs font-semibold">
              Your full name
              <input className="scc-field mt-1" autoComplete="name" required value={draft.declarant_name ?? ""}
                placeholder="As it appears on your ID" onChange={(e) => set("declarant_name", e.target.value)} />
            </label>
            <label className="block text-xs font-semibold">
              Your role in relation to the device
              <input className="scc-field mt-1" required value={draft.declarant_position ?? ""}
                placeholder={answers.use === "Only me" ? "e.g. Sole user of the device" : "e.g. Regular user of the device"}
                onChange={(e) => set("declarant_position", e.target.value)} />
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="block text-xs font-semibold">
                Device make
                <input className="scc-field mt-1" required value={draft.device_make ?? ""} placeholder="e.g. Samsung"
                  onChange={(e) => set("device_make", e.target.value)} />
              </label>
              <label className="block text-xs font-semibold">
                Device model
                <input className="scc-field mt-1" required value={draft.device_model ?? ""} placeholder="e.g. Galaxy A14"
                  onChange={(e) => set("device_model", e.target.value)} />
              </label>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="block min-w-0 text-xs font-semibold">
                Country code
                <select className="scc-field mt-1" autoComplete="tel-country-code" value={draft.phoneRegion}
                  onChange={(e) => set("phoneRegion", e.target.value)}>
                  {phoneCountries.map(({ region, name, code }) => (
                    <option key={region} value={region}>{name} ({code})</option>
                  ))}
                </select>
              </label>
              <label className="block min-w-0 text-xs font-semibold">
                Phone number <span className="font-normal">(optional)</span>
                <input className="scc-field mt-1" type="tel" autoComplete="tel-national" maxLength={32}
                  value={draft.phoneInput} placeholder={draft.phoneRegion === "KE" ? "e.g. 0712 345 678" : undefined}
                  onChange={(e) => set("phoneInput", e.target.value)} />
              </label>
            </div>
            <label className="block text-xs font-semibold">
              How was the evidence produced?
              <select className="scc-field mt-1" required value={draft.production_method ?? ""}
                onChange={(e) => set("production_method", e.target.value as ProductionMethod)}>
                <option value="" disabled>Choose a method</option>
                {METHODS.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <details className="scc-device-checks scc-disclosure rounded-lg border border-stone-200 p-3" open>
              <summary>Review the device checks</summary>
              <div className="mt-3 space-y-2">
                {CHECKS.map(([key, label]) => (
                  <Choices key={key} name={key} legend={label} value={answer(draft[key])} options={YES_NO}
                    onChange={(value) => set(key, value === "yes")} />
                ))}
                {malfunction && (
                  <>
                    <label className="block text-xs font-semibold">
                      What went wrong with the device?
                      <textarea className="scc-field mt-1" required rows={2} value={draft.malfunction_details ?? ""}
                        onChange={(e) => set("malfunction_details", e.target.value)} />
                    </label>
                    <Choices name="malfunction-affects-record" legend="Did the problem affect the accuracy of the records?"
                      value={answer(draft.malfunction_affects_record ?? undefined)} options={YES_NO}
                      onChange={(value) => set("malfunction_affects_record", value === "yes")} />
                  </>
                )}
              </div>
            </details>
            <label className="scc-note flex items-start gap-2 rounded-lg p-3">
              <input type="checkbox" required className="mt-0.5" checked={draft.declaration_confirmed ?? false}
                onChange={(e) => set("declaration_confirmed", e.target.checked)} />
              <span>I confirm that these details are true to the best of my knowledge and belief. I understand that I must review and sign the certificate by hand before using it in court.</span>
            </label>
          </fieldset>

          <div className="space-y-2 md:sticky md:top-4">
            <p className="scc-kicker">Certificate preview</p>
            <div className="scc-card scc-preview space-y-4">
              <div className="space-y-1 text-center uppercase">
                <p className="text-xs">Republic of Kenya</p>
                <p className="font-bold">Certificate under Section 106B of the Evidence Act</p>
              </div>
              <p>I, <mark>{draft.declarant_name?.trim() || "[your full name]"}</mark>, acting as <mark>{draft.declarant_position?.trim() || "[your role]"}</mark>, provide the following details about the electronic evidence.</p>
              <ol className="list-decimal space-y-2 pl-5">
                <li>The records listed are {fileNames.length} screenshot{fileNames.length === 1 ? "" : "s"}.
                  <ul className="list-disc break-all pl-5 text-xs text-stone-700">
                    {fileNames.map((name, index) => <li key={index}>{name}</li>)}
                  </ul>
                </li>
                <li>Device used: <mark>{device || "[device make and model]"}</mark>{phoneNumber && <> · phone number <mark>{phoneNumber}</mark></>}.</li>
                <li>Production method: <mark>{METHODS.find((method) => method.value === draft.production_method)?.label ?? "[production method]"}</mark>.</li>
                <li>Device operation: <mark>{draft.operating_properly === undefined ? "[device condition]" : draft.operating_properly ? "Working normally" : "Device problem reported"}</mark>.</li>
              </ol>
              <div className="grid grid-cols-2 gap-5 pt-6 text-xs">
                <div className="border-t border-stone-400 pt-1">Signature</div>
                <div className="border-t border-stone-400 pt-1">Date</div>
              </div>
            </div>
            <p className="text-xs leading-relaxed text-stone-600">Highlighted text comes from your answers. Your answers and evidence are checked before a certificate or a report is generated.</p>
          </div>
        </div>
      )}
      {error && <p className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800" role="alert">{error}</p>}
      {message && <p className="scc-note mt-3 rounded-lg p-3" role="status">{message}</p>}
      <div className="scc-actions scc-footer">
        <button type="button" className="scc-button mr-auto" onClick={onClose} disabled={busy}><ArrowLeft size={15} aria-hidden="true" /> Back</button>
        {!record && <button type="button" className="scc-button" onClick={saveDraft} disabled={busy}>Save draft</button>}
        <button type="submit" className="scc-button scc-primary" disabled={busy}>
          {busy ? <><Loader2 size={15} className="animate-spin" aria-hidden="true" /> Preparing PDF…</> : <><Download size={15} aria-hidden="true" /> Download {documentLabel} (PDF)</>}
        </button>
      </div>
    </form>
  );
}
