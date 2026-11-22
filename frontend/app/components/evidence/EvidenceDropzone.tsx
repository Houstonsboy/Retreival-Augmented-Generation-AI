"use client";

import { useRef, useCallback, useState, useEffect } from "react";
import { Upload, X, FileText, ImageIcon, Mail, Loader2 } from "lucide-react";
import {
  MAX_FILES_PER_BATCH,
  MAX_FILE_SIZE_BYTES,
  ACCEPTED_EXTENSIONS,
  ACCEPTED_MIME_TYPES,
  isAcceptedExtension,
} from "@/app/lib/evidenceConfig";
import {
  type EvidenceFileDraft,
  type EvidenceType,
  EVIDENCE_TYPE_LABELS,
} from "@/app/lib/evidenceTypes";

// ── Design tokens ──────────────────────────────────────────────────────────
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const CREAM2  = "#F5F0E4";
const INK     = "#151515";
const MUTED   = "#6B6660";
const SURFACE = "#FFFDF8";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileTypeIcon(file: File) {
  if (file.type.startsWith("image/")) return <ImageIcon className="h-5 w-5" style={{ color: GOLD }} aria-hidden="true" />;
  if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"))
    return <FileText className="h-5 w-5" style={{ color: GOLD }} aria-hidden="true" />;
  if (file.name.toLowerCase().endsWith(".eml"))
    return <Mail className="h-5 w-5" style={{ color: GOLD }} aria-hidden="true" />;
  return <FileText className="h-5 w-5" style={{ color: MUTED }} aria-hidden="true" />;
}

function isPdf(file: File) {
  return (
    file.type === "application/pdf" ||
    file.name.toLowerCase().endsWith(".pdf")
  );
}

function isImage(file: File) {
  return (
    file.type.startsWith("image/") ||
    [".png", ".jpg", ".jpeg", ".webp", ".heic"].some((e) =>
      file.name.toLowerCase().endsWith(e)
    )
  );
}

async function buildThumbnail(file: File): Promise<string | null> {
  if (!isImage(file)) return null;
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
}

interface ValidationError {
  fileName: string;
  reason: string;
}

export interface EvidenceDropzoneProps {
  files: EvidenceFileDraft[];
  onFilesChange: (files: EvidenceFileDraft[]) => void;
  isSubmitting?: boolean;
}

export function EvidenceDropzone({
  files,
  onFilesChange,
  isSubmitting = false,
}: EvidenceDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const filesRef = useRef(files);
  useEffect(() => { filesRef.current = files; }, [files]);

  const [isDragging, setIsDragging] = useState(false);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);

  const addFiles = useCallback(
    async (incoming: File[]) => {
      const errors: ValidationError[] = [];
      const toAdd: EvidenceFileDraft[] = [];
      const current = filesRef.current;

      for (const file of incoming) {
        // Too many files
        if (current.length + toAdd.length >= MAX_FILES_PER_BATCH) {
          errors.push({
            fileName: file.name,
            reason: `Batch is full (max ${MAX_FILES_PER_BATCH} files).`,
          });
          continue;
        }

        // Duplicate (same name + size)
        const isDuplicate =
          current.some((f) => f.file.name === file.name && f.file.size === file.size) ||
          toAdd.some((d) => d.file.name === file.name && d.file.size === file.size);
        if (isDuplicate) {
          errors.push({ fileName: file.name, reason: "Already added." });
          continue;
        }

        // File size
        if (file.size > MAX_FILE_SIZE_BYTES) {
          errors.push({
            fileName: file.name,
            reason: `Too large (${formatBytes(file.size)} — max ${formatBytes(MAX_FILE_SIZE_BYTES)}).`,
          });
          continue;
        }

        // Extension / MIME
        const extOk = isAcceptedExtension(file.name);
        const mimeOk = !file.type || ACCEPTED_MIME_TYPES.has(file.type);
        if (!extOk || !mimeOk) {
          errors.push({
            fileName: file.name,
            reason: `Unsupported file type. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}.`,
          });
          continue;
        }

        const thumbnail = await buildThumbnail(file);
        toAdd.push({
          id: `${file.name}-${file.size}-${Date.now()}-${Math.random()}`,
          file,
          evidenceType: "",
          passwordProtected: false,
          password: "",
          thumbnail,
          processingStatus: "queued",
        });
      }

      setValidationErrors(errors);
      if (toAdd.length) onFilesChange([...filesRef.current, ...toAdd]);
    },
    [onFilesChange]
  );

  const removeFile = useCallback(
    (id: string) => {
      onFilesChange(filesRef.current.filter((f) => f.id !== id));
    },
    [onFilesChange]
  );

  const updateFile = useCallback(
    (id: string, patch: Partial<EvidenceFileDraft>) => {
      onFilesChange(filesRef.current.map((f) => (f.id === id ? { ...f, ...patch } : f)));
    },
    [onFilesChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      addFiles(Array.from(e.dataTransfer.files));
    },
    [addFiles]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) addFiles(Array.from(e.target.files));
      e.target.value = "";
    },
    [addFiles]
  );

  const isFull = files.length >= MAX_FILES_PER_BATCH;
  const acceptAttr = ACCEPTED_EXTENSIONS.join(",");

  return (
    <div>
      {/* ── Drop zone ──────────────────────────────────────────────── */}
      <div
        role="button"
        tabIndex={isFull || isSubmitting ? -1 : 0}
        aria-label="Upload evidence files — click or drag and drop"
        aria-disabled={isFull || isSubmitting}
        onClick={() => !isFull && !isSubmitting && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !isFull && !isSubmitting)
            inputRef.current?.click();
        }}
        onDragOver={(e) => { e.preventDefault(); if (!isFull && !isSubmitting) setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={isFull || isSubmitting ? undefined : handleDrop}
        className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed transition-all py-10 px-6 cursor-pointer focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
        style={{
          borderColor: isDragging ? GOLD : BORDER,
          background: isDragging ? "#F6E7B833" : isFull ? CREAM2 : SURFACE,
          cursor: isFull || isSubmitting ? "not-allowed" : "pointer",
          opacity: isFull || isSubmitting ? 0.65 : 1,
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={acceptAttr}
          className="hidden"
          onChange={handleInputChange}
          disabled={isFull || isSubmitting}
          aria-hidden="true"
        />
        <div
          className="h-12 w-12 rounded-full flex items-center justify-center mb-4"
          style={{ background: `${GOLD}18` }}
          aria-hidden="true"
        >
          <Upload className="h-6 w-6" style={{ color: GOLD }} />
        </div>
        <p className="text-sm font-semibold text-center" style={{ color: INK }}>
          {isFull
            ? `Maximum ${MAX_FILES_PER_BATCH} files reached`
            : "Drop files here or click to browse"}
        </p>
        <p className="text-xs mt-1.5 text-center" style={{ color: MUTED }}>
          PDF, PNG, JPG, WEBP, HEIC, EML · max{" "}
          {formatBytes(MAX_FILE_SIZE_BYTES)} per file
        </p>
        {!isFull && (
          <p className="text-xs mt-1" style={{ color: MUTED }}>
            {files.length} of {MAX_FILES_PER_BATCH} files added
          </p>
        )}
      </div>

      {/* ── Validation errors ──────────────────────────────────────── */}
      {validationErrors.length > 0 && (
        <ul
          className="mt-3 space-y-1"
          role="alert"
          aria-label="File validation errors"
        >
          {validationErrors.map((err, i) => (
            <li
              key={i}
              className="text-xs px-3 py-2 rounded-lg flex items-start gap-2"
              style={{ background: "#FEF2F2", border: "1px solid #FECACA", color: "#B91C1C" }}
            >
              <span className="font-semibold shrink-0">{err.fileName}:</span>
              {err.reason}
            </li>
          ))}
        </ul>
      )}

      {/* ── File cards ─────────────────────────────────────────────── */}
      {files.length > 0 && (
        <ul className="mt-4 space-y-3" aria-label="Selected files">
          {files.map((draft) => (
            <FileCard
              key={draft.id}
              draft={draft}
              onRemove={removeFile}
              onUpdate={updateFile}
              isSubmitting={isSubmitting}
            />
          ))}
        </ul>
      )}

      {/* ── File counter ───────────────────────────────────────────── */}
      {files.length > 0 && (
        <p className="mt-3 text-xs text-right" style={{ color: MUTED }}>
          {files.length} of {MAX_FILES_PER_BATCH} files
        </p>
      )}
    </div>
  );
}

// ── Individual file card ───────────────────────────────────────────────────
function FileCard({
  draft,
  onRemove,
  onUpdate,
  isSubmitting,
}: {
  draft: EvidenceFileDraft;
  onRemove: (id: string) => void;
  onUpdate: (id: string, patch: Partial<EvidenceFileDraft>) => void;
  isSubmitting: boolean;
}) {
  const isPdfFile = isPdf(draft.file);
  const isProcessing =
    draft.processingStatus === "queued" ||
    draft.processingStatus === "reading" ||
    draft.processingStatus === "extracting" ||
    draft.processingStatus === "analysing";

  return (
    <li
      className="rounded-2xl p-4"
      style={{ background: SURFACE, border: `1px solid ${BORDER}` }}
      aria-label={`Evidence file: ${draft.file.name}`}
    >
      <div className="flex items-start gap-3">
        {/* Thumbnail or icon */}
        <div
          className="shrink-0 h-12 w-12 rounded-xl overflow-hidden flex items-center justify-center"
          style={{ background: CREAM2 }}
        >
          {draft.thumbnail ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={draft.thumbnail}
              alt=""
              className="h-full w-full object-cover"
              aria-hidden="true"
            />
          ) : (
            fileTypeIcon(draft.file)
          )}
        </div>

        {/* Name + size */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold truncate" style={{ color: INK }}>
            {draft.file.name}
          </p>
          <p className="text-xs mt-0.5" style={{ color: MUTED }}>
            {formatBytes(draft.file.size)}
          </p>

          {/* Processing chip */}
          {isSubmitting && isProcessing && (
            <span
              className="inline-flex items-center gap-1 mt-2 text-[10px] font-bold tracking-[.12em] uppercase px-2 py-0.5 rounded-full"
              style={{ background: `${GOLD}18`, color: GOLD_DK }}
            >
              <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
              {draft.processingStatus}
            </span>
          )}
        </div>

        {/* Remove button */}
        {!isSubmitting && (
          <button
            type="button"
            onClick={() => onRemove(draft.id)}
            className="shrink-0 h-7 w-7 rounded-lg flex items-center justify-center transition-colors hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-brand-gold"
            aria-label={`Remove ${draft.file.name}`}
          >
            <X className="h-4 w-4" style={{ color: "#B91C1C" }} aria-hidden="true" />
          </button>
        )}
      </div>

      {/* ── Evidence type select ─────────────────────────────────── */}
      <div className="mt-3">
        <label
          htmlFor={`ev-type-${draft.id}`}
          className="block text-xs font-semibold mb-1"
          style={{ color: MUTED }}
        >
          Evidence type <span aria-hidden="true" style={{ color: "#EF4444" }}>*</span>
          <span className="sr-only">(required)</span>
        </label>
        <select
          id={`ev-type-${draft.id}`}
          value={draft.evidenceType}
          onChange={(e) =>
            onUpdate(draft.id, { evidenceType: e.target.value as EvidenceType | "" })
          }
          disabled={isSubmitting}
          required
          className="w-full text-sm rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-gold disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            background: "white",
            border: `1px solid ${draft.evidenceType ? GOLD : BORDER}`,
            color: draft.evidenceType ? INK : MUTED,
          }}
          aria-required="true"
        >
          <option value="" disabled>
            Select evidence type…
          </option>
          {(Object.keys(EVIDENCE_TYPE_LABELS) as EvidenceType[]).map((key) => (
            <option key={key} value={key}>
              {EVIDENCE_TYPE_LABELS[key]}
            </option>
          ))}
        </select>
      </div>

      {/* ── PDF password (PDFs only) ─────────────────────────────── */}
      {isPdfFile && (
        <div className="mt-3">
          <label className="flex items-center gap-2 text-xs font-medium cursor-pointer select-none" style={{ color: MUTED }}>
            <input
              type="checkbox"
              checked={draft.passwordProtected}
              onChange={(e) => onUpdate(draft.id, { passwordProtected: e.target.checked })}
              disabled={isSubmitting}
              className="rounded"
              style={{ accentColor: GOLD }}
            />
            This PDF is password-protected
          </label>
          {draft.passwordProtected && (
            <div className="mt-2">
              <label
                htmlFor={`ev-pw-${draft.id}`}
                className="block text-xs font-semibold mb-1"
                style={{ color: MUTED }}
              >
                PDF password
              </label>
              <input
                id={`ev-pw-${draft.id}`}
                type="password"
                autoComplete="off"
                value={draft.password}
                onChange={(e) => onUpdate(draft.id, { password: e.target.value })}
                disabled={isSubmitting}
                placeholder="e.g. your ID number"
                className="w-full text-sm rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-gold disabled:opacity-50"
                style={{
                  background: "white",
                  border: `1px solid ${BORDER}`,
                  color: INK,
                }}
              />
              <p className="mt-1 text-xs" style={{ color: MUTED }}>
                Safaricom statements are usually protected with your ID number.
                Your password is never stored or logged.
              </p>
            </div>
          )}
        </div>
      )}
    </li>
  );
}
