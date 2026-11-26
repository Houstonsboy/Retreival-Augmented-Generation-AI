"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import {
  collection,
  query,
  orderBy,
  onSnapshot,
  Timestamp,
} from "firebase/firestore";
import { db } from "@/app/lib/firebase";
import { useAuth } from "@/app/context/AuthContext";
import { apiFetch } from "@/app/lib/api";
import { UserMenu } from "@/app/components/UserMenu";
import {
  Scale,
  Upload,
  FileText,
  ArrowLeft,
  ExternalLink,
  X,
  File,
  AlertCircle,
  Loader2,
} from "lucide-react";


interface CaseLawDoc {
  id: string;
  title: string;
  fileName: string;
  fileType: "pdf" | "docx";
  localPath: string;
  uploadedBy: string;
  uploadedAt: Timestamp | null;
  fileSizeBytes: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(ts: Timestamp | null): string {
  if (!ts) return "—";
  const d = ts.toDate();
  return d.toLocaleDateString("en-KE", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function FileBadge({ type }: { type: string }) {
  const isPdf = type === "pdf";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium uppercase tracking-wide ${
        isPdf
          ? "bg-red-50 text-red-600 dark:bg-red-900/20 dark:text-red-400"
          : "bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
      }`}
    >
      {type}
    </span>
  );
}

export default function CaseLawsPage() {
  const { profile } = useAuth();
  const [docs, setDocs] = useState<CaseLawDoc[]>([]);
  const [loading, setLoading] = useState(true);

  // Upload modal state
  const [showUpload, setShowUpload] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

  // Real-time listener on CaseLaws collection
  useEffect(() => {
    const q = query(
      collection(db, "CaseLaws"),
      orderBy("uploadedAt", "desc")
    );
    const unsub = onSnapshot(q, (snap) => {
      setDocs(
        snap.docs.map((d) => ({ id: d.id, ...(d.data() as Omit<CaseLawDoc, "id">) }))
      );
      setLoading(false);
    });
    return unsub;
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setSelectedFile(f);
    setUploadError(null);
    setUploadSuccess(false);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const ext = selectedFile.name.split(".").pop()?.toLowerCase();
    if (ext !== "pdf" && ext !== "docx") {
      setUploadError("Only PDF and DOCX files are allowed.");
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      const form = new FormData();
      form.append("file", selectedFile);

      const res = await apiFetch("/api/case-laws/upload", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error ?? `Upload failed (${res.status})`);
      }

      setUploadSuccess(true);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setTimeout(() => {
        setShowUpload(false);
        setUploadSuccess(false);
      }, 1500);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const closeModal = () => {
    setShowUpload(false);
    setSelectedFile(null);
    setUploadError(null);
    setUploadSuccess(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Top bar */}
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 sticky top-0 z-10">
        <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gray-900 dark:bg-gray-100">
              <Scale className="h-4 w-4 text-white dark:text-gray-900" />
            </div>
            <span className="text-base font-semibold text-gray-900 dark:text-gray-100">
              LegalAI
            </span>
          </div>
          <UserMenu />
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        {/* Breadcrumb + actions */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <Link
              href="/repository"
              className="flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors mb-2"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Repository
            </Link>
            <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <FileText className="h-5 w-5 text-blue-500" />
              Case Laws
            </h1>
            <p className="mt-0.5 text-sm text-gray-500 dark:text-gray-400">
              {loading ? "Loading…" : `${docs.length} document${docs.length !== 1 ? "s" : ""}`}
            </p>
          </div>

          <button
            onClick={() => setShowUpload(true)}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition-colors shadow-sm"
          >
            <Upload className="h-4 w-4" />
            Upload
          </button>
        </div>

        {/* Document list */}
        {loading ? (
          <div className="flex items-center justify-center py-24">
            <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
          </div>
        ) : docs.length === 0 ? (
          <div className="rounded-xl border border-dashed border-gray-300 dark:border-gray-700 py-20 text-center">
            <FileText className="mx-auto h-8 w-8 text-gray-300 dark:text-gray-600 mb-3" />
            <p className="text-sm text-gray-500 dark:text-gray-400">No documents yet.</p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              Run <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">python scripts/seed_caselaws.py</code> to seed existing files, or upload one above.
            </p>
          </div>
        ) : (
          <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
                  <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide w-12">
                    Type
                  </th>
                  <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Case Name
                  </th>
                  <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide hidden sm:table-cell">
                    Uploaded by
                  </th>
                  <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide hidden md:table-cell">
                    Date
                  </th>
                  <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide hidden lg:table-cell">
                    Size
                  </th>
                  <th className="px-5 py-3 w-16" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {docs.map((doc) => (
                  <tr
                    key={doc.id}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800/40 transition-colors"
                  >
                    <td className="px-5 py-4">
                      <FileBadge type={doc.fileType} />
                    </td>
                    <td className="px-5 py-4">
                      <p className="font-medium text-gray-900 dark:text-gray-100 truncate max-w-xs lg:max-w-sm">
                        {doc.title}
                      </p>
                    </td>
                    <td className="px-5 py-4 text-gray-500 dark:text-gray-400 hidden sm:table-cell">
                      {doc.uploadedBy === "system" ? (
                        <span className="text-xs text-gray-400 dark:text-gray-500 italic">system</span>
                      ) : (
                        doc.uploadedBy
                      )}
                    </td>
                    <td className="px-5 py-4 text-gray-500 dark:text-gray-400 hidden md:table-cell whitespace-nowrap">
                      {formatDate(doc.uploadedAt)}
                    </td>
                    <td className="px-5 py-4 text-gray-400 dark:text-gray-500 hidden lg:table-cell whitespace-nowrap text-xs">
                      {formatBytes(doc.fileSizeBytes)}
                    </td>
                    <td className="px-5 py-4 text-right">
                      <a
                        href={`${apiBase}/api/case-laws/file/${encodeURIComponent(doc.fileName)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                      >
                        View <ExternalLink className="h-3 w-3" />
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>

      {/* Upload modal */}
      {showUpload && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm px-4"
          onClick={(e) => e.target === e.currentTarget && closeModal()}
        >
          <div className="w-full max-w-md rounded-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-xl p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100">
                Upload Case Document
              </h2>
              <button
                onClick={closeModal}
                className="rounded-lg p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              PDF and DOCX files only. The document will be stored in the SCC
              cases folder and made available for retrieval.
            </p>

            {/* Drop zone */}
            <label
              htmlFor="case-upload"
              className={`flex flex-col items-center justify-center w-full rounded-xl border-2 border-dashed cursor-pointer transition-colors py-8 ${
                selectedFile
                  ? "border-blue-400 dark:border-blue-600 bg-blue-50 dark:bg-blue-900/10"
                  : "border-gray-300 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-600 bg-gray-50 dark:bg-gray-800/50"
              }`}
            >
              {selectedFile ? (
                <div className="flex flex-col items-center gap-1">
                  <File className="h-8 w-8 text-blue-500 mb-1" />
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100 max-w-xs truncate px-4 text-center">
                    {selectedFile.name}
                  </span>
                  <span className="text-xs text-gray-400">
                    {formatBytes(selectedFile.size)}
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-1 text-gray-400">
                  <Upload className="h-7 w-7 mb-1" />
                  <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
                    Click to select a file
                  </span>
                  <span className="text-xs">PDF or DOCX</span>
                </div>
              )}
              <input
                id="case-upload"
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>

            {uploadError && (
              <div className="mt-3 flex items-start gap-2 rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 px-3 py-2.5 text-xs text-red-700 dark:text-red-300">
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                {uploadError}
              </div>
            )}

            {uploadSuccess && (
              <div className="mt-3 rounded-lg border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 px-3 py-2.5 text-xs text-green-700 dark:text-green-300 text-center font-medium">
                Uploaded successfully
              </div>
            )}

            <div className="mt-5 flex gap-3">
              <button
                onClick={closeModal}
                className="flex-1 rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleUpload}
                disabled={!selectedFile || uploading || uploadSuccess}
                className="flex-1 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Uploading…
                  </>
                ) : (
                  "Upload"
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
