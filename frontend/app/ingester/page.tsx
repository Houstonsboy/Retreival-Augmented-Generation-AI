"use client";

import { useState, useRef, useEffect } from "react";
import Navigation from "../components/Navigation";

interface Document {
  name: string;
}

export default function Ingester() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [isDigesting, setIsDigesting] = useState(false);
  const [digestError, setDigestError] = useState<string | null>(null);
  const [digestSuccess, setDigestSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch documents on component mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/files");
      if (response.ok) {
        const data = await response.json();
        // Convert array of strings to array of objects with name property
        const files = (data.files || []).map((filename: string) => ({
          name: filename,
        }));
        setDocuments(files);
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    // Check if file is a PDF
    if (file.type !== "application/pdf") {
      setUploadError("Please upload a PDF file only.");
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Upload failed");
      }

      setUploadSuccess(`Successfully uploaded ${file.name}`);
      // Refresh the document list
      await fetchDocuments();
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : "Failed to upload file"
      );
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleClickUpload = () => {
    fileInputRef.current?.click();
  };

  const handleDocumentClick = (filename: string) => {
    const url = `http://localhost:5000/api/files/${encodeURIComponent(filename)}`;
    window.open(url, "_blank");
  };

  const handleDigest = async () => {
    setIsDigesting(true);
    setDigestError(null);
    setDigestSuccess(null);

    try {
      const response = await fetch("http://localhost:5000/api/digest", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Digest failed");
      }

      setDigestSuccess(
        data.message || "Documents processed successfully! They are now available for chat."
      );
    } catch (error) {
      setDigestError(
        error instanceof Error ? error.message : "Failed to process documents"
      );
    } finally {
      setIsDigesting(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-white dark:bg-gray-900">
      {/* Top Navigation */}
      <Navigation />

      {/* Header */}
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="mx-auto max-w-4xl px-4 py-4">
          <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
            Document Ingester
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Upload PDF documents to add them to the repository
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-4 py-8">
          {/* Upload Area */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={handleClickUpload}
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-colors ${
              isDragging
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 hover:border-gray-400 dark:hover:border-gray-600"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileInputChange}
              className="hidden"
              disabled={isUploading}
            />
            <div className="flex flex-col items-center justify-center px-8 py-12">
              <svg
                className="mb-4 h-12 w-12 text-gray-400 dark:text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              {isUploading ? (
                <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
                  Uploading...
                </p>
              ) : (
                <>
                  <p className="mb-2 text-lg font-medium text-gray-700 dark:text-gray-300">
                    Drop a PDF file here or click to upload
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Only PDF files are supported
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Upload Messages */}
          {uploadError && (
            <div className="mt-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3">
              <p className="text-sm text-red-800 dark:text-red-200">
                {uploadError}
              </p>
            </div>
          )}

          {uploadSuccess && (
            <div className="mt-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 px-4 py-3">
              <p className="text-sm text-green-800 dark:text-green-200">
                {uploadSuccess}
              </p>
            </div>
          )}

          {/* Digest Button */}
          <div className="mt-6">
            <button
              onClick={handleDigest}
              disabled={isDigesting || documents.length === 0}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
            >
              {isDigesting ? (
                <>
                  <svg
                    className="h-5 w-5 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  <span>Processing documents...</span>
                </>
              ) : (
                <>
                  <svg
                    className="h-5 w-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>Process Documents for Chat</span>
                </>
              )}
            </button>
            <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
              Process uploaded documents to make them searchable in the chat. Only
              new or changed files will be processed.
            </p>
          </div>

          {/* Digest Messages */}
          {digestError && (
            <div className="mt-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3">
              <p className="text-sm text-red-800 dark:text-red-200">
                {digestError}
              </p>
            </div>
          )}

          {digestSuccess && (
            <div className="mt-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 px-4 py-3">
              <p className="text-sm text-green-800 dark:text-green-200">
                {digestSuccess}
              </p>
            </div>
          )}

          {/* Documents List */}
          <div className="mt-8">
            <h2 className="mb-4 text-xl font-semibold text-gray-900 dark:text-gray-100">
              Documents in Repository
            </h2>
            {documents.length === 0 ? (
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 px-6 py-8 text-center">
                <p className="text-gray-500 dark:text-gray-400">
                  No documents found. Upload a PDF to get started.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {documents.map((doc, index) => (
                  <div
                    key={index}
                    onClick={() => handleDocumentClick(doc.name)}
                    className="group flex cursor-pointer items-center justify-between rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-3 transition-colors hover:border-blue-500 dark:hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                  >
                    <div className="flex items-center gap-3">
                      <svg
                        className="h-5 w-5 text-gray-400 group-hover:text-blue-500 dark:text-blue-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                        />
                      </svg>
                      <span className="font-medium text-gray-900 dark:text-gray-100 group-hover:text-blue-600 dark:group-hover:text-blue-400">
                        {doc.name}
                      </span>
                    </div>
                    <svg
                      className="h-5 w-5 text-gray-400 group-hover:text-blue-500 dark:text-blue-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                      />
                    </svg>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

