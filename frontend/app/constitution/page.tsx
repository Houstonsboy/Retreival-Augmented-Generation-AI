"use client";

import { useState } from "react";
import Navigation from "../components/Navigation";

export default function Constitution() {
  const [isDigesting, setIsDigesting] = useState(false);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [digestOutput, setDigestOutput] = useState<string>("");
  const [statusOutput, setStatusOutput] = useState<string>("");
  const [deleteOutput, setDeleteOutput] = useState<string>("");
  const [digestError, setDigestError] = useState<string | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleDigest = async () => {
    setIsDigesting(true);
    setDigestOutput("");
    setDigestError(null);

    try {
      const response = await fetch("http://localhost:5000/api/digest/constitution", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          force_reextract: false,
          skip_existing: true,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Digest failed");
      }

      // Format the response for display
      const formattedOutput = JSON.stringify(data, null, 2);
      setDigestOutput(formattedOutput);
    } catch (error) {
      setDigestError(
        error instanceof Error ? error.message : "Failed to digest constitution"
      );
    } finally {
      setIsDigesting(false);
    }
  };

  const handleCheckStatus = async () => {
    setIsCheckingStatus(true);
    setStatusOutput("");
    setStatusError(null);

    try {
      const response = await fetch("http://localhost:5000/api/digest/constitution/status", {
        method: "GET",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Status check failed");
      }

      // Format the response for display
      const formattedOutput = JSON.stringify(data, null, 2);
      setStatusOutput(formattedOutput);
    } catch (error) {
      setStatusError(
        error instanceof Error ? error.message : "Failed to check status"
      );
    } finally {
      setIsCheckingStatus(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete all constitution chunks? This action cannot be undone.")) {
      return;
    }

    setIsDeleting(true);
    setDeleteOutput("");
    setDeleteError(null);

    try {
      const response = await fetch("http://localhost:5000/api/digest/constitution/reset", {
        method: "DELETE",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Delete failed");
      }

      // Format the response for display
      const formattedOutput = JSON.stringify(data, null, 2);
      setDeleteOutput(formattedOutput);
    } catch (error) {
      setDeleteError(
        error instanceof Error ? error.message : "Failed to delete constitution chunks"
      );
    } finally {
      setIsDeleting(false);
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
            Constitution Management
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Extract, embed, and manage constitution articles in the vector database
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-4 py-8">
          {/* Action Buttons */}
          <div className="grid gap-4 md:grid-cols-3">
            {/* Digest Button */}
            <div className="flex flex-col">
              <button
                onClick={handleDigest}
                disabled={isDigesting}
                className="flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
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
                    <span>Processing...</span>
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
                    <span>Digest Constitution</span>
                  </>
                )}
              </button>
              <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
                Extract articles and embed into ChromaDB
              </p>
            </div>

            {/* Status Button */}
            <div className="flex flex-col">
              <button
                onClick={handleCheckStatus}
                disabled={isCheckingStatus}
                className="flex items-center justify-center gap-2 rounded-lg bg-green-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-green-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
              >
                {isCheckingStatus ? (
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
                    <span>Checking...</span>
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
                        d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
                      />
                    </svg>
                    <span>Check Status</span>
                  </>
                )}
              </button>
              <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
                View extraction and ingestion status
              </p>
            </div>

            {/* Delete Button */}
            <div className="flex flex-col">
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="flex items-center justify-center gap-2 rounded-lg bg-red-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
              >
                {isDeleting ? (
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
                    <span>Deleting...</span>
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
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                    <span>Delete All</span>
                  </>
                )}
              </button>
              <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
                Remove all constitution chunks (irreversible)
              </p>
            </div>
          </div>

          {/* Digest Output */}
          {digestOutput && (
            <div className="mt-8">
              <h2 className="mb-2 text-lg font-semibold text-gray-900 dark:text-gray-100">
                Digest Output
              </h2>
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
                <pre className="overflow-x-auto text-xs text-gray-800 dark:text-gray-200 whitespace-pre-wrap break-words">
                  {digestOutput}
                </pre>
              </div>
            </div>
          )}

          {digestError && (
            <div className="mt-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3">
              <p className="text-sm text-red-800 dark:text-red-200">
                {digestError}
              </p>
            </div>
          )}

          {/* Status Output */}
          {statusOutput && (
            <div className="mt-8">
              <h2 className="mb-2 text-lg font-semibold text-gray-900 dark:text-gray-100">
                Status Output
              </h2>
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
                <pre className="overflow-x-auto text-xs text-gray-800 dark:text-gray-200 whitespace-pre-wrap break-words">
                  {statusOutput}
                </pre>
              </div>
            </div>
          )}

          {statusError && (
            <div className="mt-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3">
              <p className="text-sm text-red-800 dark:text-red-200">
                {statusError}
              </p>
            </div>
          )}

          {/* Delete Output */}
          {deleteOutput && (
            <div className="mt-8">
              <h2 className="mb-2 text-lg font-semibold text-gray-900 dark:text-gray-100">
                Delete Output
              </h2>
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
                <pre className="overflow-x-auto text-xs text-gray-800 dark:text-gray-200 whitespace-pre-wrap break-words">
                  {deleteOutput}
                </pre>
              </div>
            </div>
          )}

          {deleteError && (
            <div className="mt-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3">
              <p className="text-sm text-red-800 dark:text-red-200">
                {deleteError}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

