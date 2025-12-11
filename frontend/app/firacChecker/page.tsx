"use client";

import { useState } from "react";
import Navigation from "../components/Navigation";

export default function FiracChecker() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [document, setDocument] = useState<string | null>(null);
  const [facts, setFacts] = useState<string | null>(null);
  const [rules, setRules] = useState<string | null>(null);

  const [issues, setIssues] = useState<string | null>(null);
  const [application, setApplication] = useState<string | null>(null);
  const [conclusion, setConclusion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<string | null>(null);

  const handleProcess = async () => {
    setIsProcessing(true);
    setError(null);
    setDocument(null);
    setFacts(null);
    setIssues(null);
    setRules(null);

    setApplication(null);
    setConclusion(null);
    setOutput(null);

    try {
      const response = await fetch("http://localhost:5000/api/firac", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await response.json();

      if (!response.ok && !data.document) {
        throw new Error(data.error || "Processing failed");
      }

      setDocument(data.document || "");
      setFacts(data.facts || "");
      setIssues(data.issues || "");
      setRules(data.rules || "");

      setApplication(data.application || "");
      setConclusion(data.conclusion || "");
      setOutput(data.output || "");

      if (data.error) {
        setError(data.error);
      }
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Failed to process document"
      );
    } finally {
      setIsProcessing(false);
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
            FIRAC Checker
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Extract and analyze Facts, Issues, Application, and Conclusion from
            the Wilson Wanjala Mkendeshwo v Republic judgment using LLM.
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-4 py-8">
          {/* Process Button */}
          <div className="mb-6">
            <button
              onClick={handleProcess}
              disabled={isProcessing}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
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
                  <span>Processing document...</span>
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
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <span>Process Wilson Wanjala Document</span>
                </>
              )}
            </button>
            <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
              Click to extract FIRAC components from the Wilson Wanjala PDF
              using Groq LLM (Llama 3.3 70B)
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-6 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3">
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}

          {/* Output Section */}
          {output && (
            <div className="mb-6">
              <h2 className="mb-2 text-lg font-semibold text-gray-900 dark:text-gray-100">
                Processing Output
              </h2>
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
                <pre className="whitespace-pre-wrap break-words text-xs text-gray-700 dark:text-gray-300 font-mono overflow-x-auto">
                  {output}
                </pre>
              </div>
            </div>
          )}

          {/* Full Document Section */}
          {document && (
            <div className="mb-8">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Full Document Content
                </h2>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {document.length.toLocaleString()} characters
                </span>
              </div>
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {document}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {/* Extracted Facts Section */}
          {facts && (
            <div className="mb-8">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Extracted Facts (LLM Analysis)
                </h2>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {facts.length.toLocaleString()} characters
                </span>
              </div>
              <div className="rounded-lg border-2 border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {facts}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {/* Extracted Issues Section */}
          {issues && (
            <div className="mb-8">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Extracted Issues (LLM Analysis)
                </h2>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {issues.length.toLocaleString()} characters
                </span>
              </div>
              <div className="rounded-lg border-2 border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {issues}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {/* Extracted Application Section */}
          {application && (
            <div className="mb-8">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Application / Analysis (LLM)
                </h2>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {application.length.toLocaleString()} characters
                </span>
              </div>
              <div className="rounded-lg border-2 border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {application}
                  </pre>
                </div>
              </div>
            </div>
          )}

           {/* Extracted rules Section */}
           {rules && (
            <div className="mb-8">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Rule applied
                </h2>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {rules.length.toLocaleString()} characters
                </span>
              </div>
              <div className="rounded-lg border-2 border-yellow-200 dark:border-yellow-700 bg-yellow-50 dark:bg-yellow-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {rules}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {/* Extracted Conclusion Section */}
          {conclusion && (
            <div className="mb-8">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Conclusion / Holding (LLM)
                </h2>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {conclusion.length.toLocaleString()} characters
                </span>
              </div>
              <div className="rounded-lg border-2 border-yellow-200 dark:border-yellow-700 bg-yellow-50 dark:bg-yellow-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {conclusion}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!document &&
            !facts &&
            !issues &&
            !rules &&
            !application &&
            !conclusion &&
            !error &&
            !isProcessing && (
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 px-6 py-12 text-center">
                <svg
                  className="mx-auto mb-4 h-12 w-12 text-gray-400 dark:text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <p className="text-gray-500 dark:text-gray-400">
                  Click the button above to process the Wilson Wanjala document
                  and view the full FIRAC breakdown.
                </p>
              </div>
            )}
        </div>
      </div>
    </div>
  );
}
