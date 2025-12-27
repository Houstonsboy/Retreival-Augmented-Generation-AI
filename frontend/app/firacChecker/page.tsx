"use client";

import { useState } from "react";
import Navigation from "../components/Navigation";



interface FiracSection {
  content: string;
  metadata: string;
}

export default function FiracChecker() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [document, setDocument] = useState<string | null>(null);
  const [facts, setFacts] = useState<FiracSection | null>(null);
  const [rules, setRules] = useState<FiracSection | null>(null);
  const [issues, setIssues] = useState<FiracSection | null>(null);
  const [metadata, setMetadata] = useState<string | null>(null);
  const [application, setApplication] = useState<FiracSection | null>(null);
  const [conclusion, setConclusion] = useState<FiracSection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<string | null>(null);
  const [embeddingMessage, setEmbeddingMessage] = useState<{ type: 'success' | 'error' | null; text: string }>({ type: null, text: '' });

  const handleProcess = async () => {
    setIsProcessing(true);
    setError(null);
    setDocument(null);
    setFacts(null);
    setIssues(null);
    setRules(null);
    setMetadata(null);          // ✅ just reset to null
    setApplication(null);
    setConclusion(null);
    setOutput(null);
    setEmbeddingMessage({ type: null, text: '' }); // Clear embedding message when processing new document

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
      setMetadata(data.metadata ?? null);

      // Handle new structure where each section has content and metadata
      if (data.facts && typeof data.facts === 'object' && 'content' in data.facts) {
        setFacts(data.facts);
      } else {
        // Fallback for old structure
        setFacts({ content: data.facts || "", metadata: data.metadata || "" });
      }

      if (data.issues && typeof data.issues === 'object' && 'content' in data.issues) {
        setIssues(data.issues);
      } else {
        setIssues({ content: data.issues || "", metadata: data.metadata || "" });
      }

      if (data.rules && typeof data.rules === 'object' && 'content' in data.rules) {
        setRules(data.rules);
      } else {
        setRules({ content: data.rules || "", metadata: data.metadata || "" });
      }

      if (data.application && typeof data.application === 'object' && 'content' in data.application) {
        setApplication(data.application);
      } else {
        setApplication({ content: data.application || "", metadata: data.metadata || "" });
      }

      if (data.conclusion && typeof data.conclusion === 'object' && 'content' in data.conclusion) {
        setConclusion(data.conclusion);
      } else {
        setConclusion({ content: data.conclusion || "", metadata: data.metadata || "" });
      }

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

  const handleEmbed = async () => {
    // Check if we have FIRAC data to embed
    if (!facts && !issues && !rules && !application && !conclusion) {
      setEmbeddingMessage({
        type: 'error',
        text: 'No FIRAC data available. Please process the document first.'
      });
      return;
    }

    setIsEmbedding(true);
    setEmbeddingMessage({ type: null, text: '' });

    try {
      // Prepare FIRAC data in the format expected by the backend
      const firacData = {
        document: document || '',
        metadata: metadata || '',
        facts: facts || { content: '', metadata: '' },
        issues: issues || { content: '', metadata: '' },
        rules: rules || { content: '', metadata: '' },
        application: application || { content: '', metadata: '' },
        conclusion: conclusion || { content: '', metadata: '' },
      };

      const response = await fetch("http://localhost:5000/api/ingest-firac", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(firacData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Embedding failed");
      }

      setEmbeddingMessage({
        type: 'success',
        text: data.message || 'FIRAC elements successfully embedded into vector database!'
      });
    } catch (error) {
      setEmbeddingMessage({
        type: 'error',
        text: error instanceof Error ? error.message : "Failed to embed FIRAC data"
      });
    } finally {
      setIsEmbedding(false);
    }
  };

  // Check if FIRAC data is available for embedding
  const hasFiracData = facts || issues || rules || application || conclusion;

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
          {/* Process and Embed Buttons */}
          <div className="mb-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Process Button */}
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

              {/* Embed Button */}
              <button
                onClick={handleEmbed}
                disabled={!hasFiracData || isEmbedding || isProcessing}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-green-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-green-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
              >
                {isEmbedding ? (
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
                    <span>Embedding into vector DB...</span>
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
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                      />
                    </svg>
                    <span>Embed FIRAC Elements</span>
                  </>
                )}
              </button>
            </div>
            <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
              First extract FIRAC components, then embed them into the vector database
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-6 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3">
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}

          {/* Embedding Success/Error Message */}
          {embeddingMessage.type && (
            <div className={`mb-6 rounded-lg border px-4 py-3 ${
              embeddingMessage.type === 'success'
                ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
            }`}>
              <p className={`text-sm ${
                embeddingMessage.type === 'success'
                  ? 'text-green-800 dark:text-green-200'
                  : 'text-red-800 dark:text-red-200'
              }`}>
                {embeddingMessage.text}
              </p>
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
          {metadata && (
            <div className="mb-8">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Extracted Metadata (LLM Analysis)
                </h2>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {metadata.length.toLocaleString()} characters
                </span>
              </div>
              <div className="rounded-lg border-2 border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {metadata}
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
                  {facts.content.length.toLocaleString()} characters
                </span>
              </div>
              {/* Metadata for Facts */}
              {facts.metadata && (
                <div className="mb-4 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800 p-4">
                  <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Metadata
                  </h3>
                  <pre className="whitespace-pre-wrap break-words text-xs text-gray-600 dark:text-gray-400 font-mono">
                    {facts.metadata}
                  </pre>
                </div>
              )}
              <div className="rounded-lg border-2 border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {facts.content}
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
                  {issues.content.length.toLocaleString()} characters
                </span>
              </div>
              {/* Metadata for Issues */}
              {issues.metadata && (
                <div className="mb-4 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800 p-4">
                  <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Metadata
                  </h3>
                  <pre className="whitespace-pre-wrap break-words text-xs text-gray-600 dark:text-gray-400 font-mono">
                    {issues.metadata}
                  </pre>
                </div>
              )}
              <div className="rounded-lg border-2 border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {issues.content}
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
                  {application.content.length.toLocaleString()} characters
                </span>
              </div>
              {/* Metadata for Application */}
              {application.metadata && (
                <div className="mb-4 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800 p-4">
                  <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Metadata
                  </h3>
                  <pre className="whitespace-pre-wrap break-words text-xs text-gray-600 dark:text-gray-400 font-mono">
                    {application.metadata}
                  </pre>
                </div>
              )}
              <div className="rounded-lg border-2 border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {application.content}
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
                  {rules.content.length.toLocaleString()} characters
                </span>
              </div>
              {/* Metadata for Rules */}
              {rules.metadata && (
                <div className="mb-4 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800 p-4">
                  <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Metadata
                  </h3>
                  <pre className="whitespace-pre-wrap break-words text-xs text-gray-600 dark:text-gray-400 font-mono">
                    {rules.metadata}
                  </pre>
                </div>
              )}
              <div className="rounded-lg border-2 border-yellow-200 dark:border-yellow-700 bg-yellow-50 dark:bg-yellow-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {rules.content}
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
                  {conclusion.content.length.toLocaleString()} characters
                </span>
              </div>
              {/* Metadata for Conclusion */}
              {conclusion.metadata && (
                <div className="mb-4 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800 p-4">
                  <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Metadata
                  </h3>
                  <pre className="whitespace-pre-wrap break-words text-xs text-gray-600 dark:text-gray-400 font-mono">
                    {conclusion.metadata}
                  </pre>
                </div>
              )}
              <div className="rounded-lg border-2 border-yellow-200 dark:border-yellow-700 bg-yellow-50 dark:bg-yellow-900/20 p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap break-words text-sm text-gray-900 dark:text-gray-100 font-sans overflow-x-auto max-h-[600px] overflow-y-auto">
                    {conclusion.content}
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
