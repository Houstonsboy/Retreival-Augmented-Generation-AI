"use client";

import { useState, useRef, useEffect } from "react";
import { FileText, Scale, Search, ChevronDown, ChevronUp, AlertCircle } from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  classification?: any;
  aiResponse?: string;
  evidence?: any[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5000/api/qretrieve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: userMessage.content }),
      });

      const data = await response.json();

      console.log("📥 Received data:", data);
      console.log("Classification:", data.classification);
      console.log("Retrieval:", data.retrieval);

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to get response");
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        classification: data.classification,
        aiResponse: data.retrieval?.results?.[0]?.ai_summary || "No relevant legal information found.",
        evidence: data.retrieval?.results || [],
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        aiResponse: error instanceof Error ? error.message : "Sorry, I encountered an error. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800">
      {/* Top Navigation */}
      <div className="border-b border-slate-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-sm">
        <div className="mx-auto max-w-7xl px-4 py-4">
          <div className="flex items-center gap-3">
            <Scale className="h-8 w-8 text-blue-600" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Kenyan Legal RAG
            </h1>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <Scale className="mx-auto h-16 w-16 text-blue-600 mb-4" />
                <h2 className="mb-2 text-2xl font-semibold text-gray-900 dark:text-gray-100">
                  How can I help you today?
                </h2>
                <p className="text-gray-500 dark:text-gray-400">
                  Ask me anything about Kenyan legal documents and cases.
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-8">
              {messages.map((message) => (
                <div key={message.id}>
                  {message.role === "user" ? (
                    <div className="flex justify-end mb-6">
                      <div className="max-w-[85%] rounded-2xl px-4 py-3 bg-blue-600 text-white">
                        <p className="whitespace-pre-wrap break-words">
                          {message.content}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <AssistantResponse message={message} />
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl bg-white dark:bg-gray-800 px-6 py-4 shadow-md border border-slate-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                      <div className="flex space-x-2">
                        <div className="h-2 w-2 animate-bounce rounded-full bg-blue-600 [animation-delay:-0.3s]"></div>
                        <div className="h-2 w-2 animate-bounce rounded-full bg-blue-600 [animation-delay:-0.15s]"></div>
                        <div className="h-2 w-2 animate-bounce rounded-full bg-blue-600"></div>
                      </div>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Analyzing legal documents...
                      </span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg">
        <div className="mx-auto max-w-7xl px-4 py-4">
          <div className="flex items-end gap-3">
            <div className="flex-1 rounded-2xl border border-slate-300 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm focus-within:border-blue-500 dark:focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-500/20">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a legal question..."
                rows={1}
                className="w-full resize-none border-0 bg-transparent px-4 py-3 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-0"
                style={{
                  maxHeight: "200px",
                  minHeight: "52px",
                }}
                onInput={(e) => {
                  const target = e.target as HTMLTextAreaElement;
                  target.style.height = "auto";
                  target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
                }}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white transition-all hover:bg-blue-700 hover:scale-105 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed disabled:scale-100"
            >
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
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function AssistantResponse({ message }: { message: Message }) {
  return (
    <div className="space-y-4">
      {/* Classification Widget */}
      {message.classification && (
        <ClassificationWidget classification={message.classification} />
      )}

      {/* AI Response Widget */}
      {message.aiResponse && (
        <AIResponseWidget response={message.aiResponse} />
      )}

      {/* Evidence Chunks Widget */}
      {message.evidence && message.evidence.length > 0 && (
        <EvidenceWidget evidence={message.evidence} />
      )}
    </div>
  );
}

function ClassificationWidget({ classification }: { classification: any }) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="rounded-xl bg-white dark:bg-gray-800 shadow-md border border-slate-200 dark:border-gray-700 overflow-hidden">
      <div
        className="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <Search className="h-5 w-5 text-purple-600 dark:text-purple-400" />
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">
            Query Classification
          </h3>
        </div>
        {isExpanded ? (
          <ChevronUp className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        ) : (
          <ChevronDown className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        )}
      </div>
      {isExpanded && (
        <div className="px-5 py-4 space-y-4">
          {/* Display Intents */}
          {classification.intents && classification.intents.length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Query Intents
              </p>
              <div className="flex flex-wrap gap-2">
                {classification.intents.map((intent: string, idx: number) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-xs font-medium"
                  >
                    {intent.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Display Legal Domains */}
          {classification.legal_domains && classification.legal_domains.length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Legal Domains
              </p>
              <div className="space-y-2">
                {classification.legal_domains.map((domain: any, idx: number) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30"
                  >
                    <div className="flex items-center gap-3 flex-1">
                      <div className="flex-1">
                        <span className="font-semibold text-purple-900 dark:text-purple-100 block">
                          {domain.domain}
                        </span>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs px-2 py-0.5 bg-purple-200 dark:bg-purple-800/40 text-purple-800 dark:text-purple-200 rounded">
                            {domain.mode}
                          </span>
                          <span className="text-xs text-gray-600 dark:text-gray-400">
                            {domain.procedural_scope}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="text-right">
                        <div className="text-lg font-bold text-purple-700 dark:text-purple-300">
                          {(domain.confidence_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          confidence
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Display Target Components */}
          {classification.target_components && classification.target_components.length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Target Components (IRAC)
              </p>
              <div className="flex flex-wrap gap-2">
                {classification.target_components.map((component: string, idx: number) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-lg text-xs font-medium border border-green-200 dark:border-green-800"
                  >
                    {component}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Display Strategy Critique */}
          {classification.strategy_critique && Object.keys(classification.strategy_critique).length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Strategy Analysis
              </p>
              <div className="space-y-2">
                {Object.entries(classification.strategy_critique).map(([key, value]) => (
                  <div 
                    key={key} 
                    className="p-3 bg-amber-50 dark:bg-amber-900/10 rounded-lg border border-amber-100 dark:border-amber-800/30"
                  >
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                      <div className="flex-1">
                        <span className="font-semibold text-amber-900 dark:text-amber-100 capitalize text-sm">
                          {key}:
                        </span>{" "}
                        <span className="text-sm text-gray-700 dark:text-gray-300">
                          {value as string}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Display Entities */}
          {classification.entities && (
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Referenced Legal Entities
              </p>
              <div className="space-y-2 bg-slate-50 dark:bg-gray-900 p-3 rounded-lg border border-slate-200 dark:border-gray-700">
                {classification.entities.statutes && classification.entities.statutes.length > 0 && (
                  <div>
                    <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">
                      Statutes:
                    </span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {classification.entities.statutes.map((statute: string, idx: number) => (
                        <span 
                          key={idx}
                          className="text-xs text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 px-2 py-1 rounded border border-slate-200 dark:border-gray-700"
                        >
                          {statute}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {classification.entities.cases && classification.entities.cases.length > 0 && (
                  <div className="mt-2">
                    <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">
                      Cases:
                    </span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {classification.entities.cases.map((caseRef: string, idx: number) => (
                        <span 
                          key={idx}
                          className="text-xs text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 px-2 py-1 rounded border border-slate-200 dark:border-gray-700"
                        >
                          {caseRef}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Display Vector Query */}
          {classification.vector_query && (
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Search Vector Query
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300 bg-slate-50 dark:bg-gray-900 p-3 rounded-lg border border-slate-200 dark:border-gray-700 italic">
                "{classification.vector_query}"
              </p>
            </div>
          )}

          {/* Display Reasoning Summary */}
          {classification.reasoning_summary && (
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Classification Reasoning
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300 bg-slate-50 dark:bg-gray-900 p-3 rounded-lg border border-slate-200 dark:border-gray-700 leading-relaxed">
                {classification.reasoning_summary}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AIResponseWidget({ response }: { response: string }) {
  return (
    <div className="rounded-xl bg-white dark:bg-gray-800 shadow-md border border-slate-200 dark:border-gray-700 overflow-hidden">
      <div className="px-5 py-3 bg-gradient-to-r from-green-50 to-emerald-100 dark:from-green-900/20 dark:to-emerald-800/20">
        <div className="flex items-center gap-3">
          <Scale className="h-5 w-5 text-green-600 dark:text-green-400" />
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">
            Legal Analysis
          </h3>
        </div>
      </div>
      <div className="px-5 py-4">
        <p className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap leading-relaxed">
          {response}
        </p>
      </div>
    </div>
  );
}

function EvidenceWidget({ evidence }: { evidence: any[] }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-xl bg-white dark:bg-gray-800 shadow-md border border-slate-200 dark:border-gray-700 overflow-hidden">
      <div
        className="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-amber-50 to-orange-100 dark:from-amber-900/20 dark:to-orange-800/20 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <FileText className="h-5 w-5 text-amber-600 dark:text-amber-400" />
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">
            Supporting Cases ({evidence.length})
          </h3>
        </div>
        {isExpanded ? (
          <ChevronUp className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        ) : (
          <ChevronDown className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        )}
      </div>
      {isExpanded && (
        <div className="px-5 py-4 space-y-4 max-h-96 overflow-y-auto">
          {evidence.map((item, idx) => (
            <div
              key={idx}
              className="p-4 bg-slate-50 dark:bg-gray-900 rounded-lg border border-slate-200 dark:border-gray-700"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="px-2 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded text-xs font-bold">
                      #{item.rank || idx + 1}
                    </span>
                    <h4 className="font-semibold text-gray-900 dark:text-gray-100 text-sm">
                      {item.case_name || `Evidence ${idx + 1}`}
                    </h4>
                  </div>
                  {item.metadata && (
                    <div className="text-xs text-gray-500 dark:text-gray-400 space-y-0.5 mt-2">
                      {item.metadata.court_level && (
                        <div>Court: {item.metadata.court_level}</div>
                      )}
                      {item.metadata.year && (
                        <div>Year: {item.metadata.year}</div>
                      )}
                      {item.metadata.judge && (
                        <div>Judge: {item.metadata.judge}</div>
                      )}
                    </div>
                  )}
                </div>
                <span className="px-3 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-full text-xs font-bold ml-2">
                  {(item.score * 100)?.toFixed(1)}%
                </span>
              </div>
              
              {item.matched_components && item.matched_components.length > 0 && (
                <div className="mb-3">
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Matched Components:
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {item.matched_components.map((comp: string, i: number) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs font-medium"
                      >
                        {comp}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {item.best_chunk && (
                <div className="mt-3 p-3 bg-white dark:bg-gray-800 rounded border border-slate-200 dark:border-gray-700">
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Relevant Excerpt:
                  </p>
                  <p className="text-sm text-gray-700 dark:text-gray-300 italic leading-relaxed">
                    "{item.best_chunk.substring(0, 300)}{item.best_chunk.length > 300 ? '...' : ''}"
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}