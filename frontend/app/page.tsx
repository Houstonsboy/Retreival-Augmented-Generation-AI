"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  FileText,
  Scale,
  Search,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  ExternalLink,
  BookOpen,
  File,
  X,
} from "lucide-react";
import { apiFetch } from "@/app/lib/api";
import { UserMenu } from "@/app/components/UserMenu";
import type { ReactNode } from "react";

// ── Design tokens ─────────────────────────────────────────────────────────────
const BG       = "#FAF7F0";
const GOLD     = "#C9971A";
const GOLD_DK  = "#A87A0C";
const HIGHLIGHT= "#F6E7B8";
const BORDER   = "#DDD3BE";
const CREAM_2  = "#F0EAE0";
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Coverage {
  in_corpus: string[];
  not_in_corpus: string[];
}

interface ChunkSource {
  alias: string;
  chunk_id: string;
  label: string;
  type: "case" | "statute" | "knowledge";
  used: boolean;
  precision?: "chunk" | "document";
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  classification?: any;
  aiResponse?: string;
  evidence?: any[];
  rules?: any[];
  coverage?: Coverage;
  premiseCheck?: string;
  unverifiedClaims?: string[];
  sources?: ChunkSource[];
  aliasMap?: Record<string, string>;
}

const EXAMPLE_QUESTIONS = [
  { category: "DIFFICULT JUDGMENT", question: "When will the SCC set aside an ex parte judgment after service by WhatsApp?" },
  { category: "ELECTRONIC EVIDENCE", question: "Are M-Pesa statements admissible without a section 106B certificate?" },
  { category: "EXECUTION", question: "What must I show to stay execution of a warrant of attachment?" },
  { category: "BAILMENT", question: "Is a bailee for reward liable for theft by an independent security contractor?" },
];

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput]       = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [panelMessage, setPanelMessage] = useState<Message | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef       = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-show sources panel for the latest assistant message
  useEffect(() => {
    const last = [...messages].reverse().find(
      (m) => m.role === "assistant" && m.sources && m.sources.length > 0
    );
    if (last) setPanelMessage(last);
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
      const response = await apiFetch("/api/qretrieve", {
        method: "POST",
        body: JSON.stringify({ query: userMessage.content }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || "Retrieval failed");

      let aiResponse = "No relevant legal information found.";
      let coverage: Coverage | undefined;
      let premiseCheck: string | undefined;
      let unverifiedClaims: string[] | undefined;
      let sources: ChunkSource[] | undefined;
      let aliasMap: Record<string, string> | undefined;

      try {
        const ansRes = await apiFetch("/api/generate-legal-answer", {
          method: "POST",
          body: JSON.stringify({
            query: userMessage.content,
            retrieval: data.retrieval,
            retrieval_mode: data.classification?.retrieval_mode || "",
            classification: data.classification || {},
          }),
        });
        if (ansRes.ok) {
          const ans = await ansRes.json();
          if (ans.success) {
            aiResponse       = ans.answer;
            coverage         = ans.coverage;
            premiseCheck     = ans.premise_check;
            unverifiedClaims = ans.unverified_claims_removed;
            sources          = ans.sources;
            aliasMap         = ans.alias_map;
          }
        }
      } catch { /* keep fallback aiResponse */ }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        classification: data.classification,
        aiResponse,
        evidence: data.retrieval?.results || [],
        rules:    data.retrieval?.rules   || [],
        coverage,
        premiseCheck,
        unverifiedClaims,
        sources,
        aliasMap,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "",
          aiResponse: err instanceof Error ? err.message : "An error occurred. Please try again.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };
  const autoResize = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const t = e.target as HTMLTextAreaElement;
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 200)}px`;
  };

  const hasMessages  = messages.length > 0;
  const hasSources   = !!(panelMessage?.sources?.length);

  return (
    <div className="flex h-screen flex-col" style={{ backgroundColor: BG }}>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="shrink-0 z-10" style={{ backgroundColor: BG, borderBottom: `1px solid ${BORDER}` }}>
        <div className="mx-auto max-w-full px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
              style={{ background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)` }}>
              <Scale className="h-4 w-4 text-white" />
            </div>
            <span className="text-sm font-bold tracking-tight text-gray-900">
              Kenyan Legal{" "}<span style={{ color: GOLD }} className="font-black">RAG</span>
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="text-sm font-medium text-gray-500 hover:text-gray-800 transition-colors">
              Dashboard
            </Link>
            <UserMenu />
          </div>
        </div>
      </header>

      {/* ── Body (split layout) ─────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">

        {/* ── Left: chat column ──────────────────────────────────────────── */}
        <div
          className="flex flex-col overflow-hidden"
          style={{
            width: hasSources ? "33.33%" : "100%",
            transition: "width 0.45s cubic-bezier(.4,0,.2,1)",
            minWidth: hasSources ? 320 : undefined,
          }}
        >
          <div className="flex-1 overflow-y-auto">
            {!hasMessages ? (
              /* ── Landing ── */
              <div className="flex flex-col items-center justify-center min-h-full py-16 px-6">
                <div className="w-16 h-16 rounded-full mb-6 flex items-center justify-center"
                  style={{ background: "radial-gradient(circle at 35% 30%, #E8A84E, #A85A1A)", boxShadow: "0 6px 24px rgba(180,90,20,.35)" }}>
                  <Scale className="h-7 w-7 text-white opacity-90" />
                </div>
                <p className="text-[10px] font-semibold tracking-[.28em] uppercase mb-4" style={{ color: GOLD }}>
                  Small Claims Court · Research Assistant
                </p>
                <h1 className="text-4xl font-bold text-gray-900 mb-4 text-center leading-tight">
                  Ask the{" "}
                  <em style={{ fontFamily: "Georgia,'Times New Roman',serif", fontStyle: "italic", fontWeight: 700 }}>
                    law library
                  </em>
                </h1>
                <p className="text-gray-500 text-center max-w-sm text-sm leading-relaxed mb-10">
                  Describe your client&apos;s facts. You get a FIRAC opinion grounded in Kenyan case law and statutes,
                  with every source open beside it.
                </p>
                <div className="w-full max-w-2xl">
                  <InputArea
                    inputRef={inputRef} value={input} onChange={setInput}
                    onKeyDown={handleKeyDown} onInput={autoResize}
                    onSubmit={handleSubmit} isLoading={isLoading}
                    placeholder="Ask a legal question…"
                  />
                  <div className="mt-8">
                    <p className="text-[10px] uppercase tracking-[.22em] text-gray-400 mb-3 font-semibold">Try a question</p>
                    <div className="grid grid-cols-2 gap-3">
                      {EXAMPLE_QUESTIONS.map((ex, i) => (
                        <button key={i} onClick={() => { setInput(ex.question); inputRef.current?.focus(); }}
                          className="text-left p-3 rounded-xl bg-white transition-all hover:shadow-md"
                          style={{ border: `1px solid ${BORDER}` }}>
                          <p className="text-[9px] uppercase tracking-[.18em] text-gray-400 mb-1.5 font-semibold">{ex.category}</p>
                          <p className="text-[11px] text-gray-700 leading-snug">{ex.question}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              /* ── Chat messages ── */
              <div className={`py-6 space-y-8 ${hasSources ? "px-4" : "mx-auto max-w-3xl px-6"}`}>
                {messages.map((msg) => (
                  <div key={msg.id}>
                    {msg.role === "user" ? (
                      <div className="flex justify-end mb-2">
                        <div className="max-w-[85%] rounded-2xl px-4 py-3 text-white" style={{ background: GOLD }}>
                          <p className="whitespace-pre-wrap break-words text-sm">{msg.content}</p>
                        </div>
                      </div>
                    ) : (
                      <AssistantMessage
                        message={msg}
                        isActive={panelMessage?.id === msg.id}
                        onShowSources={() => setPanelMessage(msg)}
                        compact={hasSources}
                      />
                    )}
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="rounded-2xl bg-white px-5 py-4 shadow-sm border" style={{ borderColor: BORDER }}>
                      <div className="flex items-center gap-3">
                        <div className="flex space-x-1.5">
                          {["-0.3s", "-0.15s", "0s"].map((d, i) => (
                            <div key={i} className="h-2 w-2 rounded-full animate-bounce" style={{ background: GOLD, animationDelay: d }} />
                          ))}
                        </div>
                        <span className="text-sm text-gray-500">Analyzing legal documents…</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Bottom input bar (chat mode) */}
          {hasMessages && (
            <div className="shrink-0 bg-white" style={{ borderTop: `1px solid ${BORDER}` }}>
              <div className="px-4 py-3">
                <InputArea
                  inputRef={inputRef} value={input} onChange={setInput}
                  onKeyDown={handleKeyDown} onInput={autoResize}
                  onSubmit={handleSubmit} isLoading={isLoading}
                  placeholder="Ask a follow-up question…"
                />
              </div>
            </div>
          )}
        </div>

        {/* ── Right: source panel ─────────────────────────────────────────── */}
        <div
          className="flex flex-col overflow-hidden"
          style={{
            width: hasSources ? "66.67%" : "0%",
            transition: "width 0.45s cubic-bezier(.4,0,.2,1)",
            borderLeft: hasSources ? `1px solid ${BORDER}` : "none",
            backgroundColor: BG,
          }}
        >
          {hasSources && panelMessage && (
            <SourcePanel
              message={panelMessage}
              onClose={() => setPanelMessage(null)}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Shared input area ─────────────────────────────────────────────────────────
function InputArea({
  inputRef, value, onChange, onKeyDown, onInput, onSubmit, isLoading, placeholder,
}: {
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  value: string;
  onChange: (v: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onInput: (e: React.FormEvent<HTMLTextAreaElement>) => void;
  onSubmit: () => void;
  isLoading: boolean;
  placeholder: string;
}) {
  return (
    <div className="flex items-end gap-3">
      <div className="flex-1 flex items-start rounded-2xl bg-white px-4 py-3"
        style={{ border: `1px solid ${BORDER}`, boxShadow: "0 1px 4px rgba(0,0,0,.06)" }}>
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          onInput={onInput}
          placeholder={placeholder}
          rows={1}
          disabled={isLoading}
          className="flex-1 resize-none bg-transparent text-gray-800 placeholder-gray-400 focus:outline-none text-sm"
          style={{ minHeight: "24px", maxHeight: "160px" }}
        />
        <div className="flex items-center gap-1 ml-3 text-[10px] font-bold tracking-[.18em] shrink-0 text-[#B5A090] mt-0.5">
          <span>KENYA</span><span className="opacity-50">·</span><span>SCC</span>
        </div>
      </div>
      <button
        onClick={onSubmit}
        disabled={!value.trim() || isLoading}
        className="w-10 h-10 rounded-full flex items-center justify-center text-white shrink-0 transition-all hover:brightness-110 hover:scale-105 disabled:opacity-40 disabled:cursor-not-allowed disabled:scale-100"
        style={{ background: GOLD }}
      >
        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
        </svg>
      </button>
    </div>
  );
}

// ── Assistant message (chat column) ──────────────────────────────────────────
function AssistantMessage({
  message, isActive, onShowSources, compact,
}: {
  message: Message;
  isActive: boolean;
  onShowSources: () => void;
  compact: boolean;
}) {
  const hasSrcs = !!(message.sources?.length);
  return (
    <div className="space-y-3">
      {message.classification && !compact && (
        <ClassificationWidget classification={message.classification} />
      )}
      <CoverageNotice coverage={message.coverage} removed={message.unverifiedClaims} />
      {message.aiResponse && <AIResponseWidget response={message.aiResponse} />}
      {hasSrcs && (
        <button
          onClick={onShowSources}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all hover:shadow-sm"
          style={{
            background: isActive ? HIGHLIGHT : "white",
            border: `1px solid ${isActive ? GOLD : BORDER}`,
            color: isActive ? GOLD_DK : "#555",
          }}
        >
          <BookOpen className="h-4 w-4" />
          {isActive ? "Viewing sources →" : "Open sources panel →"}
          <span className="ml-auto text-xs opacity-60">
            {message.sources!.filter((s) => s.used).length} cited ·{" "}
            {message.sources!.length} total
          </span>
        </button>
      )}
    </div>
  );
}

// ── Source panel (right column) ───────────────────────────────────────────────
function SourcePanel({ message, onClose }: { message: Message; onClose: () => void }) {
  const [tab, setTab] = useState<"reader" | "original">("reader");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);

  const sources  = message.sources  || [];
  const evidence = message.evidence || [];
  const rules    = message.rules    || [];
  const aliasMap = message.aliasMap || {};

  // Build reverse map: chunk_id → alias
  const chunkIdToAlias = Object.fromEntries(
    Object.entries(aliasMap).map(([alias, cid]) => [cid, alias])
  );
  const usedAliases = new Set(sources.filter((s) => s.used).map((s) => s.alias));

  // Case list for OriginalPane
  const cases = evidence.map((ev, i) => ({
    caseId: ev.case_name || ev.case_identifier || `Case ${i + 1}`,
    label: ev.case_name || `Case ${i + 1}`,
    pdfUrl: ev.pdf_url || null,
  }));

  return (
    <div className="flex flex-col h-full">
      {/* ── Panel header ────────────────────────────────────────────────── */}
      <div className="shrink-0 flex items-center justify-between px-5 py-3"
        style={{ borderBottom: `1px solid ${BORDER}`, backgroundColor: CREAM_2 }}>
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4" style={{ color: GOLD }} />
          <span className="text-sm font-semibold text-gray-800">Sources</span>
          <span className="text-xs px-2 py-0.5 rounded-full font-medium"
            style={{ background: HIGHLIGHT, color: GOLD_DK }}>
            {sources.filter((s) => s.used).length} cited
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Tab switcher */}
          <div className="flex rounded-lg overflow-hidden" style={{ border: `1px solid ${BORDER}` }}>
            {(["reader", "original"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="px-3 py-1.5 text-xs font-medium transition-colors capitalize"
                style={{
                  background: tab === t ? GOLD : "white",
                  color: tab === t ? "white" : "#555",
                }}
              >
                {t === "reader" ? "Reader" : "PDF"}
              </button>
            ))}
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* ── Cluster bar ────────────────────────────────────────────────── */}
      <ClusterBar sources={sources} evidence={evidence} rules={rules} />

      {/* ── Pane content ────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {tab === "reader" ? (
          <ReaderPane
            sources={sources}
            evidence={evidence}
            rules={rules}
            chunkIdToAlias={chunkIdToAlias}
            usedAliases={usedAliases}
          />
        ) : (
          <OriginalPane
            cases={cases}
            selectedCaseId={selectedCaseId}
            onSelectCase={setSelectedCaseId}
            sources={sources}
          />
        )}
      </div>
    </div>
  );
}

// ── Cluster widget ────────────────────────────────────────────────────────────
type DocCard = {
  title: string;
  citation: string;
  kind: "Judgment" | "Statute" | "Constitution" | "Summary";
  relevance: number;
  url?: string;
};

function ClusterBar({
  sources, evidence, rules,
}: {
  sources: ChunkSource[];
  evidence: any[];
  rules: any[];
}) {
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const docs: DocCard[] = [
    ...evidence.map((ev) => ({
      title: ev.case_name || "Case",
      citation: ev.metadata?.citation || ev.metadata?.year || "",
      kind: "Judgment" as const,
      relevance: ev.score ?? 0,
      url: ev.pdf_url ? `${API_BASE}${ev.pdf_url}` : undefined,
    })),
    ...rules.map((r) => ({
      title: r.document_label || r.source_document || "Statute",
      citation: `s.${r.article_number}${r.article_header ? ` — ${r.article_header}` : ""}`,
      kind: "Statute" as const,
      relevance: r.score ?? 0,
      url: r.pdf_url ? `${API_BASE}${r.pdf_url}` : undefined,
    })),
  ];

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  if (!docs.length) return null;

  const CARD_W = 228;

  const kindBadge = (kind: DocCard["kind"], isStatute: boolean) => (
    <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
      style={{ background: isStatute ? "#EDE8DC" : HIGHLIGHT, color: isStatute ? "#6B6660" : GOLD_DK }}>
      {kind}
    </span>
  );

  return (
    <div className="shrink-0 px-4 py-2 flex items-center justify-between"
      style={{ borderBottom: `1px solid ${BORDER}`, background: "#FFFDF8" }}>
      <span className="text-[10px] uppercase tracking-[.18em] text-gray-400 font-semibold">
        {docs.length} source{docs.length !== 1 ? "s" : ""}
      </span>

      <div ref={containerRef} className="relative">
        {/* Fanned stack button */}
        <button
          aria-label="Toggle document cluster"
          aria-expanded={open}
          onClick={() => setOpen((p) => !p)}
          onMouseEnter={() => setOpen(true)}
          className="relative focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
          style={{ width: 36 + Math.min(docs.length - 1, 4) * 7, height: 40 }}
        >
          {docs.slice(0, 5).map((doc, i) => {
            const isStatute = doc.kind === "Statute";
            const isActive = activeIdx === i;
            return (
              <div
                key={i}
                className="absolute rounded-lg shadow-sm transition-transform duration-200"
                style={{
                  width: 36,
                  height: 40,
                  left: i * 7,
                  background: isStatute ? "#EDE8DC" : "#FFFDF8",
                  border: `1.5px solid ${isActive ? GOLD : BORDER}`,
                  boxShadow: isActive ? `0 0 0 2px ${GOLD}` : "0 1px 3px rgba(0,0,0,.06)",
                  zIndex: i,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {isStatute
                  ? <Scale className="h-4 w-4" style={{ color: "#6B6660" }} />
                  : <FileText className="h-4 w-4" style={{ color: GOLD_DK }} />
                }
              </div>
            );
          })}
          <span
            className="absolute -top-1.5 rounded-full text-[10px] font-bold text-center"
            style={{
              right: -6, background: GOLD, color: "white",
              minWidth: 18, lineHeight: "18px", height: 18, padding: "0 3px", zIndex: 10,
            }}
          >
            {docs.length}
          </span>
        </button>

        {/* Expanded cards */}
        {open && (
          <div
            className="absolute right-0 z-50 flex flex-col gap-2 py-2"
            style={{ top: 48, width: CARD_W }}
            onMouseLeave={() => setOpen(false)}
          >
            {docs.map((doc, i) => {
              const isStatute = doc.kind === "Statute";
              const isActive = activeIdx === i;
              return (
                <div
                  key={i}
                  role="button"
                  tabIndex={0}
                  aria-label={`${doc.title}`}
                  onClick={() => { setActiveIdx(i === activeIdx ? null : i); if (doc.url) window.open(doc.url, "_blank"); }}
                  onKeyDown={(e) => e.key === "Enter" && setActiveIdx(i === activeIdx ? null : i)}
                  className="rounded-xl p-3 cursor-pointer transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
                  style={{
                    background: isActive ? HIGHLIGHT : "#FFFDF8",
                    border: `1.5px solid ${isActive ? GOLD : BORDER}`,
                    boxShadow: isActive ? `inset 3px 0 0 ${GOLD}` : "0 1px 4px rgba(0,0,0,.06)",
                    color: "#151515",
                  }}
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <p className="text-xs font-semibold leading-tight line-clamp-2 flex-1" style={{ color: "#151515" }}>
                      {doc.title}
                    </p>
                    {kindBadge(doc.kind, isStatute)}
                  </div>
                  <p className="text-[11px] leading-snug line-clamp-1 mb-1.5" style={{ color: "#6B6660" }}>
                    {doc.citation}
                  </p>
                  <div className="flex items-center gap-2">
                    <div className="h-1 flex-1 rounded-full overflow-hidden" style={{ background: BORDER }}>
                      <div className="h-full rounded-full transition-all" style={{ width: `${Math.round(doc.relevance * 100)}%`, background: GOLD }} />
                    </div>
                    <span className="text-[10px] tabular-nums" style={{ color: "#6B6660" }}>
                      {Math.round(doc.relevance * 100)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Reader pane ───────────────────────────────────────────────────────────────
function ReaderPane({
  sources, evidence, rules, chunkIdToAlias, usedAliases,
}: {
  sources: ChunkSource[];
  evidence: any[];
  rules: any[];
  chunkIdToAlias: Record<string, string>;
  usedAliases: Set<string>;
}) {
  return (
    <div className="px-5 py-4 space-y-6">
      {/* Case law chunks */}
      {evidence.map((ev, caseIdx) => (
        <div key={caseIdx}>
          <div className="flex items-center gap-2 mb-3">
            <FileText className="h-4 w-4 text-gray-400" />
            <h3 className="text-sm font-semibold text-gray-800 truncate">{ev.case_name || `Case ${caseIdx + 1}`}</h3>
            {ev.metadata?.year && (
              <span className="text-xs text-gray-400">{ev.metadata.year}</span>
            )}
            {ev.pdf_url && (
              <a href={`${API_BASE}${ev.pdf_url}`} target="_blank" rel="noopener noreferrer"
                className="ml-auto flex items-center gap-1 text-xs px-2 py-1 rounded text-white transition-all hover:brightness-110"
                style={{ background: GOLD }}>
                <ExternalLink className="h-3 w-3" />
                PDF
              </a>
            )}
          </div>
          <div className="space-y-3">
            {(ev.chunks || []).slice(0, 3).map((ch: any, chIdx: number) => {
              const alias = chunkIdToAlias[ch.chunk_id || ch.id] || `C${caseIdx + 1}.${chIdx + 1}`;
              const parentAlias = `C${caseIdx + 1}`;
              const srcEntry = sources.find((s) => s.alias === alias);
              const parentEntry = sources.find((s) => s.alias === parentAlias);
              const isUsedDirect = usedAliases.has(alias);
              const isUsedParent = usedAliases.has(parentAlias);
              const isUsed = isUsedDirect || isUsedParent;
              const precision: "chunk" | "document" = isUsedDirect ? "chunk" : "document";
              return (
                <ChunkCard
                  key={chIdx}
                  alias={alias}
                  text={ch.text || ""}
                  isUsed={isUsed}
                  label={ch.component || ""}
                  precision={precision}
                />
              );
            })}
          </div>
        </div>
      ))}

      {/* Statute chunks */}
      {rules.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Scale className="h-4 w-4 text-gray-400" />
            <h3 className="text-sm font-semibold text-gray-800">Statutory Provisions</h3>
          </div>
          <div className="space-y-3">
            {rules.map((rule, rIdx) => {
              const parentAlias = `S${rIdx + 1}`;
              const ruleChunks: any[] = rule.chunks || [];
              // Show individual chunk cards if available, else fall back to full text
              if (ruleChunks.length > 0) {
                return ruleChunks.map((sch: any, schIdx: number) => {
                  const alias = `S${rIdx + 1}.${schIdx + 1}`;
                  const isUsedDirect = usedAliases.has(alias);
                  const isUsedParent = usedAliases.has(parentAlias);
                  const isUsed = isUsedDirect || isUsedParent;
                  const precision: "chunk" | "document" = isUsedDirect ? "chunk" : "document";
                  const secLabel = `${rule.document_label || ""} s.${sch.section_num || rule.article_number}${sch.section_heading ? ` — ${sch.section_heading}` : ""}`;
                  return (
                    <ChunkCard
                      key={`${rIdx}-${schIdx}`}
                      alias={alias}
                      text={sch.text || ""}
                      isUsed={isUsed}
                      label={secLabel}
                      precision={precision}
                    />
                  );
                });
              }
              // Fallback: single card
              const isUsed = usedAliases.has(parentAlias);
              const text = rule.full_text || rule.text_preview || "";
              return (
                <ChunkCard
                  key={rIdx}
                  alias={parentAlias}
                  text={text.slice(0, 600) + (text.length > 600 ? "…" : "")}
                  isUsed={isUsed}
                  label={`${rule.document_label || ""} s.${rule.article_number} — ${rule.article_header || ""}`}
                  precision={isUsed ? "document" : undefined}
                />
              );
            })}
          </div>
        </div>
      )}

      {!evidence.length && !rules.length && (
        <p className="text-sm text-gray-400 text-center py-12">No source chunks available.</p>
      )}
    </div>
  );
}

// ── Chunk card ────────────────────────────────────────────────────────────────
function ChunkCard({
  alias, text, isUsed, label, precision,
}: {
  alias: string;
  text: string;
  isUsed: boolean;
  label?: string;
  precision?: "chunk" | "document";
}) {
  const [expanded, setExpanded] = useState(false);
  const preview = text.slice(0, 280);
  const showToggle = text.length > 280;
  const isDocLevel = isUsed && precision === "document";
  const isChunkLevel = isUsed && precision !== "document";

  return (
    <div
      className="rounded-xl p-4 cursor-pointer transition-all"
      style={{
        background: isChunkLevel ? HIGHLIGHT : isDocLevel ? "#FDF8EC" : "white",
        border: isChunkLevel
          ? `1.5px solid ${GOLD}`
          : isDocLevel
          ? `1.5px solid #E5D9A8`
          : `1.5px dashed ${BORDER}`,
        borderLeft: isChunkLevel ? `3px solid ${GOLD}` : undefined,
      }}
      onClick={() => showToggle && setExpanded((e) => !e)}
    >
      <div className="flex items-start gap-2 mb-2">
        <span className="shrink-0 text-xs font-bold px-1.5 py-0.5 rounded"
          style={{
            background: isChunkLevel ? GOLD : isDocLevel ? "#E8D99A" : CREAM_2,
            color: isChunkLevel ? "white" : isDocLevel ? "#7A6520" : "#888",
          }}>
          {alias}
        </span>
        {label && (
          <span className="text-xs text-gray-500 truncate leading-tight mt-0.5">{label}</span>
        )}
        {isDocLevel && (
          <span className="ml-auto shrink-0 text-[10px] italic" style={{ color: "#9A8550" }}>
            cited at case level
          </span>
        )}
        {isChunkLevel && (
          <span className="ml-auto shrink-0 text-[10px] font-semibold uppercase tracking-widest"
            style={{ color: GOLD_DK }}>
            cited
          </span>
        )}
      </div>
      <p className="text-xs text-gray-700 leading-relaxed" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
        {expanded ? text : preview}
        {!expanded && showToggle && "…"}
      </p>
      {showToggle && (
        <button className="mt-1.5 text-[10px] font-semibold" style={{ color: GOLD }}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

// ── Original pane (PDF viewer) ────────────────────────────────────────────────
function OriginalPane({
  cases, selectedCaseId, onSelectCase, sources,
}: {
  cases: Array<{ caseId: string; label: string; pdfUrl: string | null }>;
  selectedCaseId: string | null;
  onSelectCase: (id: string) => void;
  sources: ChunkSource[];
}) {
  const [ocrInfo, setOcrInfo] = useState<Record<string, boolean>>({});

  const active = cases.find((c) => c.caseId === selectedCaseId) ?? cases[0] ?? null;

  // Fetch OCR status for the active case's first used chunk
  useEffect(() => {
    if (!active) return;
    const usedChunk = sources.find(
      (s) => s.used && s.type === "case" && s.chunk_id
    );
    if (!usedChunk?.chunk_id || ocrInfo[active.caseId] !== undefined) return;
    apiFetch(`/api/chunks/${encodeURIComponent(usedChunk.chunk_id)}/location`)
      .then((r) => r.json())
      .then((d) => {
        if (d.ocr !== undefined) {
          setOcrInfo((prev) => ({ ...prev, [active.caseId]: d.ocr }));
        }
      })
      .catch(() => {});
  }, [active, sources, ocrInfo]);

  if (!cases.length) {
    return (
      <div className="flex items-center justify-center h-64 text-sm text-gray-400">
        No case PDF available for this query.
      </div>
    );
  }

  const pdfSrc = active?.pdfUrl
    ? `${API_BASE}${active.pdfUrl}`
    : active
    ? `${API_BASE}/api/case-pdf?case_id=${encodeURIComponent(active.caseId)}`
    : null;

  const isOcr = active ? ocrInfo[active.caseId] : false;

  return (
    <div className="flex flex-col h-full">
      {/* Case selector */}
      {cases.length > 1 && (
        <div className="flex gap-2 px-4 py-2 flex-wrap shrink-0"
          style={{ borderBottom: `1px solid ${BORDER}` }}>
          {cases.map((c) => (
            <button
              key={c.caseId}
              onClick={() => onSelectCase(c.caseId)}
              className="text-xs px-2.5 py-1.5 rounded-lg font-medium transition-colors"
              style={{
                background: (selectedCaseId ?? cases[0]?.caseId) === c.caseId ? GOLD : CREAM_2,
                color: (selectedCaseId ?? cases[0]?.caseId) === c.caseId ? "white" : "#555",
              }}
            >
              {c.label}
            </button>
          ))}
        </div>
      )}

      {/* OCR chip */}
      {isOcr && (
        <div className="shrink-0 px-4 py-2 flex items-center gap-2"
          style={{ background: "#FFF8E1", borderBottom: `1px solid #FFE082` }}>
          <span className="text-xs font-semibold text-amber-700">OCR</span>
          <span className="text-xs text-amber-600">This document was scanned — text was extracted via OCR and may contain errors.</span>
        </div>
      )}

      {/* PDF embed */}
      {pdfSrc ? (
        <div className="flex-1 relative">
          <object
            data={pdfSrc}
            type="application/pdf"
            className="w-full h-full"
            style={{ minHeight: 500 }}
          >
            <div className="flex flex-col items-center justify-center h-64 gap-3">
              <File className="h-10 w-10 text-gray-300" />
              <p className="text-sm text-gray-400">PDF preview unavailable in this browser.</p>
              <a href={pdfSrc} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-white text-sm"
                style={{ background: GOLD }}>
                <ExternalLink className="h-4 w-4" /> Open PDF
              </a>
            </div>
          </object>
        </div>
      ) : (
        <div className="flex items-center justify-center h-64 text-sm text-gray-400">
          No PDF available for this case.
        </div>
      )}
    </div>
  );
}

// ── Coverage notice ───────────────────────────────────────────────────────────
function CoverageNotice({ coverage, removed }: { coverage?: Coverage; removed?: string[] }) {
  const missing = coverage?.not_in_corpus ?? [];
  if (!missing.length && !(removed?.length)) return null;
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 overflow-hidden">
      <div className="flex items-center gap-2.5 px-4 py-3 border-b border-amber-200">
        <AlertCircle className="h-4 w-4 text-amber-600 shrink-0" />
        <span className="text-sm font-semibold text-amber-900">Knowledge Base Coverage</span>
      </div>
      <div className="px-4 py-3 space-y-2 text-sm">
        {missing.length > 0 && (
          <p className="text-amber-800">
            <span className="font-medium">Not in knowledge base: </span>
            {missing.join("; ")}.
          </p>
        )}
        {removed && removed.length > 0 && (
          <p className="text-amber-700 text-xs">
            {removed.length} unverified citation{removed.length > 1 ? "s" : ""} removed.
          </p>
        )}
      </div>
    </div>
  );
}

// ── Classification widget ─────────────────────────────────────────────────────
function ClassificationWidget({ classification }: { classification: any }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded-xl bg-white shadow-sm border overflow-hidden" style={{ borderColor: BORDER }}>
      <div className="flex items-center justify-between px-4 py-2.5 cursor-pointer"
        style={{ background: "#F5F0FF" }} onClick={() => setExpanded((e) => !e)}>
        <div className="flex items-center gap-2">
          <Search className="h-4 w-4 text-purple-500" />
          <span className="text-sm font-semibold text-gray-800">Classification</span>
          {classification.legal_domains?.length > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 font-medium">
              {classification.legal_domains[0]?.domain}
            </span>
          )}
        </div>
        {expanded ? <ChevronUp className="h-4 w-4 text-gray-400" /> : <ChevronDown className="h-4 w-4 text-gray-400" />}
      </div>
      {expanded && (
        <div className="px-4 py-3 space-y-3">
          {classification.intents?.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {classification.intents.map((intent: string, i: number) => (
                <span key={i} className="px-2.5 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium border border-blue-100">
                  {intent.replace(/_/g, " ")}
                </span>
              ))}
            </div>
          )}
          {classification.reasoning_summary && (
            <p className="text-xs text-gray-600 leading-relaxed bg-gray-50 px-3 py-2 rounded-lg">
              {classification.reasoning_summary}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Inline markdown ───────────────────────────────────────────────────────────
function renderInline(text: string): ReactNode[] {
  // Split on bold, code, italic, and citation badges [C1.1] [S2]
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[CS]\d+(?:\.\d+)?\])/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={i} className="font-semibold text-gray-900">{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`"))
      return <code key={i} className="bg-slate-100 text-green-700 px-1.5 py-0.5 rounded text-xs font-mono">{part.slice(1, -1)}</code>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={i} className="italic text-gray-600">{part.slice(1, -1)}</em>;
    if (/^\[[CS]\d+/.test(part))
      return (
        <span key={i} className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-bold mx-0.5"
          style={{ background: HIGHLIGHT, color: GOLD_DK }}>
          {part}
        </span>
      );
    return part;
  });
}

function parseMd(raw: string): ReactNode[] {
  const lines = raw.split("\n");
  const out: ReactNode[] = [];
  let ulItems: string[] = [];
  let olItems: string[] = [];

  const flushUl = () => {
    if (!ulItems.length) return;
    out.push(
      <ul key={out.length} className="list-disc list-outside pl-5 mb-3 space-y-1.5">
        {ulItems.map((t, i) => <li key={i} className="text-sm text-gray-800 leading-relaxed">{renderInline(t)}</li>)}
      </ul>
    );
    ulItems = [];
  };
  const flushOl = () => {
    if (!olItems.length) return;
    out.push(
      <ol key={out.length} className="list-decimal list-outside pl-5 mb-3 space-y-1.5">
        {olItems.map((t, i) => <li key={i} className="text-sm text-gray-800 leading-relaxed">{renderInline(t)}</li>)}
      </ol>
    );
    olItems = [];
  };

  for (const line of lines) {
    if (/^#{4}\s/.test(line)) { flushUl(); flushOl(); out.push(<h4 key={out.length} className="text-sm font-semibold text-gray-700 mt-4 mb-1">{renderInline(line.slice(5))}</h4>); continue; }
    if (/^#{3}\s/.test(line)) { flushUl(); flushOl(); out.push(<h3 key={out.length} className="text-base font-semibold text-gray-900 mt-5 mb-2 border-b pb-1" style={{ borderColor: BORDER }}>{renderInline(line.slice(4))}</h3>); continue; }
    if (/^#{2}\s/.test(line)) { flushUl(); flushOl(); out.push(<h2 key={out.length} className="text-lg font-bold text-gray-900 mt-6 mb-2">{renderInline(line.slice(3))}</h2>); continue; }
    if (/^>\s/.test(line)) { flushUl(); flushOl(); out.push(<blockquote key={out.length} className="border-l-4 border-green-400 pl-4 my-3 bg-green-50 py-2 pr-3 rounded-r-lg"><p className="text-sm text-green-900 italic leading-relaxed">{renderInline(line.slice(2))}</p></blockquote>); continue; }
    const ulM = line.match(/^\s*[*\-]\s{1,3}(.+)$/);
    if (ulM) { flushOl(); ulItems.push(ulM[1]); continue; }
    const olM = line.match(/^\s*\d+\.\s{1,3}(.+)$/);
    if (olM) { flushUl(); olItems.push(olM[1]); continue; }
    if (line.trim() === "") { flushUl(); flushOl(); continue; }
    flushUl(); flushOl();
    out.push(<p key={out.length} className="text-sm text-gray-800 leading-relaxed mb-2.5">{renderInline(line)}</p>);
  }
  flushUl(); flushOl();
  return out;
}

// ── AI response widget ────────────────────────────────────────────────────────
function AIResponseWidget({ response }: { response: string }) {
  return (
    <div className="rounded-xl bg-white shadow-sm border overflow-hidden" style={{ borderColor: BORDER }}>
      <div className="px-4 py-2.5 flex items-center gap-2" style={{ background: "#F0FDF4" }}>
        <Scale className="h-4 w-4 text-green-600" />
        <span className="text-sm font-semibold text-gray-800">Legal Analysis</span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 font-medium ml-1">
          Source-grounded
        </span>
      </div>
      <div className="px-4 py-4 space-y-0">{parseMd(response)}</div>
    </div>
  );
}
