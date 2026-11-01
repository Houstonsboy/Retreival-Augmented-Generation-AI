"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Scale, FileText, ChevronDown, ChevronUp, AlertCircle,
  ExternalLink, BookOpen, X, Grid2X2, Copy, LayoutGrid,
  Menu, Plus, Pin, Trash2, Pencil, Check, RotateCcw, Clock,
} from "lucide-react";
import dynamic from "next/dynamic";
import { apiFetch } from "@/app/lib/api";
import { UserMenu } from "@/app/components/UserMenu";
import { useAuth } from "@/app/context/AuthContext";
import {
  listConversations, getConversation, patchConversation,
  deleteConversation as apiDeleteConversation,
  type ConvSummary, type HistoryTurn,
} from "@/app/lib/historyApi";
import type { ReactNode } from "react";
import type { ChunkLocation } from "@/app/components/PdfViewer";
import { ChatPanelSkeleton } from "@/app/components/skeleton/ChatPanelSkeleton";
import { SidebarSkeleton } from "@/app/components/skeleton/SidebarSkeleton";
import { WorkspaceSkeleton } from "@/app/components/skeleton/WorkspaceSkeleton";
import { Reveal } from "@/app/components/skeleton/Reveal";
import { DevReplay } from "@/app/components/skeleton/DevReplay";

const PdfViewer = dynamic(() => import("@/app/components/PdfViewer"), { ssr: false });

// ── Dev-only simulated latency ──────────────────────────────────────────────────
async function devDelay(ms = 1800): Promise<void> {
  if (process.env.NODE_ENV !== "development") return;
  if (typeof window === "undefined") return;
  if (!new URLSearchParams(window.location.search).has("simulateLatency")) return;
  await new Promise<void>((r) => setTimeout(r, ms));
}

// ── Design tokens ───────────────────────────────────────────────────────────────
const BG        = "#FAF7F0";
const WS_BG     = "#F5F0E4";
const SURFACE   = "#FFFDF8";
const GOLD      = "#C9971A";
const GOLD_DK   = "#A87A0C";
const HIGHLIGHT = "#F6E7B8";
const BORDER    = "#ECE6D8";
const CREAM_2   = "#F5F0E4";
const INK       = "#151515";
const MUTED     = "#6B6660";
const API_BASE  = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

const NR  = "var(--font-newsreader), Georgia, 'Times New Roman', serif";
const MNR = "var(--font-manrope), system-ui, sans-serif";

// ── Types ───────────────────────────────────────────────────────────────────────
interface Coverage   { in_corpus: string[]; not_in_corpus: string[] }
interface Prospect   { label: string; rating: "Strong" | "Moderate" | "Weak" | string }
interface ChunkSource {
  alias: string; chunk_id: string; label: string;
  type: "case" | "statute" | "knowledge" | "constitution";
  used: boolean; precision?: "chunk" | "document";
  display_title?: string; short_label?: string;
  doc_type?: string; court?: string; year?: string; relevance?: number;
  passages?: Array<{ chunk_id: string; alias: string; label: string; precision: string }>;
  // History-enriched fields
  case_identifier?: string | null;
  parties?: string | null;
  court_level?: string | null;
  document_label?: string | null;
  article_number?: string | null;
  article_header?: string | null;
  pdf_url?: string | null;
  snippet?: string | null;
}

interface Message {
  id: string; role: "user" | "assistant"; content: string; timestamp: Date;
  classification?: any; aiResponse?: string;
  evidence?: any[]; rules?: any[];
  coverage?: Coverage; premiseCheck?: string; unverifiedClaims?: string[];
  sources?: ChunkSource[]; aliasMap?: Record<string, string>;
  prospects?: Prospect[];
  // History fields
  turnId?: string;
  query?: string;
  historyStatus?: "complete" | "failed" | "not_saved";
  errorCode?: string;
  errorMessage?: string;
  fromHistory?: boolean;
}

// ── Example questions ───────────────────────────────────────────────────────────
const EXAMPLES = [
  { category: "DIFFICULT JUDGMENT", question: "When will the SCC set aside an ex parte judgment after service by WhatsApp?" },
  { category: "ELECTRONIC EVIDENCE", question: "Are M-Pesa statements admissible without a section 106B certificate?" },
  { category: "EXECUTION",           question: "What must I show to stay execution of a warrant of attachment?" },
  { category: "BAILMENT",            question: "Is a bailee for reward liable for theft by an independent security contractor?" },
];

// ── FIRAC parser ────────────────────────────────────────────────────────────────
interface FiracSection { letter: string; label: string; content: string }

function parseFirac(raw: string): FiracSection[] | null {
  const MARKERS: Array<{ letter: string; label: string; re: RegExp }> = [
    { letter: "F", label: "FACTS",       re: /^#{1,3}\s*FACTS?\s*$|^\*\*FACTS?\*\*\s*$|^FACTS?:\s*$/im },
    { letter: "I", label: "ISSUES",      re: /^#{1,3}\s*ISSUES?\s*$|^\*\*ISSUES?\*\*\s*$|^ISSUES?:\s*$/im },
    { letter: "R", label: "RULES",       re: /^#{1,3}\s*(RULES?|RELEVANT LAW|LAW)\s*$|^\*\*RULES?\*\*\s*$|^RULES?:\s*$/im },
    { letter: "A", label: "APPLICATION", re: /^#{1,3}\s*(APPLICATION|ANALYSIS|APPLYING)\s*$|^\*\*APPLICATION\*\*\s*$/im },
    { letter: "C", label: "CONCLUSION",  re: /^#{1,3}\s*CONCLUSIONS?\s*$|^\*\*CONCLUSIONS?\*\*\s*$|^CONCLUSIONS?:\s*$/im },
  ];
  const lines = raw.split("\n");
  const found: Array<{ idx: number; letter: string; label: string }> = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i].trim();
    for (const m of MARKERS) {
      if (m.re.test(l)) { found.push({ idx: i, ...m }); break; }
    }
  }
  if (found.length < 2) return null;
  return found.map((f, fi) => ({
    letter: f.letter, label: f.label,
    content: lines.slice(f.idx + 1, found[fi + 1]?.idx ?? lines.length).join("\n").trim(),
  }));
}

// ── Relative time helper ────────────────────────────────────────────────────────
function relativeTime(isoStr: string | null): string {
  if (!isoStr) return "";
  try {
    const ms = Date.now() - new Date(isoStr).getTime();
    const s = Math.floor(ms / 1000);
    if (s < 60) return "just now";
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.floor(h / 24);
    if (d < 7) return `${d}d ago`;
    return new Date(isoStr).toLocaleDateString();
  } catch { return ""; }
}

// ── Reconstruct evidence/rules from stored history sources ──────────────────────
function buildEvidenceFromSources(sources: ChunkSource[]): { evidence: any[]; rules: any[] } {
  const caseMap = new Map<string, any>();
  const ruleMap = new Map<string, any>();
  for (const src of sources) {
    if (src.type === "case") {
      const key = src.case_identifier || src.label || src.chunk_id;
      if (!caseMap.has(key)) {
        caseMap.set(key, {
          case_name: src.case_identifier || src.label,
          case_identifier: src.case_identifier,
          metadata: { court_level: src.court_level, year: src.year, parties: src.parties },
          pdf_url: src.pdf_url || null,
          chunks: [],
        });
      }
      caseMap.get(key).chunks.push({
        chunk_id: src.chunk_id,
        id: src.chunk_id,
        text: src.snippet || "(text not stored in history)",
      });
    } else {
      const key = src.document_label || src.label || src.chunk_id;
      if (!ruleMap.has(key)) {
        ruleMap.set(key, {
          document_label: src.document_label || src.label,
          document_type: src.type,
          article_number: src.article_number,
          article_header: src.article_header,
          pdf_url: src.pdf_url || null,
          chunks: [],
        });
      }
      ruleMap.get(key).chunks.push({
        chunk_id: src.chunk_id,
        text: src.snippet || "(text not stored in history)",
      });
    }
  }
  return { evidence: Array.from(caseMap.values()), rules: Array.from(ruleMap.values()) };
}

// ── Main component ──────────────────────────────────────────────────────────────
export function ChatView({ initialConversationId }: { initialConversationId?: string }) {
  const router = useRouter();
  const { user } = useAuth();

  // ── Existing state ────────────────────────────────────────────────────────────
  const [messages,      setMessages]      = useState<Message[]>([]);
  const [input,         setInput]         = useState("");
  const [isLoading,     setIsLoading]     = useState(false);
  const [panelMessage,  setPanelMessage]  = useState<Message | null>(null);
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const [activeDocKey,  setActiveDocKey]  = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef       = useRef<HTMLTextAreaElement>(null);

  // ── History state ─────────────────────────────────────────────────────────────
  const [conversationId,    setConversationId]    = useState<string | null>(initialConversationId ?? null);
  const [conversationTitle, setConversationTitle] = useState<string | null>(null);
  const [sidebarOpen,       setSidebarOpen]       = useState(false);
  const [conversations,     setConversations]     = useState<ConvSummary[]>([]);
  const [sidebarNextCursor, setSidebarNextCursor] = useState<string | null>(null);
  const [sidebarLoading,    setSidebarLoading]    = useState(false);
  const [notSavedNote,      setNotSavedNote]      = useState(false);
  const [deleteConfirmId,   setDeleteConfirmId]   = useState<string | null>(null);
  const [renamingId,        setRenamingId]        = useState<string | null>(null);
  const [renameValue,       setRenameValue]       = useState("");
  // true while loadConversation() is in flight (shows ChatPanelSkeleton)
  const [isConversationLoading, setIsConversationLoading] = useState(!!initialConversationId);
  const [announceMsg, setAnnounceMsg] = useState(initialConversationId ? "Loading document…" : "");

  // ── Refs ──────────────────────────────────────────────────────────────────────
  const requestCounterRef      = useRef(0);
  const abortControllerRef     = useRef<AbortController | null>(null);
  const titleUnsubscribeRef    = useRef<(() => void) | null>(null);
  const titleTimeoutRef        = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [chatWidth, setChatWidth] = useState(() =>
    typeof window !== "undefined" ? Math.round(window.innerWidth * 0.4) : 480
  );
  const dragStateRef = useRef<{ startX: number; startW: number } | null>(null);

  // ── Load conversation and sidebar on mount ────────────────────────────────────
  useEffect(() => {
    if (!user) return;
    loadSidebar();
    if (initialConversationId) {
      loadConversation(initialConversationId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.uid, initialConversationId]);

  // ── Clear state on logout ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!user) {
      setMessages([]);
      setConversationId(null);
      setConversationTitle(null);
      setConversations([]);
      titleUnsubscribeRef.current?.();
      titleUnsubscribeRef.current = null;
    }
  }, [user]);

  // ── Cleanup on unmount ────────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      titleUnsubscribeRef.current?.();
      if (titleTimeoutRef.current) clearTimeout(titleTimeoutRef.current);
      abortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const last = [...messages].reverse().find(
      (m) => m.role === "assistant" && m.sources && m.sources.length > 0
    );
    if (last) {
      setPanelMessage(last);
      const firstCase = last.evidence?.[0];
      if (firstCase) setActiveDocKey(firstCase.case_name || firstCase.case_identifier || "case-0");
    }
  }, [messages]);

  // ── Sidebar ───────────────────────────────────────────────────────────────────
  const loadSidebar = async (cursor?: string | null) => {
    if (!user) return;
    setSidebarLoading(true);
    await devDelay(1200);
    try {
      const res = await listConversations(cursor);
      if (cursor) {
        setConversations((prev) => [...prev, ...res.conversations]);
      } else {
        setConversations(res.conversations);
      }
      setSidebarNextCursor(res.next_cursor ?? null);
    } catch (e) {
      console.error("[Sidebar] load failed:", e);
    } finally {
      setSidebarLoading(false);
    }
  };

  const loadConversation = async (cid: string) => {
    if (!user) return;
    setIsConversationLoading(true);
    setAnnounceMsg("Loading document…");
    await devDelay(1800);
    try {
      const res = await getConversation(cid);
      setConversationId(cid);
      setConversationTitle(res.conversation.title);

      const msgs: Message[] = [];
      for (const turn of res.turns) {
        msgs.push({
          id: `hist-user-${turn.id}`,
          role: "user",
          content: turn.query,
          timestamp: turn.created_at ? new Date(turn.created_at) : new Date(),
          query: turn.query,
          turnId: turn.id,
        });
        const src = (turn.sources || []) as ChunkSource[];
        const { evidence, rules } = buildEvidenceFromSources(src);
        msgs.push({
          id: `hist-asst-${turn.id}`,
          role: "assistant",
          content: "",
          aiResponse: turn.answer_markdown ||
            (turn.status === "failed" ? `[Error] ${turn.error || "Unknown error"}` : ""),
          sources: src,
          aliasMap: turn.alias_map || {},
          coverage: turn.coverage,
          premiseCheck: turn.answer_structured?.premise_check,
          evidence,
          rules,
          timestamp: turn.created_at ? new Date(turn.created_at) : new Date(),
          turnId: turn.id,
          query: turn.query,
          historyStatus: turn.status === "complete" ? "complete" : "failed",
          errorCode: turn.error_code ?? undefined,
          errorMessage: turn.error ?? undefined,
          fromHistory: true,
        });
      }
      setMessages(msgs);
    } catch (e: any) {
      if (e?.status === 404) router.replace("/");
      else console.error("[History] loadConversation failed:", e);
    } finally {
      setIsConversationLoading(false);
      setAnnounceMsg("Document loaded");
    }
  };

  // ── Title listener (polls REST API — no client-side Firestore rules needed) ───
  const startTitleListener = useCallback((cid: string) => {
    titleUnsubscribeRef.current?.();
    if (titleTimeoutRef.current) clearTimeout(titleTimeoutRef.current);

    let cancelled = false;
    const startedAt = Date.now();

    const poll = async () => {
      if (cancelled) return;
      if (Date.now() - startedAt > 30_000) return;
      try {
        const res = await getConversation(cid, 0);
        if (cancelled) return;
        const { title, title_source } = res.conversation;
        setConversationTitle(title);
        setConversations((prev) =>
          prev.map((c) => c.id === cid ? { ...c, title, title_source: title_source as any } : c)
        );
        if (title_source !== "fallback") return;
      } catch { /* ignore */ }
      if (!cancelled) titleTimeoutRef.current = setTimeout(poll, 2_000);
    };

    titleUnsubscribeRef.current = () => {
      cancelled = true;
      if (titleTimeoutRef.current) clearTimeout(titleTimeoutRef.current);
    };
    titleTimeoutRef.current = setTimeout(poll, 2_000);
  }, []);

  // ── Conversation management ───────────────────────────────────────────────────
  const handleNewChat = useCallback(() => {
    abortControllerRef.current?.abort();
    titleUnsubscribeRef.current?.();
    titleUnsubscribeRef.current = null;
    setMessages([]);
    setConversationId(null);
    setConversationTitle(null);
    setNotSavedNote(false);
    setInput("");
    setSidebarOpen(false);
    if (initialConversationId) router.push("/");
  }, [initialConversationId, router]);

  const handleSelectConversation = useCallback((cid: string) => {
    if (cid === conversationId) { setSidebarOpen(false); return; }
    abortControllerRef.current?.abort();
    titleUnsubscribeRef.current?.();
    titleUnsubscribeRef.current = null;
    setSidebarOpen(false);
    router.push(`/chat/${cid}`);
  }, [conversationId, router]);

  const handleDeleteConversation = useCallback(async (cid: string) => {
    setDeleteConfirmId(null);
    try {
      await apiDeleteConversation(cid);
      setConversations((prev) => prev.filter((c) => c.id !== cid));
      if (cid === conversationId) handleNewChat();
    } catch (e) {
      console.error("[History] delete failed:", e);
    }
  }, [conversationId, handleNewChat]);

  const handleRenameSubmit = useCallback(async (cid: string) => {
    const title = renameValue.trim();
    setRenamingId(null);
    if (!title || title.length > 120) return;
    try {
      await patchConversation(cid, { title });
      setConversations((prev) =>
        prev.map((c) => c.id === cid ? { ...c, title, title_source: "user" } : c)
      );
      if (cid === conversationId) setConversationTitle(title);
    } catch (e) {
      console.error("[History] rename failed:", e);
    }
  }, [renameValue, conversationId]);

  const handleTogglePin = useCallback(async (cid: string, pinned: boolean) => {
    try {
      await patchConversation(cid, { pinned });
      setConversations((prev) =>
        prev.map((c) => c.id === cid ? { ...c, pinned } : c)
      );
    } catch (e) {
      console.error("[History] pin failed:", e);
    }
  }, []);

  // Upsert a conversation at the top of the sidebar list
  const upsertSidebarConv = useCallback((summary: Partial<ConvSummary> & { id: string }) => {
    setConversations((prev) => {
      const existing = prev.find((c) => c.id === summary.id);
      const updated: ConvSummary = {
        id: summary.id,
        title: summary.title ?? existing?.title ?? "New conversation",
        title_source: summary.title_source ?? existing?.title_source ?? "fallback",
        pinned: summary.pinned ?? existing?.pinned ?? false,
        turn_count: (summary.turn_count ?? existing?.turn_count ?? 0),
        updated_at: summary.updated_at ?? new Date().toISOString(),
      };
      return [updated, ...prev.filter((c) => c.id !== summary.id)];
    });
  }, []);

  // ── Core request logic ────────────────────────────────────────────────────────
  const doRequest = useCallback(async (query: string, reuseTurnId?: string) => {
    const thisTurnId    = reuseTurnId ?? crypto.randomUUID();
    const thisRequestId = ++requestCounterRef.current;
    const thisConvId    = conversationId;

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setNotSavedNote(false);

    try {
      const response = await apiFetch("/api/qretrieve", {
        method: "POST",
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || "Retrieval failed");
      if (requestCounterRef.current !== thisRequestId) return;

      let aiResponse = "No relevant legal information found.";
      let coverage: Coverage | undefined, premiseCheck: string | undefined;
      let unverifiedClaims: string[] | undefined, sources: ChunkSource[] | undefined;
      let aliasMap: Record<string, string> | undefined, prospects: Prospect[] | undefined;
      let savedConvId = thisConvId, savedTurnId: string | undefined;
      let historyStatus: Message["historyStatus"] = "not_saved";

      try {
        const ansRes = await apiFetch("/api/generate-legal-answer", {
          method: "POST",
          body: JSON.stringify({
            query,
            retrieval: data.retrieval,
            retrieval_mode: data.classification?.retrieval_mode || "",
            classification: data.classification || {},
            conversation_id: thisConvId,
            client_turn_id: thisTurnId,
          }),
          signal: controller.signal,
        });
        if (requestCounterRef.current !== thisRequestId) return;

        if (ansRes.ok) {
          const ans = await ansRes.json();
          if (ans.success) {
            aiResponse = ans.answer;
            coverage = ans.coverage; premiseCheck = ans.premise_check;
            unverifiedClaims = ans.unverified_claims_removed;
            sources = ans.sources; aliasMap = ans.alias_map; prospects = ans.prospects;

            if (ans.history_saved) {
              historyStatus = "complete";
              savedConvId   = ans.conversation_id;
              savedTurnId   = ans.turn_id;
              const newTitle = ans.conversation_title || "New conversation";
              setConversationTitle(newTitle);

              if (!thisConvId && savedConvId) {
                setConversationId(savedConvId);
                // replaceState avoids a Next.js navigation (which would remount
                // ChatView and reload from history, wiping the in-memory answer)
                window.history.replaceState(null, "", `/chat/${savedConvId}`);
              }
              upsertSidebarConv({
                id: savedConvId!,
                title: newTitle,
                turn_count: 1,
                updated_at: new Date().toISOString(),
              });
              if (savedConvId) {
                startTitleListener(savedConvId);
              }
            } else {
              historyStatus = "not_saved";
              setNotSavedNote(true);
            }
          }
        }
      } catch (inner) {
        if ((inner as any)?.name === "AbortError") return;
        // keep fallback answer
      }

      if (requestCounterRef.current !== thisRequestId) return;

      const newMsg: Message = {
        id: `${Date.now() + 1}`,
        role: "assistant", content: "",
        classification: data.classification,
        aiResponse,
        evidence: data.retrieval?.results || [],
        rules: data.retrieval?.rules    || [],
        coverage, premiseCheck, unverifiedClaims, sources, aliasMap, prospects,
        timestamp: new Date(),
        turnId: savedTurnId || thisTurnId,
        query,
        historyStatus,
      };

      if (reuseTurnId) {
        // Replace the failed turn's assistant message in place
        setMessages((prev) => {
          for (let i = prev.length - 1; i >= 0; i--) {
            if (prev[i].role === "assistant" && prev[i].turnId === reuseTurnId) {
              const next = [...prev];
              next[i] = newMsg;
              return next;
            }
          }
          return [...prev, newMsg];
        });
      } else {
        setMessages((prev) => [...prev, newMsg]);
      }

    } catch (err) {
      if ((err as any)?.name === "AbortError") return;
      if (requestCounterRef.current !== thisRequestId) return;
      const errMsg = err instanceof Error ? err.message : "An error occurred.";
      const failedMsg: Message = {
        id: `${Date.now() + 1}`, role: "assistant", content: "",
        aiResponse: errMsg, timestamp: new Date(),
        turnId: reuseTurnId || thisTurnId, query,
        historyStatus: "failed", errorMessage: errMsg,
      };
      if (reuseTurnId) {
        setMessages((prev) => {
          for (let i = prev.length - 1; i >= 0; i--) {
            if (prev[i].role === "assistant" && prev[i].turnId === reuseTurnId) {
              const next = [...prev]; next[i] = failedMsg; return next;
            }
          }
          return [...prev, failedMsg];
        });
      } else {
        setMessages((prev) => [...prev, failedMsg]);
      }
    } finally {
      if (requestCounterRef.current === thisRequestId) setIsLoading(false);
    }
  }, [conversationId, user, router, startTitleListener, upsertSidebarConv]);

  const handleSubmit = useCallback(() => {
    const q = input.trim();
    if (!q || isLoading) return;
    setMessages((prev) => [...prev, {
      id: Date.now().toString(), role: "user", content: q, timestamp: new Date(),
    }]);
    setInput("");
    doRequest(q);
  }, [input, isLoading, doRequest]);

  const handleRetry = useCallback((msg: Message) => {
    if (!msg.query || !msg.turnId || isLoading) return;
    doRequest(msg.query, msg.turnId);
  }, [isLoading, doRequest]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };
  const autoResize = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const t = e.target as HTMLTextAreaElement;
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 200)}px`;
  };

  // ── Citation click ────────────────────────────────────────────────────────────
  const handleCitationClick = useCallback((alias: string, msg: Message) => {
    if (!msg.aliasMap) return;
    const chunkId = msg.aliasMap[alias];
    if (chunkId) setActiveChunkId(chunkId);
    if (msg.evidence) {
      for (const ev of msg.evidence) {
        const match = (ev.chunks || []).find((ch: any) => ch.chunk_id === chunkId || ch.id === chunkId);
        if (match) { setActiveDocKey(ev.case_name || ev.case_identifier || "case-0"); break; }
      }
    }
    if (msg.sources) {
      const src = msg.sources.find((s) => s.alias === alias);
      if (src?.type === "statute" || src?.type === "constitution") setActiveDocKey(`statute-${alias}`);
    }
    setPanelMessage(msg);
  }, []);

  const buildCitLabel = (alias: string, msg: Message): string => {
    const src = msg.sources?.find((s) => s.alias === alias);
    if (src?.short_label) return src.short_label;
    if (src?.display_title) return src.display_title.slice(0, 30) + (src.display_title.length > 30 ? "…" : "");
    if (src?.label) return src.label.slice(0, 30);
    if (alias.startsWith("C") && msg.evidence) {
      const idx = parseInt(alias.replace(/\D+.*/, "").slice(1)) - 1;
      const ev = msg.evidence[idx];
      if (ev) {
        const parts = ev.case_name?.split(" v ") || ev.case_name?.split(" v. ");
        return parts ? parts[parts.length - 1]?.trim().slice(0, 25) || ev.case_name?.slice(0, 25) : alias;
      }
    }
    if (alias.startsWith("S") && msg.rules) {
      const idx = parseInt(alias.replace(/\D+.*/, "").slice(1)) - 1;
      const r = msg.rules[idx];
      if (r) return `${r.document_label || r.act_title || ""} s.${r.article_number || r.section_num}`.slice(0, 30);
    }
    return alias;
  };

  // ── Resizable splitter ────────────────────────────────────────────────────────
  const onResizerMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragStateRef.current = { startX: e.clientX, startW: chatWidth };
    const onMove = (ev: MouseEvent) => {
      if (!dragStateRef.current) return;
      const delta = ev.clientX - dragStateRef.current.startX;
      const maxW  = typeof window !== "undefined" ? Math.round(window.innerWidth * 0.65) : 720;
      setChatWidth(Math.max(260, Math.min(maxW, dragStateRef.current.startW + delta)));
    };
    const onUp = () => {
      dragStateRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [chatWidth]);

  const hasMessages = messages.length > 0;
  const hasSources  = !!(panelMessage?.sources?.length);

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div
      className="flex h-screen flex-col"
      style={{ backgroundColor: BG, fontFamily: MNR }}
      aria-busy={isConversationLoading}
    >
      {/* Screen-reader live region — announces load start and completion */}
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announceMsg}
      </p>

      {/* Dev-only replay button (stripped from production builds) */}
      <DevReplay />

      {/* ── Sidebar backdrop ──────────────────────────────────────────────── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40"
          style={{ background: "rgba(21,21,21,0.35)" }}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Left sidebar drawer ───────────────────────────────────────────── */}
      <aside
        className="fixed left-0 top-0 bottom-0 z-50 flex flex-col overflow-hidden"
        style={{
          width: 322,
          background: SURFACE,
          borderRight: `1px solid ${BORDER}`,
          boxShadow: "4px 0 24px rgba(60,45,10,0.12)",
          transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)",
          transition: "transform 0.22s cubic-bezier(.4,0,.2,1)",
        }}
      >
        {/* Sidebar header */}
        <div className="shrink-0 flex items-center justify-between px-4 py-3"
          style={{ borderBottom: `1px solid ${BORDER}` }}>
          <span className="text-xs font-bold tracking-[.15em] uppercase" style={{ color: MUTED }}>
            Chat History
          </span>
          <button onClick={() => setSidebarOpen(false)}
            className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-amber-50 transition-colors">
            <X className="h-4 w-4" style={{ color: MUTED }} />
          </button>
        </div>

        {/* New chat button */}
        <div className="shrink-0 px-3 py-2.5" style={{ borderBottom: `1px solid ${BORDER}` }}>
          <button onClick={handleNewChat}
            className="w-full flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm font-semibold transition-all hover:shadow-sm"
            style={{ background: GOLD, color: "white" }}>
            <Plus className="h-4 w-4 shrink-0" />
            New chat
          </button>
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto py-2">
          {/* Skeleton while the first page of conversations loads */}
          {sidebarLoading && conversations.length === 0 && <SidebarSkeleton />}
          {conversations.length === 0 && !sidebarLoading && (
            <p className="px-4 py-6 text-xs text-center" style={{ color: MUTED }}>
              No conversations yet. Start by asking a question.
            </p>
          )}
          {conversations.map((conv) => {
            const isActive  = conv.id === conversationId;
            const isRenaming = renamingId === conv.id;
            const isDeleting = deleteConfirmId === conv.id;
            return (
              <div key={conv.id} className="relative group px-2 mb-0.5">
                <div
                  onClick={() => !isRenaming && !isDeleting && handleSelectConversation(conv.id)}
                  className="w-full rounded-xl px-3 py-2.5 transition-all cursor-pointer"
                  style={{
                    background: isActive ? HIGHLIGHT : "transparent",
                    border: `1.5px solid ${isActive ? GOLD : "transparent"}`,
                  }}
                >
                  {/* Title row */}
                  {isRenaming ? (
                    <form onSubmit={(e) => { e.preventDefault(); handleRenameSubmit(conv.id); }}
                      className="flex items-center gap-1">
                      <input
                        autoFocus
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onBlur={() => handleRenameSubmit(conv.id)}
                        onKeyDown={(e) => e.key === "Escape" && setRenamingId(null)}
                        maxLength={120}
                        className="flex-1 min-w-0 text-xs rounded-md px-1.5 py-1 focus:outline-none"
                        style={{ border: `1px solid ${GOLD}`, background: "white", color: INK }}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <button type="submit" className="shrink-0 w-5 h-5 flex items-center justify-center"
                        onClick={(e) => e.stopPropagation()}>
                        <Check className="h-3.5 w-3.5" style={{ color: GOLD_DK }} />
                      </button>
                    </form>
                  ) : (
                    <p className="text-xs font-medium leading-snug truncate pr-1"
                      style={{ color: isActive ? GOLD_DK : INK }}>
                      {conv.pinned && <Pin className="inline h-2.5 w-2.5 mr-1 mb-0.5" style={{ color: GOLD }} />}
                      {conv.title}
                    </p>
                  )}

                  {/* Meta row */}
                  {!isRenaming && (
                    <div className="flex items-center justify-between mt-0.5">
                      <span className="text-[10px]" style={{ color: MUTED }}>
                        <Clock className="inline h-2.5 w-2.5 mr-0.5 mb-0.5" />
                        {relativeTime(conv.updated_at)}
                      </span>
                      <span className="text-[10px]" style={{ color: MUTED }}>
                        {conv.turn_count} turn{conv.turn_count !== 1 ? "s" : ""}
                      </span>
                    </div>
                  )}

                  {/* Delete confirm */}
                  {isDeleting && (
                    <div className="mt-2 flex items-center gap-1.5"
                      onClick={(e) => e.stopPropagation()}>
                      <span className="text-[10px]" style={{ color: MUTED }}>Delete?</span>
                      <button onClick={() => handleDeleteConversation(conv.id)}
                        className="text-[10px] font-semibold rounded-full px-2 py-0.5 transition-colors"
                        style={{ background: "#FEE2E2", color: "#B91C1C" }}>Yes</button>
                      <button onClick={() => setDeleteConfirmId(null)}
                        className="text-[10px] font-semibold rounded-full px-2 py-0.5 transition-colors"
                        style={{ background: CREAM_2, color: MUTED }}>No</button>
                    </div>
                  )}
                </div>

                {/* Action icons (visible on hover) */}
                {!isRenaming && !isDeleting && (
                  <div className="absolute right-4 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center gap-1">
                    <button title={conv.pinned ? "Unpin" : "Pin"}
                      onClick={(e) => { e.stopPropagation(); handleTogglePin(conv.id, !conv.pinned); }}
                      className="w-5 h-5 flex items-center justify-center rounded hover:bg-amber-50 transition-colors">
                      <Pin className="h-3 w-3" style={{ color: conv.pinned ? GOLD : MUTED }} />
                    </button>
                    <button title="Rename"
                      onClick={(e) => { e.stopPropagation(); setRenamingId(conv.id); setRenameValue(conv.title); }}
                      className="w-5 h-5 flex items-center justify-center rounded hover:bg-amber-50 transition-colors">
                      <Pencil className="h-3 w-3" style={{ color: MUTED }} />
                    </button>
                    <button title="Delete"
                      onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(conv.id); }}
                      className="w-5 h-5 flex items-center justify-center rounded hover:bg-red-50 transition-colors">
                      <Trash2 className="h-3 w-3" style={{ color: "#B91C1C" }} />
                    </button>
                  </div>
                )}
              </div>
            );
          })}

          {/* Load more */}
          {sidebarNextCursor && (
            <div className="px-3 pt-1 pb-2">
              <button onClick={() => loadSidebar(sidebarNextCursor)} disabled={sidebarLoading}
                className="w-full text-xs rounded-lg py-2 font-medium transition-colors"
                style={{ background: CREAM_2, color: sidebarLoading ? MUTED : GOLD_DK }}>
                {sidebarLoading ? "Loading…" : "Load more"}
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* ── Top bar ───────────────────────────────────────────────────────── */}
      <header className="shrink-0 z-20" style={{ backgroundColor: BG, borderBottom: `1px solid ${BORDER}` }}>
        <div className="px-5 py-2.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen((p) => !p)}
              className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-amber-50 transition-colors"
              style={{ border: `1px solid ${BORDER}` }}
              title="Chat history">
              <Menu className="h-4 w-4" style={{ color: MUTED }} />
            </button>
            <div className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0"
              style={{ background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)` }}>
              <Scale className="h-4 w-4 text-white" />
            </div>
            <span className="text-sm font-bold tracking-tight" style={{ color: INK, fontFamily: MNR }}>
              Kenyan Legal <span style={{ color: GOLD }}>RAG</span>
            </span>
            {conversationTitle && (
              <span className="text-xs truncate max-w-[200px]" style={{ color: MUTED }}>
                · {conversationTitle}
              </span>
            )}
            <Link href="/dashboard"
              className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold transition-colors hover:bg-amber-50"
              style={{ border: `1px solid ${BORDER}`, color: MUTED }}>
              <LayoutGrid className="h-3 w-3" />
              Dashboard
            </Link>
          </div>
          <UserMenu />
        </div>
      </header>

      {/* ── Body ──────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">

        {/* ── LEFT: chat column ─────────────────────────────────────────── */}
        <div className="flex flex-col overflow-hidden"
          style={{
            width: hasSources ? chatWidth : "100%",
            minWidth: hasSources ? 260 : undefined,
            flexShrink: 0,
            transition: hasSources ? "none" : "width 0.5s cubic-bezier(.4,0,.2,1)",
          }}>
          <div className="flex-1 overflow-y-auto">
            {/* Skeleton while an existing conversation loads from the API */}
            {isConversationLoading ? (
              <ChatPanelSkeleton />
            ) : !hasMessages ? (
              <div className="flex flex-col items-center justify-center min-h-full py-16 px-6">
                <div className="w-16 h-16 rounded-full mb-6 flex items-center justify-center"
                  style={{ background: "radial-gradient(circle at 35% 30%, #E8A84E, #A85A1A)", boxShadow: "0 6px 24px rgba(180,90,20,.35)" }}>
                  <Scale className="h-7 w-7 text-white opacity-90" />
                </div>
                <p className="text-[10px] font-semibold tracking-[.28em] uppercase mb-4" style={{ color: GOLD }}>
                  Small Claims Court · Research Assistant
                </p>
                <h1 className="text-4xl font-bold mb-4 text-center leading-tight" style={{ color: INK }}>
                  Ask the{" "}
                  <em style={{ fontFamily: NR, fontStyle: "italic" }}>law library</em>
                </h1>
                <p className="text-center max-w-sm text-sm leading-relaxed mb-10" style={{ color: MUTED }}>
                  Describe your client&apos;s facts. You get a FIRAC opinion grounded in Kenyan case law and statutes,
                  with every source open beside it.
                </p>
                <div className="w-full max-w-xl">
                  <InputArea inputRef={inputRef} value={input} onChange={setInput}
                    onKeyDown={handleKeyDown} onInput={autoResize} onSubmit={handleSubmit}
                    isLoading={isLoading} placeholder="Ask a legal question…" />
                  <div className="mt-8">
                    <p className="text-[10px] uppercase tracking-[.22em] mb-3 font-semibold" style={{ color: MUTED }}>Try a question</p>
                    <div className="grid grid-cols-2 gap-3">
                      {EXAMPLES.map((ex, i) => (
                        <button key={i} onClick={() => { setInput(ex.question); inputRef.current?.focus(); }}
                          className="text-left p-3 rounded-2xl transition-all hover:shadow-md"
                          style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
                          <p className="text-[9px] uppercase tracking-[.18em] mb-1.5 font-semibold" style={{ color: MUTED }}>{ex.category}</p>
                          <p className="text-[11px] leading-snug" style={{ color: INK }}>{ex.question}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-5 space-y-6 px-4 sk-fade-in">
                {/* "not saved" banner */}
                {notSavedNote && (
                  <div className="rounded-xl px-4 py-2.5 flex items-center gap-2 text-xs"
                    style={{ background: "#FFF8E6", border: `1px solid ${HIGHLIGHT}`, color: GOLD_DK }}>
                    <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                    <span>This answer was not saved to history due to a temporary error. It is still accurate.</span>
                    <button onClick={() => setNotSavedNote(false)} className="ml-auto shrink-0">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}

                {messages.map((msg, idx) => (
                  <Reveal key={msg.id} delay={Math.min(idx * 45, 360)}>
                    {msg.role === "user"
                      ? <UserBubble msg={msg} />
                      : <AssistantCard
                          msg={msg}
                          isActive={panelMessage?.id === msg.id}
                          onShowSources={() => { setPanelMessage(msg); }}
                          onCitationClick={(alias) => handleCitationClick(alias, msg)}
                          buildCitLabel={(alias) => buildCitLabel(alias, msg)}
                          activeAlias={activeChunkId
                            ? Object.entries(msg.aliasMap || {}).find(([, cid]) => cid === activeChunkId)?.[0] ?? null
                            : null}
                          onRetry={() => handleRetry(msg)}
                        />
                    }
                  </Reveal>
                ))}
                {isLoading && <LoadingCard />}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Composer */}
          {hasMessages && (
            <div className="shrink-0" style={{ borderTop: `1px solid ${BORDER}`, background: "white" }}>
              <div className="px-4 py-3">
                <InputArea inputRef={inputRef} value={input} onChange={setInput}
                  onKeyDown={handleKeyDown} onInput={autoResize} onSubmit={handleSubmit}
                  isLoading={isLoading} placeholder="Ask a follow-up question…" />
                {isLoading && (
                  <p className="text-[10px] mt-1.5 text-center" style={{ color: MUTED }}>
                    Send is disabled while an answer is generating.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── Resizer handle ──────────────────────────────────────────────── */}
        {hasSources && (
          <div onMouseDown={onResizerMouseDown}
            className="group shrink-0 relative flex items-center justify-center"
            style={{ width: 8, cursor: "col-resize", backgroundColor: "transparent", zIndex: 10 }}
            title="Drag to resize">
            <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-px transition-colors duration-150"
              style={{ background: BORDER }} />
            <div className="absolute inset-y-0 left-0 right-0 opacity-0 group-hover:opacity-100 transition-opacity duration-150 rounded"
              style={{ background: `${GOLD}22` }} />
            <div className="relative flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
              {[0,1,2].map((i) => (
                <div key={i} className="w-1 h-1 rounded-full" style={{ background: GOLD }} />
              ))}
            </div>
          </div>
        )}

        {/* ── RIGHT: workspace ──────────────────────────────────────────── */}
        <div className="flex-1 flex flex-col overflow-hidden"
          style={{
            backgroundColor: WS_BG,
            opacity: (hasSources || isConversationLoading) ? 1 : 0,
            pointerEvents: (hasSources || isConversationLoading) ? "auto" : "none",
            transition: "opacity 0.4s ease",
          }}>
          {/* Skeleton for the document viewer while history loads */}
          {isConversationLoading && !hasSources && (
            <DocumentViewerSkeleton />
          )}
          {hasSources && panelMessage && (
            <Reveal>
            <WorkspaceColumn
              message={panelMessage}
              activeDocKey={activeDocKey}
              onSelectDoc={setActiveDocKey}
              activeChunkId={activeChunkId}
              onChunkClick={setActiveChunkId}
            />
            </Reveal>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Document viewer skeleton (embedded in right panel, not full-page) ───────────
import { DocumentHeaderSkeleton } from "@/app/components/skeleton/DocumentHeaderSkeleton";
import { DocumentBodySkeleton }   from "@/app/components/skeleton/DocumentBodySkeleton";
import { SourcesCardSkeleton }    from "@/app/components/skeleton/SourcesCardSkeleton";
import { Skeleton as Sk }         from "@/app/components/loading/Skeleton";

function DocumentViewerSkeleton() {
  return (
    <div className="flex flex-col h-full" aria-hidden="true">
      {/* Workspace header */}
      <div className="shrink-0 flex items-start justify-between px-5 py-3"
        style={{ borderBottom: `1px solid ${BORDER}`, background: WS_BG }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
          <Sk variant="line" width="130px" height="0.6rem" />
          <Sk variant="line" width="200px" height="0.75rem" />
        </div>
        <SourcesCardSkeleton />
      </div>
      {/* Tab bar */}
      <div className="shrink-0 flex items-center gap-2 px-5 py-2"
        style={{ borderBottom: `1px solid ${BORDER}`, background: WS_BG }}>
        <div className="flex rounded-full overflow-hidden" style={{ border: `1px solid ${BORDER}` }}>
          <Sk variant="line" width="60px" height="1.75rem" radius="0px" />
          <Sk variant="line" width="42px" height="1.75rem" radius="0px" style={{ marginLeft: 1 }} />
        </div>
        <div style={{ marginLeft: "auto" }}>
          <Sk variant="pill" width="80px" />
        </div>
      </div>
      {/* Document card */}
      <div className="flex-1 overflow-y-auto">
        <div className="px-6 py-8 flex justify-center">
          <div className="w-full rounded-2xl bg-white"
            style={{ maxWidth: "42rem", border: `1px solid ${BORDER}`, boxShadow: "0 8px 32px rgba(60,45,10,0.08)" }}>
            <DocumentHeaderSkeleton />
            <DocumentBodySkeleton />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Input area ──────────────────────────────────────────────────────────────────
function InputArea({ inputRef, value, onChange, onKeyDown, onInput, onSubmit, isLoading, placeholder }: {
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  value: string; onChange: (v: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onInput: (e: React.FormEvent<HTMLTextAreaElement>) => void;
  onSubmit: () => void; isLoading: boolean; placeholder: string;
}) {
  return (
    <div className="flex items-end gap-3">
      <div className="flex-1 flex items-start rounded-2xl bg-white px-4 py-3"
        style={{ border: `1px solid ${BORDER}`, boxShadow: "0 1px 4px rgba(0,0,0,.05)" }}>
        <textarea ref={inputRef} value={value} onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown} onInput={onInput} placeholder={placeholder}
          rows={1} disabled={isLoading}
          className="flex-1 resize-none bg-transparent focus:outline-none text-sm"
          style={{ minHeight: "24px", maxHeight: "160px", color: INK, fontFamily: MNR }}
        />
        <div className="flex items-center gap-1 ml-3 text-[10px] font-bold tracking-[.18em] shrink-0 mt-0.5" style={{ color: "#C4B49A" }}>
          <span>KENYA</span><span className="opacity-40">·</span><span>SCC</span>
        </div>
      </div>
      <button onClick={onSubmit} disabled={!value.trim() || isLoading}
        className="w-10 h-10 rounded-full flex items-center justify-center text-white shrink-0 transition-all hover:brightness-110 hover:scale-105 disabled:opacity-30 disabled:cursor-not-allowed disabled:scale-100"
        style={{ background: isLoading ? "#C4B49A" : `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})` }}>
        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
        </svg>
      </button>
    </div>
  );
}

// ── User bubble ─────────────────────────────────────────────────────────────────
function UserBubble({ msg }: { msg: Message }) {
  const [expanded, setExpanded] = useState(false);
  const LIMIT = 320;
  const isLong = msg.content.length > LIMIT;
  const shown  = isLong && !expanded ? msg.content.slice(0, LIMIT) + "…" : msg.content;
  const caseNo = msg.content.match(/(?:SCC|SCCCOMM|ELC|HCCC|CIVIL|CRIMINAL)[\s.-]*(CASE\s*)?(?:NO\.?\s*)?([A-Z0-9\/]+(?:\/\d{4})?)/i)?.[0]?.toUpperCase();
  const ts = msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return (
    <div className="flex flex-col items-end gap-1">
      <div className="flex items-center gap-2 pr-1">
        {caseNo && <span className="text-[9px] font-bold tracking-[.15em] uppercase" style={{ color: GOLD }}>{caseNo}</span>}
        <span className="text-[10px]" style={{ color: MUTED }}>{ts}</span>
      </div>
      <div className="max-w-[88%] rounded-2xl px-4 py-3" style={{ background: "#F3E9CF", color: INK }}>
        <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">{shown}</p>
        {isLong && (
          <button className="mt-1 text-[11px] font-semibold" style={{ color: GOLD }}
            onClick={() => setExpanded((v) => !v)}>
            {expanded ? "Show less" : "Show full question"}
          </button>
        )}
      </div>
    </div>
  );
}

// ── Loading card ────────────────────────────────────────────────────────────────
function LoadingCard() {
  return (
    <div className="rounded-2xl p-4" style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
      <div className="flex items-center gap-3 mb-4">
        <span className="relative flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: GOLD }} />
          <span className="relative inline-flex h-3 w-3 rounded-full" style={{ background: GOLD }} />
        </span>
        <span className="text-sm font-medium" style={{ color: MUTED }}>Drafting grounds from authorities…</span>
      </div>
      {[80, 60, 72].map((w, i) => (
        <div key={i} className="h-3 rounded-full mb-2 animate-pulse" style={{ width: `${w}%`, background: BORDER }} />
      ))}
    </div>
  );
}

// ── Prospects strip ─────────────────────────────────────────────────────────────
function ProspectsStrip({ prospects }: { prospects: Prospect[] }) {
  const ratingStyle = (r: string) => {
    if (r === "Strong")   return { background: "#EDFBEE", color: "#2D7A3A", border: "1px solid #B5E0BA" };
    if (r === "Moderate") return { background: "#FFF8E6", color: "#8A6200", border: `1px solid ${HIGHLIGHT}` };
    return { background: "#FFF0F0", color: "#8A2020", border: "1px solid #F0C0C0" };
  };
  return (
    <div className="flex gap-2 flex-wrap mb-3">
      {prospects.map((p, i) => (
        <div key={i} className="rounded-xl px-3 py-2 text-xs font-medium flex flex-col gap-0.5 min-w-[90px]"
          style={{ background: CREAM_2, border: `1px solid ${BORDER}` }}>
          <span style={{ color: MUTED, fontSize: 9, letterSpacing: ".1em", textTransform: "uppercase" }}>{p.label}</span>
          <span className="font-bold rounded-full px-2 py-0.5 self-start" style={ratingStyle(p.rating)}>{p.rating}</span>
        </div>
      ))}
    </div>
  );
}

// ── Citation pill ───────────────────────────────────────────────────────────────
function CitPill({ alias, label, isActive, onClick }: { alias: string; label: string; isActive: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} title={alias}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium transition-all mx-0.5"
      style={isActive ? { background: INK, color: "white" } : { background: HIGHLIGHT, color: GOLD_DK, border: `1px solid #E2C673` }}>
      <FileText className="h-3 w-3 shrink-0" />
      {label}
    </button>
  );
}

// ── Inline markdown with citation pills ─────────────────────────────────────────
function renderInline(text: string, activeAlias: string | null, onCit: (a: string) => void, buildLabel: (a: string) => string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[CS]\d+(?:\.\d+)?\])/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={i} className="font-semibold" style={{ color: INK }}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`"))
      return <code key={i} className="px-1.5 py-0.5 rounded text-xs font-mono" style={{ background: CREAM_2, color: GOLD_DK }}>{part.slice(1, -1)}</code>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={i} style={{ color: MUTED }}>{part.slice(1, -1)}</em>;
    if (/^\[[CS]\d+/.test(part)) {
      const alias = part.slice(1, -1);
      return <CitPill key={i} alias={alias} label={buildLabel(alias)} isActive={activeAlias === alias} onClick={() => onCit(alias)} />;
    }
    return part;
  });
}

function parseMdFirac(raw: string, activeAlias: string | null, onCit: (a: string) => void, buildLabel: (a: string) => string): ReactNode[] {
  const lines = raw.split("\n");
  const out: ReactNode[] = [];
  let ulItems: string[] = [], olItems: string[] = [];
  const ri = (t: string) => renderInline(t, activeAlias, onCit, buildLabel);
  const flushUl = () => {
    if (!ulItems.length) return;
    out.push(<ul key={out.length} className="list-disc list-outside pl-5 mb-3 space-y-1.5">{ulItems.map((t, i) => <li key={i} className="text-sm leading-relaxed" style={{ color: INK }}>{ri(t)}</li>)}</ul>);
    ulItems = [];
  };
  const flushOl = () => {
    if (!olItems.length) return;
    out.push(<ol key={out.length} className="list-decimal list-outside pl-5 mb-3 space-y-1.5">{olItems.map((t, i) => <li key={i} className="text-sm leading-relaxed" style={{ color: INK }}>{ri(t)}</li>)}</ol>);
    olItems = [];
  };
  for (const line of lines) {
    const flush = () => { flushUl(); flushOl(); };
    if (/^#{5}\s/.test(line)) { flush(); out.push(<p key={out.length} className="text-[10px] font-bold tracking-[.2em] uppercase mt-4 mb-1" style={{ color: MUTED }}>{ri(line.slice(6))}</p>); continue; }
    if (/^#{4}\s/.test(line)) { flush(); out.push(<div key={out.length} className="flex items-center gap-2 mt-5 mb-2"><p className="text-[10px] font-bold tracking-[.2em] uppercase shrink-0" style={{ color: GOLD }}>{ri(line.slice(5))}</p><div className="h-px flex-1" style={{ background: BORDER }} /></div>); continue; }
    if (/^#{3}\s/.test(line)) { flush(); out.push(<h3 key={out.length} className="text-sm font-bold mt-4 mb-1.5 border-b pb-1" style={{ color: INK, borderColor: BORDER }}>{ri(line.slice(4))}</h3>); continue; }
    if (/^#{2}\s/.test(line)) { flush(); out.push(<h2 key={out.length} className="text-base font-bold mt-5 mb-2" style={{ color: INK }}>{ri(line.slice(3))}</h2>); continue; }
    if (/^#\s/.test(line)) { flush(); out.push(<h1 key={out.length} className="text-lg font-bold mt-6 mb-2" style={{ color: INK }}>{ri(line.slice(2))}</h1>); continue; }
    if (/^-{3,}$/.test(line.trim())) { flush(); out.push(<hr key={out.length} className="my-4" style={{ borderColor: BORDER }} />); continue; }
    if (/^>\s/.test(line)) { flush(); out.push(<blockquote key={out.length} className="border-l-4 pl-4 my-3 py-2 pr-3 rounded-r-xl" style={{ borderColor: GOLD, background: HIGHLIGHT }}><p className="text-sm italic leading-relaxed" style={{ color: GOLD_DK }}>{ri(line.slice(2))}</p></blockquote>); continue; }
    const ulM = line.match(/^\s*[*\-]\s{1,3}(.+)$/);
    if (ulM) { flushOl(); ulItems.push(ulM[1]); continue; }
    const olM = line.match(/^\s*\d+\.\s{1,3}(.+)$/);
    if (olM) { flushUl(); olItems.push(olM[1]); continue; }
    if (line.trim() === "") { flush(); continue; }
    flush();
    out.push(<p key={out.length} className="text-sm leading-relaxed mb-2" style={{ color: INK }}>{ri(line)}</p>);
  }
  flushUl(); flushOl();
  return out;
}

// ── FIRAC tile ──────────────────────────────────────────────────────────────────
function FiracTile({ letter }: { letter: string }) {
  return (
    <div className="w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-bold shrink-0"
      style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}>
      {letter}
    </div>
  );
}

// ── Assistant card ──────────────────────────────────────────────────────────────
function AssistantCard({ msg, isActive, onShowSources, onCitationClick, buildCitLabel, activeAlias, onRetry }: {
  msg: Message; isActive: boolean; onShowSources: () => void;
  onCitationClick: (alias: string) => void;
  buildCitLabel: (alias: string) => string;
  activeAlias: string | null;
  onRetry?: () => void;
}) {
  const isFailed   = msg.historyStatus === "failed";
  const isHistory  = msg.fromHistory === true;
  const firac      = isFailed ? null : parseFirac(msg.aiResponse || "");
  const numSrcs    = msg.sources?.filter((s) => s.used).length ?? 0;
  const usedSrcs   = msg.sources?.filter((s) => s.used) ?? [];

  if (isFailed) {
    return (
      <div className="rounded-2xl p-4 space-y-3" style={{ background: SURFACE, border: `1.5px solid #F0C0C0` }}>
        <div className="flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" style={{ color: "#B91C1C" }} />
          <span className="text-sm font-semibold" style={{ color: "#B91C1C" }}>Request failed</span>
          {msg.errorCode && (
            <span className="text-[10px] px-2 py-0.5 rounded-full font-mono"
              style={{ background: "#FEE2E2", color: "#B91C1C" }}>
              {msg.errorCode}
            </span>
          )}
        </div>
        <p className="text-sm leading-relaxed" style={{ color: MUTED }}>
          {msg.errorMessage || msg.aiResponse || "An error occurred."}
        </p>
        {onRetry && (
          <button onClick={onRetry}
            className="flex items-center gap-1.5 text-xs font-semibold rounded-full px-3 py-1.5 transition-all hover:shadow-sm"
            style={{ background: GOLD, color: "white" }}>
            <RotateCcw className="h-3 w-3" />
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <CoverageCard coverage={msg.coverage} removed={msg.unverifiedClaims} />
      <div className="rounded-2xl overflow-hidden" style={{ background: SURFACE, border: `1px solid ${BORDER}`, boxShadow: "0 8px 30px rgba(60,45,10,0.08)" }}>
        <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: `1px solid ${BORDER}` }}>
          <div className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})` }}>
            <Scale className="h-3.5 w-3.5 text-white" />
          </div>
          <span className="text-sm font-semibold" style={{ color: INK }}>Legal Analysis</span>
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full ml-1" style={{ background: HIGHLIGHT, color: GOLD_DK }}>
            AI-generated from retrieved sources
          </span>
          {isHistory && (
            <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full ml-auto"
              style={{ background: CREAM_2, color: MUTED }}>
              from history · text excerpts only
            </span>
          )}
        </div>
        <div className="px-4 py-4">
          {msg.prospects && msg.prospects.length > 0 && <ProspectsStrip prospects={msg.prospects} />}
          {firac ? (
            <div className="space-y-4">
              {firac.map((sec) => (
                <div key={sec.letter}>
                  <div className="flex items-center gap-2 mb-2">
                    <FiracTile letter={sec.letter} />
                    <span className="text-[10px] font-bold tracking-[.18em] uppercase" style={{ color: GOLD }}>{sec.label}</span>
                  </div>
                  <div className="pl-8">
                    {parseMdFirac(sec.content, activeAlias, onCitationClick, buildCitLabel)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            parseMdFirac(msg.aiResponse || "", activeAlias, onCitationClick, buildCitLabel)
          )}
          {usedSrcs.length > 0 && (
            <div className="mt-4 pt-3" style={{ borderTop: `1px solid ${BORDER}` }}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[9px] font-bold tracking-[.18em] uppercase shrink-0" style={{ color: MUTED }}>
                  SOURCES · {numSrcs}
                </span>
                {usedSrcs.map((src) => (
                  <CitPill key={src.alias} alias={src.alias} label={buildCitLabel(src.alias)}
                    isActive={activeAlias === src.alias} onClick={() => onCitationClick(src.alias)} />
                ))}
              </div>
            </div>
          )}
        </div>
        {msg.sources && msg.sources.length > 0 && (
          <button onClick={onShowSources}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-all text-left"
            style={{ borderTop: `1px solid ${BORDER}`, background: isActive ? HIGHLIGHT : "transparent", color: isActive ? GOLD_DK : MUTED }}>
            <BookOpen className="h-4 w-4 shrink-0" />
            {isActive ? "Viewing sources →" : "Open sources →"}
            <span className="ml-auto text-xs opacity-60">{numSrcs} cited · {msg.sources.length} total</span>
          </button>
        )}
      </div>
    </div>
  );
}

// ── Coverage card ───────────────────────────────────────────────────────────────
function CoverageCard({ coverage, removed }: { coverage?: Coverage; removed?: string[] }) {
  const missing = coverage?.not_in_corpus ?? [];
  if (!missing.length && !removed?.length) return null;
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: SURFACE, border: `1px solid ${BORDER}`, borderLeft: "3px solid #C9971A" }}>
      <div className="flex items-center gap-2 px-4 py-2.5">
        <AlertCircle className="h-4 w-4 shrink-0" style={{ color: GOLD }} />
        <span className="text-xs font-semibold" style={{ color: INK }}>Knowledge Base Coverage</span>
      </div>
      <div className="px-4 pb-3 space-y-1 text-xs" style={{ color: MUTED }}>
        {missing.length > 0 && <p><span className="font-semibold" style={{ color: INK }}>Not in knowledge base: </span>{missing.join("; ")}.</p>}
        {removed && removed.length > 0 && <p>{removed.length} unverified citation{removed.length > 1 ? "s" : ""} removed.</p>}
      </div>
    </div>
  );
}

// ── Workspace column ────────────────────────────────────────────────────────────
function WorkspaceColumn({ message, activeDocKey, onSelectDoc, activeChunkId, onChunkClick }: {
  message: Message; activeDocKey: string | null;
  onSelectDoc: (k: string) => void;
  activeChunkId: string | null;
  onChunkClick: (id: string) => void;
}) {
  const [readerTab, setReaderTab] = useState<"reader" | "pdf">("reader");
  const evidence = message.evidence || [];
  const rules    = message.rules    || [];
  const sources  = message.sources  || [];
  const aliasMap = message.aliasMap || {};
  const chunkToAlias = Object.fromEntries(Object.entries(aliasMap).map(([a, c]) => [c, a]));
  const usedChunkIds = new Set(sources.filter((s) => s.used).map((s) => s.chunk_id));

  interface ClusterDoc {
    key: string; title: string; subtitle: string;
    kind: "JUDGMENT" | "STATUTE" | "CONSTITUTION" | "SUMMARY";
    court: string; year: string; relevance: number; pdfUrl?: string;
    chunks: any[]; dbKey: string;
  }

  const clusterDocs: ClusterDoc[] = [
    ...evidence.map((ev, i): ClusterDoc => ({
      key: ev.case_name || ev.case_identifier || `case-${i}`,
      title: ev.case_name || `Case ${i + 1}`,
      subtitle: ev.case_identifier || "",
      kind: "JUDGMENT",
      court: ev.metadata?.court_level || "",
      year: ev.metadata?.year || ev.year || "",
      relevance: ev.score ?? 0,
      pdfUrl: ev.pdf_url ? `${API_BASE}${ev.pdf_url}` : undefined,
      chunks: ev.chunks || [],
      dbKey: ev.case_identifier || ev.case_name || `case-${i}`,
    })),
    ...rules.map((r, i): ClusterDoc => ({
      key: `statute-S${i + 1}`,
      title: r.document_label || r.act_title || r.source_document || "Statute",
      subtitle: `s.${r.article_number || r.section_num}${r.article_header ? ` — ${r.article_header}` : ""}`,
      kind: "STATUTE",
      court: "", year: r.act_version || "",
      relevance: r.score ?? 0,
      pdfUrl: r.pdf_url ? `${API_BASE}${r.pdf_url}` : undefined,
      chunks: r.chunks || [],
      dbKey: r.source_document || r.act_title || r.document_label || `statute-${i}`,
    })),
  ];

  const activeDoc  = clusterDocs.find((d) => d.key === activeDocKey) ?? clusterDocs[0] ?? null;
  const matterTitle = evidence[0]?.case_name || rules[0]?.act_title || "Current query";
  const caseList = evidence.map((ev, i) => ({
    caseId: ev.case_name || ev.case_identifier || `Case ${i + 1}`,
    label:  ev.case_name || `Case ${i + 1}`,
    pdfUrl: ev.pdf_url || null,
  }));
  const [selectedPdfCase, setSelectedPdfCase] = useState<string | null>(null);

  return (
    <div className="flex flex-col h-full">
      <div className="shrink-0 flex items-start justify-between px-5 py-3"
        style={{ borderBottom: `1px solid ${BORDER}`, background: WS_BG }}>
        <div>
          <p className="text-[9px] font-bold tracking-[.2em] uppercase mb-0.5" style={{ color: GOLD }}>RETRIEVED AUTHORITIES</p>
          <p className="text-xs font-medium" style={{ color: MUTED }}>{matterTitle}</p>
        </div>
        <ClusterWidget docs={clusterDocs} activeDocKey={activeDoc?.key ?? null} onSelectDoc={(key) => { onSelectDoc(key); }} />
      </div>
      {activeDoc && (
        <div className="shrink-0 flex items-center gap-2 px-5 py-2" style={{ borderBottom: `1px solid ${BORDER}`, background: WS_BG }}>
          <div className="flex rounded-full overflow-hidden" style={{ border: `1px solid ${BORDER}` }}>
            {(["reader", "pdf"] as const).map((t) => (
              <button key={t} onClick={() => setReaderTab(t)}
                className="px-4 py-1.5 text-xs font-semibold transition-colors capitalize"
                style={{ background: readerTab === t ? GOLD : "transparent", color: readerTab === t ? "white" : MUTED }}>
                {t === "reader" ? "Reader" : "PDF"}
              </button>
            ))}
          </div>
          {activeDoc.pdfUrl && readerTab === "reader" && (
            <a href={activeDoc.pdfUrl} target="_blank" rel="noopener noreferrer"
              className="ml-auto flex items-center gap-1 text-xs rounded-full px-3 py-1 transition-colors hover:bg-amber-50"
              style={{ color: MUTED, border: `1px solid ${BORDER}` }}>
              <ExternalLink className="h-3 w-3" /> Open original
            </a>
          )}
        </div>
      )}
      <div className="flex-1 overflow-y-auto">
        {!activeDoc ? (
          <EmptyReader />
        ) : readerTab === "reader" ? (
          <DocumentReader doc={activeDoc} usedChunkIds={usedChunkIds} activeChunkId={activeChunkId}
            chunkToAlias={chunkToAlias} sources={sources} />
        ) : (
          <OriginalPane cases={caseList} selectedCaseId={selectedPdfCase}
            onSelectCase={setSelectedPdfCase} sources={sources} scrollToChunkId={activeChunkId} />
        )}
      </div>
    </div>
  );
}

// ── Cluster widget ──────────────────────────────────────────────────────────────
function ClusterWidget({ docs, activeDocKey, onSelectDoc }: {
  docs: Array<{ key: string; title: string; subtitle: string; kind: string; court: string; year: string; relevance: number }>;
  activeDocKey: string | null; onSelectDoc: (k: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [open]);
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, []);
  if (!docs.length) return null;
  const reading = docs.find((d) => d.key === activeDocKey) ?? docs[0];
  const kindBg = (_k: string) => ({ background: "#EEEBE3", color: MUTED });
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((p) => !p)} onMouseEnter={() => setOpen(true)}
        className="flex items-center gap-2 rounded-full px-3 py-1.5 transition-all hover:shadow-sm"
        style={{ background: "white", border: `1px solid ${BORDER}`, boxShadow: "0 1px 4px rgba(0,0,0,.05)" }}>
        <div className="relative flex" style={{ width: 16 + Math.min(docs.length - 1, 3) * 6, height: 20 }}>
          {docs.slice(0, 4).map((d, i) => (
            <div key={i} className="absolute w-5 h-5 rounded flex items-center justify-center shadow-sm"
              style={{ left: i * 6, background: d.kind === "JUDGMENT" ? HIGHLIGHT : CREAM_2, border: `1px solid ${BORDER}`, zIndex: i }}>
              {d.kind === "JUDGMENT" ? <FileText className="h-2.5 w-2.5" style={{ color: GOLD_DK }} /> : <Scale className="h-2.5 w-2.5" style={{ color: MUTED }} />}
            </div>
          ))}
        </div>
        <div className="flex flex-col items-start ml-1">
          <span className="text-[11px] font-bold leading-none" style={{ color: INK }}>{docs.length} sources</span>
          <span className="text-[9px] leading-none mt-0.5" style={{ color: MUTED }}>
            Reading: {reading?.title.split(" ").slice(0, 3).join(" ")}{reading?.title.split(" ").length > 3 ? "…" : ""}
          </span>
        </div>
        <span className="ml-1 rounded-full text-[9px] font-bold w-4 h-4 flex items-center justify-center"
          style={{ background: GOLD, color: "white" }}>{docs.length}</span>
      </button>
      {open && (
        <div className="absolute right-0 top-12 z-50 rounded-2xl shadow-2xl"
          style={{ width: 520, background: "white", border: `1px solid ${BORDER}`, boxShadow: "0 16px 48px rgba(60,45,10,0.15)" }}
          onMouseLeave={() => setOpen(false)}>
          <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: `1px solid ${BORDER}` }}>
            <span className="text-[9px] font-bold tracking-[.2em] uppercase" style={{ color: GOLD }}>RANKED BY RELEVANCE</span>
            <span className="text-[10px]" style={{ color: MUTED }}>Click a source to read it</span>
          </div>
          <div className="p-3 grid grid-cols-3 gap-2 max-h-96 overflow-y-auto">
            {docs.map((doc, i) => {
              const isActive = doc.key === activeDocKey;
              return (
                <button key={i} onClick={() => { onSelectDoc(doc.key); setOpen(false); }}
                  className="text-left rounded-xl p-3 transition-all hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
                  style={{ background: isActive ? HIGHLIGHT : SURFACE, border: `1.5px solid ${isActive ? GOLD : BORDER}`, boxShadow: isActive ? `0 0 0 2px ${GOLD}33` : undefined }}>
                  <div className="flex items-start justify-between mb-1.5 gap-1">
                    <span className="text-[9px] font-bold tracking-[.12em] uppercase px-1.5 py-0.5 rounded-md" style={kindBg(doc.kind)}>{doc.kind}</span>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: isActive ? GOLD_DK : MUTED }}>{Math.round(doc.relevance * 100)}</span>
                  </div>
                  <p className="text-xs font-semibold leading-snug line-clamp-2 mb-1" style={{ fontFamily: NR, color: INK }}>{doc.title}</p>
                  {doc.subtitle && <p className="text-[10px] line-clamp-1 mb-2" style={{ color: MUTED }}>{doc.subtitle}</p>}
                  {(doc.court || doc.year) && <p className="text-[9px] line-clamp-1 mb-2" style={{ color: MUTED }}>{[doc.court, doc.year].filter(Boolean).join(" · ")}</p>}
                  <div className="h-0.5 rounded-full overflow-hidden" style={{ background: BORDER }}>
                    <div className="h-full rounded-full" style={{ width: `${Math.round(doc.relevance * 100)}%`, background: GOLD }} />
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Document reader ─────────────────────────────────────────────────────────────
function DocumentReader({ doc, usedChunkIds, activeChunkId, chunkToAlias, sources }: {
  doc: { key: string; title: string; subtitle: string; kind: string; court: string; year: string; relevance: number; pdfUrl?: string; chunks: any[]; dbKey: string };
  usedChunkIds: Set<string>; activeChunkId: string | null;
  chunkToAlias: Record<string, string>; sources: ChunkSource[];
}) {
  const chunkRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [allChunks, setAllChunks] = useState<any[] | null>(null);
  const [loadingFull, setLoadingFull] = useState(false);
  const scrolledToFirst = useRef(false);
  const isStatute = doc.kind === "STATUTE";

  useEffect(() => {
    setAllChunks(null);
    scrolledToFirst.current = false;
    if (!doc.dbKey) return;
    setLoadingFull(true);
    const path = isStatute
      ? `/api/statutes/${encodeURIComponent(doc.dbKey)}/all-chunks`
      : `/api/cases/${encodeURIComponent(doc.dbKey)}/all-chunks`;
    apiFetch(path)
      .then((r) => r.json())
      .then((d) => { if (d.chunks && d.chunks.length > 0) setAllChunks(d.chunks); else setAllChunks(null); })
      .catch(() => setAllChunks(null))
      .finally(() => setLoadingFull(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc.key]);

  const scrollToChunk = useCallback((chunkId: string) => {
    const attempt = () => { const el = chunkRefs.current[chunkId]; if (el) el.scrollIntoView({ behavior: "smooth", block: "center" }); };
    attempt();
    const t = setTimeout(attempt, 120);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => { if (!activeChunkId) return; return scrollToChunk(activeChunkId); }, [activeChunkId, allChunks, scrollToChunk]);

  useEffect(() => {
    if (!allChunks || scrolledToFirst.current || activeChunkId) return;
    const firstCited = allChunks.find((ch) => usedChunkIds.has(ch.chunk_id));
    if (!firstCited) return;
    scrolledToFirst.current = true;
    scrollToChunk(firstCited.chunk_id);
  }, [allChunks, usedChunkIds, activeChunkId, scrollToChunk]);

  const retrievedChunks = [...(doc.chunks || [])].sort((a, b) => (a.chunk_index ?? 0) - (b.chunk_index ?? 0));
  const chunks = allChunks ?? retrievedChunks;
  const citedChunks = chunks.filter((ch) => usedChunkIds.has(ch.chunk_id || ch.id));
  const jumpLabels: Array<{ chunkId: string; label: string }> = citedChunks.map((ch) => {
    const alias = chunkToAlias[ch.chunk_id || ch.id] || "";
    const src   = sources.find((s) => s.alias === alias);
    return { chunkId: ch.chunk_id || ch.id, label: src?.short_label || alias || "para" };
  }).filter((j) => j.chunkId);

  if (!chunks.length && !loadingFull) {
    const fallback = (doc as any).best_chunk || (doc as any).text_preview || (doc as any).full_text;
    if (fallback) {
      return (
        <div className="px-6 py-8 flex justify-center">
          <div className="w-full max-w-2xl rounded-2xl bg-white shadow-sm p-8" style={{ border: `1px solid ${BORDER}` }}>
            <SheetHeader doc={doc} jumpLabels={[]} onJump={() => {}} />
            <p className="text-sm leading-loose" style={{ fontFamily: NR, color: INK }}>{fallback}</p>
          </div>
        </div>
      );
    }
    return <EmptyReader />;
  }

  return (
    <div className="px-6 py-8 flex justify-center">
      <div className="w-full max-w-2xl rounded-2xl bg-white shadow-sm" style={{ border: `1px solid ${BORDER}`, boxShadow: "0 8px 32px rgba(60,45,10,0.08)" }}>
        <SheetHeader doc={doc} jumpLabels={jumpLabels} onJump={(cid) => { const el = chunkRefs.current[cid]; if (el) el.scrollIntoView({ behavior: "smooth", block: "center" }); }} />
        {loadingFull && (
          <div className="px-8 py-4 flex items-center gap-2 text-xs" style={{ color: MUTED, borderBottom: `1px solid ${BORDER}` }}>
            <span className="relative flex h-2 w-2"><span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: GOLD }} /><span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: GOLD }} /></span>
            Loading full document…
          </div>
        )}
        {!loadingFull && (
          <div className="px-8 py-2 flex items-center gap-2 text-[10px]" style={{ borderBottom: `1px solid ${BORDER}`, background: CREAM_2 }}>
            {allChunks ? (
              <><span className="font-bold tracking-[.12em] uppercase" style={{ color: GOLD }}>Full document</span><span style={{ color: MUTED }}>— cited passages highlighted · {usedChunkIds.size} of {allChunks.length} sections cited</span></>
            ) : (
              <><span className="font-bold tracking-[.12em] uppercase" style={{ color: MUTED }}>Retrieved excerpts only</span><span style={{ color: MUTED }}>— full document unavailable</span></>
            )}
          </div>
        )}
        <div className="px-8 py-6 space-y-0">
          {chunks.map((ch, idx) => {
            const chId    = ch.chunk_id || ch.id || `chunk-${idx}`;
            const isCited = usedChunkIds.has(chId);
            const isActive = chId === activeChunkId;
            const text    = ch.text || ch.content || "";
            const paras   = text.split(/\n\n+/).filter(Boolean);
            const alias   = chunkToAlias[chId];
            return (
              <div key={chId} ref={(el) => { chunkRefs.current[chId] = el; }}
                className="relative transition-all duration-200"
                style={isCited ? { background: isActive ? "#F0DC9C" : HIGHLIGHT, borderLeft: `3px solid ${GOLD}`, borderRadius: 10, padding: "12px 16px", marginBottom: 4 } : { borderLeft: "2px dashed transparent", padding: "4px 16px" }}>
                {isCited && alias && (
                  <span className="absolute top-2 right-3 text-[9px] font-bold px-1.5 py-0.5 rounded"
                    style={{ background: GOLD, color: "white" }}>{alias}</span>
                )}
                {(paras as string[]).map((para: string, pi: number) => {
                  const paraNumMatch = para.match(/^(\d+)\.\s+/);
                  const isHeading    = /^[A-Z][A-Z\s]{4,}$/.test(para.trim().slice(0, 50)) && para.trim().length < 80;
                  return (
                    <div key={pi} className="flex gap-4">
                      <span className="shrink-0 w-6 text-right text-[10px] mt-0.5 select-none" style={{ color: BORDER }}>
                        {paraNumMatch ? paraNumMatch[1] : ""}
                      </span>
                      {isHeading ? (
                        <p className="flex-1 text-sm font-bold mt-4 mb-2 tracking-wide" style={{ color: INK, fontFamily: NR }}>{para.trim()}</p>
                      ) : (
                        <p className="flex-1 text-[15px] leading-[1.75] mb-3" style={{ fontFamily: NR, color: INK }}>{para.replace(/^\d+\.\s+/, "")}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Sheet header ────────────────────────────────────────────────────────────────
function SheetHeader({ doc, jumpLabels, onJump }: {
  doc: { title: string; subtitle: string; kind: string; court: string; year: string; relevance: number; pdfUrl?: string };
  jumpLabels: Array<{ chunkId: string; label: string }>;
  onJump: (chunkId: string) => void;
}) {
  return (
    <div className="px-8 pt-7 pb-4" style={{ borderBottom: `1px solid ${BORDER}` }}>
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-bold tracking-[.15em] uppercase px-2 py-0.5 rounded-md" style={{ background: CREAM_2, color: MUTED }}>{doc.kind}</span>
          <span className="text-[11px] font-semibold tabular-nums" style={{ color: MUTED }}>Relevance {Math.round(doc.relevance * 100) / 100}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-amber-50 transition-colors"
            style={{ border: `1px solid ${BORDER}` }}
            onClick={() => navigator.clipboard?.writeText(doc.title)}>
            <Copy className="h-3.5 w-3.5" style={{ color: MUTED }} />
          </button>
          {doc.pdfUrl && (
            <a href={doc.pdfUrl} target="_blank" rel="noopener noreferrer"
              className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-amber-50 transition-colors"
              style={{ border: `1px solid ${BORDER}` }}>
              <ExternalLink className="h-3.5 w-3.5" style={{ color: MUTED }} />
            </a>
          )}
        </div>
      </div>
      <h2 className="text-2xl font-bold leading-tight mb-1" style={{ fontFamily: NR, color: INK }}>{doc.title}</h2>
      {doc.subtitle && <p className="text-sm italic mb-1" style={{ fontFamily: NR, color: MUTED }}>{doc.subtitle}</p>}
      {(doc.court || doc.year) && <p className="text-xs" style={{ color: MUTED }}>{[doc.court, doc.year].filter(Boolean).join(" · ")}</p>}
      {jumpLabels.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-3">
          <span className="text-[10px] font-semibold self-center" style={{ color: MUTED }}>Jump to cited passage</span>
          {jumpLabels.map(({ chunkId, label }) => (
            <button key={chunkId} onClick={() => onJump(chunkId)}
              className="flex items-center gap-1 text-[11px] font-medium rounded-full px-2.5 py-1 transition-all hover:shadow-sm"
              style={{ background: HIGHLIGHT, color: GOLD_DK, border: `1px solid #E2C673` }}>
              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: GOLD }} />
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Empty reader ────────────────────────────────────────────────────────────────
function EmptyReader() {
  return (
    <div className="flex flex-col items-center justify-center h-full py-20 px-8 text-center">
      <BookOpen className="h-10 w-10 mb-4" style={{ color: BORDER }} />
      <p className="text-sm font-medium" style={{ color: MUTED }}>Select a source to read it here.</p>
    </div>
  );
}

// ── Original pane (PDF viewer) ──────────────────────────────────────────────────
function OriginalPane({ cases, selectedCaseId, onSelectCase, sources, scrollToChunkId }: {
  cases: Array<{ caseId: string; label: string; pdfUrl: string | null }>;
  selectedCaseId: string | null; onSelectCase: (id: string) => void;
  sources: ChunkSource[]; scrollToChunkId?: string | null;
}) {
  const [ocrInfo, setOcrInfo] = useState<Record<string, boolean>>({});
  const [locations, setLocations] = useState<ChunkLocation[]>([]);
  const active = cases.find((c) => c.caseId === selectedCaseId) ?? cases[0] ?? null;

  useEffect(() => {
    const caseChunks = sources.filter((s) => s.type === "case" && s.chunk_id);
    if (!caseChunks.length) return;
    Promise.all(caseChunks.map((s) =>
      apiFetch(`/api/chunks/${encodeURIComponent(s.chunk_id)}/location`)
        .then((r) => r.json())
        .then((d): ChunkLocation | null => {
          if (d.error) return null;
          if (d.ocr && active) setOcrInfo((prev) => ({ ...prev, [active.caseId]: true }));
          return { chunk_id: s.chunk_id, page_start: d.page_start ?? 1, page_end: d.page_end ?? 1, boxes: (d.boxes ?? []).map((b: any) => ({ page: b.page ?? 0, bbox: b.bbox })), ocr: !!d.ocr };
        }).catch(() => null)
    )).then((r) => setLocations(r.filter(Boolean) as ChunkLocation[]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sources.map((s) => s.chunk_id).join(",")]);

  if (!cases.length) return <div className="flex items-center justify-center h-64 text-sm" style={{ color: MUTED }}>No PDF available.</div>;
  const pdfSrc = active?.pdfUrl ? `${API_BASE}${active.pdfUrl}` : active ? `${API_BASE}/api/case-pdf?case_id=${encodeURIComponent(active.caseId)}` : null;
  const isOcr = active ? ocrInfo[active.caseId] : false;
  const citedChunkIds  = new Set(sources.filter((s) => s.used && s.type === "case").map((s) => s.chunk_id));
  const unusedChunkIds = new Set(sources.filter((s) => !s.used && s.type === "case").map((s) => s.chunk_id));

  return (
    <div className="flex flex-col h-full">
      {cases.length > 1 && (
        <div className="flex gap-2 px-4 py-2 flex-wrap shrink-0" style={{ borderBottom: `1px solid ${BORDER}` }}>
          {cases.map((c) => (
            <button key={c.caseId} onClick={() => onSelectCase(c.caseId)}
              className="text-xs px-3 py-1.5 rounded-full font-semibold transition-colors"
              style={{ background: (selectedCaseId ?? cases[0]?.caseId) === c.caseId ? GOLD : CREAM_2, color: (selectedCaseId ?? cases[0]?.caseId) === c.caseId ? "white" : MUTED }}>
              {c.label}
            </button>
          ))}
        </div>
      )}
      {isOcr && (
        <div className="shrink-0 px-4 py-2 flex items-center gap-2" style={{ background: HIGHLIGHT, borderBottom: `1px solid #E2C673` }}>
          <span className="text-xs font-bold" style={{ color: GOLD_DK }}>OCR</span>
          <span className="text-xs" style={{ color: GOLD_DK }}>Scanned document — text extracted via OCR.</span>
        </div>
      )}
      {pdfSrc ? (
        <div className="flex-1 overflow-y-auto">
          <PdfViewer pdfUrl={pdfSrc} locations={locations} citedChunkIds={citedChunkIds}
            unusedChunkIds={unusedChunkIds} scrollToChunkId={scrollToChunkId} />
        </div>
      ) : (
        <div className="flex items-center justify-center h-64 text-sm" style={{ color: MUTED }}>No PDF for this case.</div>
      )}
    </div>
  );
}
