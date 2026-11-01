import { apiFetch } from "@/app/lib/api";

// ── Shared interfaces ──────────────────────────────────────────────────────────

export interface ConvSummary {
  id: string;
  title: string;
  title_source: "fallback" | "generated" | "user";
  pinned: boolean;
  turn_count: number;
  updated_at: string | null;
}

export interface HistorySource {
  alias: string;
  chunk_id: string;
  type: "case" | "statute" | "constitution";
  document_type?: string;
  label: string;
  used: boolean;
  precision?: string;
  case_identifier?: string | null;
  parties?: string | null;
  court_level?: string | null;
  year?: string | null;
  document_label?: string | null;
  article_number?: string | null;
  article_header?: string | null;
  pdf_url?: string | null;
  snippet?: string | null;
}

export interface HistoryTurn {
  id: string;
  turn_index: number;
  status: "complete" | "failed";
  query: string;
  answer_markdown?: string | null;
  answer_structured?: Record<string, any> | null;
  sources: HistorySource[];
  alias_map: Record<string, string>;
  coverage?: any;
  error?: string | null;
  error_code?: string | null;
  created_at: string | null;
}

export interface ConvDetail {
  conversation: ConvSummary & { last_query_preview?: string };
  turns: HistoryTurn[];
  has_more_turns: boolean;
}

// ── API functions ──────────────────────────────────────────────────────────────

export async function listConversations(cursor?: string | null): Promise<{ conversations: ConvSummary[]; next_cursor?: string | null }> {
  const params = new URLSearchParams({ limit: "20" });
  if (cursor) params.set("cursor", cursor);
  const res = await apiFetch(`/api/conversations?${params}`);
  if (!res.ok) throw new Error(`listConversations: ${res.status}`);
  return res.json();
}

export async function getConversation(id: string, turnLimit = 50, beforeIndex?: number): Promise<ConvDetail> {
  const params = new URLSearchParams({ turn_limit: String(turnLimit) });
  if (beforeIndex !== undefined) params.set("before_index", String(beforeIndex));
  const res = await apiFetch(`/api/conversations/${id}?${params}`);
  if (!res.ok) {
    if (res.status === 404) throw Object.assign(new Error("Not found"), { status: 404 });
    throw new Error(`getConversation: ${res.status}`);
  }
  return res.json();
}

export async function patchConversation(id: string, patch: { title?: string; pinned?: boolean }): Promise<void> {
  const res = await apiFetch(`/api/conversations/${id}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`patchConversation: ${res.status}`);
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await apiFetch(`/api/conversations/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`deleteConversation: ${res.status}`);
}

export async function regenerateTitle(id: string, force = false): Promise<void> {
  const res = await apiFetch(`/api/conversations/${id}/regenerate-title`, {
    method: "POST",
    body: JSON.stringify({ force }),
  });
  if (!res.ok) throw new Error(`regenerateTitle: ${res.status}`);
}
