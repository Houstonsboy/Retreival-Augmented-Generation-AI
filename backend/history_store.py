"""
Chat history persistence for the Kenyan Legal RAG chatbot.

Firestore schema:
  users/{uid}/conversations/{cid}       — conversation metadata
  users/{uid}/conversations/{cid}/turns/{tid} — individual Q/A turns

All public functions accept uid from g.uid (never from the request body).
The module-level ThreadPoolExecutor runs title generation without blocking
HTTP responses; Flask's request context is NOT available in those threads.
"""

import atexit
import json
import logging
import re
import uuid as _uuid_mod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from firebase_admin import firestore as fb_firestore
from google.cloud.firestore import transactional as _transactional

from groq_client import call_groq, GROQ_MODEL

logger = logging.getLogger(__name__)

# ── Module-level executor for background title generation ─────────────────────
_executor = ThreadPoolExecutor(max_workers=2)
atexit.register(_executor.shutdown, wait=True)

# ── Constants ─────────────────────────────────────────────────────────────────
_TITLE_MAX = 60          # max chars in any conversation title
_PREVIEW_MAX = 120       # max chars for last_query_preview
_SNIPPET_MAX = 300       # max chars for source snippets stored in turns
_MAX_DOC_BYTES = 900_000 # guard against the 1 MB Firestore document limit

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ALIAS_RE = re.compile(r"\[[CS]\d+(?:\.\d+)?\]")
_JUNK_RE  = re.compile(r'[*`\[\]#]')  # markdown/bracket noise in generated titles

_TITLE_SYSTEM_PROMPT = (
    "You write short titles for a Kenyan legal research chat history. "
    "Return ONLY the title: 3 to 7 words, Title Case, no quotes, no trailing "
    "punctuation, no citation aliases like [S1.1]. Name the legal issue and "
    "the key statute/section if one is central, e.g. "
    "'M-Pesa Records as Evidence (s.106B)', "
    "'Small Claims Court Jurisdiction Limits', "
    "'Landlord Eviction Notice Requirements'. "
    "Prefer the legal issue over personal or company names."
)

# ── Firestore helpers ─────────────────────────────────────────────────────────

def _db():
    """Firestore client — always the same singleton after init_firebase_admin()."""
    return fb_firestore.client()


def _conv_ref(uid: str, cid: str):
    return _db().collection("users").document(uid).collection("conversations").document(cid)


def _turns_col(uid: str, cid: str):
    return _conv_ref(uid, cid).collection("turns")


# ── Title helpers ─────────────────────────────────────────────────────────────

def make_fallback_title(query: str) -> str:
    """Trim query to 60 chars at a word boundary; append '...' if cut."""
    q = query.strip()
    if len(q) <= _TITLE_MAX:
        return q
    cut = q[:_TITLE_MAX].rsplit(" ", 1)[0]
    return cut + "..."


def clean_generated_title(raw: str) -> str:
    """
    Extract and sanitise the first non-empty line from a Groq title response.
    Strips <think> blocks, markdown, alias brackets, quotes, trailing
    punctuation; collapses whitespace; caps at 60 chars at a word boundary.
    Returns "" if nothing usable remains.
    """
    text = _THINK_RE.sub("", raw).strip()
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if not line:
        return ""
    line = re.sub(r'^["\']|["\']$', "", line).strip()  # remove wrapping quotes
    line = _JUNK_RE.sub("", line)                        # remove markdown noise
    line = _ALIAS_RE.sub("", line)                       # remove [S1.1] aliases
    line = re.sub(r'[.,;:!]+$', "", line).strip()        # trailing punctuation
    line = re.sub(r'\s+', " ", line).strip()             # collapse whitespace
    if len(line) > _TITLE_MAX:
        line = line[:_TITLE_MAX].rsplit(" ", 1)[0]
    return line


# ── Source builder ────────────────────────────────────────────────────────────

def _find_source_in_retrieval(chunk_id: str, retrieval_data: dict) -> dict:
    """
    Return {"kind": "rule"|"case", "source": <parent dict>, "chunk": <chunk dict>}
    for the first hit by chunk_id, or {} if not found.
    """
    for rule in retrieval_data.get("rules", []):
        if rule.get("chunk_id") == chunk_id:
            return {"kind": "rule", "source": rule, "chunk": {}}
        for ch in rule.get("chunks", []):
            if ch.get("chunk_id") == chunk_id:
                return {"kind": "rule", "source": rule, "chunk": ch}
    for case in retrieval_data.get("results", []):
        for ch in case.get("chunks", []):
            if ch.get("chunk_id") == chunk_id:
                return {"kind": "case", "source": case, "chunk": ch}
    return {}


def build_sources_for_history(response_sources: list, retrieval_data: dict) -> list:
    """
    Enrich response_sources with display metadata from retrieval_data.
    Never stores full chunk text; snippets are capped at 300 chars.
    Missing fields are stored as null — never blocks the save.
    """
    result = []
    for src in response_sources:
        chunk_id = src.get("chunk_id", "")
        entry = {
            "alias":          src.get("alias", ""),
            "chunk_id":       chunk_id,
            "type":           src.get("type", ""),
            "label":          src.get("label", ""),
            "used":           src.get("used", False),
            "precision":      src.get("precision", ""),
            # display metadata — overwritten below if found
            "case_identifier": None,
            "parties":         None,
            "court_level":     None,
            "year":            None,
            "document_label":  None,
            "article_number":  None,
            "article_header":  None,
            "pdf_url":         None,
            "snippet":         None,
        }

        hit = _find_source_in_retrieval(chunk_id, retrieval_data)
        if hit:
            source = hit["source"]
            chunk  = hit["chunk"]
            if hit["kind"] == "case":
                meta = source.get("metadata", {})
                entry["case_identifier"] = source.get("case_name") or None
                entry["parties"]         = meta.get("parties") or None
                entry["court_level"]     = meta.get("court_level") or None
                entry["year"]            = str(meta.get("year", "")) or None
                entry["pdf_url"]         = source.get("pdf_url") or None
                raw_text                 = chunk.get("text", "")
            else:  # rule (statute or constitution)
                doc_type = source.get("document_type", "")
                if doc_type == "constitution":
                    entry["type"] = "constitution"
                entry["document_label"] = (
                    source.get("document_label") or source.get("source_document") or None
                )
                entry["article_number"] = str(source.get("article_number", "")) or None
                entry["article_header"] = source.get("article_header") or None
                entry["pdf_url"]        = source.get("pdf_url") or None
                raw_text = chunk.get("text", "") or source.get("text_preview", "")

            entry["snippet"] = raw_text[:_SNIPPET_MAX] if raw_text else None

        result.append(entry)
    return result


# ── Firestore transaction ─────────────────────────────────────────────────────

@_transactional
def _exec_turn_transaction(
    transaction,
    conv_ref,
    turn_ref,
    turn_doc: dict,
    is_new_conv: bool,
    fallback_title: str,
    query_preview: str,
):
    """
    Atomic write: create-or-update conversation + write turn.
    Returns the turn_index assigned to this turn.
    """
    if is_new_conv:
        transaction.set(conv_ref, {
            "title":              fallback_title,
            "title_source":       "fallback",
            "title_generated_at": None,
            "created_at":         fb_firestore.SERVER_TIMESTAMP,
            "updated_at":         fb_firestore.SERVER_TIMESTAMP,
            "turn_count":         1,
            "last_query_preview": query_preview,
            "pinned":             False,
            "schema_version":     1,
        })
        turn_index = 0
    else:
        snap = conv_ref.get(transaction=transaction)
        if not snap.exists:
            raise LookupError("Conversation disappeared inside transaction")
        turn_index = snap.get("turn_count") or 0
        transaction.update(conv_ref, {
            "updated_at":         fb_firestore.SERVER_TIMESTAMP,
            "last_query_preview": query_preview,
            "turn_count":         fb_firestore.Increment(1),
        })

    final_doc = dict(turn_doc)
    final_doc["turn_index"] = turn_index
    transaction.set(turn_ref, final_doc)
    return turn_index


# ── Public persistence API ────────────────────────────────────────────────────

def get_or_create_conversation(uid: str, conversation_id: str, first_query: str):
    """
    Validate that conversation_id exists for uid.
    Returns (conv_ref, is_new=False).
    Raises LookupError when not found — callers convert this to 404.
    Never creates a conversation from a client-supplied id.
    """
    ref = _conv_ref(uid, conversation_id)
    snap = ref.get()
    if not snap.exists:
        raise LookupError(f"Conversation {conversation_id!r} not found for user")
    return ref, False


def save_turn(
    uid: str,
    conversation_id: str | None,
    turn_data: dict,
    client_turn_id: str | None = None,
) -> dict:
    """
    Persist a turn to Firestore inside a transaction.

    conversation_id=None  → create a new conversation (auto-UUID).
    client_turn_id        → used as the turn doc ID for idempotency; if a
                            turn with that id already exists, return its ids
                            without writing a duplicate.

    Returns {"conversation_id": str, "turn_id": str, "is_new_conversation": bool}.
    """
    db = _db()
    is_new_conv = conversation_id is None

    if is_new_conv:
        conversation_id = str(_uuid_mod.uuid4())

    conv_ref  = _conv_ref(uid, conversation_id)
    turn_id   = client_turn_id or str(_uuid_mod.uuid4())
    turn_ref  = _turns_col(uid, conversation_id).document(turn_id)

    # Idempotency: if the turn doc already exists, return early
    if client_turn_id and not is_new_conv:
        snap = turn_ref.get()
        if snap.exists:
            return {
                "conversation_id":    conversation_id,
                "turn_id":            turn_id,
                "is_new_conversation": False,
            }

    first_query    = turn_data.get("query", "")
    fallback_title = make_fallback_title(first_query)
    query_preview  = first_query[:_PREVIEW_MAX]

    turn_doc = {
        "created_at":     fb_firestore.SERVER_TIMESTAMP,
        "schema_version": 1,
        "client_turn_id": client_turn_id,
        "truncated":      False,
        **turn_data,
    }
    turn_doc = _size_guard(turn_doc)

    txn = db.transaction()
    _exec_turn_transaction(
        txn,
        conv_ref,
        turn_ref,
        turn_doc,
        is_new_conv,
        fallback_title,
        query_preview,
    )

    return {
        "conversation_id":     conversation_id,
        "turn_id":             turn_id,
        "is_new_conversation": is_new_conv,
    }


def _size_guard(doc: dict) -> dict:
    """
    If the document would exceed _MAX_DOC_BYTES when serialised, truncate
    answer_markdown and source snippets and set truncated=True.
    """
    try:
        raw = json.dumps(doc, default=str).encode("utf-8")
    except Exception:
        return doc

    if len(raw) <= _MAX_DOC_BYTES:
        return doc

    overage = len(raw) - _MAX_DOC_BYTES
    doc = dict(doc)  # shallow copy so callers are not mutated
    doc["truncated"] = True

    if "answer_markdown" in doc and doc["answer_markdown"]:
        max_ans = max(100, len(doc["answer_markdown"]) - overage - 1000)
        doc["answer_markdown"] = doc["answer_markdown"][:max_ans]

    for src in doc.get("sources", []):
        if src.get("snippet"):
            src["snippet"] = src["snippet"][:100]

    return doc


# ── Background title generation ───────────────────────────────────────────────

def submit_title_generation(
    uid: str,
    conversation_id: str,
    query: str,
    short_answer: str,
) -> None:
    """Submit background title generation. Does not block the HTTP response."""
    _executor.submit(generate_conversation_title, uid, conversation_id, query, short_answer)


def generate_conversation_title(
    uid: str,
    conversation_id: str,
    query: str,
    short_answer: str,
) -> None:
    """
    Call Groq to generate a short title, then write it to Firestore only if
    title_source is still "fallback" (so a user rename is never overwritten).
    Flask's g / request context is NOT available here — all data is explicit.
    Never raises — all errors are caught and logged.
    """
    try:
        raw = call_groq(
            [
                {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
                {"role": "user",   "content": f"{query}\n\n{short_answer[:500]}"},
            ],
            temperature=0.2,
            max_tokens=300,
            timeout=10,
        )
        title = clean_generated_title(raw)
        if not title:
            print(f"[HISTORY] Title generation empty for conv {conversation_id!r}")
            return
        set_generated_title(uid, conversation_id, title)
        print(f"[HISTORY] Title generated for conv {conversation_id!r}: {title!r}")
    except Exception as exc:
        print(f"[HISTORY] Title generation failed for conv {conversation_id!r}: {exc}")


@_transactional
def _set_title_in_transaction(transaction, conv_ref, title: str):
    """Update title only when title_source is still 'fallback'."""
    snap = conv_ref.get(transaction=transaction)
    if not snap.exists:
        return
    if snap.get("title_source") != "fallback":
        return  # user has renamed — never overwrite
    transaction.update(conv_ref, {
        "title":              title,
        "title_source":       "generated",
        "title_generated_at": fb_firestore.SERVER_TIMESTAMP,
    })


def set_generated_title(uid: str, conversation_id: str, title: str) -> None:
    """Conditional title update (safe with concurrent user renames)."""
    db  = _db()
    ref = _conv_ref(uid, conversation_id)
    txn = db.transaction()
    _set_title_in_transaction(txn, ref, title)


# ── Read / manage endpoints ───────────────────────────────────────────────────

def _ts_to_iso(ts) -> str | None:
    """Convert a Firestore Timestamp (or None) to an ISO-8601 UTC string."""
    if ts is None:
        return None
    try:
        # google.cloud.firestore Timestamp has .seconds / .nanos or is a datetime
        if hasattr(ts, "isoformat"):
            dt = ts
        else:
            dt = ts.date_value if hasattr(ts, "date_value") else ts.ToDatetime(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return str(ts)


def _conv_to_dict(doc_id: str, data: dict) -> dict:
    return {
        "id":                 doc_id,
        "title":              data.get("title", ""),
        "title_source":       data.get("title_source", "fallback"),
        "pinned":             data.get("pinned", False),
        "turn_count":         data.get("turn_count", 0),
        "last_query_preview": data.get("last_query_preview", ""),
        "created_at":         _ts_to_iso(data.get("created_at")),
        "updated_at":         _ts_to_iso(data.get("updated_at")),
    }


def list_conversations(uid: str, limit: int = 20, cursor: str | None = None) -> dict:
    """
    List conversations ordered by updated_at desc.
    cursor is the ISO-8601 updated_at of the last returned item.
    Returns {"conversations": [...], "next_cursor": str|null}.
    """
    db  = _db()
    col = db.collection("users").document(uid).collection("conversations")
    q   = col.order_by("updated_at", direction="DESCENDING").limit(limit + 1)

    if cursor:
        try:
            # Parse ISO cursor back to datetime for Firestore comparison
            cursor_dt = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
            q = q.start_after([cursor_dt])
        except Exception:
            pass  # bad cursor → ignore and return from start

    docs = list(q.stream())
    next_cursor = None
    if len(docs) > limit:
        last = docs[limit]
        raw_ts = last.to_dict().get("updated_at")
        next_cursor = _ts_to_iso(raw_ts)
        docs = docs[:limit]

    return {
        "conversations": [_conv_to_dict(d.id, d.to_dict()) for d in docs],
        "next_cursor":   next_cursor,
    }


def get_conversation(
    uid: str,
    conversation_id: str,
    turn_limit: int = 50,
    before_index: int | None = None,
) -> dict:
    """
    Fetch a conversation and its turns ordered by turn_index asc.
    before_index: only return turns with turn_index < before_index (for paging).
    Returns {"conversation": {...}, "turns": [...], "has_more_turns": bool}.
    """
    conv_ref = _conv_ref(uid, conversation_id)
    snap = conv_ref.get()
    if not snap.exists:
        raise LookupError(f"Conversation {conversation_id!r} not found")

    turns_col = _turns_col(uid, conversation_id)
    q = turns_col.order_by("turn_index", direction="ASCENDING")

    if before_index is not None:
        q = q.where("turn_index", "<", before_index)

    q = q.limit(turn_limit + 1)
    turn_docs = list(q.stream())

    has_more = len(turn_docs) > turn_limit
    if has_more:
        turn_docs = turn_docs[:turn_limit]

    turns = []
    for td in turn_docs:
        d = td.to_dict()
        turns.append({
            "id":               td.id,
            "turn_index":       d.get("turn_index"),
            "status":           d.get("status"),
            "query":            d.get("query", ""),
            "answer_markdown":  d.get("answer_markdown"),
            "answer_structured":d.get("answer_structured"),
            "sources":          d.get("sources", []),
            "alias_map":        d.get("alias_map", {}),
            "coverage":         d.get("coverage", {}),
            "error":            d.get("error"),
            "error_code":       d.get("error_code"),
            "created_at":       _ts_to_iso(d.get("created_at")),
        })

    return {
        "conversation":    _conv_to_dict(conversation_id, snap.to_dict()),
        "turns":           turns,
        "has_more_turns":  has_more,
    }


def rename_conversation(uid: str, conversation_id: str, title: str) -> None:
    """Set a user-supplied title (title_source = 'user')."""
    _conv_ref(uid, conversation_id).update({
        "title":        title,
        "title_source": "user",
        "updated_at":   fb_firestore.SERVER_TIMESTAMP,
    })


def set_pinned(uid: str, conversation_id: str, pinned: bool) -> None:
    _conv_ref(uid, conversation_id).update({
        "pinned":     pinned,
        "updated_at": fb_firestore.SERVER_TIMESTAMP,
    })


def delete_conversation(uid: str, conversation_id: str) -> None:
    """Recursively delete the conversation and all its turns."""
    db  = _db()
    ref = _conv_ref(uid, conversation_id)
    db.recursive_delete(ref)


def regenerate_title(uid: str, conversation_id: str, force: bool = False) -> None:
    """
    Re-run title generation from the first turn.
    Skipped when title_source == "user" unless force=True.
    """
    conv_ref = _conv_ref(uid, conversation_id)
    snap = conv_ref.get()
    if not snap.exists:
        raise LookupError(f"Conversation {conversation_id!r} not found")

    if snap.get("title_source") == "user" and not force:
        return  # respect the user rename

    # Fetch turn 0
    turns = list(
        _turns_col(uid, conversation_id)
        .where("turn_index", "==", 0)
        .limit(1)
        .stream()
    )
    if not turns:
        return

    turn_data = turns[0].to_dict()
    query        = turn_data.get("query", "")
    short_answer = (turn_data.get("answer_structured") or {}).get("short_answer", "")

    # Reset to fallback so set_generated_title will write
    conv_ref.update({"title_source": "fallback"})

    submit_title_generation(uid, conversation_id, query, short_answer)
