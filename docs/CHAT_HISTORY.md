# Chat History — Data Model & API Contract

## Firestore data model

```
users/{uid}/
  conversations/{cid}            ← conversation document
    turns/{client_turn_id}       ← turn document; ID = client_turn_id (idempotency)
```

### Conversation document

| Field            | Type      | Notes                                          |
|------------------|-----------|------------------------------------------------|
| `uid`            | string    | Owner; always set from server-side `g.uid`     |
| `title`          | string    | ≤ 60 chars                                     |
| `title_source`   | string    | `"fallback"` / `"generated"` / `"user"`        |
| `turn_count`     | integer   | Incremented atomically inside a transaction    |
| `pinned`         | boolean   | Client-controlled via PATCH                    |
| `created_at`     | timestamp |                                                |
| `updated_at`     | timestamp | Set on every successful turn save              |

### Turn document

| Field              | Type    | Notes                                                        |
|--------------------|---------|--------------------------------------------------------------|
| `turn_index`       | integer | 0-based; assigned inside transaction on `turn_count`         |
| `status`           | string  | `"ok"` or `"failed"`                                         |
| `query`            | string  | Raw user query                                               |
| `answer`           | string  | LLM answer (omitted on failed turns)                         |
| `coverage`         | string  | `"full"` / `"partial"` / `"none"`                            |
| `sources`          | array   | See **Source object** below                                  |
| `elapsed_ms`       | integer | Wall-clock time for the generate call                        |
| `error`            | string  | Sanitised error message (failed turns only)                  |
| `error_code`       | string  | `"no_sources"` / `"llm_error"` / `"internal_error"`         |
| `created_at`       | timestamp |                                                            |

### Source object (inside `sources[]`)

| Field           | Type          | Notes                                                |
|-----------------|---------------|------------------------------------------------------|
| `chunk_id`      | string        | ChromaDB chunk identifier                            |
| `type`          | string        | `"case"` / `"statute"` / `"constitution"`            |
| `document_type` | string        | Raw value from retrieval payload                     |
| `case_identifier` | string \| null | Case citation / statute identifier                |
| `section`       | string \| null | Statute section or case paragraph reference          |
| `pdf_url`       | string \| null | Signed URL or static path to the source PDF          |
| `snippet`       | string \| null | ≤ 300 chars from the chunk text                      |
| `score`         | number \| null | Retrieval similarity score                           |

---

## API endpoints

All endpoints require a valid Firebase ID token in `Authorization: Bearer <token>`.
Only users with role `"advocate"` or `"admin"` may access these routes.
`uid` is always resolved server-side from the token — never from the request body.

### GET `/api/conversations`

List conversations for the authenticated user, ordered by pinned desc, then `updated_at` desc.

**Query params**

| Param    | Default | Range  | Description                          |
|----------|---------|--------|--------------------------------------|
| `limit`  | 20      | 1–50   | Number of conversations to return    |
| `cursor` | —       | —      | ISO-8601 `updated_at` of last item   |

**Response 200**

```json
{
  "conversations": [
    {
      "id": "abc123",
      "title": "Tenant eviction notice requirements",
      "title_source": "generated",
      "turn_count": 4,
      "pinned": false,
      "created_at": "2026-10-31T08:14:00Z",
      "updated_at": "2026-10-31T08:22:41Z"
    }
  ],
  "next_cursor": "2026-10-31T07:00:00Z"
}
```

`next_cursor` is omitted when there are no further pages.

---

### GET `/api/conversations/<id>`

Fetch a single conversation with its turns.

**Query params**

| Param          | Default | Range   | Description                                    |
|----------------|---------|---------|------------------------------------------------|
| `turn_limit`   | 50      | 1–100   | Max turns to return                            |
| `before_index` | —       | integer | Return turns with `turn_index < before_index`  |

**Response 200**

```json
{
  "id": "abc123",
  "title": "Tenant eviction notice requirements",
  "title_source": "generated",
  "turn_count": 4,
  "pinned": false,
  "created_at": "2026-10-31T08:14:00Z",
  "updated_at": "2026-10-31T08:22:41Z",
  "turns": [
    {
      "id": "client-uuid-v4",
      "turn_index": 0,
      "status": "ok",
      "query": "What notice period is required?",
      "answer": "Under the Landlord and Tenant Act...",
      "coverage": "full",
      "sources": [
        {
          "chunk_id": "chunk-xyz",
          "type": "statute",
          "document_type": "statute",
          "case_identifier": "S12",
          "section": "12(1)",
          "pdf_url": "https://...",
          "snippet": "The landlord shall give not less than...",
          "score": 0.87
        }
      ],
      "elapsed_ms": 3241,
      "created_at": "2026-10-31T08:14:02Z"
    }
  ]
}
```

**Response 404** — conversation not found or not owned by caller.

---

### PATCH `/api/conversations/<id>`

Rename and/or pin a conversation. Send at least one field.

**Request body**

```json
{ "title": "New title up to 120 chars", "pinned": true }
```

**Response 200**

```json
{ "ok": true }
```

**Response 400** — `title` empty, > 120 chars, or `pinned` not boolean.  
**Response 404** — conversation not found.

---

### DELETE `/api/conversations/<id>`

Recursively delete the conversation and all its turns.

**Response 200**

```json
{ "ok": true }
```

**Response 404** — conversation not found or not owned by caller.

---

### POST `/api/conversations/<id>/regenerate-title`

Queue background AI title regeneration.

**Request body** *(optional)*

```json
{ "force": false }
```

`force: false` (default) skips regeneration if `title_source` is already `"generated"` or `"user"`.  
`force: true` regenerates even for user-set titles.

**Response 202**

```json
{ "ok": true, "message": "Title regeneration queued" }
```

**Response 404** — conversation not found.

---

## Enriched `/api/generate-legal-answer` response

After implementing history, the generate endpoint returns additional fields:

```json
{
  "answer": "...",
  "sources": [...],
  "coverage": "full",
  "conversation_id": "abc123",
  "turn_id": "client-uuid-v4",
  "history_saved": true,
  "conversation_title": "Tenant eviction notice requirements"
}
```

| Field                | Type    | Notes                                                    |
|----------------------|---------|----------------------------------------------------------|
| `conversation_id`    | string  | Echo of the conversation (new UUID if caller sent none)  |
| `turn_id`            | string  | The `client_turn_id` used as the Firestore document ID   |
| `history_saved`      | boolean | `false` if Firestore write failed (answer still returned)|
| `conversation_title` | string  | Current title (fallback or generated)                    |

---

## Frontend integration guide

### Sending conversation context

Include these two fields in every `POST /api/generate-legal-answer` request:

```json
{
  "query": "...",
  "retrieval": { ... },
  "conversation_id": "abc123",
  "client_turn_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

- **`conversation_id`** — omit (or send `null`) to start a new conversation. Use the `conversation_id` returned by the previous response to continue an existing one.
- **`client_turn_id`** — generate a UUID v4 client-side before the request. Idempotency: if the same ID is sent twice (retry), the second save is a no-op.

### Sidebar

1. On mount, call `GET /api/conversations?limit=20` to populate the sidebar list.
2. When the user selects a conversation, call `GET /api/conversations/<id>` to load its turns and render the chat history.
3. Paginate turns backwards with `before_index` as the user scrolls up.
4. Paginate the sidebar list with the `next_cursor` from the list response.

### Live title updates

After a successful generate call:

- If `conversation_title` in the response differs from the sidebar entry (e.g. it changed from a fallback to a generated title), update the sidebar item in-place.
- Alternatively, listen to `users/{uid}/conversations/{cid}` via `onSnapshot` for real-time updates — the title field changes once the background job completes (typically 2–5 s after the answer is returned).

### Resuming a chat

When the user reopens an existing conversation from the sidebar:

1. Load turns via `GET /api/conversations/<id>`.
2. Set `conversation_id` in local state.
3. Each new query in that conversation sends `conversation_id` along with a fresh `client_turn_id`.
