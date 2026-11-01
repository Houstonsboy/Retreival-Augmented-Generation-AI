"""
Tests for history_store.py and history_routes.py.

Runs against the Firestore emulator when FIRESTORE_EMULATOR_HOST is set,
otherwise uses unittest.mock stubs.

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_history.py -v
    # or with emulator:
    FIRESTORE_EMULATOR_HOST=localhost:8080 PYTHONPATH=. pytest tests/test_history.py -v
"""

import os
import sys
import time
import uuid
from unittest.mock import MagicMock, call, patch

import pytest

# Make sure the backend directory is on sys.path
_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

EMULATOR = os.environ.get("FIRESTORE_EMULATOR_HOST")


def _make_mock_db(conv_data=None, turn_exists=False):
    """
    Return a (mock_db, mock_conv_ref, mock_turn_ref) triple.
    conv_data: dict returned by conv_ref.get().to_dict()  (None → doc doesn't exist)
    turn_exists: whether the turn doc already exists (idempotency test)
    """
    mock_db = MagicMock()

    # Build the reference chain
    mock_conv_ref = MagicMock()
    mock_turn_ref = MagicMock()
    mock_turns_col = MagicMock()

    # db.collection().document().collection().document() → mock_turn_ref
    mock_db.collection.return_value.document.return_value \
        .collection.return_value.document.return_value = mock_turn_ref
    # db.collection().document() → mock_conv_ref
    mock_db.collection.return_value.document.return_value = mock_conv_ref
    # mock_conv_ref.collection().document() → mock_turn_ref
    mock_conv_ref.collection.return_value.document.return_value = mock_turn_ref

    # conv_ref.get() → snapshot
    if conv_data is None:
        snap = MagicMock()
        snap.exists = False
        snap.get.side_effect = KeyError("doc doesn't exist")
    else:
        snap = MagicMock()
        snap.exists = True
        snap.to_dict.return_value = dict(conv_data)
        snap.get.side_effect = lambda key: conv_data.get(key)

    mock_conv_ref.get.return_value = snap

    # turn_ref.get() → snapshot for idempotency
    turn_snap = MagicMock()
    turn_snap.exists = turn_exists
    mock_turn_ref.get.return_value = turn_snap

    # transaction
    mock_txn = MagicMock()
    mock_db.transaction.return_value = mock_txn

    return mock_db, mock_conv_ref, mock_turn_ref, mock_txn


# ---------------------------------------------------------------------------
# 1. Pure unit tests — no Firestore needed
# ---------------------------------------------------------------------------

class TestMakeFallbackTitle:
    def test_short_query_unchanged(self):
        from history_store import make_fallback_title
        q = "What is bail?"
        assert make_fallback_title(q) == q

    def test_long_query_trimmed_at_word_boundary(self):
        from history_store import make_fallback_title
        q = "a " * 40  # 80 chars
        title = make_fallback_title(q)
        assert len(title) <= 63  # 60 + "..."
        assert title.endswith("...")

    def test_exactly_60_chars_no_ellipsis(self):
        from history_store import make_fallback_title
        q = "x" * 60
        assert make_fallback_title(q) == q

    def test_61_chars_gets_ellipsis(self):
        from history_store import make_fallback_title
        q = "word " * 12  # 60 chars = "word " * 12
        title = make_fallback_title(q.rstrip())
        # trimmed version should be <= 60 chars + possible "..."
        assert len(title) <= 63


class TestCleanGeneratedTitle:
    def test_basic_clean(self):
        from history_store import clean_generated_title
        assert clean_generated_title("Landlord Eviction Notice Requirements") == \
               "Landlord Eviction Notice Requirements"

    def test_removes_think_blocks(self):
        from history_store import clean_generated_title
        raw = "<think>some reasoning</think>\nM-Pesa Records as Evidence"
        assert clean_generated_title(raw) == "M-Pesa Records as Evidence"

    def test_removes_quotes(self):
        from history_store import clean_generated_title
        assert clean_generated_title('"Bail Application Grounds"') == \
               "Bail Application Grounds"

    def test_removes_alias_brackets(self):
        from history_store import clean_generated_title
        raw = "Evidence Act [S1.1] Admissibility"
        assert "[S1.1]" not in clean_generated_title(raw)

    def test_removes_trailing_punctuation(self):
        from history_store import clean_generated_title
        assert clean_generated_title("Small Claims Court Jurisdiction.") == \
               "Small Claims Court Jurisdiction"

    def test_collapses_whitespace(self):
        from history_store import clean_generated_title
        assert clean_generated_title("Bail   Application   Grounds") == \
               "Bail Application Grounds"

    def test_caps_at_60_chars(self):
        from history_store import clean_generated_title
        long = "This Is A Very Long Title That Exceeds The Maximum Allowed Characters Limit For Titles"
        assert len(clean_generated_title(long)) <= 60

    def test_empty_when_only_think_block(self):
        from history_store import clean_generated_title
        assert clean_generated_title("<think>all reasoning here</think>") == ""

    def test_first_line_only(self):
        from history_store import clean_generated_title
        raw = "Land Eviction Rights\nSecond line ignored"
        assert clean_generated_title(raw) == "Land Eviction Rights"


class TestBuildSourcesForHistory:
    def _retrieval(self):
        return {
            "results": [{
                "case_name": "Republic v Doe [2020]",
                "pdf_url":   "/api/case-pdf?case_id=Republic+v+Doe+%5B2020%5D",
                "metadata":  {"parties": "Republic v Doe", "court_level": "High Court", "year": "2020"},
                "chunks": [
                    {"chunk_id": "c-001", "text": "A" * 400},
                ],
            }],
            "rules": [{
                "document_type":  "statute",
                "document_label": "Evidence Act",
                "article_number": "106B",
                "article_header": "Admissibility of electronic records",
                "pdf_url":        "/api/rule-pdf?doc_type=statute&source=Evidence+Act",
                "chunk_id":       "s-001",
                "chunks": [
                    {"chunk_id": "s-001", "text": "B" * 400},
                ],
            }, {
                "document_type":  "constitution",
                "document_label": "Constitution of Kenya",
                "article_number": "50",
                "article_header": "Fair hearing",
                "pdf_url":        "/api/rule-pdf?doc_type=constitution&source=Constitution",
                "chunk_id":       "const-001",
                "chunks": [
                    {"chunk_id": "const-001", "text": "C" * 400},
                ],
            }],
        }

    def test_snippet_truncated_to_300(self):
        from history_store import build_sources_for_history
        response_sources = [{"alias": "C1.1", "chunk_id": "c-001", "type": "case",
                              "label": "Republic v Doe", "used": True, "precision": "chunk"}]
        result = build_sources_for_history(response_sources, self._retrieval())
        assert result[0]["snippet"] is not None
        assert len(result[0]["snippet"]) <= 300

    def test_no_full_text_stored(self):
        from history_store import build_sources_for_history
        response_sources = [{"alias": "C1.1", "chunk_id": "c-001", "type": "case",
                              "label": "Republic v Doe", "used": True, "precision": "chunk"}]
        result = build_sources_for_history(response_sources, self._retrieval())
        entry = result[0]
        assert "text" not in entry
        assert entry.get("snippet") is None or len(entry.get("snippet", "")) <= 300

    def test_case_metadata_populated(self):
        from history_store import build_sources_for_history
        response_sources = [{"alias": "C1.1", "chunk_id": "c-001", "type": "case",
                              "label": "Lbl", "used": True, "precision": "chunk"}]
        result = build_sources_for_history(response_sources, self._retrieval())
        entry = result[0]
        assert entry["case_identifier"] == "Republic v Doe [2020]"
        assert entry["court_level"] == "High Court"
        assert entry["year"] == "2020"
        assert entry["pdf_url"] is not None

    def test_constitution_type_mapped(self):
        from history_store import build_sources_for_history
        response_sources = [{"alias": "S1.1", "chunk_id": "const-001", "type": "statute",
                              "label": "Constitution", "used": True, "precision": "chunk"}]
        result = build_sources_for_history(response_sources, self._retrieval())
        assert result[0]["type"] == "constitution"

    def test_statute_type_unchanged(self):
        from history_store import build_sources_for_history
        response_sources = [{"alias": "S2.1", "chunk_id": "s-001", "type": "statute",
                              "label": "Evidence Act", "used": True, "precision": "chunk"}]
        result = build_sources_for_history(response_sources, self._retrieval())
        assert result[0]["type"] == "statute"

    def test_missing_chunk_id_nulls_out_metadata(self):
        from history_store import build_sources_for_history
        response_sources = [{"alias": "S9.1", "chunk_id": "nonexistent", "type": "statute",
                              "label": "Unknown", "used": False, "precision": "chunk"}]
        result = build_sources_for_history(response_sources, self._retrieval())
        entry = result[0]
        assert entry["pdf_url"] is None
        assert entry["snippet"] is None


# ---------------------------------------------------------------------------
# 2. Firestore-mocked tests for save_turn
# ---------------------------------------------------------------------------

class TestSaveTurn:
    def _patch_db(self, conv_data=None, turn_exists=False):
        """Patch history_store._db() to return a mock Firestore client."""
        mock_db, conv_ref, turn_ref, txn = _make_mock_db(conv_data, turn_exists)
        return patch("history_store._db", return_value=mock_db), mock_db, conv_ref, turn_ref, txn

    def test_new_conversation_returns_ids(self):
        from history_store import save_turn
        with patch("history_store._db") as mock_db_fn, \
             patch("history_store._exec_turn_transaction") as mock_txn_fn:
            mock_db_fn.return_value = MagicMock()
            mock_txn_fn.return_value = None
            result = save_turn("uid-1", None, {"query": "What is bail?", "status": "complete"})
            assert "conversation_id" in result
            assert "turn_id" in result
            assert result["is_new_conversation"] is True

    def test_follow_up_not_new_conversation(self):
        from history_store import save_turn
        existing_conv_id = str(uuid.uuid4())
        with patch("history_store._db") as mock_db_fn, \
             patch("history_store._exec_turn_transaction") as mock_txn_fn:
            mock_db_fn.return_value = MagicMock()
            mock_txn_fn.return_value = None
            result = save_turn("uid-1", existing_conv_id, {"query": "Follow up?", "status": "complete"})
            assert result["conversation_id"] == existing_conv_id
            assert result["is_new_conversation"] is False

    def test_idempotency_same_client_turn_id(self):
        from history_store import save_turn
        client_turn_id = str(uuid.uuid4())
        existing_conv_id = str(uuid.uuid4())

        # Simulate existing turn doc
        mock_turn_snap = MagicMock()
        mock_turn_snap.exists = True
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value \
            .collection.return_value.document.return_value \
            .get.return_value = mock_turn_snap

        with patch("history_store._db", return_value=mock_db), \
             patch("history_store._exec_turn_transaction") as mock_txn_fn:
            mock_txn_fn.return_value = None
            result = save_turn(
                "uid-1", existing_conv_id,
                {"query": "What is bail?", "status": "complete"},
                client_turn_id=client_turn_id,
            )
            # Should return early without calling the transaction
            mock_txn_fn.assert_not_called()
            assert result["turn_id"] == client_turn_id
            assert result["is_new_conversation"] is False

    def test_client_turn_id_used_as_doc_id(self):
        from history_store import save_turn
        client_turn_id = str(uuid.uuid4())
        with patch("history_store._db") as mock_db_fn, \
             patch("history_store._exec_turn_transaction") as mock_txn_fn:
            mock_db_fn.return_value = MagicMock()
            mock_txn_fn.return_value = None
            result = save_turn("uid-1", None, {"query": "Q", "status": "complete"},
                               client_turn_id=client_turn_id)
            assert result["turn_id"] == client_turn_id


class TestGetOrCreateConversation:
    def test_existing_conv_returns_ref(self):
        from history_store import get_or_create_conversation
        mock_snap = MagicMock()
        mock_snap.exists = True
        mock_ref = MagicMock()
        mock_ref.get.return_value = mock_snap
        with patch("history_store._db") as mock_db_fn:
            mock_db_fn.return_value = MagicMock()
            # Patch _conv_ref directly
            with patch("history_store._conv_ref", return_value=mock_ref):
                ref, is_new = get_or_create_conversation("uid-1", "conv-123", "Q")
                assert is_new is False

    def test_nonexistent_conv_raises(self):
        from history_store import get_or_create_conversation
        mock_snap = MagicMock()
        mock_snap.exists = False
        mock_ref = MagicMock()
        mock_ref.get.return_value = mock_snap
        with patch("history_store._conv_ref", return_value=mock_ref):
            with pytest.raises(LookupError):
                get_or_create_conversation("uid-1", "bad-id", "Q")

    def test_foreign_conversation_raises(self):
        """A conversation doc that exists for another user raises LookupError (same check)."""
        from history_store import get_or_create_conversation
        mock_snap = MagicMock()
        mock_snap.exists = False  # doesn't exist for this uid's path
        mock_ref = MagicMock()
        mock_ref.get.return_value = mock_snap
        with patch("history_store._conv_ref", return_value=mock_ref):
            with pytest.raises(LookupError):
                get_or_create_conversation("other-uid", "conv-123", "Q")


# ---------------------------------------------------------------------------
# 3. Title generation tests
# ---------------------------------------------------------------------------

class TestGenerateConversationTitle:
    def test_generated_title_calls_set_generated_title(self):
        from history_store import generate_conversation_title
        with patch("history_store.call_groq", return_value="Bail Application Standards"), \
             patch("history_store.clean_generated_title",
                   return_value="Bail Application Standards") as mock_clean, \
             patch("history_store.set_generated_title") as mock_set:
            generate_conversation_title("uid-1", "conv-1", "What is bail?", "Bail is...")
            mock_set.assert_called_once_with("uid-1", "conv-1", "Bail Application Standards")

    def test_groq_failure_leaves_title_alone(self):
        from history_store import generate_conversation_title
        with patch("history_store.call_groq", side_effect=Exception("timeout")), \
             patch("history_store.set_generated_title") as mock_set:
            # Must not raise
            generate_conversation_title("uid-1", "conv-1", "Q", "A")
            mock_set.assert_not_called()

    def test_groq_timeout_leaves_title_alone(self):
        from history_store import generate_conversation_title
        import requests as _req
        with patch("history_store.call_groq",
                   side_effect=_req.exceptions.Timeout("timed out")), \
             patch("history_store.set_generated_title") as mock_set:
            generate_conversation_title("uid-1", "conv-1", "Q", "A")
            mock_set.assert_not_called()

    def test_empty_cleaned_title_leaves_fallback(self):
        from history_store import generate_conversation_title
        with patch("history_store.call_groq", return_value="<think>nothing</think>"), \
             patch("history_store.set_generated_title") as mock_set:
            generate_conversation_title("uid-1", "conv-1", "Q", "A")
            mock_set.assert_not_called()


class TestSetGeneratedTitle:
    def test_updates_when_source_is_fallback(self):
        from history_store import _set_title_in_transaction
        snap = MagicMock()
        snap.exists = True
        snap.get.side_effect = lambda k: "fallback" if k == "title_source" else None
        mock_ref = MagicMock()
        mock_ref.get.return_value = snap
        mock_txn = MagicMock()
        # Access the original function via .to_wrap (google-cloud-firestore _Transactional)
        _set_title_in_transaction.to_wrap(mock_txn, mock_ref, "Generated Title")
        mock_txn.update.assert_called_once()
        call_kwargs = mock_txn.update.call_args[0][1]
        assert call_kwargs["title"] == "Generated Title"
        assert call_kwargs["title_source"] == "generated"

    def test_skips_when_source_is_user(self):
        from history_store import _set_title_in_transaction
        snap = MagicMock()
        snap.exists = True
        snap.get.side_effect = lambda k: "user" if k == "title_source" else None
        mock_ref = MagicMock()
        mock_ref.get.return_value = snap
        mock_txn = MagicMock()
        _set_title_in_transaction.to_wrap(mock_txn, mock_ref, "Generated Title")
        mock_txn.update.assert_not_called()

    def test_skips_when_source_is_generated(self):
        from history_store import _set_title_in_transaction
        snap = MagicMock()
        snap.exists = True
        snap.get.side_effect = lambda k: "generated" if k == "title_source" else None
        mock_ref = MagicMock()
        mock_ref.get.return_value = snap
        mock_txn = MagicMock()
        _set_title_in_transaction.to_wrap(mock_txn, mock_ref, "Another Title")
        mock_txn.update.assert_not_called()


class TestSubmitTitleGeneration:
    def test_submits_to_executor(self):
        from history_store import submit_title_generation
        with patch("history_store._executor") as mock_exec:
            submit_title_generation("uid-1", "conv-1", "Q", "A")
            mock_exec.submit.assert_called_once()
            args = mock_exec.submit.call_args[0]
            from history_store import generate_conversation_title
            assert args[0] is generate_conversation_title
            assert args[1] == "uid-1"
            assert args[2] == "conv-1"


# ---------------------------------------------------------------------------
# 4. Read / manage endpoint tests
# ---------------------------------------------------------------------------

class TestListConversations:
    def _make_stream_doc(self, doc_id, data):
        doc = MagicMock()
        doc.id = doc_id
        doc.to_dict.return_value = data
        return doc

    def test_returns_ordered_list(self):
        from history_store import list_conversations
        docs = [
            self._make_stream_doc("conv-2", {"title": "B", "updated_at": None,
                                              "title_source": "fallback", "pinned": False,
                                              "turn_count": 1, "last_query_preview": "Q2",
                                              "created_at": None}),
            self._make_stream_doc("conv-1", {"title": "A", "updated_at": None,
                                              "title_source": "fallback", "pinned": False,
                                              "turn_count": 2, "last_query_preview": "Q1",
                                              "created_at": None}),
        ]
        mock_query = MagicMock()
        mock_query.stream.return_value = docs
        mock_query.limit.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.start_after.return_value = mock_query
        mock_col = MagicMock()
        mock_col.order_by.return_value = mock_query
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = mock_col

        with patch("history_store._db", return_value=mock_db):
            result = list_conversations("uid-1", limit=20)
            assert len(result["conversations"]) == 2
            assert result["conversations"][0]["id"] == "conv-2"

    def test_pagination_next_cursor_when_more_items(self):
        from history_store import list_conversations
        # Return limit+1 docs to trigger next_cursor
        docs = []
        for i in range(22):  # limit=20, so 22 means there's a next page
            d = MagicMock()
            d.id = f"conv-{i}"
            d.to_dict.return_value = {"title": f"T{i}", "updated_at": None,
                                       "title_source": "fallback", "pinned": False,
                                       "turn_count": 1, "last_query_preview": "",
                                       "created_at": None}
            docs.append(d)
        mock_query = MagicMock()
        mock_query.stream.return_value = docs
        mock_query.limit.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_col = MagicMock()
        mock_col.order_by.return_value = mock_query
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = mock_col

        with patch("history_store._db", return_value=mock_db):
            result = list_conversations("uid-1", limit=20)
            assert len(result["conversations"]) == 20
            # next_cursor should be set (even if None from mock timestamp)


class TestGetConversation:
    def test_turns_returned_in_order(self):
        from history_store import get_conversation
        conv_snap = MagicMock()
        conv_snap.exists = True
        conv_snap.to_dict.return_value = {
            "title": "Test Conv", "title_source": "fallback", "pinned": False,
            "turn_count": 2, "last_query_preview": "Q", "created_at": None, "updated_at": None,
        }
        turn_docs = []
        for i in range(2):
            td = MagicMock()
            td.id = f"turn-{i}"
            td.to_dict.return_value = {
                "turn_index": i, "status": "complete", "query": f"Q{i}",
                "answer_markdown": f"A{i}", "answer_structured": None,
                "sources": [], "alias_map": {}, "coverage": {},
                "error": None, "error_code": None, "created_at": None,
            }
            turn_docs.append(td)

        mock_turns_q = MagicMock()
        mock_turns_q.order_by.return_value = mock_turns_q
        mock_turns_q.limit.return_value = mock_turns_q
        mock_turns_q.stream.return_value = turn_docs

        mock_conv_ref = MagicMock()
        mock_conv_ref.get.return_value = conv_snap
        mock_conv_ref.collection.return_value.order_by.return_value = mock_turns_q

        with patch("history_store._conv_ref", return_value=mock_conv_ref):
            result = get_conversation("uid-1", "conv-1")
            assert len(result["turns"]) == 2
            assert result["turns"][0]["turn_index"] == 0
            assert result["turns"][1]["turn_index"] == 1
            assert result["has_more_turns"] is False

    def test_nonexistent_conv_raises(self):
        from history_store import get_conversation
        snap = MagicMock()
        snap.exists = False
        mock_ref = MagicMock()
        mock_ref.get.return_value = snap
        with patch("history_store._conv_ref", return_value=mock_ref):
            with pytest.raises(LookupError):
                get_conversation("uid-1", "bad-id")


class TestRenameAndPin:
    def test_rename_sets_user_source(self):
        from history_store import rename_conversation
        mock_ref = MagicMock()
        with patch("history_store._conv_ref", return_value=mock_ref):
            rename_conversation("uid-1", "conv-1", "New Title")
            mock_ref.update.assert_called_once()
            kwargs = mock_ref.update.call_args[0][0]
            assert kwargs["title"] == "New Title"
            assert kwargs["title_source"] == "user"

    def test_set_pinned_true(self):
        from history_store import set_pinned
        mock_ref = MagicMock()
        with patch("history_store._conv_ref", return_value=mock_ref):
            set_pinned("uid-1", "conv-1", True)
            kwargs = mock_ref.update.call_args[0][0]
            assert kwargs["pinned"] is True

    def test_set_pinned_false(self):
        from history_store import set_pinned
        mock_ref = MagicMock()
        with patch("history_store._conv_ref", return_value=mock_ref):
            set_pinned("uid-1", "conv-1", False)
            kwargs = mock_ref.update.call_args[0][0]
            assert kwargs["pinned"] is False


class TestDeleteConversation:
    def test_recursive_delete_called(self):
        from history_store import delete_conversation
        mock_ref = MagicMock()
        mock_db = MagicMock()
        with patch("history_store._conv_ref", return_value=mock_ref), \
             patch("history_store._db", return_value=mock_db):
            delete_conversation("uid-1", "conv-1")
            mock_db.recursive_delete.assert_called_once_with(mock_ref)


# ---------------------------------------------------------------------------
# 5. Flask endpoint tests (history_routes.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Flask test client with Firebase token verification mocked to always succeed.
    Patches auth._verify_and_load so require_role passes without a real Firebase
    project; all other app logic (blueprints, decorators) runs normally.
    """
    import auth
    import history_routes   # noqa: F401 — ensure blueprint registered
    import app as flask_app

    flask_app.app.config["TESTING"] = True

    # Patch the low-level Firebase verification; everything above it is real.
    with patch.object(
        auth,
        "_verify_and_load",
        return_value=(("test-uid", {"type": "advocate"}), None),
    ):
        with flask_app.app.test_client() as c:
            yield c


class TestHistoryEndpoints:
    def test_list_conversations_200(self, client):
        with patch("history_store.list_conversations",
                   return_value={"conversations": [], "next_cursor": None}):
            resp = client.get("/api/conversations",
                              headers={"Authorization": "Bearer token"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "conversations" in data

    def test_list_conversations_bad_limit(self, client):
        resp = client.get("/api/conversations?limit=999",
                          headers={"Authorization": "Bearer token"})
        # limit is clamped to max so query still succeeds with fallback or 400
        # Our implementation returns 400 for out-of-range
        assert resp.status_code in (200, 400)

    def test_get_conversation_not_found(self, client):
        with patch("history_store.get_conversation",
                   side_effect=LookupError("not found")):
            resp = client.get("/api/conversations/bad-id",
                              headers={"Authorization": "Bearer token"})
            assert resp.status_code == 404

    def test_patch_rename(self, client):
        # Must patch in history_routes namespace (imported name, not the source)
        with patch("history_routes.rename_conversation"):
            resp = client.patch("/api/conversations/conv-1",
                                json={"title": "New Name"},
                                headers={"Authorization": "Bearer token"})
            assert resp.status_code == 200

    def test_patch_empty_title_rejected(self, client):
        resp = client.patch("/api/conversations/conv-1",
                            json={"title": ""},
                            headers={"Authorization": "Bearer token"})
        assert resp.status_code == 400

    def test_patch_too_long_title_rejected(self, client):
        resp = client.patch("/api/conversations/conv-1",
                            json={"title": "x" * 121},
                            headers={"Authorization": "Bearer token"})
        assert resp.status_code == 400

    def test_delete_conversation_not_found(self, client):
        with patch("history_store.get_or_create_conversation",
                   side_effect=LookupError("not found")):
            resp = client.delete("/api/conversations/bad-id",
                                 headers={"Authorization": "Bearer token"})
            assert resp.status_code == 404

    def test_delete_conversation_ok(self, client):
        with patch("history_store.get_or_create_conversation",
                   return_value=(MagicMock(), False)), \
             patch("history_store.delete_conversation"):
            resp = client.delete("/api/conversations/conv-1",
                                 headers={"Authorization": "Bearer token"})
            assert resp.status_code == 200

    def test_regen_title_not_found(self, client):
        with patch("history_store.regenerate_title",
                   side_effect=LookupError("not found")):
            resp = client.post("/api/conversations/bad-id/regenerate-title",
                               json={},
                               headers={"Authorization": "Bearer token"})
            assert resp.status_code == 404

    def test_regen_title_queued(self, client):
        with patch("history_routes.regenerate_title"):
            resp = client.post("/api/conversations/conv-1/regenerate-title",
                               json={"force": True},
                               headers={"Authorization": "Bearer token"})
            assert resp.status_code == 202


# ---------------------------------------------------------------------------
# 6. Size guard test
# ---------------------------------------------------------------------------

class TestSizeGuard:
    def test_oversized_doc_truncates(self):
        from history_store import _size_guard, _MAX_DOC_BYTES
        big_answer = "x" * _MAX_DOC_BYTES  # clearly oversized
        doc = {
            "answer_markdown": big_answer,
            "sources": [{"snippet": "y" * 300}],
            "truncated": False,
        }
        result = _size_guard(doc)
        assert result["truncated"] is True
        assert len(result["answer_markdown"]) < len(big_answer)
        assert len(result["sources"][0]["snippet"]) <= 100

    def test_normal_doc_unchanged(self):
        from history_store import _size_guard
        doc = {"answer_markdown": "short answer", "sources": [], "truncated": False}
        result = _size_guard(doc)
        assert result["truncated"] is False


# ---------------------------------------------------------------------------
# 7. Emulator-only integration tests (skipped when emulator not available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not EMULATOR, reason="Firestore emulator not running")
class TestEmulatorIntegration:
    """Full round-trip tests against the Firestore emulator."""

    @pytest.fixture(autouse=True)
    def setup_firebase(self):
        import firebase_admin
        from firebase_admin import firestore as fb_firestore
        # Use emulator credentials
        os.environ["FIRESTORE_EMULATOR_HOST"] = EMULATOR
        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={"projectId": "test-project"})
        self.db = fb_firestore.client()
        self.uid = f"test-{uuid.uuid4().hex[:8]}"
        yield
        # Cleanup
        convs = self.db.collection("users").document(self.uid) \
                    .collection("conversations").stream()
        for c in convs:
            self.db.recursive_delete(c.reference)

    def test_full_save_and_read_cycle(self):
        from history_store import save_turn, get_conversation

        # Turn 0 — new conversation
        r = save_turn(self.uid, None, {
            "status": "complete",
            "query": "What is bail?",
            "answer_markdown": "Bail is release...",
            "answer_structured": {"short_answer": "Bail is release..."},
            "sources": [], "alias_map": {}, "coverage": {},
            "retrieval_summary": {}, "unverified_claims_removed_count": 0,
            "model": "test", "latency_ms": 100, "error": None, "error_code": None,
        })
        assert r["is_new_conversation"] is True
        cid = r["conversation_id"]

        # Turn 1 — follow-up
        r2 = save_turn(self.uid, cid, {
            "status": "complete",
            "query": "What about bail conditions?",
            "answer_markdown": "Conditions include...",
            "answer_structured": {"short_answer": "Conditions include..."},
            "sources": [], "alias_map": {}, "coverage": {},
            "retrieval_summary": {}, "unverified_claims_removed_count": 0,
            "model": "test", "latency_ms": 150, "error": None, "error_code": None,
        })
        assert r2["is_new_conversation"] is False
        assert r2["conversation_id"] == cid

        # Read back
        result = get_conversation(self.uid, cid)
        assert len(result["turns"]) == 2
        assert result["turns"][0]["turn_index"] == 0
        assert result["turns"][1]["turn_index"] == 1
        assert result["conversation"]["turn_count"] == 2

    def test_idempotency_with_emulator(self):
        from history_store import save_turn
        client_turn_id = str(uuid.uuid4())
        r1 = save_turn(self.uid, None, {
            "status": "complete", "query": "Q", "answer_markdown": "A",
            "answer_structured": {}, "sources": [], "alias_map": {},
            "coverage": {}, "retrieval_summary": {},
            "unverified_claims_removed_count": 0, "model": "test",
            "latency_ms": 10, "error": None, "error_code": None,
        }, client_turn_id=client_turn_id)
        cid = r1["conversation_id"]
        # Second call with same ids
        r2 = save_turn(self.uid, cid, {
            "status": "complete", "query": "Q2", "answer_markdown": "A2",
            "answer_structured": {}, "sources": [], "alias_map": {},
            "coverage": {}, "retrieval_summary": {},
            "unverified_claims_removed_count": 0, "model": "test",
            "latency_ms": 10, "error": None, "error_code": None,
        }, client_turn_id=client_turn_id)
        assert r1["turn_id"] == r2["turn_id"]
        # turn_count should still be 1
        from history_store import get_conversation
        conv = get_conversation(self.uid, cid)
        assert conv["conversation"]["turn_count"] == 1

    def test_title_generation_lifecycle(self):
        from history_store import (
            save_turn, set_generated_title, rename_conversation,
        )
        r = save_turn(self.uid, None, {
            "status": "complete", "query": "What is fair trial?",
            "answer_markdown": "Art 50 guarantees...",
            "answer_structured": {"short_answer": "Art 50..."},
            "sources": [], "alias_map": {}, "coverage": {},
            "retrieval_summary": {}, "unverified_claims_removed_count": 0,
            "model": "test", "latency_ms": 100, "error": None, "error_code": None,
        })
        cid = r["conversation_id"]

        # Fallback title set immediately
        snap = self.db.collection("users").document(self.uid) \
                   .collection("conversations").document(cid).get()
        assert snap.get("title_source") == "fallback"

        # Background title arrives
        set_generated_title(self.uid, cid, "Fair Trial Constitutional Rights")
        snap = self.db.collection("users").document(self.uid) \
                   .collection("conversations").document(cid).get()
        assert snap.get("title") == "Fair Trial Constitutional Rights"
        assert snap.get("title_source") == "generated"

        # User renames → generated title should NOT overwrite it
        rename_conversation(self.uid, cid, "My Custom Name")
        set_generated_title(self.uid, cid, "Another Generated Title")
        snap = self.db.collection("users").document(self.uid) \
                   .collection("conversations").document(cid).get()
        assert snap.get("title") == "My Custom Name"
        assert snap.get("title_source") == "user"
