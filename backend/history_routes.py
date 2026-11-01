"""
Flask Blueprint: chat history read/manage endpoints.

Registered in app.py as: app.register_blueprint(history_bp)

All routes require @require_role("advocate","admin"); uid comes from g.uid
set by that decorator — never from the request body.
"""

from flask import Blueprint, g, jsonify, request

from auth import require_role
from history_store import (
    delete_conversation,
    get_conversation,
    list_conversations,
    regenerate_title,
    rename_conversation,
    set_pinned,
)

history_bp = Blueprint("history", __name__)

_MAX_LIMIT      = 50
_MAX_TURN_LIMIT = 100


# ── Validation helpers ────────────────────────────────────────────────────────

def _int_param(name: str, default: int, lo: int, hi: int) -> tuple[int, str | None]:
    raw = request.args.get(name, default)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default, None
    if not lo <= val <= hi:
        return default, f"'{name}' must be between {lo} and {hi}"
    return val, None


# ── GET /api/conversations ────────────────────────────────────────────────────

@history_bp.route("/api/conversations", methods=["GET"])
@require_role("advocate", "admin")
def list_convs():
    limit, err = _int_param("limit", 20, 1, _MAX_LIMIT)
    if err:
        return jsonify({"error": err}), 400

    cursor = request.args.get("cursor") or None

    try:
        result = list_conversations(g.uid, limit=limit, cursor=cursor)
        return jsonify(result), 200
    except Exception as exc:
        print(f"[HISTORY] list_conversations error: {exc}")
        return jsonify({"error": "Failed to list conversations"}), 500


# ── GET /api/conversations/<id> ───────────────────────────────────────────────

@history_bp.route("/api/conversations/<conv_id>", methods=["GET"])
@require_role("advocate", "admin")
def get_conv(conv_id: str):
    turn_limit, err = _int_param("turn_limit", 50, 1, _MAX_TURN_LIMIT)
    if err:
        return jsonify({"error": err}), 400

    before_index = None
    raw_bi = request.args.get("before_index")
    if raw_bi is not None:
        try:
            before_index = int(raw_bi)
        except (TypeError, ValueError):
            return jsonify({"error": "'before_index' must be an integer"}), 400

    try:
        result = get_conversation(
            g.uid, conv_id,
            turn_limit=turn_limit,
            before_index=before_index,
        )
        return jsonify(result), 200
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as exc:
        print(f"[HISTORY] get_conversation error: {exc}")
        return jsonify({"error": "Failed to fetch conversation"}), 500


# ── PATCH /api/conversations/<id> ─────────────────────────────────────────────

@history_bp.route("/api/conversations/<conv_id>", methods=["PATCH"])
@require_role("advocate", "admin")
def patch_conv(conv_id: str):
    body = request.get_json(silent=True) or {}

    updated = False
    try:
        if "title" in body:
            title = str(body["title"]).strip()
            if not title:
                return jsonify({"error": "title must be a non-empty string"}), 400
            if len(title) > 120:
                return jsonify({"error": "title must be <= 120 characters"}), 400
            rename_conversation(g.uid, conv_id, title)
            updated = True

        if "pinned" in body:
            if not isinstance(body["pinned"], bool):
                return jsonify({"error": "'pinned' must be a boolean"}), 400
            set_pinned(g.uid, conv_id, body["pinned"])
            updated = True

        if not updated:
            return jsonify({"error": "Nothing to update; send 'title' or 'pinned'"}), 400

        return jsonify({"ok": True}), 200

    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as exc:
        print(f"[HISTORY] patch_conv error: {exc}")
        return jsonify({"error": "Failed to update conversation"}), 500


# ── DELETE /api/conversations/<id> ───────────────────────────────────────────

@history_bp.route("/api/conversations/<conv_id>", methods=["DELETE"])
@require_role("advocate", "admin")
def delete_conv(conv_id: str):
    try:
        # Verify ownership first (raises LookupError if not owned by this user)
        from history_store import get_or_create_conversation
        get_or_create_conversation(g.uid, conv_id, "")

        delete_conversation(g.uid, conv_id)
        return jsonify({"ok": True}), 200
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as exc:
        print(f"[HISTORY] delete_conv error: {exc}")
        return jsonify({"error": "Failed to delete conversation"}), 500


# ── POST /api/conversations/<id>/regenerate-title ─────────────────────────────

@history_bp.route("/api/conversations/<conv_id>/regenerate-title", methods=["POST"])
@require_role("advocate", "admin")
def regen_title(conv_id: str):
    body  = request.get_json(silent=True) or {}
    force = bool(body.get("force", False))

    try:
        regenerate_title(g.uid, conv_id, force=force)
        return jsonify({"ok": True, "message": "Title regeneration queued"}), 202
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as exc:
        print(f"[HISTORY] regen_title error: {exc}")
        return jsonify({"error": "Failed to regenerate title"}), 500
