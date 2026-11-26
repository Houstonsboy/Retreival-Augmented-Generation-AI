import os
import functools

import firebase_admin
from firebase_admin import credentials, auth as fb_auth, firestore as fb_firestore
from flask import request, jsonify, g

_db = None


_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


_BUCKET = "legalai-scc.firebasestorage.app"
_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "legalai-scc")


def init_firebase_admin():
    """Initialize Firebase Admin SDK — key file if present, else Application Default Credentials."""
    global _db
    if firebase_admin._apps:
        _db = fb_firestore.client()
        return
    sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if sa_path and not os.path.isabs(sa_path):
        sa_path = os.path.join(_BACKEND_DIR, sa_path)
    try:
        if sa_path and os.path.exists(sa_path):
            cred = credentials.Certificate(sa_path)
            src = f"key file {sa_path}"
        else:
            cred = credentials.ApplicationDefault()
            src = "Application Default Credentials"
        firebase_admin.initialize_app(cred, {"projectId": _PROJECT, "storageBucket": _BUCKET})
        _db = fb_firestore.client()
        print(f"Firebase Admin initialised via {src}")
    except Exception as e:
        print(f"WARNING: Firebase Admin init failed ({e}) — protected routes will return 503")


def _get_db():
    global _db
    if _db is None:
        _db = fb_firestore.client()
    return _db


def _verify_and_load(token):
    """Return ((uid, profile_dict), None) or (None, error_response_tuple)."""
    if not firebase_admin._apps:
        return None, (jsonify({"error": "Firebase Admin not configured on server"}), 503)

    try:
        decoded = fb_auth.verify_id_token(token, clock_skew_seconds=60)
    except fb_auth.ExpiredIdTokenError:
        return None, (jsonify({"error": "Token expired"}), 401)
    except Exception:
        return None, (jsonify({"error": "Invalid token"}), 401)

    uid = decoded["uid"]
    doc = _get_db().collection("users").document(uid).get()
    if not doc.exists:
        return None, (jsonify({"error": "Profile incomplete", "code": "PROFILE_REQUIRED"}), 403)

    return (uid, doc.to_dict()), None


def _extract_bearer():
    """Return (token_string, None) or (None, error_response_tuple)."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing Authorization header"}), 401)
    return header[7:].strip(), None


def require_auth(f):
    """Verify token and load Firestore profile; sets g.uid, g.profile, g.user_type."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token, err = _extract_bearer()
        if err:
            return err
        result, err = _verify_and_load(token)
        if err:
            return err
        uid, profile = result
        g.uid = uid
        g.profile = profile
        g.user_type = profile.get("type", "")
        return f(*args, **kwargs)
    return decorated


def require_role(*roles):
    """Verify token, load profile, and enforce role membership.

    Usage:
        @require_role("admin")
        @require_role("advocate", "admin")
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            token, err = _extract_bearer()
            if err:
                return err
            result, err = _verify_and_load(token)
            if err:
                return err
            uid, profile = result
            user_type = profile.get("type", "")
            if user_type not in roles:
                return jsonify({"error": f"Forbidden: requires role {' or '.join(roles)}"}), 403
            g.uid = uid
            g.profile = profile
            g.user_type = user_type
            return f(*args, **kwargs)
        return decorated
    return decorator
