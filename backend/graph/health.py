"""Flask blueprint: GET /graph/health

Returns the graph health dict. When GRAPH_ENABLED=false, returns
{"enabled": false} with HTTP 200 and does not connect to Neo4j.
"""
from __future__ import annotations

from flask import Blueprint, jsonify

from . import client as _client_module

graph_health_bp = Blueprint("graph_health", __name__)


@graph_health_bp.route("/graph/health", methods=["GET"])
def graph_health():
    payload = _client_module.health()
    return jsonify(payload), 200
