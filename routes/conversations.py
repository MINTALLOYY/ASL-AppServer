import logging
import sys

from flask import Blueprint, jsonify, request
from config import db as config_db

logger = logging.getLogger(__name__)

bp = Blueprint("conversations", __name__)


def _get_db():
    app_module = sys.modules.get("app")
    if app_module is not None and hasattr(app_module, "db"):
        return app_module.db
    return config_db

@bp.post("/conversations")
def conversations_create():
    """
    Create a new captioning conversation.

    Optional JSON payload:
        {"conversation_id": "my-custom-id"}

    If conversation_id is omitted Firestore auto-generates one.

    Returns:
        JSON: {"conversation_id": "...", "status": "active"}  HTTP 201
        Error 503 if Firestore is unavailable.
        Error 500 on unexpected errors.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id") or None
    try:
        cid = db.create_conversation(conversation_id=conversation_id)
        return jsonify({"conversation_id": cid, "status": "active"}), 201
    except Exception as e:
        logger.exception("conversations_create error: %s", e)
        return jsonify({"error": str(e)}), 500

@bp.get("/conversations")
def conversations_list():
    """
    List captioning conversations, most recently updated first.

    Query parameters:
        - limit (optional, default 50, max 100): number of conversations to return.

    Returns:
        JSON: {"conversations": [...]}
        Error 503 if Firestore is unavailable.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    try:
        convs = db.list_conversations(limit=limit)
        return jsonify({"conversations": convs})
    except Exception as e:
        logger.exception("conversations_list error: %s", e)
        return jsonify({"error": str(e)}), 500

@bp.get("/conversations/<conversation_id>")
def conversations_get(conversation_id):
    """
    Retrieve a conversation and all of its messages in chronological order.

    Path parameters:
        - conversation_id: Firestore document ID.

    Returns:
        JSON: {conversation metadata} + "messages": [...]
        Error 404 if the conversation does not exist.
        Error 503 if Firestore is unavailable.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503
    try:
        conv = db.get_conversation(conversation_id)
        if conv is None:
            return jsonify({"error": "Conversation not found"}), 404
        conv["messages"] = db.get_messages(conversation_id)
        return jsonify(conv)
    except Exception as e:
        logger.exception("conversations_get error: %s", e)
        return jsonify({"error": str(e)}), 500

@bp.get("/conversations/<conversation_id>/messages")
def conversations_messages(conversation_id):
    """
    Retrieve messages for a conversation in chronological order.

    Path parameters:
        - conversation_id: Firestore document ID.

    Returns:
        JSON: {"conversation_id": "...", "messages": [...]}
        Error 503 if Firestore is unavailable.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503
    try:
        messages = db.get_messages(conversation_id)
        return jsonify({"conversation_id": conversation_id, "messages": messages})
    except Exception as e:
        logger.exception("conversations_messages error: %s", e)
        return jsonify({"error": str(e)}), 500
