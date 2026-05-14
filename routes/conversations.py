import logging
import sys

from flask import Blueprint, jsonify, request

import config
from utils import _get_accessor_uuid, normalize_uuid, validate_uuid

logger = logging.getLogger(__name__)

bp = Blueprint("conversations", __name__)


def _get_db():
    app_module = sys.modules.get("app")
    if app_module is not None and hasattr(app_module, "db"):
        return app_module.db
    return config.db


def _resolve_request_uuid():
    raw_uuid = _get_accessor_uuid(request)
    if raw_uuid is None:
        if config.ENFORCE_CONVERSATION_UUID:
            return None, jsonify({"error": "missing_conversation_uuid"}), 400
        return None, None, None
    if not validate_uuid(raw_uuid):
        return None, jsonify({"error": "invalid_uuid"}), 400
    return normalize_uuid(raw_uuid), None, None


@bp.post("/conversations")
def conversations_create():
    """
    Create a new captioning conversation.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id") or None
    owner_hint = data.get("owner_hint") or None
    conversation_uuid = data.get("conversation_uuid") or _get_accessor_uuid(request)
    if conversation_uuid is not None and not validate_uuid(conversation_uuid):
        return jsonify({"error": "invalid_uuid"}), 400
    if conversation_uuid is None and config.ENFORCE_CONVERSATION_UUID:
        return jsonify({"error": "missing_conversation_uuid"}), 400
    try:
        created = db.create_conversation(
            conversation_id=conversation_id,
            conversation_uuid=normalize_uuid(conversation_uuid) if conversation_uuid else None,
            owner_hint=owner_hint,
        )
        return jsonify({
            "conversation_id": created["conversation_id"],
            "conversation_uuid": created["conversation_uuid"],
            "status": "active",
        }), 201
    except Exception as e:
        logger.exception("conversations_create error: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.get("/conversations")
def conversations_list():
    """
    List captioning conversations, scoped by the account identifier when provided.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50

    accessor_uuid, error_response, status_code = _resolve_request_uuid()
    if error_response is not None:
        return error_response, status_code

    try:
        if accessor_uuid:
            convs = db.list_conversations(limit=limit, conversation_uuid=accessor_uuid)
        else:
            convs = db.list_conversations(limit=limit)
        return jsonify({"conversations": convs})
    except Exception as e:
        logger.exception("conversations_list error: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.get("/conversations/<conversation_id>")
def conversations_get(conversation_id):
    """
    Retrieve a conversation and all of its messages in chronological order.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    accessor_uuid, error_response, status_code = _resolve_request_uuid()
    if error_response is not None:
        return error_response, status_code

    try:
        conv = db.get_conversation(conversation_id, conversation_uuid=accessor_uuid)
        if conv is None:
            return jsonify({"error": "Conversation not found"}), 404
        conv["messages"] = db.get_messages(conversation_id, conversation_uuid=accessor_uuid)
        return jsonify(conv)
    except PermissionError:
        return jsonify({"error": "access_denied"}), 403
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as e:
        logger.exception("conversations_get error: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.get("/conversations/<conversation_id>/messages")
def conversations_messages(conversation_id):
    """
    Retrieve messages for a conversation in chronological order.
    """
    db = _get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    accessor_uuid, error_response, status_code = _resolve_request_uuid()
    if error_response is not None:
        return error_response, status_code

    try:
        messages = db.get_messages(conversation_id, conversation_uuid=accessor_uuid)
        return jsonify({"conversation_id": conversation_id, "messages": messages})
    except PermissionError:
        return jsonify({"error": "access_denied"}), 403
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as e:
        logger.exception("conversations_messages error: %s", e)
        return jsonify({"error": str(e)}), 500
