from flask import Blueprint, jsonify, request
import logging
import sys

import config
from utils import _get_accessor_uuid, normalize_uuid, validate_uuid

logger = logging.getLogger(__name__)

bp = Blueprint("speech", __name__)


def _get_shared_state():
    app_module = sys.modules.get("app")
    if app_module is None:
        return config.db, config.session_modes, config.identified_labels, config.speaker_registry
    return (
        getattr(app_module, "db", config.db),
        getattr(app_module, "session_modes", config.session_modes),
        getattr(app_module, "identified_labels", config.identified_labels),
        getattr(app_module, "speaker_registry", config.speaker_registry),
    )


def _resolve_request_uuid():
    raw_uuid = _get_accessor_uuid(request)
    if raw_uuid is None:
        if config.ENFORCE_CONVERSATION_UUID:
            return None, jsonify({"error": "missing_conversation_uuid"}), 400
        return None, None, None
    if not validate_uuid(raw_uuid):
        return None, jsonify({"error": "invalid_uuid"}), 400
    return normalize_uuid(raw_uuid), None, None


def _validate_conversation_access(db, conversation_id: str, conversation_uuid: str):
    if not db or not conversation_id:
        return
    if conversation_uuid is None:
        if config.ENFORCE_CONVERSATION_UUID:
            raise PermissionError("access denied")
        return
    db.get_conversation(conversation_id, conversation_uuid=conversation_uuid)


@bp.post("/speech/finalize")
def speech_finalize():
    """
    Finalize a speech session by marking the conversation as completed in Firestore.
    """
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    db, session_modes, identified_labels, speaker_registry = _get_shared_state()
    accessor_uuid = data.get("conversation_uuid") or _get_accessor_uuid(request)
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    if accessor_uuid is None and config.ENFORCE_CONVERSATION_UUID:
        return jsonify({"error": "missing_conversation_uuid"}), 400
    if accessor_uuid is not None and not validate_uuid(accessor_uuid):
        return jsonify({"error": "invalid_uuid"}), 400
    accessor_uuid = normalize_uuid(accessor_uuid) if accessor_uuid else None
    try:
        if db:
            db.finalize_conversation(conversation_id, conversation_uuid=accessor_uuid)
        speaker_map = speaker_registry.get(conversation_id, {})
        return jsonify({
            "status": "finalized",
            "conversation_id": conversation_id,
            "conversation_uuid": accessor_uuid,
            "speaker_map": speaker_map,
        })
    except PermissionError:
        return jsonify({"error": "access_denied"}), 403
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up all session state
        speaker_registry.pop(conversation_id, None)
        session_modes.pop(conversation_id, None)
        identified_labels.pop(conversation_id, None)


@bp.post("/speech/register_speakers")
def register_speakers_post():
    """
    Register speaker name mappings for a conversation.
    """
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    db, session_modes, identified_labels, speaker_registry = _get_shared_state()
    accessor_uuid = data.get("conversation_uuid") or _get_accessor_uuid(request)
    if not conversation_id or not isinstance(conversation_id, str):
        return jsonify({"error": "conversation_id is required"}), 400
    if accessor_uuid is None and config.ENFORCE_CONVERSATION_UUID:
        return jsonify({"error": "missing_conversation_uuid"}), 400
    if accessor_uuid is not None and not validate_uuid(accessor_uuid):
        return jsonify({"error": "invalid_uuid"}), 400
    accessor_uuid = normalize_uuid(accessor_uuid) if accessor_uuid else None

    speakers = data.get("speakers", [])
    if not isinstance(speakers, list):
        return jsonify({"error": "speakers must be a list"}), 400

    try:
        _validate_conversation_access(db, conversation_id, accessor_uuid)
    except PermissionError:
        return jsonify({"error": "access_denied"}), 403
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404

    mapping: dict[str, str] = {}
    for entry in speakers:
        if isinstance(entry, dict):
            label = entry.get("label")
            name = entry.get("name")
            if isinstance(label, str) and isinstance(name, str) and label and name:
                mapping[label] = name

    speaker_registry[conversation_id] = mapping
    logger.info(
        "Registered %d speaker(s) for conversation_id=%s: %s",
        len(mapping), conversation_id, mapping,
    )
    return jsonify({
        "status": "ok",
        "registered": len(mapping),
        "conversation_uuid": accessor_uuid,
    })


@bp.get("/speech/register_speakers")
def register_speakers_get():
    """
    Debug endpoint: return the current speaker mapping for a conversation.
    """
    conversation_id = request.args.get("conversation_id")
    db, session_modes, identified_labels, speaker_registry = _get_shared_state()
    accessor_uuid = _get_accessor_uuid(request)
    if not conversation_id:
        return jsonify({"error": "conversation_id query parameter is required"}), 400
    if accessor_uuid is None and config.ENFORCE_CONVERSATION_UUID:
        return jsonify({"error": "missing_conversation_uuid"}), 400
    if accessor_uuid is not None and not validate_uuid(accessor_uuid):
        return jsonify({"error": "invalid_uuid"}), 400
    accessor_uuid = normalize_uuid(accessor_uuid) if accessor_uuid else None

    try:
        _validate_conversation_access(db, conversation_id, accessor_uuid)
    except PermissionError:
        return jsonify({"error": "access_denied"}), 403
    except LookupError:
        return jsonify({"error": "Conversation not found"}), 404

    mapping = speaker_registry.get(conversation_id, {})
    return jsonify({
        "conversation_id": conversation_id,
        "conversation_uuid": accessor_uuid,
        "speakers": mapping,
    })
