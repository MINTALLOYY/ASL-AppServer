from flask import Blueprint, jsonify, request
import logging
import sys

from config import db as config_db
from config import identified_labels as config_identified_labels
from config import session_modes as config_session_modes
from config import speaker_registry as config_speaker_registry

logger = logging.getLogger(__name__)

bp = Blueprint("speech", __name__)

def _get_shared_state():
    app_module = sys.modules.get("app")
    if app_module is None:
        return config_db, config_session_modes, config_identified_labels, config_speaker_registry
    return (
        getattr(app_module, "db", config_db),
        getattr(app_module, "session_modes", config_session_modes),
        getattr(app_module, "identified_labels", config_identified_labels),
        getattr(app_module, "speaker_registry", config_speaker_registry),
    )

@bp.post("/speech/finalize")
def speech_finalize():
    """
    Finalize a speech session by marking the conversation as completed in Firestore.
    
    Expected JSON payload:
        {"conversation_id": "abc123", "captions": [...]}
    
    Returns:
        JSON: {"status": "finalized", "conversation_id": "abc123", "speaker_map": {...}}
        Error 400 if conversation_id is missing.
        Error 500 if Firestore operation fails.
    """
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    db, session_modes, identified_labels, speaker_registry = _get_shared_state()
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    try:
        speaker_map = speaker_registry.get(conversation_id, {})
        if db:
            db.finalize_conversation(conversation_id)
        return jsonify({
            "status": "finalized",
            "conversation_id": conversation_id,
            "speaker_map": speaker_map,
        })
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

    Expected JSON payload:
        {
            "conversation_id": "abc123",
            "speakers": [
                {"label": "Speaker_0", "name": "Marcus"},
                {"label": "Speaker_1", "name": "Priya"}
            ]
        }

    Returns:
        JSON: {"status": "ok", "registered": <count>}
        Error 400 if conversation_id is missing or speakers is not a list.
    """
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    db, session_modes, identified_labels, speaker_registry = _get_shared_state()
    if not conversation_id or not isinstance(conversation_id, str):
        return jsonify({"error": "conversation_id is required"}), 400

    speakers = data.get("speakers", [])
    if not isinstance(speakers, list):
        return jsonify({"error": "speakers must be a list"}), 400

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
    return jsonify({"status": "ok", "registered": len(mapping)})

@bp.get("/speech/register_speakers")
def register_speakers_get():
    """
    Debug endpoint: return the current speaker mapping for a conversation.

    Query parameters:
        - conversation_id (required): the conversation to look up.

    Returns:
        JSON: {"conversation_id": "...", "speakers": {...}}
        Error 400 if conversation_id is missing.
    """
    conversation_id = request.args.get("conversation_id")
    db, session_modes, identified_labels, speaker_registry = _get_shared_state()
    if not conversation_id:
        return jsonify({"error": "conversation_id query parameter is required"}), 400

    mapping = speaker_registry.get(conversation_id, {})
    return jsonify({"conversation_id": conversation_id, "speakers": mapping})
