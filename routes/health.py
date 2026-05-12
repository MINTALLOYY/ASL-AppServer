import logging
from flask import Blueprint, jsonify, request
from config import creds, db, FIREBASE_PROJECT_ID

logger = logging.getLogger(__name__)

bp = Blueprint("health", __name__)

@bp.get("/health")
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON: {"status": "ok"} if the server is running.
    """
    try:
        remote = request.remote_addr
    except Exception:
        remote = None
    logger.info("Health check requested. remote=%s, creds_present=%s", remote, bool(creds))
    if creds:
        return jsonify({
            "status": "ok and creds",
            "firestore_project_id": FIREBASE_PROJECT_ID,
            "db_available": bool(db),
        })
    return jsonify({"status": "server running but no credentials"}), 500

@bp.get("/ws-info")
def ws_info():
    """
    Returns WebSocket connection info for debugging.
    """
    host = request.host
    # Build the correct WebSocket URL (no port for production, wss for https)
    if request.is_secure or "onrender.com" in host:
        ws_scheme = "wss"
    else:
        ws_scheme = "ws"
    
    # Remove any port from host for production
    if "onrender.com" in host:
        host = host.split(":")[0]  # Strip port if present
    
    return jsonify({
        "ws_echo_url": f"{ws_scheme}://{host}/ws/echo",
        "ws_speech_url": f"{ws_scheme}://{host}/speech/ws",
        "request_host": request.host,
        "is_secure": request.is_secure,
    })

@bp.get("/ws-hello")
def ws_hello():
    """
    Just testing if jsonify or if this app.py works
    """
    logger.info("ws_hello endpoint hit")
    return jsonify({
        "hello" : "hi"
    })
