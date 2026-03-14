import json
import os
import tempfile
import threading
import time
from typing import Optional
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_sock import Sock
from flask_socketio import SocketIO, emit
import logging
import sys

from firebase.db import FirestoreDB
from speech.chirp_stream import ChirpStreamer, speaker_label_from_result
from asl.asl_inference import transcribe_video
from asl.predictor import ASLPredictor

load_dotenv()

# Configure structured logging to stdout (Render + Gunicorn)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# When running under Gunicorn, reuse its handlers so logs show up
gunicorn_error_logger = logging.getLogger("gunicorn.error")
if gunicorn_error_logger.handlers:
    root_logger.handlers = gunicorn_error_logger.handlers
    root_logger.setLevel(gunicorn_error_logger.level)

logger = logging.getLogger(__name__)

# Startup diagnostics for Render/Gunicorn
logger.info(
    "Startup config: PORT=%s WEB_CONCURRENCY=%s PYTHONUNBUFFERED=%s GUNICORN_CMD_ARGS=%s SERVER_SOFTWARE=%s",
    os.environ.get("PORT"),
    os.environ.get("WEB_CONCURRENCY"),
    os.environ.get("PYTHONUNBUFFERED"),
    os.environ.get("GUNICORN_CMD_ARGS"),
    os.environ.get("SERVER_SOFTWARE"),
)

app = Flask(__name__)
sock = Sock(app)
socketio_async_mode = os.environ.get("SOCKETIO_ASYNC_MODE", "threading")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=socketio_async_mode)
logger.info("Socket.IO initialized with async_mode=%s", socketio.async_mode)

# FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID")
# db = FirestoreDB(project_id=FIREBASE_PROJECT_ID)
db = False

# --- ASL real-time predictor ---
_asl_model_path = os.path.join(os.path.dirname(__file__), "asl", "model.h5")
asl_predictor = None
try:
    if os.path.exists(_asl_model_path):
        asl_predictor = ASLPredictor(model_path=_asl_model_path)
        logger.info("ASL predictor loaded from %s", _asl_model_path)
    else:
        logger.warning("ASL model not found at %s — /asl/ws will return errors until model is placed", _asl_model_path)
except Exception:
    logger.exception("Failed to load ASL predictor")


@socketio.on("asl_frame")
def handle_asl_frame(data):
    """
    Receives one JPEG frame from the client and emits a prediction when available.
    Accepts frame payload as raw bytes, bytearray, list[int], or base64 string.
    """
    import base64

    if asl_predictor is None:
        emit("asl_error", {"message": "ASL model not loaded on server"})
        return

    if not isinstance(data, dict):
        return

    frame_payload = data.get("frame")
    if frame_payload is None:
        return

    frame_bytes = None
    if isinstance(frame_payload, (bytes, bytearray)):
        frame_bytes = bytes(frame_payload)
    elif isinstance(frame_payload, list):
        try:
            frame_bytes = bytes(frame_payload)
        except Exception:
            return
    elif isinstance(frame_payload, str):
        # Support base64 payloads for clients that cannot emit binary attachments.
        try:
            frame_bytes = base64.b64decode(frame_payload)
        except Exception:
            return

    if not frame_bytes:
        return

    word = asl_predictor.process_frame(frame_bytes)
    if word:
        emit("asl_result", {"word": word})

creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if creds and creds.strip().startswith("{"):
    try:
        tmp_json = os.path.join(tempfile.gettempdir(), "gcp_creds.json")
        with open(tmp_json, "w") as f:
            f.write(creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_json
    except Exception:
        pass


@app.get("/health")
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON: {"status": "ok"} if the server is running.
    """
    if creds:
        return jsonify({"status": "ok and creds"})
    return jsonify({"status": "server running but no credentials"}), 500


@app.get("/ws-info")
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


@sock.route("/ws/echo")
def ws_echo(ws):
    """
    Simple WebSocket echo test endpoint.
    Connect to wss://yourserver/ws/echo and send any message to get it echoed back.
    """
    logger.info("WebSocket ECHO connection opened")
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                logger.info("WebSocket ECHO client disconnected")
                break
            logger.info("WebSocket ECHO received: %s", msg[:100] if msg else None)
            ws.send(f"echo: {msg}")
    except Exception as e:
        logger.exception("WebSocket ECHO error: %s", e)
    finally:
        logger.info("WebSocket ECHO connection closed")


@app.post("/speech/finalize")
def speech_finalize():
    """
    Finalize a speech session by marking the conversation as completed in Firestore.
    
    Expected JSON payload:
        {"conversation_id": "abc123"}
    
    Returns:
        JSON: {"status": "finalized", "conversation_id": "abc123"}
        Error 400 if conversation_id is missing.
        Error 500 if Firestore operation fails.
    """
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    try:
        if db:
            db.finalize_conversation(conversation_id)
        return jsonify({"status": "finalized", "conversation_id": conversation_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/asl/transcribe")
def asl_transcribe():
    """
    Transcribe a recorded ASL video into text.
    
    Expects multipart/form-data:
        - video: video file (e.g., .mp4)
        - conversation_id (optional): Firestore conversation ID to save the result
    
    Returns:
        JSON: {"text": "Transcribed text"}
        Error 400 if video file is missing.
        Error 500 if transcription fails.
    """
    conversation_id = request.form.get("conversation_id")
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "video file is required (form field 'video')"}), 400

    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        # Run ASL inference (stubbed for now)
        text = transcribe_video(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    # Save result to Firestore if conversation_id provided and db is initialized
    try:
        if conversation_id and text and db:
            db.save_message(conversation_id=conversation_id, text=text, source="asl")
    except Exception:
        pass

    return jsonify({"text": text})


@sock.route("/speech/ws")
def speech_ws(ws):
    """
    WebSocket endpoint for live speech-to-text streaming.
    
    Query parameters:
        - conversation_id (optional): Firestore conversation ID to save transcripts
    
    Expected messages from client (JSON):
        - {"event": "audio_chunk", "data": "<base64 audio>", "conversation_id": "abc123"}
        - {"event": "set_conversation", "conversation_id": "abc123"}
        - {"event": "end"} or {"event": "finish"} or {"event": "close"}
    
    Server responses (JSON):
        - {"event": "final_transcript", "text": "...", "speaker": "Speaker A"}
    
    Audio format: base64-encoded 16 kHz, 16-bit mono PCM (LINEAR16).
    """
    # Get conversation_id from query params (optional)
    conversation_id: Optional[str] = request.args.get("conversation_id")
    # Initialize Google Speech streaming client (with restart capability)
    streamer_state = {"streamer": ChirpStreamer(), "active": True}

    # Log when a new WebSocket connection opens
    try:
        remote = request.remote_addr
    except Exception:
        remote = None
    logger.info("WebSocket connection opened. conversation_id=%s, remote=%s", conversation_id, remote)

    # Background thread: consume responses from Google Speech and send to client
    def consume_responses():
        try:
            response_count = 0
            logger.info("consume_responses started. Waiting for Google Speech responses...")
            for response in streamer_state["streamer"].responses():
                response_count += 1
                try:
                    results_len = len(response.results)
                except Exception:
                    results_len = "unknown"
                if response_count <= 3 or response_count % 20 == 0 or results_len == 0:
                    logger.info("Received response #%s. Results count: %s", response_count, results_len)
                for result in response.results:
                    if result.is_final:
                        try:
                            alt = result.alternatives[0]
                            transcript = (alt.transcript or "").strip()
                        except Exception:
                            transcript = ""
                        speaker = speaker_label_from_result(result)
                        if transcript:
                            try:
                                message = json.dumps({
                                    "event": "final_transcript",
                                    "text": transcript,
                                    "speaker": speaker
                                })
                                logger.debug("Sending message: %s", message)
                                ws.send(message)
                            except Exception as e:
                                logger.error("WebSocket send error: %s", e)
                                logger.error(traceback.format_exc())
                                streamer_state["streamer"].finish()
                                return
                            try:
                                if conversation_id and db:
                                    db.save_message(conversation_id=conversation_id, text=transcript, source="speech", speaker=speaker)
                            except Exception as e:
                                logger.error("Firestore save error: %s", e)
            logger.debug("consume_responses finished. Total responses: %s", response_count)
        except Exception as e:
            logger.error("Error in consume_responses: %s", e)
            logger.error(traceback.format_exc())
            # If the stream errored due to audio timeout, mark inactive so we can restart on next audio
            if "Audio Timeout" in str(e) or "Audio Timeout Error" in str(e):
                streamer_state["active"] = False

    # Start response consumer in background
    t = threading.Thread(target=consume_responses, daemon=True)
    t.start()
    logger.info("consume_responses thread started. thread=%s alive=%s", t.name, t.is_alive())

    # Main loop: receive audio chunks from Flutter client
    logger.info("Entering WebSocket main loop. Waiting for messages from client...")
    msg_count = 0
    last_idle_log_at = 0.0
    receive_timeout_sec = 15
    try:
        while True:
            msg_count += 1
            if msg_count <= 3:
                logger.info("ws.receive() call #%s - waiting for client message...", msg_count)
            try:
                msg = ws.receive(timeout=receive_timeout_sec)
            except TimeoutError:
                now = time.time()
                if now - last_idle_log_at >= 30:
                    logger.info(
                        "WebSocket idle for %ss waiting for client message (conversation_id=%s)",
                        receive_timeout_sec,
                        conversation_id,
                    )
                    last_idle_log_at = now
                continue
            if msg_count <= 5:
                logger.info("ws.receive() returned. msg_count=%s msg_type=%s msg_len=%s", 
                            msg_count, type(msg).__name__, len(msg) if msg else 0)
            if msg is None:
                logger.info("WebSocket receive returned None — client disconnected")
                break
            try:
                data = json.loads(msg)
            except Exception:
                try:
                    logger.debug("Received non-JSON message: %s", repr(msg)[:200])
                except Exception:
                    pass
                continue

            event = data.get("event")
            if event == "audio_chunk":
                # Decode base64 audio and feed to streamer
                b64 = data.get("data")
                try:
                    b64_len = len(b64) if b64 is not None else 0
                except Exception:
                    b64_len = "unknown"
                audio_chunk_count = streamer_state.get("audio_chunk_count", 0) + 1
                streamer_state["audio_chunk_count"] = audio_chunk_count
                if audio_chunk_count <= 3 or audio_chunk_count % 25 == 0:
                    logger.info(
                        "event=audio_chunk conversation_id=%s chunk_count=%s b64_len=%s streamer_active=%s",
                        conversation_id,
                        audio_chunk_count,
                        b64_len,
                        streamer_state["active"],
                    )
                if not conversation_id:
                    cid = data.get("conversation_id")
                    if cid:
                        conversation_id = cid
                # If previous stream errored (timeout), restart a new stream and consumer
                if not streamer_state["active"]:
                    logger.warning("Restarting speech stream due to previous timeout...")
                    try:
                        # Finish any old streamer
                        try:
                            streamer_state["streamer"].finish()
                        except Exception:
                            pass
                        # Replace streamer and restart consumer thread
                        streamer_state["streamer"] = ChirpStreamer()
                        streamer_state["active"] = True
                        t = threading.Thread(target=consume_responses, daemon=True)
                        t.start()
                    except Exception as e:
                        logger.exception("Failed to restart speech stream: %s", e)
                streamer_state["streamer"].add_audio_base64(b64 or "")
            elif event in ("end", "finish", "close"):
                # End the session
                if not conversation_id:
                    cid = data.get("conversation_id")
                    if cid:
                        conversation_id = cid
                logger.info("event=%s — finishing streamer for conversation_id=%s", event, conversation_id)
                streamer_state["streamer"].finish()
                break
            elif event == "set_conversation":
                # Set or update conversation_id mid-session
                cid = data.get("conversation_id")
                if cid:
                    conversation_id = cid
                    logger.debug("set_conversation updated conversation_id=%s", conversation_id)
            else:
                # Ignore unknown events
                logger.debug("Unknown websocket event received: %s", event)
    except Exception:
        logger.exception("Exception in WebSocket main loop")
    finally:
        # Clean up streaming resources
        try:
            streamer_state["streamer"].finish()
        except Exception:
            pass
        try:
            t.join(timeout=2)
        except Exception:
            pass
        logger.debug("WebSocket handler cleanup complete for conversation_id=%s", conversation_id)


@sock.route("/asl/ws")
def asl_ws(ws):
    """
    WebSocket endpoint for real-time ASL sign language recognition.

    Client sends JSON messages:
        {"event": "frame", "data": "<base64 JPEG>"}
        {"event": "end"}

    Server responds with JSON:
        {"event": "asl_result", "word": "hello"}
    """
    import base64

    logger.info("ASL WebSocket connection opened")

    if asl_predictor is None:
        ws.send(json.dumps({"event": "error", "message": "ASL model not loaded on server"}))
        logger.error("ASL WebSocket rejected — no model loaded")
        return

    try:
        while True:
            msg = ws.receive(timeout=30)
            if msg is None:
                logger.info("ASL WebSocket client disconnected")
                break

            try:
                data = json.loads(msg)
            except Exception:
                continue

            event = data.get("event")

            if event == "frame":
                b64 = data.get("data")
                if not b64:
                    continue
                try:
                    frame_bytes = base64.b64decode(b64)
                except Exception:
                    continue

                word = asl_predictor.process_frame(frame_bytes)
                if word:
                    ws.send(json.dumps({"event": "asl_result", "word": word}))

            elif event in ("end", "close", "finish"):
                logger.info("ASL WebSocket session ended by client")
                break

    except Exception:
        logger.exception("Error in ASL WebSocket handler")
    finally:
        logger.info("ASL WebSocket connection closed")


# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
