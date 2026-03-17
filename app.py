import base64
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
import logging
import sys

from firebase.db import FirestoreDB
from speech.chirp_stream import ChirpStreamer, speaker_label_from_result

from asl.asl_inference import transcribe_video

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

# FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID")
# db = FirestoreDB(project_id=FIREBASE_PROJECT_ID)
db = False

# In-memory speaker registration store (MVP – not persisted across restarts).
# Keys: conversation_id, Values: dict mapping diarization label to participant name.
speaker_registry: dict[str, dict[str, str]] = {}

# Session mode state machine for speaker identification flow.
# Keys: conversation_id, Values: 'identifying' | 'captioning'
session_modes: dict[str, str] = {}

# Speaker labels already reported during identification phase.
# Keys: conversation_id, Values: set of labels (e.g. {'Speaker_1', 'Speaker_2'})
identified_labels: dict[str, set] = {}

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
    try:
        remote = request.remote_addr
    except Exception:
        remote = None
    logger.info("Health check requested. remote=%s, creds_present=%s", remote, bool(creds))
    # Ensure at least one unbuffered output for quick terminal verification
    print(f"Health check requested. remote={remote}, creds_present={bool(creds)}", flush=True)
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
        {"conversation_id": "abc123", "captions": [...]}
    
    Returns:
        JSON: {"status": "finalized", "conversation_id": "abc123", "speaker_map": {...}}
        Error 400 if conversation_id is missing.
        Error 500 if Firestore operation fails.
    """
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
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


@app.post("/speech/register_speakers")
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


@app.get("/speech/register_speakers")
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
    if not conversation_id:
        return jsonify({"error": "conversation_id query parameter is required"}), 400

    mapping = speaker_registry.get(conversation_id, {})
    return jsonify({"conversation_id": conversation_id, "speakers": mapping})



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
    WebSocket endpoint for live speech-to-text streaming with speaker identification.
    
    Query parameters:
        - conversation_id (optional): Firestore conversation ID to save transcripts
        - mode (optional): 'identifying' or 'captioning' (default: 'captioning')
    
    Expected messages from client (JSON):
        - {"event": "audio_chunk", "data": "<base64 audio>", "conversation_id": "abc123"}
        - {"event": "set_conversation", "conversation_id": "abc123"}
        - {"event": "begin_captioning"}  — switches mode from identifying to captioning
        - {"event": "end"} or {"event": "finish"} or {"event": "close"}
    
    Server responses (JSON):
        - {"event": "speaker_detected", "label": "Speaker_1"}  (identifying mode only)
        - {"event": "captioning_started"}  (after begin_captioning)
        - {"event": "final_transcript", "text": "...", "speaker": "Speaker A"}  (captioning mode)
    
    Audio format: base64-encoded 16 kHz, 16-bit mono PCM (LINEAR16).
    """
    # Get conversation_id from query params (optional)
    conversation_id: Optional[str] = request.args.get("conversation_id")
    # Mode state machine: 'identifying' or 'captioning'
    initial_mode = request.args.get("mode", "captioning")
    try:
        num_speakers = int(request.args.get("num_speakers", 2))
    except Exception:
        num_speakers = 2
    num_speakers = max(2, min(6, num_speakers))
    if conversation_id:
        session_modes[conversation_id] = initial_mode
        identified_labels[conversation_id] = set()
    # Initialize Google Speech streaming client (with restart capability)
    streamer_state = {"streamer": ChirpStreamer(diarization_speaker_count=num_speakers), "active": True}

    # Log when a new WebSocket connection opens
    try:
        remote = request.remote_addr
    except Exception:
        remote = None
    logger.info(
        "WebSocket connection opened. conversation_id=%s, remote=%s, mode=%s, num_speakers=%s",
        conversation_id, remote, initial_mode, num_speakers,
    )

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
                        # ── Identifying mode: emit speaker_detected, skip transcript ──
                        current_mode = session_modes.get(conversation_id, "captioning") if conversation_id else "captioning"
                        if current_mode == "identifying":
                            try:
                                alt = result.alternatives[0]
                                transcript = (alt.transcript or "").strip()
                                if alt.words:
                                    # Log all word-level speaker tags for debugging
                                    word_tags = [(w.word, int(getattr(w, "speaker_tag", 0) or 0)) for w in alt.words]
                                    tag_counts = {}
                                    for word in alt.words:
                                        tag = int(getattr(word, "speaker_tag", 0) or 0)
                                        if tag > 0:
                                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                                    logger.info(
                                        "IDENTIFY result: conversation_id=%s transcript='%s' word_tags=%s tag_counts=%s seen=%s",
                                        conversation_id, transcript, word_tags, tag_counts,
                                        identified_labels.get(conversation_id, set()),
                                    )
                                    if tag_counts:
                                        dominant_tag = max(tag_counts, key=tag_counts.get)
                                        label = f"Speaker_{dominant_tag}"
                                        seen = identified_labels.get(conversation_id, set())
                                        if label not in seen:
                                            seen.add(label)
                                            identified_labels[conversation_id] = seen
                                            ws.send(json.dumps({
                                                "event": "speaker_detected",
                                                "label": label,
                                            }))
                                            logger.info(
                                                "speaker_detected SENT: label=%s conversation_id=%s tag_counts=%s",
                                                label, conversation_id, tag_counts,
                                            )
                                        else:
                                            logger.info(
                                                "speaker_detected SKIPPED (already seen): label=%s conversation_id=%s seen=%s",
                                                label, conversation_id, seen,
                                            )
                                else:
                                    logger.info(
                                        "IDENTIFY result: no words in result. transcript='%s' conversation_id=%s",
                                        transcript, conversation_id,
                                    )
                            except Exception as e:
                                logger.error("Error in identifying mode: %s", e)
                            continue  # do NOT emit final_transcript during identification
                        # ── Captioning mode (existing logic) ──
                        try:
                            alt = result.alternatives[0]
                            transcript = (alt.transcript or "").strip()
                        except Exception:
                            transcript = ""
                        speaker = speaker_label_from_result(result)
                        # Log diarization details for debugging
                        try:
                            if alt.words:
                                word_tags = [(w.word, int(getattr(w, "speaker_tag", 0) or 0)) for w in alt.words[-5:]]
                                logger.info("CAPTION result: speaker=%s transcript='%s' last_word_tags=%s", speaker, transcript[:80], word_tags)
                        except Exception:
                            pass
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
                        streamer_state["streamer"] = ChirpStreamer(diarization_speaker_count=num_speakers)
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
            elif event == "begin_captioning":
                # Mode switch: identifying -> captioning
                if conversation_id:
                    session_modes[conversation_id] = "captioning"
                    logger.info("Mode switched to captioning for conversation_id=%s", conversation_id)
                ws.send(json.dumps({"event": "captioning_started"}))
            elif event == "reset_identification":
                # Clear seen labels so retries/re-identify work
                if conversation_id:
                    identified_labels[conversation_id] = set()
                    logger.info("Reset identified_labels for conversation_id=%s", conversation_id)
                ws.send(json.dumps({"event": "identification_reset"}))
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
        # Clean up session state
        if conversation_id:
            session_modes.pop(conversation_id, None)
            identified_labels.pop(conversation_id, None)
        logger.debug("WebSocket handler cleanup complete for conversation_id=%s", conversation_id)


# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
