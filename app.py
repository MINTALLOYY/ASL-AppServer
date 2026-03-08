import json
import os
import tempfile
import threading
from typing import Optional
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_sock import Sock

from firebase.db import FirestoreDB
from speech.chirp_stream import ChirpStreamer, speaker_label_from_result
from asl.asl_inference import transcribe_video

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

# FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID")
# db = FirestoreDB(project_id=FIREBASE_PROJECT_ID)
db = False

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
    print(f"[DEBUG] WebSocket connection opened. conversation_id={conversation_id}, remote={remote}")

    # Background thread: consume responses from Google Speech and send to client
    def consume_responses():
        try:
            response_count = 0
            print("[DEBUG] consume_responses started. Waiting for responses...")
            for response in streamer_state["streamer"].responses():
                response_count += 1
                try:
                    results_len = len(response.results)
                except Exception:
                    results_len = "unknown"
                print(f"[DEBUG] Received response #{response_count}. Results count: {results_len}")
                if results_len == 0:
                    print(f"[DEBUG] Response had no results: {response}")
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
                                print(f"[DEBUG] Sending message: {message}")  # Debug log for sent message
                                ws.send(message)
                            except Exception as e:
                                print(f"[ERROR] WebSocket send error: {e}")  # Debug log for WebSocket error
                                print(traceback.format_exc())
                                streamer_state["streamer"].finish()
                                return
                            try:
                                if conversation_id and db:
                                    db.save_message(conversation_id=conversation_id, text=transcript, source="speech", speaker=speaker)
                            except Exception as e:
                                print(f"Firestore save error: {e}")  # Debug log for Firestore error
            print(f"[DEBUG] consume_responses finished. Total responses: {response_count}")
        except Exception as e:
            print(f"Error in consume_responses: {e}")  # Debug log for consume_responses error
            # If the stream errored due to audio timeout, mark inactive so we can restart on next audio
            if "Audio Timeout" in str(e) or "Audio Timeout Error" in str(e):
                streamer_state["active"] = False

    # Start response consumer in background
    t = threading.Thread(target=consume_responses, daemon=True)
    t.start()

    # Main loop: receive audio chunks from Flutter client
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                print("[DEBUG] WebSocket receive returned None — client disconnected")
                break
            try:
                data = json.loads(msg)
            except Exception:
                try:
                    print(f"[DEBUG] Received non-JSON message: {repr(msg)[:200]}")
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
                print(f"[DEBUG] event=audio_chunk conversation_id={conversation_id} b64_len={b64_len} streamer_active={streamer_state['active']}")
                if not conversation_id:
                    cid = data.get("conversation_id")
                    if cid:
                        conversation_id = cid
                # If previous stream errored (timeout), restart a new stream and consumer
                if not streamer_state["active"]:
                    print("[DEBUG] Restarting speech stream due to previous timeout...")
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
                        print(f"[ERROR] Failed to restart speech stream: {e}")
                streamer_state["streamer"].add_audio_base64(b64 or "")
            elif event in ("end", "finish", "close"):
                # End the session
                if not conversation_id:
                    cid = data.get("conversation_id")
                    if cid:
                        conversation_id = cid
                print(f"[DEBUG] event={event} — finishing streamer for conversation_id={conversation_id}")
                streamer_state["streamer"].finish()
                break
            elif event == "set_conversation":
                # Set or update conversation_id mid-session
                cid = data.get("conversation_id")
                if cid:
                    conversation_id = cid
                    print(f"[DEBUG] set_conversation updated conversation_id={conversation_id}")
            else:
                # Ignore unknown events
                print(f"[DEBUG] Unknown websocket event received: {event}")
    except Exception:
        print("[ERROR] Exception in WebSocket main loop")
        print(traceback.format_exc())
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
        print(f"[DEBUG] WebSocket handler cleanup complete for conversation_id={conversation_id}")


# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
