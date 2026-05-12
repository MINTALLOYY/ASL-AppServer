from flask import Blueprint, jsonify, request, send_file
import logging
import os
import platform
import sys
import tempfile

from asl.asl_inference import get_predictor as config_get_predictor
from asl.asl_inference import transcribe_video_details as config_transcribe_video_details
from config import db as config_db

logger = logging.getLogger(__name__)

bp = Blueprint("asl", __name__)


def _get_shared_state():
    app_module = sys.modules.get("app")
    if app_module is None:
        return config_db, config_get_predictor, config_transcribe_video_details
    return (
        getattr(app_module, "db", config_db),
        getattr(app_module, "get_predictor", config_get_predictor),
        getattr(app_module, "transcribe_video_details", config_transcribe_video_details),
    )

@bp.get("/asl/diagnostics")
def asl_diagnostics():
    """
    Quick readiness endpoint for ASL model + MediaPipe compatibility.
    Loads predictor by default. Use ?load_predictor=0 to skip model initialization.
    """
    py_version = platform.python_version()
    info = {
        "python_version": py_version,
        "recommended_python": "3.11.x",
        "model_file": "asl/asl_classifier.pkl",
        "load_predictor_url": f"{request.base_url}?load_predictor=1",
    }
    try:
        major, minor = [int(x) for x in py_version.split(".")[:2]]
        info["python_compatible"] = (major, minor) <= (3, 11)
    except Exception:
        info["python_compatible"] = None
    load_predictor_arg = request.args.get("load_predictor")
    if load_predictor_arg is None:
        load_predictor = True
    else:
        load_predictor = load_predictor_arg.strip().lower() not in {"0", "false", "no"}
    info["predictor_load_requested"] = load_predictor
    info["backend"] = "sklearn-mediapipe"

    if not load_predictor:
        info["predictor_loaded"] = False
        info["predictor_status"] = "skipped (set load_predictor=1 to initialize model)"
        return jsonify(info)

    try:
        _, get_predictor_fn, _ = _get_shared_state()
        predictor = get_predictor_fn()
        info["predictor_loaded"] = True
        info["predictor_backend"] = "sklearn-mediapipe"
        info["num_classes"] = int(getattr(predictor, "num_classes", -1))
        info["runtime_inference_ok"] = bool(getattr(predictor, "runtime_inference_ok", True))
        info["runtime_issue"] = getattr(predictor, "runtime_issue", "")
        return jsonify(info)
    except Exception as e:
        info["predictor_loaded"] = False
        info["predictor_error"] = str(e)
        logger.exception("ASL diagnostics failed: %s", e)
        return jsonify(info), 500

@bp.route("/asl/test", methods=["GET"], strict_slashes=False)
@bp.get("/test_asl.html")
def asl_test_page():
    """Serve the ASL browser test page from this Flask app."""
    return send_file("test_asl.html")

@bp.route("/asl/upload-test", methods=["GET"], strict_slashes=False)
@bp.get("/test_asl_upload.html")
def asl_upload_test_page():
    """Serve the simple ASL video upload test page."""
    return send_file("test_asl_upload.html")

@bp.post("/asl/transcribe")
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
        db, get_predictor_fn, transcribe_video_details_fn = _get_shared_state()
        predictor = get_predictor_fn()
        if not bool(getattr(predictor, "runtime_inference_ok", True)):
            issue = getattr(predictor, "runtime_issue", "ASL predictor is not ready")
            return jsonify({"error": issue}), 503
    except Exception as e:
        logger.exception("ASL predictor preflight failed: %s", e)
        return jsonify({"error": f"ASL predictor initialization failed: {e}"}), 500

    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        # Run ASL inference using the structured clip transcription path.
        result = transcribe_video_details_fn(tmp_path, top_k=3)
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
        if conversation_id and result.get("text") and db:
            db.save_message(conversation_id=conversation_id, text=result["text"], source="asl")
    except Exception:
        pass

    top_predictions = list(result.get("top_predictions", []))[:3]
    response_payload = {
        "text": result.get("text", ""),
        "best_prediction": result.get("best_prediction"),
        "top_predictions": top_predictions,
        "predictions": [
            {
                "word": prediction.get("label"),
                "confidence": prediction.get("confidence"),
                "index": prediction.get("index"),
            }
            for prediction in top_predictions
        ],
        "frames_processed": result.get("frames_processed", 0),
        "windows_evaluated": result.get("windows_evaluated", 0),
    }
    return jsonify(response_payload)
