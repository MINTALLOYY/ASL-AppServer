import cv2
import numpy as np
import logging
from typing import Optional
from asl.predictor import ASLPredictor

logger = logging.getLogger(__name__)

# Module-level predictor instance (lazy-loaded)
_predictor: Optional[ASLPredictor] = None


def get_predictor() -> ASLPredictor:
    """Return the shared ASLPredictor singleton, loading it on first call."""
    global _predictor
    if _predictor is None:
        import os
        base = os.path.dirname(__file__)
        _predictor = ASLPredictor(
            model_path=os.path.join(base, "asl_model.keras"),
            labels_path=os.path.join(base, "label_map.json"),
        )
    return _predictor


def transcribe_video(file_path: str) -> str:
    """
    Transcribe an ASL video file to text by extracting frames
    and running them through the LSTM predictor.

    Args:
        file_path: Path to the video file (e.g., .mp4).

    Returns:
        Transcribed text string (space-separated predicted words).
    """
    predictor = get_predictor()
    predictor.reset()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error("Could not open video: %s", file_path)
        return ""

    words = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Encode frame as JPEG bytes for the predictor
            _, buf = cv2.imencode(".jpg", frame)
            word = predictor.process_frame(buf.tobytes())
            if word:
                words.append(word)
    finally:
        cap.release()
        predictor.reset()

    return " ".join(words) if words else ""
