import cv2
import numpy as np
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from asl.predictor import ASLPredictor

logger = logging.getLogger(__name__)

# Module-level predictor instance (lazy-loaded)
_predictor: Optional["ASLPredictor"] = None


def get_predictor() -> "ASLPredictor":
    """Return the shared ASLPredictor singleton, loading it on first call."""
    global _predictor
    if _predictor is None:
        from asl.predictor import ASLPredictor
        import os
        base = os.path.dirname(__file__)
        _predictor = ASLPredictor(
            model_path=os.path.join(base, "asl_model.bin"),
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
    return transcribe_video_details(file_path).get("text", "")


def transcribe_video_details(file_path: str) -> dict:
    """Return a structured ASL transcription payload for a video file."""
    predictor = get_predictor()
    predictor.reset()
    try:
        return predictor.transcribe_video_file(file_path)
    finally:
        predictor.reset()
