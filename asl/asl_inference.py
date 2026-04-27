from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from asl.predictor import ASLPredictor

_predictor: Optional["ASLPredictor"] = None


def get_predictor() -> "ASLPredictor":
    """Return the shared ASLPredictor singleton, loading it on first call."""
    global _predictor
    if _predictor is None:
        from asl.predictor import ASLPredictor
        _predictor = ASLPredictor()
    return _predictor


def transcribe_video(file_path: str) -> str:
    """Transcribe an ASL video file to text using the shared predictor."""
    return transcribe_video_details(file_path).get("text", "")


def transcribe_video_details(file_path: str) -> dict:
    """Return a structured ASL transcription payload for a video file."""
    predictor = get_predictor()
    predictor.reset()
    try:
        return predictor.transcribe_video_file(file_path)
    finally:
        predictor.reset()
