import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
	from asl.predictor import ASLPredictor

logger = logging.getLogger(__name__)
_predictor: Optional["ASLPredictor"] = None


def get_predictor() -> "ASLPredictor":
	global _predictor
	if _predictor is None:
		from asl.predictor import ASLPredictor
		_predictor = ASLPredictor()
	return _predictor


def transcribe_video(file_path: str, top_k: int = 3) -> str:
	return transcribe_video_details(file_path, top_k=top_k).get("text", "")


def transcribe_video_details(file_path: str, top_k: int = 3) -> dict:
	predictor = get_predictor()
	predictor.reset()
	try:
		return predictor.transcribe_video_file(file_path, top_k=top_k)
	finally:
		predictor.reset()
