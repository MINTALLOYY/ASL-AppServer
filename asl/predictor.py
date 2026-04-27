import cv2
import numpy as np
import torch
import threading
import logging
import os
from collections import deque
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

logger = logging.getLogger(__name__)

NUM_FRAMES = 16
SEQUENCE_LENGTH = NUM_FRAMES
CONFIDENCE = 0.45
WINDOW_STRIDE = 8
LIVE_BUFFER_MAX = 64
LIVE_STRIDE = 4
LIVE_CONSENSUS = 2


class ASLPredictor:
    def __init__(self, model_path: str | None = None, labels_path: str | None = None):
        # Keep args for compatibility with older call sites.
        del model_path
        del labels_path

        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        logger.info("ASL device: %s", self.device)

        model_id = "Shawon16/VideoMAE_WLASL_100_SR_8"
        logger.info("Loading VideoMAE from %s ...", model_id)

        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        raw_labels = getattr(self.model.config, "id2label", {}) or {}
        self.labels = {int(k): v for k, v in raw_labels.items()}
        self.num_classes = len(self.labels)
        logger.info(
            "VideoMAE loaded. %d classes. First 3: %s",
            self.num_classes,
            [self.labels.get(i, f"word_{i}") for i in range(min(3, self.num_classes))],
        )
        logger.info(
            "ASL inference params: num_frames=%d confidence=%.2f window_stride=%d live_buffer_max=%d live_stride=%d live_consensus=%d",
            NUM_FRAMES,
            CONFIDENCE,
            WINDOW_STRIDE,
            LIVE_BUFFER_MAX,
            LIVE_STRIDE,
            LIVE_CONSENSUS,
        )

        if os.getenv("ASL_SKIP_SANITY_PROBE", "0") != "1":
            self._run_sanity_probe()

        self.frame_buffer: list[np.ndarray] = []
        self._live_history = deque(maxlen=5)
        self._lock = threading.Lock()
        self.runtime_inference_ok = True
        self.runtime_issue = ""

    def reset(self) -> None:
        with self._lock:
            self.frame_buffer = []

    def _predict_from_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        frames: list of NUM_FRAMES numpy arrays, each (H, W, 3) RGB uint8.
        Returns softmax probability vector of length num_classes.
        """
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        return probs

    def _sample_frames(self, all_frames: list[np.ndarray], n: int = NUM_FRAMES) -> list[np.ndarray]:
        """Uniformly sample n frames from a list."""
        if len(all_frames) == 0:
            return []
        indices = np.linspace(0, len(all_frames) - 1, n, dtype=int)
        return [all_frames[i] for i in indices]

    def _top_predictions(self, probs: np.ndarray, top_k: int = 3) -> list[dict]:
        top_k = min(top_k, len(probs))
        ranked = np.argsort(probs)[::-1][:top_k]
        return [
            {
                "index": int(i),
                "label": self.labels.get(int(i), f"word_{i}"),
                "confidence": round(float(probs[i]), 4),
            }
            for i in ranked
        ]

    def _log_top_predictions(self, probs: np.ndarray, context: str, top_k: int = 3) -> None:
        preds = self._top_predictions(probs, top_k=top_k)
        if not preds:
            logger.info("[%s] no predictions", context)
            return
        logger.info(
            "[%s] top%d: %s",
            context,
            len(preds),
            ", ".join(f"{p['label']}={p['confidence']:.4f}" for p in preds),
        )

    def _run_sanity_probe(self) -> None:
        """Quick startup probe to detect collapsed checkpoints early."""
        try:
            black = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]
            probs_black = self._predict_from_frames(black)
            black_idx = int(np.argmax(probs_black))
            black_conf = float(probs_black[black_idx])
            black_label = self.labels.get(black_idx, f"word_{black_idx}")

            random_top = []
            for _ in range(5):
                frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]
                probs = self._predict_from_frames(frames)
                random_top.append(int(np.argmax(probs)))

            values, counts = np.unique(np.array(random_top), return_counts=True)
            dominant = int(values[np.argmax(counts)])
            dominant_ratio = float(np.max(counts) / len(random_top))
            dominant_label = self.labels.get(dominant, f"word_{dominant}")

            logger.info(
                "[sanity] black_top=%s conf=%.4f random_dominant=%s ratio=%.2f",
                black_label,
                black_conf,
                dominant_label,
                dominant_ratio,
            )

            # Heuristic: very high confidence on blank clip + high random dominance indicates collapse.
            if black_conf > 0.80 and dominant_ratio >= 0.80:
                logger.warning(
                    "[sanity] model appears collapsed: high-confidence blank prediction and random-input dominance detected. "
                    "Expected behavior is low confidence / diverse outputs on non-sign inputs."
                )
        except Exception as exc:
            logger.warning("[sanity] startup probe failed: %s", exc)

    def _predict_from_recording(self, rgb_frames: list[np.ndarray], top_k: int = 3) -> dict:
        """Score a full recording by sliding 16-frame windows and aggregating strongest windows."""
        if not rgb_frames:
            return {
                "text": "",
                "best_prediction": None,
                "top_predictions": [],
                "frames_processed": 0,
                "windows_evaluated": 0,
                "windows_selected": 0,
            }

        windows: list[list[np.ndarray]] = []
        if len(rgb_frames) <= NUM_FRAMES:
            windows.append(self._sample_frames(rgb_frames, NUM_FRAMES))
        else:
            for start in range(0, len(rgb_frames) - NUM_FRAMES + 1, WINDOW_STRIDE):
                windows.append(rgb_frames[start : start + NUM_FRAMES])
            if not windows:
                windows.append(self._sample_frames(rgb_frames, NUM_FRAMES))

        logger.info(
            "[recording] frames=%d windows=%d stride=%d",
            len(rgb_frames),
            len(windows),
            WINDOW_STRIDE,
        )

        scored: list[tuple[float, np.ndarray]] = []
        for clip in windows:
            probs = self._predict_from_frames(clip)
            scored.append((float(np.max(probs)), probs))

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = max(1, len(scored) // 2)
        selected = scored[:keep]

        logger.info(
            "[recording] selected_windows=%d best_window_conf=%.4f",
            len(selected),
            selected[0][0] if selected else 0.0,
        )

        accum = np.zeros((self.num_classes,), dtype=np.float32)
        for _, probs in selected:
            accum += probs
        final_probs = accum / float(len(selected))

        top_preds = self._top_predictions(final_probs, top_k)
        best = top_preds[0] if top_preds else None
        text = best["label"] if best and best["confidence"] >= CONFIDENCE else ""
        self._log_top_predictions(final_probs, context="recording_aggregated", top_k=3)

        return {
            "text": text,
            "best_prediction": best,
            "top_predictions": top_preds,
            "frames_processed": len(rgb_frames),
            "windows_evaluated": len(windows),
            "windows_selected": len(selected),
        }

    def transcribe_video_file(self, file_path: str, top_k: int = 3) -> dict:
        """Main entry point — takes a video file path, returns prediction dict."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {
                "text": "",
                "best_prediction": None,
                "top_predictions": [],
                "frames_processed": 0,
                "windows_evaluated": 0,
            }

        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not raw_frames:
            return {
                "text": "",
                "best_prediction": None,
                "top_predictions": [],
                "frames_processed": 0,
                "windows_evaluated": 0,
            }

        return self._predict_from_recording(raw_frames, top_k=top_k)

    def process_frame(self, frame_bytes: bytes) -> str | None:
        """
        Live streaming path — feed JPEG frames one at a time.
        Returns a word string when buffer hits NUM_FRAMES, else None.
        """
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            self.frame_buffer.append(rgb)
            if len(self.frame_buffer) > LIVE_BUFFER_MAX:
                self.frame_buffer = self.frame_buffer[-LIVE_BUFFER_MAX:]

            if len(self.frame_buffer) < NUM_FRAMES:
                return None

            if len(self.frame_buffer) % LIVE_STRIDE != 0:
                return None

            frames = self.frame_buffer[-NUM_FRAMES:]

        probs = self._predict_from_frames(frames)
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        best_label = self.labels.get(best_idx, f"word_{best_idx}")

        if conf >= CONFIDENCE:
            self._live_history.append(best_idx)
        else:
            self._live_history.append(-1)

        votes = sum(1 for idx in self._live_history if idx == best_idx)
        logger.info(
            "[live] buffer=%d best=%s conf=%.4f votes=%d/%d threshold=%.2f",
            len(self.frame_buffer),
            best_label,
            conf,
            votes,
            LIVE_CONSENSUS,
            CONFIDENCE,
        )
        if conf >= CONFIDENCE and votes >= LIVE_CONSENSUS:
            self._live_history.clear()
            self._log_top_predictions(probs, context="live_emit", top_k=3)
            return best_label
        return None
