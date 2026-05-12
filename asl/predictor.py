import cv2
import numpy as np
import pickle
import json
import threading
import logging
from collections import deque

logger = logging.getLogger(__name__)

SEQ_LEN = 50
NUM_FRAMES = SEQ_LEN
CONFIDENCE = 0.55

import mediapipe as mp

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "Invalid mediapipe installation: 'mp.solutions' missing. "
        "Reinstall with: pip install 'mediapipe==0.10.14' 'protobuf>=4.25.3,<5'"
    )

MP_HOLISTIC = mp.solutions.holistic


class ASLPredictor:
    def __init__(self, model_path: str | None = None, labels_path: str | None = None):
        del model_path
        del labels_path
        with open("asl/asl_classifier.pkl", "rb") as f:
            saved = pickle.load(f)
        self.model = saved["model"]
        self.le = saved["label_encoder"]

        with open("asl/label_map.json") as f:
            self.labels = {int(k): v for k, v in json.load(f).items()}
        self.num_classes = len(self.labels)

        self.holistic = MP_HOLISTIC.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self._lock = threading.Lock()
        self.runtime_inference_ok = True
        self.runtime_issue = ""
        logger.info("ASL sklearn predictor loaded. %d classes.", self.num_classes)

    def reset(self):
        with self._lock:
            self.frame_buffer.clear()

    def _extract_landmarks(self, rgb_frame: np.ndarray) -> np.ndarray:
        result = self.holistic.process(rgb_frame)

        def lm(landmark_list, n):
            if landmark_list:
                return np.array([[l.x, l.y] for l in landmark_list.landmark], dtype=np.float32)
            return np.zeros((n, 2), dtype=np.float32)

        pose = lm(result.pose_landmarks, 33)[:13]
        lh = lm(result.left_hand_landmarks, 21)
        rh = lm(result.right_hand_landmarks, 21)
        return np.concatenate([pose, lh, rh], axis=0).flatten()

    def _normalize_landmarks(self, flat_landmarks: np.ndarray) -> np.ndarray:
        points = flat_landmarks.reshape(-1, 2).astype(np.float32)
        pose = points[:13]

        left_shoulder = pose[11]
        right_shoulder = pose[12]
        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        points = (points - center) / scale
        return points.flatten()

    def _predict_from_seq(self, seq: np.ndarray) -> np.ndarray:
        """seq: (SEQ_LEN, 110) -> probability array"""
        x_data = seq.flatten().reshape(1, -1)
        return self.model.predict_proba(x_data)[0]

    def _select_most_informative_window(self, landmarks_seq: np.ndarray) -> np.ndarray:
        t = landmarks_seq.shape[0]
        if t <= SEQ_LEN:
            if t < SEQ_LEN:
                pad = np.repeat(landmarks_seq[-1:, :], SEQ_LEN - t, axis=0)
                return np.concatenate([landmarks_seq, pad], axis=0)
            return landmarks_seq

        hands = landmarks_seq[:, 26:]
        hand_presence = (np.abs(hands).sum(axis=1) > 1e-6).astype(np.float32)
        motion = np.zeros((t,), dtype=np.float32)
        motion[1:] = np.linalg.norm(landmarks_seq[1:] - landmarks_seq[:-1], axis=1)
        frame_score = hand_presence + 0.4 * motion

        best_start = 0
        best_score = frame_score[:SEQ_LEN].sum()
        running = best_score
        for start in range(1, t - SEQ_LEN + 1):
            running += frame_score[start + SEQ_LEN - 1] - frame_score[start - 1]
            if running > best_score:
                best_score = running
                best_start = start

        return landmarks_seq[best_start : best_start + SEQ_LEN]

    def _top_predictions(self, probs: np.ndarray, top_k: int = 3) -> list:
        top_k = min(top_k, len(probs))
        ranked = np.argsort(probs)[::-1][:top_k]
        return [
            {
                "index": int(i),
                "label": self.le.classes_[i],
                "confidence": round(float(probs[i]), 4),
            }
            for i in ranked
        ]

    def _predict_from_recording(self, rgb_frames: list[np.ndarray], top_k: int = 3) -> dict:
        """
        Backward-compatible helper used by test scripts.
        Accepts RGB frames and returns prediction details.
        """
        if not rgb_frames:
            return {
                "text": "",
                "best_prediction": None,
                "top_predictions": [],
                "frames_processed": 0,
                "windows_evaluated": 0,
                "windows_selected": 0,
            }

        pre_sample_n = min(len(rgb_frames), 60)
        if len(rgb_frames) > pre_sample_n:
            idx = np.linspace(0, len(rgb_frames) - 1, pre_sample_n, dtype=int)
            sampled = [rgb_frames[i] for i in idx]
        else:
            sampled = rgb_frames

        all_lm = np.array(
            [self._normalize_landmarks(self._extract_landmarks(frame)) for frame in sampled],
            dtype=np.float32,
        )
        seq = self._select_most_informative_window(all_lm)
        probs = self._predict_from_seq(seq)
        top_preds = self._top_predictions(probs, top_k)
        best = top_preds[0] if top_preds else None
        text = best["label"] if best and best["confidence"] >= CONFIDENCE else ""

        return {
            "text": text,
            "best_prediction": best,
            "top_predictions": top_preds,
            "frames_processed": len(rgb_frames),
            "windows_evaluated": 1,
            "windows_selected": 1,
        }

    def transcribe_video_file(self, file_path: str, top_k: int = 3) -> dict:
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

        result = self._predict_from_recording(raw_frames, top_k=top_k)
        result.pop("windows_selected", None)
        return result

    def process_frame(self, frame_bytes: bytes) -> str | None:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self._extract_landmarks(rgb)

        with self._lock:
            self.frame_buffer.append(landmarks)
            if len(self.frame_buffer) < SEQ_LEN:
                return None
            seq = np.array(self.frame_buffer)
            self.frame_buffer.clear()

        probs = self._predict_from_seq(seq)
        best_idx = int(np.argmax(probs))
        if float(probs[best_idx]) >= CONFIDENCE:
            return self.le.classes_[best_idx]
        return None
