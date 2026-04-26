# asl/predictor.py
import cv2
import json
import os
import numpy as np
import mediapipe as mp
import platform
import tempfile
import threading
import urllib.request
from collections import deque
import logging

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 30   # frames per inference window
STRIDE          = 15   # run inference every N frames
CONFIDENCE      = 0.85  # min confidence to emit a prediction

# Canonical 21-point hand landmark connection graph.
HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
]


class ASLPredictor:
    def __init__(self, model_path: str, labels_path: str):
        from tensorflow.keras.models import load_model

        logger.info("Loading ASL model from %s", model_path)
        # Inference-only server path: avoid optimizer state allocation.
        self.model = load_model(model_path, compile=False)
        logger.info("ASL model loaded. Input shape: %s", self.model.input_shape)
        # Supports (batch, seq, features), usually features=126 (hands) or 225 (pose+hands)
        self.feature_dim = int(self.model.input_shape[-1])
        logger.info("ASL model feature dimension: %s", self.feature_dim)
        self.runtime_inference_ok = True
        self.runtime_issue = ""
        self._saved_weights_are_valid = self._validate_saved_weights()

        with open(labels_path, "r") as f:
            label_map = json.load(f)
        # label_map.json maps string index -> sign name, e.g. {"0": "TV", "1": "after", ...}
        self.labels = {int(k): v for k, v in label_map.items()}
        self.num_classes = len(self.labels)
        logger.info("Loaded %d ASL labels", self.num_classes)

        self._backend = "holistic"
        self._tasks_image_cls = None
        self._tasks_image_format = None
        self._tasks_hand = None
        self._tasks_pose = None

        # MediaPipe package layouts vary by runtime/build; support both paths.
        try:
            holistic_cls = mp.solutions.holistic.Holistic
            logger.info("Using MediaPipe holistic from mp.solutions.holistic")
        except AttributeError:
            try:
                from mediapipe.python.solutions import holistic as mp_holistic
                holistic_cls = mp_holistic.Holistic
                logger.info("Using MediaPipe holistic from mediapipe.python.solutions.holistic")
            except Exception as exc:
                mp_version = getattr(mp, "__version__", "unknown")
                has_solutions = hasattr(mp, "solutions")
                py_version = platform.python_version()
                logger.warning(
                    "MediaPipe Holistic API not available. mediapipe=%s python=%s has_mp_solutions=%s",
                    mp_version,
                    py_version,
                    has_solutions,
                )
                logger.info("Falling back to MediaPipe Tasks landmarkers")
                self._init_tasks_backend()
                holistic_cls = None

        if holistic_cls is not None:
            self.holistic = holistic_cls(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.holistic = None
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
        self._infer_lock = threading.Lock()
        self._run_runtime_sanity_check()

    def _validate_saved_weights(self) -> bool:
        """Check whether the loaded checkpoint contains only finite weights."""
        try:
            weights = self.model.get_weights()
        except Exception as exc:
            self.runtime_inference_ok = False
            self.runtime_issue = f"Could not inspect model weights: {exc}"
            logger.exception("ASL weight validation failed: %s", exc)
            return False

        has_nan = any(np.isnan(weight).any() for weight in weights)
        has_inf = any(np.isinf(weight).any() for weight in weights)
        if has_nan or has_inf:
            self.runtime_inference_ok = False
            self.runtime_issue = (
                "Saved ASL model weights contain NaN/Inf values. "
                "The checkpoint is corrupted or incompatible with this runtime."
            )
            logger.error(
                "ASL checkpoint invalid: weights contain non-finite values (nan=%s inf=%s)",
                has_nan,
                has_inf,
            )
            return False

        return True

    def _predict_probs(self, seq: np.ndarray) -> np.ndarray:
        """Run a single sequence through the model and return a normalized probability vector."""
        if not self.runtime_inference_ok:
            raise RuntimeError(self.runtime_issue or "ASL model runtime is not healthy")
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)
        if seq.ndim != 3:
            raise ValueError(f"Expected sequence rank 3, got shape {seq.shape}")

        with self._infer_lock:
            probs = self.model.predict(seq, verbose=0)[0]

        if not np.isfinite(probs).all():
            logger.warning(
                "ASL predict_probs produced non-finite values nan=%s posinf=%s neginf=%s; sanitizing",
                int(np.isnan(probs).sum()),
                int(np.isposinf(probs).sum()),
                int(np.isneginf(probs).sum()),
            )
            probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        total_prob = float(np.sum(probs))
        if total_prob > 0.0:
            probs = probs / total_prob
        else:
            probs = np.zeros_like(probs)

        return probs.astype(np.float32)

    def _top_predictions(self, probs: np.ndarray, top_k: int = 3) -> list[dict]:
        """Convert a probability vector into a JSON-friendly ranked prediction list."""
        if probs.size == 0:
            return []

        top_k = max(1, min(int(top_k), int(probs.shape[0])))
        ranked = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in ranked:
            results.append(
                {
                    "index": int(idx),
                    "label": self.labels.get(int(idx), f"unknown_{int(idx)}"),
                    "confidence": round(float(probs[idx]), 4),
                }
            )
        return results

    def predict_sequence(self, seq: np.ndarray, top_k: int = 3) -> dict:
        """Predict a single temporal window of keypoint features."""
        probs = self._predict_probs(seq)
        top_predictions = self._top_predictions(probs, top_k=top_k)
        best = top_predictions[0] if top_predictions else None
        return {
            "best_prediction": best,
            "top_predictions": top_predictions,
        }

    def transcribe_video_file(self, file_path: str, top_k: int = 3) -> dict:
        """Transcribe a video file into a ranked ASL prediction payload."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error("Could not open video: %s", file_path)
            return {
                "text": "",
                "best_prediction": None,
                "top_predictions": [],
                "frames_processed": 0,
                "windows_evaluated": 0,
            }

        frame_features: list[np.ndarray] = []
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                frame_features.append(self._extract_keypoints(frame))
        finally:
            cap.release()

        if not frame_features:
            return {
                "text": "",
                "best_prediction": None,
                "top_predictions": [],
                "frames_processed": frame_count,
                "windows_evaluated": 0,
            }

        if len(frame_features) < SEQUENCE_LENGTH:
            last_frame = frame_features[-1]
            frame_features.extend([last_frame.copy() for _ in range(SEQUENCE_LENGTH - len(frame_features))])

        windows: list[np.ndarray] = []
        for start in range(0, len(frame_features) - SEQUENCE_LENGTH + 1, STRIDE):
            window = np.stack(frame_features[start : start + SEQUENCE_LENGTH]).astype(np.float32)
            windows.append(window)

        if not windows:
            windows.append(np.stack(frame_features[-SEQUENCE_LENGTH:]).astype(np.float32))

        accumulated_probs = None
        for window in windows:
            probs = self._predict_probs(window[np.newaxis, ...])
            if accumulated_probs is None:
                accumulated_probs = np.zeros_like(probs)
            accumulated_probs += probs

        if accumulated_probs is None:
            return {
                "text": "",
                "best_prediction": None,
                "top_predictions": [],
                "frames_processed": frame_count,
                "windows_evaluated": 0,
            }

        averaged_probs = accumulated_probs / float(len(windows))
        top_predictions = self._top_predictions(averaged_probs, top_k=top_k)
        best_prediction = top_predictions[0] if top_predictions else None

        text = ""
        if best_prediction and best_prediction["confidence"] >= CONFIDENCE:
            text = best_prediction["label"]

        return {
            "text": text,
            "best_prediction": best_prediction,
            "top_predictions": top_predictions,
            "frames_processed": frame_count,
            "windows_evaluated": len(windows),
        }

    @property
    def backend(self) -> str:
        return self._backend

    def _run_runtime_sanity_check(self) -> None:
        """Detect broken TF runtime early (e.g. all-NaN outputs on valid tensors)."""
        # Keep an earlier, more specific startup failure reason (e.g., corrupted weights).
        if not self.runtime_inference_ok:
            return
        try:
            zero_seq = np.zeros((1, SEQUENCE_LENGTH, self.feature_dim), dtype=np.float32)
            rand_seq = np.random.rand(1, SEQUENCE_LENGTH, self.feature_dim).astype(np.float32)
            with self._infer_lock:
                out_zero = self.model.predict(zero_seq, verbose=0)[0]
                out_rand = self.model.predict(rand_seq, verbose=0)[0]

            zero_ok = bool(np.isfinite(out_zero).all())
            rand_ok = bool(np.isfinite(out_rand).all())
            if not (zero_ok and rand_ok):
                self.runtime_inference_ok = False
                self.runtime_issue = (
                    "Model outputs are non-finite (NaN/Inf). "
                    "This often indicates an incompatible TensorFlow/Python runtime or an invalid checkpoint. "
                    "Use Python 3.11.x and verify the model file."
                )
                logger.error(
                    "ASL runtime sanity check failed: zero_finite=%s rand_finite=%s zero_nan=%s rand_nan=%s",
                    zero_ok,
                    rand_ok,
                    int(np.isnan(out_zero).sum()),
                    int(np.isnan(out_rand).sum()),
                )
        except Exception as exc:
            self.runtime_inference_ok = False
            self.runtime_issue = f"Runtime sanity check exception: {exc}"
            logger.exception("ASL runtime sanity check raised exception: %s", exc)

    def _ensure_task_model(self, filename: str, url: str) -> str:
        model_dir = os.path.join(tempfile.gettempdir(), "asl_mediapipe_models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path):
            logger.info("Downloading MediaPipe task asset: %s", filename)
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def _init_tasks_backend(self) -> None:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        hand_model_path = self._ensure_task_model(
            "hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        )

        # Pose model is optional for low-memory deployments.
        enable_pose = os.environ.get("ASL_ENABLE_POSE", "0").strip().lower() in {"1", "true", "yes"}
        pose_model_path = None
        if self.feature_dim != 126 and enable_pose:
            pose_model_path = self._ensure_task_model(
                "pose_landmarker_lite.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            )
        elif self.feature_dim != 126:
            logger.info(
                "Pose landmarker disabled (ASL_ENABLE_POSE!=1). Using zeroed pose features to reduce memory usage."
            )

        self._tasks_hand = mp_tasks_vision.HandLandmarker.create_from_options(
            mp_tasks_vision.HandLandmarkerOptions(
                base_options=mp_tasks_python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=mp_tasks_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.2,
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.2,
            )
        )

        if pose_model_path:
            self._tasks_pose = mp_tasks_vision.PoseLandmarker.create_from_options(
                mp_tasks_vision.PoseLandmarkerOptions(
                    base_options=mp_tasks_python.BaseOptions(model_asset_path=pose_model_path),
                    running_mode=mp_tasks_vision.RunningMode.IMAGE,
                    num_poses=1,
                )
            )

        self._tasks_image_cls = mp.Image
        self._tasks_image_format = mp.ImageFormat.SRGB
        self._backend = "tasks"
        logger.info("Initialized MediaPipe Tasks backend for ASL keypoint extraction")

    def _extract_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run MediaPipe on a BGR frame and return vector sized to model feature_dim."""
        feats, _ = self.extract_features_and_debug(frame_bgr)
        return feats

    def extract_features_and_debug(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return feature vector and optional normalized hand landmarks for debug overlays."""
        with self._infer_lock:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self._backend == "tasks":
                return self._extract_keypoints_tasks_with_debug(rgb)

            results = self.holistic.process(rgb)

        def lm_array(lm_list, n):
            if lm_list:
                return np.array(
                    [[l.x, l.y, l.z] for l in lm_list.landmark]
                ).flatten()
            return np.zeros(n * 3)

        pose = lm_array(results.pose_landmarks, 33)       # 99 values
        lh = lm_array(results.left_hand_landmarks, 21)   # 63 values
        rh = lm_array(results.right_hand_landmarks, 21)   # 63 values

        debug = {
            "left": self._normalize_hand_landmarks(results.left_hand_landmarks),
            "right": self._normalize_hand_landmarks(results.right_hand_landmarks),
        }

        feats = self._compose_features(pose, lh, rh)
        return feats.astype(np.float32), debug

    def _compose_features(self, pose: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """Compose pose/hand vectors to match the model feature dimension."""

        # Match training pipeline shape:
        # 126 -> hands only, 225 -> pose + hands, fallback -> pad/trim.
        if self.feature_dim == 126:
            return np.concatenate([lh, rh])
        if self.feature_dim == 225:
            return np.concatenate([pose, lh, rh])

        raw = np.concatenate([pose, lh, rh])
        if raw.shape[0] >= self.feature_dim:
            return raw[:self.feature_dim]
        return np.pad(raw, (0, self.feature_dim - raw.shape[0]), mode="constant")

    def _normalize_hand_landmarks(self, lm_list) -> list[list[float]]:
        if not lm_list:
            return []
        points = []
        for lm in lm_list.landmark:
            points.append([float(lm.x), float(lm.y)])
        return points

    def _extract_keypoints_tasks(self, frame_rgb: np.ndarray) -> np.ndarray:
        feats, _ = self._extract_keypoints_tasks_with_debug(frame_rgb)
        return feats

    def _extract_keypoints_tasks_with_debug(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        image = self._tasks_image_cls(
            image_format=self._tasks_image_format,
            data=frame_rgb,
        )
        hand_result = self._tasks_hand.detect(image)

        lh = np.zeros(63, dtype=np.float32)
        rh = np.zeros(63, dtype=np.float32)

        left_pts: list[list[float]] = []
        right_pts: list[list[float]] = []

        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
            points_2d = [[float(lm.x), float(lm.y)] for lm in landmarks]
            hand_name = ""
            try:
                classes = hand_result.handedness[idx]
                if classes:
                    hand_name = (classes[0].category_name or "").lower()
            except Exception:
                hand_name = ""

            if hand_name == "left":
                lh = coords
                left_pts = points_2d
            elif hand_name == "right":
                rh = coords
                right_pts = points_2d
            elif not np.any(lh):
                lh = coords
                left_pts = points_2d
            elif not np.any(rh):
                rh = coords
                right_pts = points_2d

        pose = np.zeros(99, dtype=np.float32)
        if self._tasks_pose is not None:
            pose_result = self._tasks_pose.detect(image)
            if pose_result.pose_landmarks:
                pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]],
                    dtype=np.float32,
                ).flatten()
                if pose.shape[0] < 99:
                    pose = np.pad(pose, (0, 99 - pose.shape[0]), mode="constant")
                elif pose.shape[0] > 99:
                    pose = pose[:99]

        feats = self._compose_features(pose, lh, rh)
        debug = {
            "left": left_pts,
            "right": right_pts,
            "detected": int(len(hand_result.hand_landmarks)),
        }
        return feats.astype(np.float32), debug

    def process_frame(self, frame_bytes: bytes) -> str | None:
        """
        Feed one JPEG frame as bytes.
        Returns a predicted word string, or None if not ready/confident.
        """
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("ASL frame decode failed (cv2.imdecode returned None)")
            return None

        keypoints = self._extract_keypoints(frame)
        self.frame_buffer.append(keypoints)
        self.frame_count += 1
        if self.frame_count % 15 == 0:
            hand_signal = int(np.count_nonzero(keypoints))
            logger.info(
                "ASL frame=%s buffer=%s/%s nonzero_keypoints=%s",
                self.frame_count,
                len(self.frame_buffer),
                SEQUENCE_LENGTH,
                hand_signal,
            )

        if (len(self.frame_buffer) == SEQUENCE_LENGTH
            and self.frame_count % STRIDE == 0):
            seq = np.expand_dims(np.array(self.frame_buffer), axis=0)  # (1, 30, 126)
            probs = self._predict_probs(seq)
            best = int(np.argmax(probs))
            conf = float(probs[best])
            candidate = self.labels.get(best, f"unknown_{best}")
            logger.info("ASL infer best=%s conf=%.3f threshold=%.2f", candidate, conf, CONFIDENCE)
            if conf >= CONFIDENCE:
                word = self.labels.get(best, f"unknown_{best}")
                logger.info("ASL prediction=%s conf=%.3f", word, conf)
                return word

        return None

    def reset(self):
        """Clear the frame buffer and counter."""
        self.frame_buffer.clear()
        self.frame_count = 0
