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
from tensorflow.keras.models import load_model
from collections import deque
import logging

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 30   # frames per inference window
STRIDE          = 15   # run inference every N frames
CONFIDENCE      = 0.85  # min confidence to emit a prediction


class ASLPredictor:
    def __init__(self, model_path: str, labels_path: str):
        logger.info("Loading ASL model from %s", model_path)
        self.model = load_model(model_path)
        logger.info("ASL model loaded. Input shape: %s", self.model.input_shape)
        # Supports (batch, seq, features), usually features=126 (hands) or 225 (pose+hands)
        self.feature_dim = int(self.model.input_shape[-1])
        logger.info("ASL model feature dimension: %s", self.feature_dim)

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

    @property
    def backend(self) -> str:
        return self._backend

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

        pose_model_path = None
        if self.feature_dim != 126:
            pose_model_path = self._ensure_task_model(
                "pose_landmarker_lite.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            )

        self._tasks_hand = mp_tasks_vision.HandLandmarker.create_from_options(
            mp_tasks_vision.HandLandmarkerOptions(
                base_options=mp_tasks_python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=mp_tasks_vision.RunningMode.IMAGE,
                num_hands=2,
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
        with self._infer_lock:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self._backend == "tasks":
                return self._extract_keypoints_tasks(rgb)

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

        # Match training pipeline shape:
        # 126 -> hands only, 225 -> pose + hands, fallback -> pad/trim.
        if self.feature_dim == 126:
            feats = np.concatenate([lh, rh])
        elif self.feature_dim == 225:
            feats = np.concatenate([pose, lh, rh])
        else:
            raw = np.concatenate([pose, lh, rh])
            if raw.shape[0] >= self.feature_dim:
                feats = raw[:self.feature_dim]
            else:
                feats = np.pad(raw, (0, self.feature_dim - raw.shape[0]), mode="constant")

        return feats.astype(np.float32)

    def _extract_keypoints_tasks(self, frame_rgb: np.ndarray) -> np.ndarray:
        image = self._tasks_image_cls(
            image_format=self._tasks_image_format,
            data=frame_rgb,
        )
        hand_result = self._tasks_hand.detect(image)

        lh = np.zeros(63, dtype=np.float32)
        rh = np.zeros(63, dtype=np.float32)

        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
            hand_name = ""
            try:
                classes = hand_result.handedness[idx]
                if classes:
                    hand_name = (classes[0].category_name or "").lower()
            except Exception:
                hand_name = ""

            if hand_name == "left":
                lh = coords
            elif hand_name == "right":
                rh = coords
            elif not np.any(lh):
                lh = coords
            elif not np.any(rh):
                rh = coords

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

        if self.feature_dim == 126:
            feats = np.concatenate([lh, rh])
        elif self.feature_dim == 225:
            feats = np.concatenate([pose, lh, rh])
        else:
            raw = np.concatenate([pose, lh, rh])
            if raw.shape[0] >= self.feature_dim:
                feats = raw[:self.feature_dim]
            else:
                feats = np.pad(raw, (0, self.feature_dim - raw.shape[0]), mode="constant")

        return feats.astype(np.float32)

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
            with self._infer_lock:
                probs = self.model.predict(seq, verbose=0)[0]
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
