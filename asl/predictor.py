# asl/predictor.py
import cv2
import json
import numpy as np
import mediapipe as mp
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

        # MediaPipe package layouts vary by version/build; support both paths.
        try:
            holistic_cls = mp.solutions.holistic.Holistic
            logger.info("Using MediaPipe holistic from mp.solutions.holistic")
        except AttributeError:
            try:
                from mediapipe.python.solutions import holistic as mp_holistic
                holistic_cls = mp_holistic.Holistic
                logger.info("Using MediaPipe holistic from mediapipe.python.solutions.holistic")
            except Exception as exc:
                logger.exception("MediaPipe Holistic API not available in installed mediapipe package")
                raise RuntimeError(
                    "Installed mediapipe build does not expose Holistic API required for ASL inference."
                ) from exc

        self.holistic = holistic_cls(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0

    def _extract_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run MediaPipe on a BGR frame and return vector sized to model feature_dim."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
