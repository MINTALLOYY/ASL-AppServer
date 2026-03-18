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
DEBUG_PRINT     = True


class ASLPredictor:
    def __init__(self, model_path: str, labels_path: str):
        logger.info("Loading ASL model from %s", model_path)
        if DEBUG_PRINT:
            print(f"[ASL] Loading model: {model_path}", flush=True)
        self.model = load_model(model_path)
        logger.info("ASL model loaded. Input shape: %s", self.model.input_shape)
        if DEBUG_PRINT:
            print(f"[ASL] Model loaded. input_shape={self.model.input_shape}", flush=True)

        with open(labels_path, "r") as f:
            label_map = json.load(f)
        # label_map.json maps string index -> sign name, e.g. {"0": "TV", "1": "after", ...}
        self.labels = {int(k): v for k, v in label_map.items()}
        self.num_classes = len(self.labels)
        logger.info("Loaded %d ASL labels", self.num_classes)
        if DEBUG_PRINT:
            print(f"[ASL] Labels loaded: {self.num_classes}", flush=True)

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0

    def _extract_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run MediaPipe on a BGR frame -> flat float32 vector (126,)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        def lm_array(lm_list, n):
            if lm_list:
                return np.array(
                    [[l.x, l.y, l.z] for l in lm_list.landmark]
                ).flatten()
            return np.zeros(n * 3)

        lh = lm_array(results.left_hand_landmarks, 21)   # 63 values
        rh = lm_array(results.right_hand_landmarks, 21)   # 63 values
        return np.concatenate([lh, rh])  # (126,)

    def process_frame(self, frame_bytes: bytes) -> str | None:
        """
        Feed one JPEG frame as bytes.
        Returns a predicted word string, or None if not ready/confident.
        """
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            if DEBUG_PRINT:
                print("[ASL] Frame decode failed (None).", flush=True)
            return None

        keypoints = self._extract_keypoints(frame)
        self.frame_buffer.append(keypoints)
        self.frame_count += 1
        if DEBUG_PRINT and self.frame_count % 15 == 0:
            hand_signal = int(np.count_nonzero(keypoints))
            print(
                f"[ASL] frame={self.frame_count} buffer={len(self.frame_buffer)}/{SEQUENCE_LENGTH} nonzero_keypoints={hand_signal}",
                flush=True,
            )

        if (len(self.frame_buffer) == SEQUENCE_LENGTH
                and self.frame_count % STRIDE == 0):
            seq = np.expand_dims(np.array(self.frame_buffer), axis=0)  # (1, 30, 126)
            probs = self.model.predict(seq, verbose=0)[0]
            best = int(np.argmax(probs))
            conf = float(probs[best])
            if DEBUG_PRINT:
                candidate = self.labels.get(best, f"unknown_{best}")
                print(f"[ASL] infer best={candidate} conf={conf:.3f} threshold={CONFIDENCE}", flush=True)
            if conf >= CONFIDENCE:
                word = self.labels.get(best, f"unknown_{best}")
                logger.debug("ASL prediction: %s (%.2f)", word, conf)
                if DEBUG_PRINT:
                    print(f"[ASL] PREDICTION={word} conf={conf:.3f}", flush=True)
                return word

        return None

    def reset(self):
        """Clear the frame buffer and counter."""
        self.frame_buffer.clear()
        self.frame_count = 0
