import logging
import os
from collections import Counter, deque

import cv2
import numpy as np
import tf_keras as keras

logger = logging.getLogger(__name__)

CONFIDENCE = 0.85
PREDICTION_WINDOW = 5
MIN_CONSENSUS = 3
DEFAULT_IMAGE_SIZE = 224

DEFAULT_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "EXCUSE_ME", "HELLO", "HELP", "HOW", "LISTEN", "NICE",
    "PLEASE", "SORRY", "THANKS", "WELCOME",
]


def _load_labels(model_path: str) -> list[str]:
    labels_path = os.path.join(os.path.dirname(model_path), "labels.txt")
    if os.path.exists(labels_path):
        labels = []
        with open(labels_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
        if labels:
            return labels

    return DEFAULT_LABELS


class ASLPredictor:
    def __init__(self, model_path: str):
        logger.info("Loading ASL model from %s", model_path)
        self.model = keras.models.load_model(model_path, compile=False)
        self.labels = _load_labels(model_path)
        self.input_size = int(self.model.input_shape[1] or DEFAULT_IMAGE_SIZE)
        self.prediction_history: deque[str] = deque(maxlen=PREDICTION_WINDOW)
        self.last_emitted_word: str | None = None
        logger.info(
            "ASL model loaded successfully: input_size=%s labels=%s",
            self.input_size,
            len(self.labels),
        )

    def _make_square_crop(self, crop: np.ndarray) -> np.ndarray:
        canvas = np.full((self.input_size, self.input_size, 3), 255, dtype=np.uint8)
        height, width = crop.shape[:2]
        if height == 0 or width == 0:
            return canvas

        scale = min(self.input_size / width, self.input_size / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        resized = cv2.resize(crop, (new_width, new_height))

        x_offset = (self.input_size - new_width) // 2
        y_offset = (self.input_size - new_height) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        return canvas

    def _extract_hand_image(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        frame_height, frame_width = frame_bgr.shape[:2]
        crop_size = int(min(frame_height, frame_width) * 0.75)
        if crop_size <= 0:
            return None

        x0 = max(0, (frame_width - crop_size) // 2)
        y0 = max(0, (frame_height - crop_size) // 2)
        x1 = min(frame_width, x0 + crop_size)
        y1 = min(frame_height, y0 + crop_size)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        return self._make_square_crop(crop)

    def _prepare_input(self, hand_image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(hand_image, (self.input_size, self.input_size))
        image_array = resized.astype(np.float32)
        image_array = (image_array / 127.5) - 1.0
        return np.expand_dims(image_array, axis=0)

    def _reset_prediction_state(self):
        self.prediction_history.clear()
        self.last_emitted_word = None

    def process_frame(self, frame_bytes: bytes):
        """
        Feed one JPEG frame. Returns a predicted sign label or None.
        The loaded model is a single-frame image classifier, not a sequence model.
        """
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        hand_image = self._extract_hand_image(frame)
        if hand_image is None:
            self._reset_prediction_state()
            return None

        probs = self.model.predict(self._prepare_input(hand_image), verbose=0)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        if confidence < CONFIDENCE or best_idx >= len(self.labels):
            self.prediction_history.clear()
            return None

        word = self.labels[best_idx]
        self.prediction_history.append(word)
        top_word, count = Counter(self.prediction_history).most_common(1)[0]
        if count < MIN_CONSENSUS or top_word != word:
            return None

        if word == self.last_emitted_word:
            return None

        self.last_emitted_word = word
        logger.info("ASL prediction: '%s' (confidence=%.3f)", word, confidence)
        return word
