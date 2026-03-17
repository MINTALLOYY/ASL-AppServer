import base64
import queue
import threading
import traceback
import logging
import time
from typing import Generator, Optional

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

logger = logging.getLogger(__name__)


class ChirpStreamer:
    """
    Streaming speech-to-text client using Google Cloud Speech v2.

    Supports:
        - Real-time audio streaming via gRPC
        - Speaker diarization (labels speakers 1, 2, etc.)
        - Base64 audio input (16 kHz, 16-bit mono PCM)
    """
    def __init__(
        self,
        project_id: str = "",
        language_code: str = "en-US",
        sample_rate_hz: int = 16000,
        diarization_speaker_count: int = 2,
        model: str = "long",
        location: str = "global",
        audio_queue_maxsize: int = 256,
    ) -> None:
        self.project_id = project_id
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.diarization_speaker_count = diarization_speaker_count
        self.model = model
        self.location = location
        self.audio_queue_maxsize = max(16, int(audio_queue_maxsize))

        self._client = SpeechClient()
        self._audio_q: queue.Queue[bytes] = queue.Queue(maxsize=self.audio_queue_maxsize)
        self._finished = threading.Event()
        self._dropped_chunks = 0
        self._last_empty_log_ts = 0.0

    def add_audio_base64(self, b64: str) -> None:
        """
        Add a base64-encoded audio chunk to the streaming queue.

        Args:
            b64: Base64-encoded audio data (16 kHz, 16-bit mono PCM).
        """
        try:
            if not b64:
                logger.debug("add_audio_base64 called with empty data")
                return
            decoded = base64.b64decode(b64)
            try:
                self._audio_q.put(decoded, block=False)
            except queue.Full:
                self._dropped_chunks += 1
                if self._dropped_chunks <= 3 or self._dropped_chunks % 25 == 0:
                    logger.warning(
                        "Audio queue full (maxsize=%s). Dropping chunk #%s (size=%s bytes)",
                        self.audio_queue_maxsize,
                        self._dropped_chunks,
                        len(decoded),
                    )
                return
            try:
                qsize = self._audio_q.qsize()
            except Exception:
                qsize = "unknown"
            if isinstance(qsize, int):
                if qsize >= int(self.audio_queue_maxsize * 0.8):
                    logger.warning("Audio queue high water mark: queue_size=%s/%s", qsize, self.audio_queue_maxsize)
                elif qsize % 20 == 0:
                    logger.debug("Enqueued audio chunk size=%s bytes, queue_size=%s", len(decoded), qsize)
            else:
                logger.debug("Enqueued audio chunk size=%s bytes, queue_size=%s", len(decoded), qsize)
        except Exception as e:
            logger.error("Failed to add audio base64: %s", e)
            logger.error(traceback.format_exc())

    def finish(self) -> None:
        """Signal that no more audio will be sent; close the stream gracefully."""
        self._finished.set()

    def _get_streaming_config(self):
        """
        Create the v2 streaming recognition config.
        """
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate_hz,
                audio_channel_count=1,
            ),
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                diarization_config=cloud_speech.SpeakerDiarizationConfig(
                    min_speaker_count=max(2, self.diarization_speaker_count),
                    max_speaker_count=max(2, self.diarization_speaker_count),
                ),
            ),
            language_codes=[self.language_code],
            model=self.model,
        )
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=False,
            ),
        )
        logger.debug("v2 streaming config created: %s", streaming_config)
        return streaming_config

    def _request_generator(self) -> Generator[cloud_speech.StreamingRecognizeRequest, None, None]:
        """
        Generate streaming requests for Google Speech v2 gRPC API.
        First request sends config, subsequent requests send audio.
        Sends silent keepalive frames when idle to prevent Audio Timeout.
        """
        recognizer = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"

        # First request: config only (no audio)
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=self._get_streaming_config(),
        )
        logger.debug("Sent v2 config request. recognizer=%s", recognizer)

        audio_passed = False
        chunk_count = 0
        last_audio_time = time.time()
        # 200ms of silence at 16kHz 16-bit mono = 6400 bytes of zeros
        silence_frame = b'\x00' * 6400
        keepalive_interval = 3.0

        while not self._finished.is_set() or not self._audio_q.empty():
            try:
                chunk = self._audio_q.get(timeout=0.5)
                if chunk:
                    audio_passed = True
                    chunk_count += 1
                    last_audio_time = time.time()
                    if chunk_count <= 3 or chunk_count % 25 == 0:
                        logger.debug(
                            "Sending audio chunk #%s of size: %s bytes. Queue size: %s",
                            chunk_count,
                            len(chunk),
                            self._audio_q.qsize(),
                        )
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
            except queue.Empty:
                now = time.time()
                if now - last_audio_time >= keepalive_interval:
                    yield cloud_speech.StreamingRecognizeRequest(audio=silence_frame)
                    last_audio_time = now
                if now - self._last_empty_log_ts >= 10:
                    logger.debug("Queue empty, sending keepalive. _finished=%s", self._finished.is_set())
                    self._last_empty_log_ts = now
                continue

        logger.debug("_request_generator finished. Total chunks sent: %s. Audio passed: %s", chunk_count, audio_passed)
        if not audio_passed:
            logger.debug("Connection made but no audio was passed through.")

    def responses(self):
        """
        Start the streaming recognize call and return the response iterator.
        """
        try:
            logger.debug("Starting streaming recognize call (v2 API)...")
            response_iterator = self._client.streaming_recognize(
                requests=self._request_generator(),
            )
            logger.debug("Streaming recognize call started successfully.")
            return response_iterator
        except Exception as e:
            logger.error("Error in streaming_recognize: %s", e)
            logger.error(traceback.format_exc())
            raise


def speaker_label_from_result(result) -> Optional[str]:
    """
    Extract a human-readable speaker label from a v2 speech recognition result.

    In v2, words have `speaker_label` (string like "1", "2") instead of v1's
    `speaker_tag` (int).

    Returns:
        Speaker label like "Speaker A", "Speaker B", etc., or None if unavailable.
    """
    try:
        alt = result.alternatives[0]
        if alt.words:
            label = alt.words[-1].speaker_label
            if label:
                try:
                    tag_num = int(label)
                    base = ord('A') - 1
                    readable = f"Speaker {chr(base + tag_num)}"
                    logger.debug("speaker_label_from_result: label=%s -> %s", label, readable)
                    return readable
                except ValueError:
                    logger.debug("speaker_label_from_result: non-numeric label=%s", label)
                    return f"Speaker {label}"
    except Exception:
        pass
    return None
