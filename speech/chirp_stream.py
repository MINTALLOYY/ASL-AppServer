import base64
import audioop
import queue
import threading
import traceback
import logging
import time
from typing import Generator, Optional

from google.cloud import speech_v1 as speech

logger = logging.getLogger(__name__)


class ChirpStreamer:
    """
    Streaming speech-to-text client using Google Cloud Speech v1.

    Supports:
        - Real-time audio streaming via gRPC
        - Speaker diarization (labels speakers 1, 2, etc.)
        - Base64 audio input (16 kHz, 16-bit mono PCM)
    """
    def __init__(
        self,
        language_code: str = "en-US",
        sample_rate_hz: int = 16000,
        diarization_speaker_count: int = 2,
        audio_queue_maxsize: int = 512,
    ) -> None:
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.diarization_speaker_count = diarization_speaker_count
        self.audio_queue_maxsize = max(16, int(audio_queue_maxsize))

        self._client = speech.SpeechClient()
        self._audio_q: queue.Queue[bytes] = queue.Queue(maxsize=self.audio_queue_maxsize)
        self._finished = threading.Event()
        self._dropped_chunks = 0
        self._last_empty_log_ts = 0.0
        self._created_at = time.time()
        self._received_b64_chunks = 0
        self._received_audio_bytes = 0
        self._sent_audio_chunks = 0
        self._sent_audio_bytes = 0
        self._keepalive_count = 0
        self._decode_error_count = 0
        self._audio_signature_warnings = 0
        self._low_energy_chunks = 0
        self._last_rms = 0
        self._last_peak = 0
        logger.info(
            "ChirpStreamer init: language=%s sample_rate_hz=%s diarization_speakers=%s queue_max=%s",
            self.language_code,
            self.sample_rate_hz,
            self.diarization_speaker_count,
            self.audio_queue_maxsize,
        )

    def add_audio_base64(self, b64: str) -> None:
        """
        Add a base64-encoded audio chunk to the streaming queue.

        Args:
            b64: Base64-encoded audio data (16 kHz, 16-bit mono PCM).
        """
        try:
            self._received_b64_chunks += 1
            b64_chunk_index = self._received_b64_chunks
            if not b64:
                if b64_chunk_index <= 3 or b64_chunk_index % 50 == 0:
                    logger.warning("add_audio_base64 received empty payload. chunk=%s", b64_chunk_index)
                return
            decoded = base64.b64decode(b64)
            decoded_len = len(decoded)
            self._received_audio_bytes += decoded_len
            try:
                rms = int(audioop.rms(decoded, 2)) if decoded_len >= 2 else 0
                peak = int(audioop.max(decoded, 2)) if decoded_len >= 2 else 0
            except Exception:
                rms = -1
                peak = -1
            self._last_rms = rms
            self._last_peak = peak
            if rms >= 0 and rms < 25:
                self._low_energy_chunks += 1
            if decoded.startswith(b"RIFF") or decoded.startswith(b"OggS") or decoded.startswith(b"fLaC"):
                self._audio_signature_warnings += 1
                if self._audio_signature_warnings <= 5:
                    logger.warning(
                        "Audio payload looks containerized/compressed (signature=%s chunk=%s len=%s). Expected raw LINEAR16 PCM.",
                        repr(decoded[:4]),
                        b64_chunk_index,
                        decoded_len,
                    )
            try:
                self._audio_q.put(decoded, block=False)
            except queue.Full:
                self._dropped_chunks += 1
                cleared = 0
                while True:
                    try:
                        self._audio_q.get_nowait()
                        cleared += 1
                    except queue.Empty:
                        break
                try:
                    self._audio_q.put(decoded, block=False)
                except queue.Full:
                    pass
                if self._dropped_chunks <= 3 or self._dropped_chunks % 25 == 0:
                    logger.warning(
                        "Audio queue full (maxsize=%s). Cleared %s queued chunks; replaced with newest chunk #%s (size=%s bytes). totals: recv_chunks=%s recv_bytes=%s",
                        self.audio_queue_maxsize,
                        cleared,
                        self._dropped_chunks,
                        decoded_len,
                        self._received_b64_chunks,
                        self._received_audio_bytes,
                    )
                return
            try:
                qsize = self._audio_q.qsize()
            except Exception:
                qsize = "unknown"
            if b64_chunk_index <= 5 or b64_chunk_index % 25 == 0:
                logger.info(
                    "RX audio accepted: chunk=%s b64_len=%s decoded_len=%s rms=%s peak=%s low_energy_chunks=%s queue_size=%s dropped=%s recv_bytes=%s",
                    b64_chunk_index,
                    len(b64),
                    decoded_len,
                    rms,
                    peak,
                    self._low_energy_chunks,
                    qsize,
                    self._dropped_chunks,
                    self._received_audio_bytes,
                )
            if rms == 0 and (b64_chunk_index <= 5 or b64_chunk_index % 50 == 0):
                logger.warning(
                    "Chunk appears silent (rms=0). chunk=%s decoded_len=%s",
                    b64_chunk_index,
                    decoded_len,
                )
            if isinstance(qsize, int):
                if qsize >= int(self.audio_queue_maxsize * 0.8):
                    logger.warning("Audio queue high water mark: queue_size=%s/%s", qsize, self.audio_queue_maxsize)
                elif qsize % 20 == 0:
                    logger.debug("Enqueued audio chunk size=%s bytes, queue_size=%s", len(decoded), qsize)
            else:
                logger.debug("Enqueued audio chunk size=%s bytes, queue_size=%s", len(decoded), qsize)
        except Exception as e:
            self._decode_error_count += 1
            logger.error("Failed to add audio base64: %s", e)
            logger.error(traceback.format_exc())

    def finish(self) -> None:
        """Signal that no more audio will be sent; close the stream gracefully."""
        self._finished.set()
        logger.info("ChirpStreamer finish requested. stats=%s", self.debug_stats())

    def _get_streaming_config(self):
        """
        Create the streaming recognition config with speaker diarization.
        Uses model=latest_long for best diarization accuracy.
        """
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=max(2, self.diarization_speaker_count),
            max_speaker_count=max(2, self.diarization_speaker_count),
        )
        rec_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
            model="latest_long",
            diarization_config=diarization_config,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=rec_config,
            interim_results=False,
            single_utterance=False,
        )
        logger.debug("Streaming config: model=latest_long, speakers=%s", self.diarization_speaker_count)
        return streaming_config

    def _request_generator(self) -> Generator[speech.StreamingRecognizeRequest, None, None]:
        """
        Generate streaming requests for Google Speech gRPC API.
        Sends silent keepalive frames when idle to prevent Audio Timeout.
        """
        audio_passed = False
        chunk_count = 0
        last_audio_time = time.time()
        # 200ms of silence at 16kHz 16-bit mono = 6400 bytes of zeros
        silence_frame = b'\x00' * 6400
        keepalive_interval = 3.0
        logger.info("_request_generator started. keepalive_interval=%ss", keepalive_interval)

        while not self._finished.is_set() or not self._audio_q.empty():
            try:
                chunk = self._audio_q.get(timeout=0.5)
                if chunk:
                    audio_passed = True
                    chunk_count += 1
                    self._sent_audio_chunks += 1
                    self._sent_audio_bytes += len(chunk)
                    last_audio_time = time.time()
                    if chunk_count <= 3 or chunk_count % 25 == 0:
                        logger.info(
                            "TX->Google audio chunk=%s size=%s queue_size=%s totals(sent_chunks=%s sent_bytes=%s recv_chunks=%s recv_bytes=%s)",
                            chunk_count,
                            len(chunk),
                            self._audio_q.qsize(),
                            self._sent_audio_chunks,
                            self._sent_audio_bytes,
                            self._received_b64_chunks,
                            self._received_audio_bytes,
                        )
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                now = time.time()
                if now - last_audio_time >= keepalive_interval:
                    self._keepalive_count += 1
                    yield speech.StreamingRecognizeRequest(audio_content=silence_frame)
                    last_audio_time = now
                    if self._keepalive_count <= 3 or self._keepalive_count % 20 == 0:
                        logger.info(
                            "TX->Google keepalive=%s size=%s finished=%s queue_empty=%s",
                            self._keepalive_count,
                            len(silence_frame),
                            self._finished.is_set(),
                            self._audio_q.empty(),
                        )
                if now - self._last_empty_log_ts >= 10:
                    logger.info(
                        "Queue empty heartbeat: finished=%s recv_chunks=%s sent_chunks=%s keepalives=%s dropped=%s decode_errors=%s",
                        self._finished.is_set(),
                        self._received_b64_chunks,
                        self._sent_audio_chunks,
                        self._keepalive_count,
                        self._dropped_chunks,
                        self._decode_error_count,
                    )
                    self._last_empty_log_ts = now
                continue

        logger.info(
            "_request_generator finished. chunk_count=%s audio_passed=%s stats=%s",
            chunk_count,
            audio_passed,
            self.debug_stats(),
        )
        if not audio_passed:
            logger.debug("Connection made but no audio was passed through.")

    def responses(self):
        """
        Start the streaming recognize call and return the response iterator.
        """
        try:
            logger.info("Starting streaming recognize call (v1 API, model=latest_long)...")
            streaming_config = self._get_streaming_config()
            response_iterator = self._client.streaming_recognize(streaming_config, self._request_generator())
            logger.info("Streaming recognize call started successfully.")
            return response_iterator
        except Exception as e:
            logger.error("Error in streaming_recognize: %s", e)
            logger.error(traceback.format_exc())
            raise

    def debug_stats(self) -> dict:
        uptime = max(0.0, time.time() - self._created_at)
        return {
            "uptime_sec": round(uptime, 3),
            "recv_chunks": self._received_b64_chunks,
            "recv_audio_bytes": self._received_audio_bytes,
            "sent_chunks": self._sent_audio_chunks,
            "sent_audio_bytes": self._sent_audio_bytes,
            "keepalives_sent": self._keepalive_count,
            "dropped_chunks": self._dropped_chunks,
            "decode_errors": self._decode_error_count,
            "audio_signature_warnings": self._audio_signature_warnings,
            "low_energy_chunks": self._low_energy_chunks,
            "last_rms": self._last_rms,
            "last_peak": self._last_peak,
            "queue_size": self._audio_q.qsize(),
            "finished": self._finished.is_set(),
        }


def speaker_label_from_result(result: speech.StreamingRecognitionResult) -> Optional[str]:
    """
    Extract a human-readable speaker label from a v1 speech recognition result.

    Returns:
        Speaker label like "Speaker_1", "Speaker_2", etc., or None if unavailable.
    """
    try:
        alt = result.alternatives[0]
        if alt.words:
            tag = alt.words[-1].speaker_tag
            if tag:
                try:
                    label = f"Speaker_{int(tag)}"
                    logger.debug("speaker_label_from_result: tag=%s -> %s", tag, label)
                    return label
                except Exception:
                    logger.debug("speaker_label_from_result: invalid tag=%s", tag)
                    pass
    except Exception:
        pass
    return None
