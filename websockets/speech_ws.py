import json
import logging
import threading
import time
import traceback
from typing import Optional

from flask import request

import config
from speech.chirp_stream import ChirpStreamer, speaker_label_from_result
from config import db, identified_labels, session_modes
from utils import _get_accessor_uuid, normalize_uuid, validate_uuid

logger = logging.getLogger(__name__)


def register_speech_ws(sock):
    """
    Register the speech websocket endpoint with the specific sock instance.
    """

    @sock.route("/speech/ws")
    def speech_ws(ws):
        """
        WebSocket endpoint for live speech-to-text streaming with speaker identification.
        """
        ws_closed = threading.Event()
        ws_closed_reason = {"value": None}

        conversation_id: Optional[str] = request.args.get("conversation_id")
        initial_mode = request.args.get("mode", "captioning")
        try:
            num_speakers = int(request.args.get("num_speakers", 2))
        except Exception:
            num_speakers = 2
        num_speakers = max(2, min(6, num_speakers))

        raw_uuid = request.args.get("conversation_uuid") or _get_accessor_uuid(request)
        if raw_uuid is not None and not validate_uuid(raw_uuid):
            try:
                ws.send(json.dumps({"event": "error", "error": "invalid_uuid"}))
            except Exception:
                pass
            return
        if raw_uuid is None and config.ENFORCE_CONVERSATION_UUID:
            try:
                ws.send(json.dumps({"event": "error", "error": "missing_conversation_uuid"}))
            except Exception:
                pass
            return
        conversation_uuid: Optional[str] = normalize_uuid(raw_uuid) if raw_uuid else None

        def _session_key() -> Optional[str]:
            return conversation_uuid or conversation_id

        def _migrate_session_state(previous_key: Optional[str]) -> None:
            key = _session_key()
            if not key or key == previous_key:
                return
            if previous_key and previous_key != key:
                if previous_key in session_modes and key not in session_modes:
                    session_modes[key] = session_modes.pop(previous_key)
                if previous_key in identified_labels and key not in identified_labels:
                    identified_labels[key] = identified_labels.pop(previous_key)
            if key not in session_modes:
                session_modes[key] = initial_mode
            if key not in identified_labels:
                identified_labels[key] = set()

        def _set_conversation_uuid(candidate: Optional[str]) -> bool:
            nonlocal conversation_uuid
            if candidate is None:
                return True
            if not validate_uuid(candidate):
                return False
            normalized = normalize_uuid(candidate)
            if conversation_uuid and conversation_uuid != normalized:
                return False
            if conversation_uuid == normalized:
                return True
            previous_key = _session_key()
            conversation_uuid = normalized
            _migrate_session_state(previous_key)
            return True

        def _mark_ws_closed(reason: str) -> None:
            if not ws_closed.is_set():
                ws_closed_reason["value"] = reason
                ws_closed.set()
                logger.info(
                    "WebSocket marked closed. reason=%s conversation_id=%s conversation_uuid=%s",
                    reason,
                    conversation_id,
                    conversation_uuid,
                )

        def _safe_ws_send(payload: dict, context: str) -> bool:
            if ws_closed.is_set():
                logger.info(
                    "Skipping WS send because socket already closed. context=%s event=%s reason=%s conversation_id=%s conversation_uuid=%s",
                    context,
                    payload.get("event"),
                    ws_closed_reason["value"],
                    conversation_id,
                    conversation_uuid,
                )
                return False
            try:
                serialized = json.dumps(payload)
                ws.send(serialized)
                logger.info(
                    "WS->client sent context=%s event=%s bytes=%s conversation_id=%s conversation_uuid=%s",
                    context,
                    payload.get("event"),
                    len(serialized),
                    conversation_id,
                    conversation_uuid,
                )
                return True
            except Exception as send_err:
                err_name = type(send_err).__name__
                err_text = str(send_err)
                if err_name == "ConnectionClosed" or "Connection closed" in err_text:
                    _mark_ws_closed(reason=f"send_failed:{context}:{err_name}")
                    logger.info(
                        "WebSocket closed during send. context=%s event=%s err=%s conversation_id=%s conversation_uuid=%s",
                        context,
                        payload.get("event"),
                        send_err,
                        conversation_id,
                        conversation_uuid,
                    )
                    return False
                logger.error("WebSocket send error context=%s err=%s payload=%s", context, send_err, payload)
                logger.error(traceback.format_exc())
                return False

        def _display_title_from_text(text: str, max_len: int = 64) -> str:
            normalized = " ".join((text or "").strip().split())
            if not normalized:
                return "Untitled conversation"
            if len(normalized) <= max_len:
                return normalized
            return f"{normalized[:max_len].rstrip()}..."

        if conversation_uuid is not None:
            _migrate_session_state(None)
        elif conversation_id:
            session_modes[conversation_id] = initial_mode
            identified_labels[conversation_id] = set()

        if db:
            try:
                created = db.create_conversation(
                    conversation_id=conversation_id,
                    conversation_uuid=conversation_uuid,
                )
                if not conversation_id:
                    conversation_id = created.get("conversation_id")
                if not conversation_uuid and created.get("conversation_uuid"):
                    _set_conversation_uuid(created.get("conversation_uuid"))
            except Exception as e:
                logger.error(f"Failed to auto-create conversation: {e}")

        streamer_state = {"streamer": ChirpStreamer(diarization_speaker_count=num_speakers), "active": True}
        title_state = {"emitted": False}

        try:
            remote = request.remote_addr
        except Exception:
            remote = None
        logger.info(
            "WebSocket connection opened. conversation_id=%s conversation_uuid=%s remote=%s mode=%s num_speakers=%s query=%s",
            conversation_id,
            conversation_uuid,
            remote,
            initial_mode,
            num_speakers,
            dict(request.args),
        )

        def consume_responses():
            try:
                response_count = 0
                logger.info("consume_responses started. Waiting for Google Speech responses...")
                for response in streamer_state["streamer"].responses():
                    response_count += 1
                    try:
                        results_len = len(response.results)
                    except Exception:
                        results_len = "unknown"
                    if response_count <= 3 or response_count % 20 == 0 or results_len == 0:
                        logger.info(
                            "Google response #%s results=%s streamer_stats=%s",
                            response_count,
                            results_len,
                            streamer_state["streamer"].debug_stats(),
                        )
                    if not response.results:
                        logger.info(
                            "Google response #%s had zero results. conversation_id=%s conversation_uuid=%s",
                            response_count,
                            conversation_id,
                            conversation_uuid,
                        )
                    for result in response.results:
                        try:
                            alt0 = result.alternatives[0] if result.alternatives else None
                            alt0_text = ((alt0.transcript or "").strip() if alt0 else "")
                            alt0_words = len(alt0.words) if alt0 and alt0.words else 0
                            logger.info(
                                "Google result detail: final=%s stability=%.3f alts=%s transcript_len=%s words=%s conversation_id=%s conversation_uuid=%s",
                                result.is_final,
                                float(getattr(result, "stability", 0.0) or 0.0),
                                len(result.alternatives),
                                len(alt0_text),
                                alt0_words,
                                conversation_id,
                                conversation_uuid,
                            )
                        except Exception:
                            logger.debug("Failed to log Google result detail")
                        if not result.is_final:
                            continue
                        current_key = _session_key()
                        current_mode = session_modes.get(current_key, "captioning") if current_key else "captioning"
                        if current_mode == "identifying":
                            try:
                                alt = result.alternatives[0]
                                if alt.words:
                                    tag_counts = {}
                                    for word in alt.words:
                                        tag = int(getattr(word, "speaker_tag", 0) or 0)
                                        if tag > 0:
                                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                                    if tag_counts:
                                        dominant_tag = max(tag_counts, key=tag_counts.get)
                                        label = f"Speaker_{dominant_tag}"
                                        seen = identified_labels.get(current_key, set()) if current_key else set()
                                        if label not in seen and current_key:
                                            seen.add(label)
                                            identified_labels[current_key] = seen
                                            sent_ok = _safe_ws_send(
                                                {
                                                    "event": "speaker_detected",
                                                    "label": label,
                                                    "conversation_id": conversation_id,
                                                    "conversation_uuid": conversation_uuid,
                                                },
                                                context="speaker_detected",
                                            )
                                            if not sent_ok:
                                                if ws_closed.is_set():
                                                    continue
                                                streamer_state["streamer"].finish()
                                                return
                            except Exception as e:
                                logger.error("Error in identifying mode: %s", e)
                            continue
                        try:
                            alt = result.alternatives[0]
                            transcript = (alt.transcript or "").strip()
                        except Exception:
                            transcript = ""
                        speaker = speaker_label_from_result(result)
                        if transcript:
                            if not title_state["emitted"]:
                                readable_title = _display_title_from_text(transcript)
                                title_sent_ok = _safe_ws_send(
                                    {
                                        "event": "conversation_title",
                                        "title": readable_title,
                                        "conversation_id": conversation_id,
                                        "conversation_uuid": conversation_uuid,
                                    },
                                    context="conversation_title",
                                )
                                if title_sent_ok or ws_closed.is_set():
                                    title_state["emitted"] = True
                            try:
                                if conversation_id and db:
                                    db.set_conversation_display_name_if_missing(
                                        conversation_id,
                                        transcript,
                                        conversation_uuid=conversation_uuid,
                                    )
                                    db.save_transcript(
                                        conversation_id=conversation_id,
                                        conversation_uuid=conversation_uuid,
                                        segment={
                                            "text": transcript,
                                            "type": "speech",
                                            "speaker": speaker,
                                        },
                                    )
                            except Exception as e:
                                logger.error("Firestore save error: %s", e)
                            try:
                                payload = {
                                    "event": "final_transcript",
                                    "text": transcript,
                                    "speaker": speaker,
                                    "conversation_id": conversation_id,
                                    "conversation_uuid": conversation_uuid,
                                }
                                sent_ok = _safe_ws_send(payload, context="final_transcript")
                                if not sent_ok:
                                    if ws_closed.is_set():
                                        continue
                                    streamer_state["streamer"].finish()
                                    return
                            except Exception as e:
                                logger.error("WebSocket send error: %s", e)
                                logger.error(traceback.format_exc())
                                streamer_state["streamer"].finish()
                                return
                logger.debug("consume_responses finished. Total responses: %s", response_count)
            except Exception as e:
                logger.error("Error in consume_responses: %s", e)
                logger.error(traceback.format_exc())
                if "Audio Timeout" in str(e) or "Audio Timeout Error" in str(e):
                    streamer_state["active"] = False

        t = threading.Thread(target=consume_responses, daemon=True)
        t.start()

        msg_count = 0
        last_idle_log_at = 0.0
        receive_timeout_sec = 15
        try:
            while True:
                msg_count += 1
                try:
                    msg = ws.receive(timeout=receive_timeout_sec)
                except TimeoutError:
                    now = time.time()
                    if now - last_idle_log_at >= 30:
                        last_idle_log_at = now
                    continue
                if msg is None:
                    _mark_ws_closed(reason="receive_none")
                    break
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                event = data.get("event") or data.get("type")
                if event == "start":
                    candidate_uuid = data.get("conversation_uuid")
                    if candidate_uuid is None and config.ENFORCE_CONVERSATION_UUID:
                        _safe_ws_send({"event": "error", "error": "missing_conversation_uuid"}, context="start")
                        break
                    if candidate_uuid is not None and not _set_conversation_uuid(candidate_uuid):
                        _safe_ws_send({"event": "error", "error": "access_denied"}, context="start")
                        break
                    if not conversation_id:
                        cid = data.get("conversation_id")
                        if cid:
                            conversation_id = cid
                    if _session_key():
                        _migrate_session_state(None)
                    _safe_ws_send({
                        "event": "connected",
                        "conversation_id": conversation_id,
                        "conversation_uuid": conversation_uuid
                    }, context="start")
                    continue

                if event == "audio_chunk":
                    b64 = data.get("data")
                    audio_chunk_count = streamer_state.get("audio_chunk_count", 0) + 1
                    streamer_state["audio_chunk_count"] = audio_chunk_count
                    if not conversation_id:
                        cid = data.get("conversation_id")
                        if cid:
                            conversation_id = cid
                    candidate_uuid = data.get("conversation_uuid")
                    if candidate_uuid is not None and not _set_conversation_uuid(candidate_uuid):
                        _safe_ws_send({"event": "error", "error": "access_denied"}, context="audio_chunk")
                        break
                    if conversation_uuid is None and config.ENFORCE_CONVERSATION_UUID:
                        _safe_ws_send({"event": "error", "error": "missing_conversation_uuid"}, context="audio_chunk")
                        break
                    if not streamer_state["active"]:
                        try:
                            try:
                                streamer_state["streamer"].finish()
                            except Exception:
                                pass
                            streamer_state["streamer"] = ChirpStreamer(diarization_speaker_count=num_speakers)
                            streamer_state["active"] = True
                            t = threading.Thread(target=consume_responses, daemon=True)
                            t.start()
                        except Exception as e:
                            logger.exception("Failed to restart speech stream: %s", e)
                    streamer_state["streamer"].add_audio_base64(b64 or "")
                elif event in ("end", "finish", "close"):
                    if not conversation_id:
                        cid = data.get("conversation_id")
                        if cid:
                            conversation_id = cid
                    candidate_uuid = data.get("conversation_uuid")
                    if candidate_uuid is not None and not _set_conversation_uuid(candidate_uuid):
                        _safe_ws_send({"event": "error", "error": "access_denied"}, context=event)
                        break
                    _mark_ws_closed(reason=f"client_event:{event}")
                    streamer_state["streamer"].finish()
                    break
                elif event == "begin_captioning":
                    key = _session_key()
                    if key:
                        session_modes[key] = "captioning"
                    _safe_ws_send({"event": "captioning_started", "conversation_uuid": conversation_uuid}, context="begin_captioning")
                elif event == "reset_identification":
                    key = _session_key()
                    if key:
                        identified_labels[key] = set()
                    _safe_ws_send({"event": "identification_reset", "conversation_uuid": conversation_uuid}, context="reset_identification")
                elif event == "set_conversation":
                    cid = data.get("conversation_id")
                    if cid:
                        conversation_id = cid
                        _migrate_session_state(None)
                    candidate_uuid = data.get("conversation_uuid")
                    if candidate_uuid is not None and not _set_conversation_uuid(candidate_uuid):
                        _safe_ws_send({"event": "error", "error": "access_denied"}, context="set_conversation")
                        break
                else:
                    pass
        except Exception:
            _mark_ws_closed(reason="main_loop_exception")
        finally:
            _mark_ws_closed(reason="handler_finally")
            try:
                streamer_state["streamer"].finish()
            except Exception:
                pass
            try:
                t.join(timeout=2)
            except Exception:
                pass
            for key in {conversation_uuid, conversation_id}:
                if key:
                    session_modes.pop(key, None)
                    identified_labels.pop(key, None)
