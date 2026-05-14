import json
import base64
import time
import logging
from flask import request
from typing import Optional
from asl.asl_inference import get_predictor
import config
from utils import _get_accessor_uuid, normalize_uuid, validate_uuid

logger = logging.getLogger(__name__)

def register_asl_ws(sock):
    """
    Register the ASL websocket endpoint with the specific sock instance.
    """
    @sock.route("/asl/ws")
    def asl_ws(ws):
        """
        WebSocket endpoint for real-time ASL sign-language translation.

        The client streams JPEG frames as base64 JSON, server runs sklearn inference
        on landmark windows, and sends predicted words back to the client.

        Query parameters:
            - conversation_uuid (optional): account-scoped user identifier.
        Expected client events:
            - {"event": "asl_frame", "frame": "<base64 JPEG>"}
            - {"event": "reset"}     # clear predictor buffer
            - {"event": "end"}       # close session

        Server responses (JSON):
            - {"event": "asl_result", "word": "hello"}
            - {"event": "connected"}
            - {"event": "error", "message": "..."}
        """
        conn_started_at = time.time()
        conn_id = f"aslws-{int(conn_started_at * 1000)}-{(id(ws) % 100000)}"
        remote = request.remote_addr
        conversation_uuid_raw = request.args.get("conversation_uuid") or _get_accessor_uuid(request)
        if conversation_uuid_raw is not None and not validate_uuid(conversation_uuid_raw):
            ws.send(json.dumps({"event": "error", "message": "invalid_uuid"}))
            return
        if conversation_uuid_raw is None and config.ENFORCE_CONVERSATION_UUID:
            ws.send(json.dumps({"event": "error", "message": "missing_conversation_uuid"}))
            return
        conversation_uuid: Optional[str] = normalize_uuid(conversation_uuid_raw) if conversation_uuid_raw else None
        logger.info("[%s] OPEN remote=%s path=/asl/ws conversation_uuid=%s", conn_id, remote, conversation_uuid)

        # Ack immediately so clients are not stuck waiting while model loads.
        ws.send(json.dumps({"event": "connected"}))
        ws.send(json.dumps({"event": "asl_status", "phase": "loading_model"}))
        logger.info("[%s] TX connected + loading_model", conn_id)

        try:
            predictor = get_predictor()
            logger.info(
                "[%s] PREDICTOR ready backend=%s classes=%s",
                conn_id,
                "sklearn-mediapipe",
                getattr(predictor, "num_classes", "unknown"),
            )
            ws.send(json.dumps({
                "event": "asl_status",
                "phase": "ready",
                "backend": "sklearn-mediapipe",
            }))
            if not bool(getattr(predictor, "runtime_inference_ok", True)):
                issue = getattr(predictor, "runtime_issue", "ASL runtime check failed")
                logger.error("[%s] PREDICTOR runtime_invalid: %s", conn_id, issue)
                ws.send(json.dumps({"event": "asl_status", "phase": "runtime_error"}))
                ws.send(json.dumps({
                    "event": "error",
                    "message": issue,
                }))
                return
        except Exception as e:
            logger.exception("[%s] PREDICTOR init_failed: %s", conn_id, e)
            ws.send(json.dumps({"event": "asl_status", "phase": "error"}))
            ws.send(json.dumps({"event": "error", "message": "Model failed to load"}))
            return

        # Shared model, per-connection session state.
        frame_count = 0
        msg_count = 0
        last_idle_log_at = 0.0
        last_ping_at = 0.0
        receive_timeout_sec = 15
        keepalive_interval_sec = 12
        last_frame_at = conn_started_at
        decode_failures = 0
        infer_calls = 0
        predictions_sent = 0
        heartbeat_sent = 0
        heartbeat_recv = 0

        try:
            while True:
                msg_count += 1
                try:
                    msg = ws.receive(timeout=receive_timeout_sec)
                except TimeoutError:
                    now = time.time()
                    if now - last_idle_log_at >= 30:
                        logger.info(
                            "[%s] IDLE wait=%ss since_last_frame=%.1fs messages=%s frames=%s",
                            conn_id,
                            receive_timeout_sec,
                            now - last_frame_at,
                            msg_count,
                            frame_count,
                        )
                        last_idle_log_at = now

                    # Keepalive for proxies/load balancers that close quiet websocket connections.
                    if now - last_ping_at >= keepalive_interval_sec:
                        ws.send(json.dumps({"event": "ping", "ts": int(now)}))
                        last_ping_at = now
                        heartbeat_sent += 1
                        logger.info("[%s] TX ping count=%s", conn_id, heartbeat_sent)
                    continue

                if msg is None:
                    logger.info("[%s] CLIENT disconnected (receive returned None)", conn_id)
                    break

                try:
                    data = json.loads(msg)
                except Exception:
                    logger.info("[%s] RX non_json len=%s", conn_id, len(msg) if msg else 0)
                    continue

                event = data.get("event")
                now = time.time()

                if now - last_ping_at >= keepalive_interval_sec:
                    ws.send(json.dumps({"event": "ping", "ts": int(now)}))
                    last_ping_at = now
                    heartbeat_sent += 1
                    logger.info("[%s] TX ping count=%s", conn_id, heartbeat_sent)

                if event == "asl_frame":
                    if frame_count % 10 == 0:
                        logger.info(
                            "[%s] RX frame idx=%s b64_len=%s",
                            conn_id,
                            frame_count,
                            len(data.get("frame") or ""),
                        )
                    frame_b64 = data.get("frame")
                    if not frame_b64:
                        logger.warning("[%s] RX frame missing_payload", conn_id)
                        continue

                    frame_bytes = base64.b64decode(frame_b64)
                    last_frame_at = time.time()
                    frame_count += 1

                    try:
                        word = predictor.process_frame(frame_bytes)
                    except Exception as infer_err:
                        decode_failures += 1
                        logger.warning("[%s] process_frame failed: %s", conn_id, infer_err)
                        continue

                    infer_calls += 1
                    if word:
                        predictions_sent += 1
                        ws.send(json.dumps({"event": "asl_result", "word": word}))
                        logger.info("[%s] TX asl_result word=%s total_predictions=%s", conn_id, word, predictions_sent)

                elif event == "reset":
                    predictor.reset()
                    logger.info("[%s] RX reset -> predictor_buffer_cleared", conn_id)

                elif event in ("pong", "heartbeat"):
                    heartbeat_recv += 1
                    logger.info("[%s] RX %s count=%s", conn_id, event, heartbeat_recv)

                elif event in ("end", "finish", "close"):
                    logger.info("[%s] RX %s -> closing session", conn_id, event)
                    break

                else:
                    logger.info("[%s] RX unknown_event=%s", conn_id, event)

        except Exception as e:
            logger.exception("[%s] EXCEPTION in ASL WebSocket handler: %s", conn_id, e)
        finally:
            elapsed = time.time() - conn_started_at
            logger.info(
                "[%s] CLOSE elapsed=%.1fs messages=%s frames=%s infer_calls=%s predictions=%s decode_failures=%s heartbeats_tx=%s heartbeats_rx=%s",
                conn_id,
                elapsed,
                msg_count,
                frame_count,
                infer_calls,
                predictions_sent,
                decode_failures,
                heartbeat_sent,
                heartbeat_recv,
            )
