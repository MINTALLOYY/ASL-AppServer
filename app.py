import os
import platform
import logging

from dotenv import load_dotenv
from flask import Flask
from flask_sock import Sock
from flask_cors import CORS

from routes.health import bp as health_bp
from routes.conversations import bp as conversations_bp
from routes.speech import bp as speech_bp
from routes.asl import bp as asl_bp

from websockets.asl_ws import register_asl_ws
from websockets.speech_ws import register_speech_ws
from config import db, speaker_registry
from asl.asl_inference import transcribe_video_details, get_predictor

load_dotenv()

logger = logging.getLogger(__name__)

logger.info(
    "Startup config: PORT=%s WEB_CONCURRENCY=%s PYTHONUNBUFFERED=%s GUNICORN_CMD_ARGS=%s SERVER_SOFTWARE=%s",
    os.environ.get("PORT"),
    os.environ.get("WEB_CONCURRENCY"),
    os.environ.get("PYTHONUNBUFFERED"),
    os.environ.get("GUNICORN_CMD_ARGS"),
    os.environ.get("SERVER_SOFTWARE"),
)

app = Flask(__name__)
sock = Sock(app)
CORS(app)

app.register_blueprint(health_bp)
app.register_blueprint(conversations_bp)
app.register_blueprint(speech_bp)
app.register_blueprint(asl_bp)

register_asl_ws(sock)
register_speech_ws(sock)

@sock.route("/ws/echo")
def ws_echo(ws):
    """
    Simple WebSocket echo test endpoint.
    Connect to wss://yourserver/ws/echo and send any message to get it echoed back.
    """
    logger.info("WebSocket ECHO connection opened")
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                logger.info("WebSocket ECHO client disconnected")
                break
            logger.info("WebSocket ECHO received: %s", msg[:100] if msg else None)
            ws.send(f"echo: {msg}")
    except Exception as e:
        logger.exception("WebSocket ECHO error: %s", e)
    finally:
        logger.info("WebSocket ECHO connection closed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
