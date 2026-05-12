import json
import logging
import os
import sys
import tempfile

# Configure structured logging to stdout (Render + Gunicorn)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# When running under Gunicorn, reuse its handlers so logs show up
gunicorn_error_logger = logging.getLogger("gunicorn.error")
if gunicorn_error_logger.handlers:
    root_logger.handlers = gunicorn_error_logger.handlers
    root_logger.setLevel(gunicorn_error_logger.level)

creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
creds_project_id = None
if creds and creds.strip().startswith("{"):
    try:
        creds_obj = json.loads(creds)
        creds_project_id = creds_obj.get("project_id")
        tmp_json = os.path.join(tempfile.gettempdir(), "gcp_creds.json")
        with open(tmp_json, "w") as f:
            f.write(creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_json
    except Exception:
        pass

# Session mode state machine for speaker identification flow.
# Keys: conversation_id, Values: 'identifying' | 'captioning'
session_modes: dict[str, str] = {}

# Speaker labels already reported during identification phase.
# Keys: conversation_id, Values: set of labels (e.g. {'Speaker_1', 'Speaker_2'})
identified_labels: dict[str, set] = {}

from firebase.db import FirestoreDB

FIREBASE_PROJECT_ID = (
    os.environ.get("FIREBASE_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCLOUD_PROJECT")
    or creds_project_id
)

try:
    db = FirestoreDB(project_id=FIREBASE_PROJECT_ID or None)
    root_logger.info("Firestore initialized successfully. project_id=%s", FIREBASE_PROJECT_ID)
except Exception as _db_init_err:
    root_logger.warning("Firestore initialization failed — db features disabled: %s", _db_init_err)
    db = None

# In-memory speaker registration store (MVP – not persisted across restarts).
# Keys: conversation_id, Values: dict mapping diarization label to participant name.
speaker_registry: dict[str, dict[str, str]] = {}
