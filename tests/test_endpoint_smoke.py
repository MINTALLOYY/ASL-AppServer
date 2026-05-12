"""Lightweight endpoint smoke tests for route wiring after refactors."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch


# Patch external/heavy dependencies before importing app.
with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


class TestEndpointSmoke(unittest.TestCase):
    """Verify HTTP endpoints are still mounted and callable."""

    def setUp(self):
        self.client = server_app.app.test_client()
        self.mock_db = MagicMock()
        self.mock_db.create_conversation.return_value = "cid-smoke"
        self.mock_db.list_conversations.return_value = []
        self.mock_db.get_conversation.return_value = None
        self.mock_db.get_messages.return_value = []

        self._orig_db = getattr(server_app, "db", None)
        self._orig_get_predictor = getattr(server_app, "get_predictor", None)
        self._orig_transcribe = getattr(server_app, "transcribe_video_details", None)

        server_app.db = self.mock_db

        predictor = MagicMock()
        predictor.runtime_inference_ok = True
        predictor.runtime_issue = ""
        server_app.get_predictor = MagicMock(return_value=predictor)
        server_app.transcribe_video_details = MagicMock(
            return_value={
                "text": "hello",
                "best_prediction": {"index": 1, "label": "hello", "confidence": 0.99},
                "top_predictions": [{"index": 1, "label": "hello", "confidence": 0.99}],
                "frames_processed": 10,
                "windows_evaluated": 1,
            }
        )

    def tearDown(self):
        server_app.db = self._orig_db
        server_app.get_predictor = self._orig_get_predictor
        server_app.transcribe_video_details = self._orig_transcribe

    def test_expected_routes_are_registered(self):
        rules = {rule.rule for rule in server_app.app.url_map.iter_rules()}
        expected = {
            "/health",
            "/ws-info",
            "/ws-hello",
            "/conversations",
            "/conversations/<conversation_id>",
            "/conversations/<conversation_id>/messages",
            "/speech/finalize",
            "/speech/register_speakers",
            "/asl/diagnostics",
            "/asl/transcribe",
            "/asl/test",
            "/test_asl.html",
            "/asl/upload-test",
            "/test_asl_upload.html",
        }
        for route in expected:
            self.assertIn(route, rules)

    def test_health_and_ws_info_endpoints(self):
        health = self.client.get("/health")
        self.assertIn(health.status_code, {200, 500})

        ws_info = self.client.get("/ws-info")
        self.assertEqual(ws_info.status_code, 200)
        body = ws_info.get_json()
        self.assertIn("ws_speech_url", body)

    def test_conversation_endpoints_callable(self):
        created = self.client.post("/conversations", data=json.dumps({}), content_type="application/json")
        self.assertEqual(created.status_code, 201)

        listed = self.client.get("/conversations")
        self.assertEqual(listed.status_code, 200)

        fetched = self.client.get("/conversations/does-not-exist")
        self.assertIn(fetched.status_code, {404, 500})

        messages = self.client.get("/conversations/does-not-exist/messages")
        self.assertEqual(messages.status_code, 200)

    def test_speech_endpoints_callable(self):
        missing = self.client.post("/speech/finalize", data=json.dumps({}), content_type="application/json")
        self.assertEqual(missing.status_code, 400)

        register = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(
                {
                    "conversation_id": "cid-smoke",
                    "speakers": [{"label": "Speaker_0", "name": "Alex"}],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(register.status_code, 200)

        lookup = self.client.get("/speech/register_speakers?conversation_id=cid-smoke")
        self.assertEqual(lookup.status_code, 200)

    def test_asl_endpoints_callable(self):
        diagnostics = self.client.get("/asl/diagnostics?load_predictor=0")
        self.assertEqual(diagnostics.status_code, 200)

        no_file = self.client.post("/asl/transcribe", data={}, content_type="multipart/form-data")
        self.assertEqual(no_file.status_code, 400)

        with_file = self.client.post(
            "/asl/transcribe",
            data={"video": (io.BytesIO(b"fake-bytes"), "clip.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(with_file.status_code, 200)


if __name__ == "__main__":
    unittest.main()
