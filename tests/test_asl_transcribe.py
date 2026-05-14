"""Tests for the ASL upload transcription route."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

import config

VALID_UUID = "123e4567-e89b-42d3-a456-426614174000"


with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


class TestAslTranscribeRoute(unittest.TestCase):
    def setUp(self):
        self.client = server_app.app.test_client()
        self.mock_db = MagicMock()
        self.mock_predictor = MagicMock()
        self._orig_db = server_app.db
        self._orig_transcribe = server_app.transcribe_video_details
        self._orig_get_predictor = server_app.get_predictor
        self._orig_enforce = config.ENFORCE_CONVERSATION_UUID
        server_app.db = self.mock_db
        self.mock_predictor.runtime_inference_ok = True
        self.mock_predictor.runtime_issue = ""
        server_app.get_predictor = MagicMock(return_value=self.mock_predictor)
        config.ENFORCE_CONVERSATION_UUID = False

    def tearDown(self):
        server_app.db = self._orig_db
        server_app.transcribe_video_details = self._orig_transcribe
        server_app.get_predictor = self._orig_get_predictor
        config.ENFORCE_CONVERSATION_UUID = self._orig_enforce

    def test_asl_transcribe_returns_structured_payload(self):
        server_app.transcribe_video_details = MagicMock(return_value={
            "text": "hello",
            "best_prediction": {"index": 113, "label": "hello", "confidence": 0.9723},
            "top_predictions": [
                {"index": 113, "label": "hello", "confidence": 0.9723},
                {"index": 214, "label": "thankyou", "confidence": 0.0151},
                {"index": 7, "label": "yes", "confidence": 0.0092},
            ],
            "frames_processed": 42,
            "windows_evaluated": 3,
        })
        payload = {
            "conversation_id": "conv-1",
            "conversation_uuid": VALID_UUID,
            "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
        }
        resp = self.client.post(
            "/asl/transcribe",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["text"], "hello")
        self.assertEqual(body["best_prediction"]["label"], "hello")
        self.assertEqual(len(body["top_predictions"]), 3)
        self.assertEqual(len(body["predictions"]), 3)
        self.assertEqual(body["predictions"][0]["word"], "hello")
        self.assertEqual(body["frames_processed"], 42)
        self.assertEqual(body["windows_evaluated"], 3)
        self.assertEqual(body["conversation_uuid"], VALID_UUID)
        self.mock_db.save_message.assert_not_called()

    def test_asl_transcribe_without_conversation_id_still_skips_firestore(self):
        server_app.transcribe_video_details = MagicMock(return_value={
            "text": "thankyou",
            "best_prediction": {"index": 214, "label": "thankyou", "confidence": 0.9512},
            "top_predictions": [{"index": 214, "label": "thankyou", "confidence": 0.9512}],
            "frames_processed": 36,
            "windows_evaluated": 2,
        })
        payload = {
            "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
        }
        resp = self.client.post(
            "/asl/transcribe",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["text"], "thankyou")
        self.mock_db.save_message.assert_not_called()

    def test_asl_transcribe_requires_video(self):
        resp = self.client.post(
            "/asl/transcribe",
            data=json.dumps({"conversation_id": "conv-1", "conversation_uuid": VALID_UUID}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_asl_transcribe_requires_uuid_when_enforced(self):
        config.ENFORCE_CONVERSATION_UUID = True
        payload = {
            "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
        }
        resp = self.client.post(
            "/asl/transcribe",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("missing_conversation_uuid", resp.get_json()["error"])

    def test_asl_transcribe_predictor_unhealthy_returns_503(self):
        self.mock_predictor.runtime_inference_ok = False
        self.mock_predictor.runtime_issue = "Saved ASL model weights contain NaN/Inf values"
        payload = {
            "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
        }
        resp = self.client.post(
            "/asl/transcribe",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertIn("error", body)
        self.assertIn("NaN/Inf", body["error"])


if __name__ == "__main__":
    unittest.main()
