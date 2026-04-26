"""Tests for the ASL upload transcription route."""
import io
import json
import unittest
from unittest.mock import MagicMock, patch


# Patch heavy dependencies before importing app so tests stay offline and deterministic.
with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


class TestAslTranscribeRoute(unittest.TestCase):
    def setUp(self):
        self.client = server_app.app.test_client()
        self.mock_db = MagicMock()
        self._orig_db = server_app.db
        self._orig_transcribe = server_app.transcribe_video_details
        server_app.db = self.mock_db

    def tearDown(self):
        server_app.db = self._orig_db
        server_app.transcribe_video_details = self._orig_transcribe

    def test_asl_transcribe_returns_structured_payload(self):
        server_app.transcribe_video_details = MagicMock(return_value={
            "text": "hello",
            "best_prediction": {"index": 113, "label": "hello", "confidence": 0.9723},
            "top_predictions": [
                {"index": 113, "label": "hello", "confidence": 0.9723},
                {"index": 214, "label": "thankyou", "confidence": 0.0151},
            ],
            "frames_processed": 42,
            "windows_evaluated": 3,
        })
        payload = {
            "conversation_id": "conv-1",
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
        self.assertEqual(len(body["top_predictions"]), 2)
        self.assertEqual(body["frames_processed"], 42)
        self.assertEqual(body["windows_evaluated"], 3)
        self.mock_db.save_message.assert_called_once_with(
            conversation_id="conv-1",
            text="hello",
            source="asl",
        )

    def test_asl_transcribe_without_conversation_id_skips_firestore(self):
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
            data=json.dumps({"conversation_id": "conv-1"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())


if __name__ == "__main__":
    unittest.main()
