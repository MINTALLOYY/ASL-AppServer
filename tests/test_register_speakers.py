import json
import unittest
from unittest.mock import patch

# Patch heavy external dependencies before importing app
with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


class TestRegisterSpeakers(unittest.TestCase):

    def setUp(self):
        self.client = server_app.app.test_client()
        # Clear registry between tests
        server_app.speaker_registry.clear()

    # ── POST valid payload ──────────────────────────────────────────

    def test_register_valid_speakers(self):
        payload = {
            "conversation_id": "cid1",
            "speakers": [
                {"label": "Speaker_0", "name": "Marcus"},
                {"label": "Speaker_1", "name": "Priya"},
            ],
        }
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["registered"], 2)
        # Verify in-memory state
        self.assertEqual(
            server_app.speaker_registry["cid1"],
            {"Speaker_0": "Marcus", "Speaker_1": "Priya"},
        )

    def test_register_empty_speakers_list(self):
        payload = {"conversation_id": "cid2", "speakers": []}
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["registered"], 0)
        self.assertEqual(server_app.speaker_registry["cid2"], {})

    def test_register_omitted_speakers_key(self):
        payload = {"conversation_id": "cid3"}
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["registered"], 0)

    # ── POST invalid payloads ──────────────────────────────────────

    def test_missing_conversation_id(self):
        payload = {"speakers": [{"label": "Speaker_0", "name": "Marcus"}]}
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("conversation_id", resp.get_json()["error"])

    def test_speakers_not_a_list(self):
        payload = {"conversation_id": "cid4", "speakers": "not-a-list"}
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("speakers must be a list", resp.get_json()["error"])

    def test_malformed_speaker_entries_skipped(self):
        payload = {
            "conversation_id": "cid5",
            "speakers": [
                {"label": "Speaker_0", "name": "Marcus"},
                {"label": 123, "name": "Bad"},          # non-string label
                {"label": "Speaker_2"},                  # missing name
                "not-a-dict",                            # not a dict at all
                {"label": "", "name": "Empty"},          # empty label
                {"label": "Speaker_3", "name": ""},      # empty name
            ],
        }
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["registered"], 1)
        self.assertEqual(
            server_app.speaker_registry["cid5"],
            {"Speaker_0": "Marcus"},
        )

    def test_empty_body(self):
        resp = self.client.post(
            "/speech/register_speakers",
            data="",
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    # ── GET debug endpoint ─────────────────────────────────────────

    def test_get_registered_speakers(self):
        server_app.speaker_registry["cid6"] = {"Speaker_0": "Alice"}
        resp = self.client.get("/speech/register_speakers?conversation_id=cid6")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "cid6")
        self.assertEqual(body["speakers"], {"Speaker_0": "Alice"})

    def test_get_unknown_conversation(self):
        resp = self.client.get("/speech/register_speakers?conversation_id=unknown")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["speakers"], {})

    def test_get_missing_conversation_id(self):
        resp = self.client.get("/speech/register_speakers")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("conversation_id", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
