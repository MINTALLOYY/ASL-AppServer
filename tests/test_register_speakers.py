import json
import unittest
from unittest.mock import MagicMock, patch

import config

VALID_UUID = "123e4567-e89b-42d3-a456-426614174000"


with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


class TestRegisterSpeakers(unittest.TestCase):
    def setUp(self):
        self.client = server_app.app.test_client()
        server_app.speaker_registry.clear()
        self._orig_db = server_app.db
        self._orig_enforce = config.ENFORCE_CONVERSATION_UUID
        self.mock_db = MagicMock()
        server_app.db = self.mock_db
        config.ENFORCE_CONVERSATION_UUID = False

    def tearDown(self):
        server_app.db = self._orig_db
        config.ENFORCE_CONVERSATION_UUID = self._orig_enforce

    def test_register_valid_speakers(self):
        self.mock_db.get_conversation.return_value = {"conversation_uuid": VALID_UUID}
        payload = {
            "conversation_id": "cid1",
            "conversation_uuid": VALID_UUID,
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
        self.assertEqual(body["conversation_uuid"], VALID_UUID)
        self.assertEqual(
            server_app.speaker_registry["cid1"],
            {"Speaker_0": "Marcus", "Speaker_1": "Priya"},
        )

    def test_register_empty_speakers_list(self):
        self.mock_db.get_conversation.return_value = {"conversation_uuid": VALID_UUID}
        payload = {"conversation_id": "cid2", "conversation_uuid": VALID_UUID, "speakers": []}
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
        self.mock_db.get_conversation.return_value = {"conversation_uuid": VALID_UUID}
        payload = {"conversation_id": "cid3", "conversation_uuid": VALID_UUID}
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["registered"], 0)

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
        self.mock_db.get_conversation.return_value = {"conversation_uuid": VALID_UUID}
        payload = {"conversation_id": "cid4", "conversation_uuid": VALID_UUID, "speakers": "not-a-list"}
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("speakers must be a list", resp.get_json()["error"])

    def test_malformed_speaker_entries_skipped(self):
        self.mock_db.get_conversation.return_value = {"conversation_uuid": VALID_UUID}
        payload = {
            "conversation_id": "cid5",
            "conversation_uuid": VALID_UUID,
            "speakers": [
                {"label": "Speaker_0", "name": "Marcus"},
                {"label": 123, "name": "Bad"},
                {"label": "Speaker_2"},
                "not-a-dict",
                {"label": "", "name": "Empty"},
                {"label": "Speaker_3", "name": ""},
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
        self.assertEqual(server_app.speaker_registry["cid5"], {"Speaker_0": "Marcus"})

    def test_empty_body(self):
        resp = self.client.post(
            "/speech/register_speakers",
            data="",
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_get_registered_speakers(self):
        self.mock_db.get_conversation.return_value = {"conversation_uuid": VALID_UUID}
        server_app.speaker_registry["cid6"] = {"Speaker_0": "Alice"}
        resp = self.client.get(f"/speech/register_speakers?conversation_id=cid6&conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "cid6")
        self.assertEqual(body["conversation_uuid"], VALID_UUID)
        self.assertEqual(body["speakers"], {"Speaker_0": "Alice"})

    def test_get_unknown_conversation(self):
        self.mock_db.get_conversation.side_effect = LookupError("missing")
        resp = self.client.get(f"/speech/register_speakers?conversation_id=unknown&conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 404)

    def test_get_missing_conversation_id(self):
        resp = self.client.get("/speech/register_speakers")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("conversation_id", resp.get_json()["error"])

    def test_requires_uuid_when_enforced(self):
        config.ENFORCE_CONVERSATION_UUID = True
        self.mock_db.get_conversation.return_value = {"conversation_uuid": VALID_UUID}
        payload = {"conversation_id": "cid7", "speakers": []}
        resp = self.client.post(
            "/speech/register_speakers",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("missing_conversation_uuid", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
