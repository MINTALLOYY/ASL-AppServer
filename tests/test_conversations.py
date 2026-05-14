"""
Tests for the /conversations HTTP routes and the FirestoreDB helper methods.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import config

VALID_UUID = "123e4567-e89b-42d3-a456-426614174000"


with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


def _make_mock_db():
    return MagicMock()


class TestConversationRoutes(unittest.TestCase):
    def setUp(self):
        self.client = server_app.app.test_client()
        self.mock_db = _make_mock_db()
        self._orig_db = server_app.db
        self._orig_enforce = config.ENFORCE_CONVERSATION_UUID
        server_app.db = self.mock_db
        config.ENFORCE_CONVERSATION_UUID = False

    def tearDown(self):
        server_app.db = self._orig_db
        config.ENFORCE_CONVERSATION_UUID = self._orig_enforce

    def test_create_conversation_returns_uuid(self):
        self.mock_db.create_conversation.return_value = {
            "conversation_id": "auto-generated-id",
            "conversation_uuid": VALID_UUID,
        }
        resp = self.client.post(
            "/conversations",
            data=json.dumps({"conversation_uuid": VALID_UUID}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "auto-generated-id")
        self.assertEqual(body["conversation_uuid"], VALID_UUID)
        self.mock_db.create_conversation.assert_called_once_with(
            conversation_id=None,
            conversation_uuid=VALID_UUID,
            owner_hint=None,
        )

    def test_create_conversation_explicit_id(self):
        self.mock_db.create_conversation.return_value = {
            "conversation_id": "my-conv-123",
            "conversation_uuid": VALID_UUID,
        }
        resp = self.client.post(
            "/conversations",
            data=json.dumps({"conversation_id": "my-conv-123", "conversation_uuid": VALID_UUID}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "my-conv-123")
        self.assertEqual(body["conversation_uuid"], VALID_UUID)
        self.mock_db.create_conversation.assert_called_once_with(
            conversation_id="my-conv-123",
            conversation_uuid=VALID_UUID,
            owner_hint=None,
        )

    def test_create_conversation_with_non_uuid_identifier(self):
        self.mock_db.create_conversation.return_value = {
            "conversation_id": "auto-generated-id",
            "conversation_uuid": "firebase_uid_abc123",
        }
        resp = self.client.post(
            "/conversations",
            data=json.dumps({"conversation_uuid": "firebase_uid_abc123"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)

    def test_create_conversation_requires_uuid_when_enforced(self):
        config.ENFORCE_CONVERSATION_UUID = True
        resp = self.client.post(
            "/conversations",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("missing_conversation_uuid", resp.get_json()["error"])

    def test_list_conversations_scoped_to_uuid(self):
        self.mock_db.list_conversations.return_value = [
            {"conversation_id": "a", "status": "active", "conversation_uuid": VALID_UUID, "updated_at": "2024-01-01T00:00:00"},
        ]
        resp = self.client.get(f"/conversations?conversation_uuid={VALID_UUID}&limit=10")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(len(body["conversations"]), 1)
        self.mock_db.list_conversations.assert_called_once_with(limit=10, conversation_uuid=VALID_UUID)

    def test_list_conversations_requires_uuid_when_enforced(self):
        config.ENFORCE_CONVERSATION_UUID = True
        resp = self.client.get("/conversations")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("missing_conversation_uuid", resp.get_json()["error"])

    def test_list_conversations_db_unavailable(self):
        server_app.db = None
        resp = self.client.get(f"/conversations?conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 503)

    def test_get_conversation_found(self):
        self.mock_db.get_conversation.return_value = {
            "conversation_id": "conv1",
            "conversation_uuid": VALID_UUID,
            "status": "active",
            "created_at": "2024-01-01T00:00:00",
        }
        self.mock_db.get_messages.return_value = [
            {"id": "m1", "text": "hello", "type": "speech", "speaker": "Alice", "created_at": "2024-01-01T00:01:00"},
            {"id": "m2", "text": "world", "type": "asl", "speaker": None, "created_at": "2024-01-01T00:02:00"},
        ]
        resp = self.client.get(f"/conversations/conv1?conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "conv1")
        self.assertEqual(len(body["messages"]), 2)
        self.mock_db.get_conversation.assert_called_once_with("conv1", conversation_uuid=VALID_UUID)
        self.mock_db.get_messages.assert_called_once_with("conv1", conversation_uuid=VALID_UUID)

    def test_get_conversation_access_denied(self):
        self.mock_db.get_conversation.side_effect = PermissionError("access denied")
        resp = self.client.get(f"/conversations/conv1?conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 403)

    def test_get_conversation_not_found(self):
        self.mock_db.get_conversation.return_value = None
        resp = self.client.get(f"/conversations/nonexistent?conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 404)

    def test_get_messages_returns_list(self):
        self.mock_db.get_messages.return_value = [
            {"id": "m1", "text": "hi", "type": "speech", "speaker": "Bob", "created_at": "2024-01-01T00:00:00"},
        ]
        resp = self.client.get(f"/conversations/conv2/messages?conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "conv2")
        self.assertEqual(len(body["messages"]), 1)
        self.mock_db.get_messages.assert_called_once_with("conv2", conversation_uuid=VALID_UUID)

    def test_get_messages_access_denied(self):
        self.mock_db.get_messages.side_effect = PermissionError("access denied")
        resp = self.client.get(f"/conversations/conv2/messages?conversation_uuid={VALID_UUID}")
        self.assertEqual(resp.status_code, 403)


class TestFirestoreDBMethods(unittest.TestCase):
    def _make_db(self):
        from firebase.db import FirestoreDB

        db = FirestoreDB.__new__(FirestoreDB)
        db.client = MagicMock()
        return db

    def test_create_conversation_auto_id(self):
        db = self._make_db()
        mock_ref = MagicMock()
        mock_ref.id = "new-id"
        db.client.collection.return_value.document.return_value = mock_ref

        result = db.create_conversation(conversation_uuid=VALID_UUID)
        self.assertEqual(result["conversation_id"], "new-id")
        self.assertEqual(result["conversation_uuid"], VALID_UUID)
        mock_ref.set.assert_called_once()
        payload = mock_ref.set.call_args[0][0]
        self.assertEqual(payload["conversation_uuid"], VALID_UUID)

    def test_create_conversation_explicit_id(self):
        db = self._make_db()
        mock_ref = MagicMock()
        mock_ref.id = "explicit-id"
        db.client.collection.return_value.document.return_value = mock_ref

        result = db.create_conversation(conversation_id="explicit-id", conversation_uuid=VALID_UUID)
        self.assertEqual(result["conversation_id"], "explicit-id")
        self.assertEqual(result["conversation_uuid"], VALID_UUID)
        db.client.collection.return_value.document.assert_called_with("explicit-id")

    def test_save_transcript_calls_set(self):
        db = self._make_db()
        conv_ref = MagicMock()
        msg_ref = MagicMock()
        doc = MagicMock()
        doc.exists = True
        doc.to_dict.return_value = {"conversation_uuid": VALID_UUID}
        db.client.collection.return_value.document.return_value = conv_ref
        conv_ref.get.return_value = doc
        conv_ref.collection.return_value.document.return_value = msg_ref

        db.save_transcript("conv1", VALID_UUID, {"text": "hello", "type": "speech", "speaker": "Alice"})
        conv_ref.set.assert_called()
        msg_ref.set.assert_called_once()
        call_args = msg_ref.set.call_args[0][0]
        self.assertEqual(call_args["text"], "hello")
        self.assertEqual(call_args["type"], "speech")
        self.assertEqual(call_args["speaker"], "Alice")
        self.assertEqual(call_args["conversation_uuid"], VALID_UUID)

    def test_save_message_wrapper(self):
        db = self._make_db()
        db.save_transcript = MagicMock()
        db.save_message("conv1", "hello", "speech", speaker="Alice", conversation_uuid=VALID_UUID)
        db.save_transcript.assert_called_once()

    def test_save_transcript_no_op_when_empty_id(self):
        db = self._make_db()
        db.save_transcript("", VALID_UUID, {"text": "hello"})
        db.client.collection.assert_not_called()

    def test_get_conversation_returns_none_when_missing(self):
        db = self._make_db()
        mock_doc = MagicMock()
        mock_doc.exists = False
        db.client.collection.return_value.document.return_value.get.return_value = mock_doc

        result = db.get_conversation("missing")
        self.assertIsNone(result)

    def test_get_conversation_rejects_uuid_mismatch(self):
        db = self._make_db()
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.id = "conv1"
        mock_doc.to_dict.return_value = {"conversation_uuid": "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"}
        db.client.collection.return_value.document.return_value.get.return_value = mock_doc

        with self.assertRaises(PermissionError):
            db.get_conversation("conv1", conversation_uuid=VALID_UUID)

    def test_get_conversation_returns_dict(self):
        db = self._make_db()
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.id = "conv1"
        mock_doc.to_dict.return_value = {"status": "active", "conversation_uuid": VALID_UUID}
        db.client.collection.return_value.document.return_value.get.return_value = mock_doc

        result = db.get_conversation("conv1", conversation_uuid=VALID_UUID)
        self.assertIsNotNone(result)
        self.assertEqual(result["conversation_id"], "conv1")
        self.assertEqual(result["status"], "active")

    def test_get_messages_returns_ordered_list(self):
        db = self._make_db()
        conv_ref = MagicMock()
        doc = MagicMock()
        doc.exists = True
        doc.to_dict.return_value = {"conversation_uuid": VALID_UUID}
        db.client.collection.return_value.document.return_value = conv_ref
        conv_ref.get.return_value = doc

        doc1, doc2 = MagicMock(), MagicMock()
        doc1.id = "m1"
        doc1.to_dict.return_value = {"text": "first", "type": "speech", "speaker": "A"}
        doc2.id = "m2"
        doc2.to_dict.return_value = {"text": "second", "type": "asl", "speaker": None}

        (conv_ref.collection.return_value.order_by.return_value.stream.return_value) = iter([doc1, doc2])

        messages = db.get_messages("conv1", conversation_uuid=VALID_UUID)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["id"], "m1")
        self.assertEqual(messages[0]["text"], "first")
        self.assertEqual(messages[1]["id"], "m2")

    def test_finalize_conversation_no_op_on_empty(self):
        db = self._make_db()
        db.finalize_conversation("")
        db.client.collection.assert_not_called()

    def test_list_conversations_scoped_by_uuid(self):
        db = self._make_db()
        (db.client.collection.return_value
         .where.return_value
         .order_by.return_value
         .limit.return_value
         .stream.return_value) = iter([])

        db.list_conversations(limit=10, conversation_uuid=VALID_UUID)
        db.client.collection.return_value.where.assert_called_with("conversation_uuid", "==", VALID_UUID)

    def test_list_conversations_respects_limit(self):
        db = self._make_db()
        (db.client.collection.return_value
         .order_by.return_value
         .limit.return_value
         .stream.return_value) = iter([])

        db.list_conversations(limit=10)
        db.client.collection.return_value.order_by.return_value.limit.assert_called_with(10)

    def test_list_conversations_caps_at_100(self):
        db = self._make_db()
        (db.client.collection.return_value
         .order_by.return_value
         .limit.return_value
         .stream.return_value) = iter([])

        db.list_conversations(limit=999)
        db.client.collection.return_value.order_by.return_value.limit.assert_called_with(100)


if __name__ == "__main__":
    unittest.main()
