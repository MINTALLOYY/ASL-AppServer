"""
Tests for the /conversations HTTP routes and the FirestoreDB helper methods.

All Firestore and external calls are mocked so these tests run fully offline.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

# Patch heavy dependencies before importing app so they are never actually loaded.
with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


def _make_mock_db():
    """Return a MagicMock that looks like a FirestoreDB instance."""
    return MagicMock()


class TestConversationRoutes(unittest.TestCase):

    def setUp(self):
        self.client = server_app.app.test_client()
        # Replace the module-level db with a fresh mock for each test.
        self.mock_db = _make_mock_db()
        self.mock_db.get_conversation.return_value = None
        self._orig_db = server_app.db
        server_app.db = self.mock_db
        self.auth_patcher = patch.object(server_app, "_require_request_user_id", return_value=("user-1", None))
        self.auth_patcher.start()

    def tearDown(self):
        self.auth_patcher.stop()
        server_app.db = self._orig_db

    # ── POST /conversations ────────────────────────────────────────────────

    def test_create_conversation_auto_id(self):
        self.mock_db.create_conversation.return_value = "auto-generated-id"
        resp = self.client.post(
            "/conversations",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "auto-generated-id")
        self.assertEqual(body["status"], "active")
        self.mock_db.create_conversation.assert_called_once_with(conversation_id=None, user_id="user-1")

    def test_create_conversation_explicit_id(self):
        self.mock_db.create_conversation.return_value = "my-conv-123"
        resp = self.client.post(
            "/conversations",
            data=json.dumps({"conversation_id": "my-conv-123"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "my-conv-123")
        self.mock_db.create_conversation.assert_called_once_with(conversation_id="my-conv-123", user_id="user-1")

    def test_create_conversation_db_error(self):
        self.mock_db.create_conversation.side_effect = RuntimeError("firestore down")
        resp = self.client.post(
            "/conversations",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 500)
        self.assertIn("error", resp.get_json())

    def test_create_conversation_db_unavailable(self):
        server_app.db = None
        resp = self.client.post("/conversations", content_type="application/json")
        self.assertEqual(resp.status_code, 503)

    # ── GET /conversations ─────────────────────────────────────────────────

    def test_list_conversations_default(self):
        self.mock_db.list_conversations.return_value = [
            {"conversation_id": "a", "status": "active", "updated_at": "2024-01-01T00:00:00"},
        ]
        resp = self.client.get("/conversations")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("conversations", body)
        self.assertEqual(len(body["conversations"]), 1)
        self.mock_db.list_conversations.assert_called_once_with(limit=50, user_id="user-1")

    def test_list_conversations_custom_limit(self):
        self.mock_db.list_conversations.return_value = []
        resp = self.client.get("/conversations?limit=10")
        self.assertEqual(resp.status_code, 200)
        self.mock_db.list_conversations.assert_called_once_with(limit=10, user_id="user-1")

    def test_list_conversations_bad_limit_uses_default(self):
        self.mock_db.list_conversations.return_value = []
        resp = self.client.get("/conversations?limit=notanumber")
        self.assertEqual(resp.status_code, 200)
        # Bad value falls back to 50
        self.mock_db.list_conversations.assert_called_once_with(limit=50, user_id="user-1")

    def test_list_conversations_limit_capped_at_100(self):
        self.mock_db.list_conversations.return_value = []
        resp = self.client.get("/conversations?limit=150")
        self.assertEqual(resp.status_code, 200)
        # Route passes the value through; the DB layer caps it at 100
        self.mock_db.list_conversations.assert_called_once_with(limit=150, user_id="user-1")

    def test_list_conversations_db_unavailable(self):
        server_app.db = None
        resp = self.client.get("/conversations")
        self.assertEqual(resp.status_code, 503)

    # ── GET /conversations/<id> ────────────────────────────────────────────

    def test_get_conversation_found(self):
        self.mock_db.get_conversation.return_value = {
            "conversation_id": "conv1",
            "status": "active",
            "user_id": "user-1",
            "created_at": "2024-01-01T00:00:00",
        }
        self.mock_db.get_messages.return_value = [
            {"id": "m1", "text": "hello", "type": "speech", "speaker": "Alice", "created_at": "2024-01-01T00:01:00"},
            {"id": "m2", "text": "world", "type": "asl", "speaker": None, "created_at": "2024-01-01T00:02:00"},
        ]
        resp = self.client.get("/conversations/conv1")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "conv1")
        self.assertEqual(len(body["messages"]), 2)
        # Messages are in chronological order (as returned by mock)
        self.assertEqual(body["messages"][0]["text"], "hello")
        self.assertEqual(body["messages"][1]["text"], "world")

    def test_get_conversation_not_found(self):
        self.mock_db.get_conversation.return_value = None
        resp = self.client.get("/conversations/nonexistent")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.get_json())

    def test_get_conversation_db_unavailable(self):
        server_app.db = None
        resp = self.client.get("/conversations/conv1")
        self.assertEqual(resp.status_code, 503)

    def test_get_conversation_db_error(self):
        self.mock_db.get_conversation.side_effect = RuntimeError("timeout")
        resp = self.client.get("/conversations/conv1")
        self.assertEqual(resp.status_code, 500)

    # ── GET /conversations/<id>/messages ───────────────────────────────────

    def test_get_messages_returns_list(self):
        self.mock_db.get_conversation.return_value = {"conversation_id": "conv2", "user_id": "user-1"}
        self.mock_db.get_messages.return_value = [
            {"id": "m1", "text": "hi", "type": "speech", "speaker": "Bob", "created_at": "2024-01-01T00:00:00"},
        ]
        resp = self.client.get("/conversations/conv2/messages")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["conversation_id"], "conv2")
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["text"], "hi")

    def test_get_messages_empty(self):
        self.mock_db.get_conversation.return_value = {"conversation_id": "conv3", "user_id": "user-1"}
        self.mock_db.get_messages.return_value = []
        resp = self.client.get("/conversations/conv3/messages")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["messages"], [])

    def test_get_messages_db_unavailable(self):
        server_app.db = None
        resp = self.client.get("/conversations/conv4/messages")
        self.assertEqual(resp.status_code, 503)

    def test_get_messages_db_error(self):
        self.mock_db.get_conversation.return_value = {"conversation_id": "conv4", "user_id": "user-1"}
        self.mock_db.get_messages.side_effect = RuntimeError("firestore error")
        resp = self.client.get("/conversations/conv4/messages")
        self.assertEqual(resp.status_code, 500)

    def test_get_conversation_forbidden_user_hidden_as_not_found(self):
        self.mock_db.get_conversation.return_value = {
            "conversation_id": "conv1",
            "status": "active",
            "user_id": "other-user",
        }
        resp = self.client.get("/conversations/conv1")
        self.assertEqual(resp.status_code, 404)

    def test_get_messages_forbidden_user_hidden_as_not_found(self):
        self.mock_db.get_conversation.return_value = {
            "conversation_id": "conv9",
            "status": "active",
            "user_id": "other-user",
        }
        resp = self.client.get("/conversations/conv9/messages")
        self.assertEqual(resp.status_code, 404)

    def test_list_conversations_requires_auth(self):
        with patch.object(
            server_app,
            "_require_request_user_id",
            return_value=(None, ({"error": "Missing Authorization header"}, 401)),
        ):
            resp = self.client.get("/conversations")
        self.assertEqual(resp.status_code, 401)


class TestFirestoreDBMethods(unittest.TestCase):
    """Unit tests for FirestoreDB read/write methods using a mock Firestore client."""

    def _make_db(self):
        """Build a FirestoreDB instance with a mocked firestore.Client."""
        from firebase.db import FirestoreDB
        db = FirestoreDB.__new__(FirestoreDB)
        db.client = MagicMock()
        return db

    def test_create_conversation_auto_id(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        mock_ref = MagicMock()
        mock_ref.id = "new-id"
        db.client.collection.return_value.document.return_value = mock_ref

        result = db.create_conversation()
        self.assertEqual(result, "new-id")
        mock_ref.set.assert_called_once()

    def test_create_conversation_explicit_id(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        mock_ref = MagicMock()
        mock_ref.id = "explicit-id"
        db.client.collection.return_value.document.return_value = mock_ref

        result = db.create_conversation(conversation_id="explicit-id")
        self.assertEqual(result, "explicit-id")
        db.client.collection.return_value.document.assert_called_with("explicit-id")

    def test_save_message_calls_set(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        conv_ref = MagicMock()
        msg_ref = MagicMock()
        db.client.collection.return_value.document.return_value = conv_ref
        conv_ref.collection.return_value.document.return_value = msg_ref

        db.save_message("conv1", "hello", "speech", speaker="Alice")
        conv_ref.set.assert_called()
        msg_ref.set.assert_called_once()
        call_args = msg_ref.set.call_args[0][0]
        self.assertEqual(call_args["text"], "hello")
        self.assertEqual(call_args["type"], "speech")
        self.assertEqual(call_args["speaker"], "Alice")

    def test_save_message_no_op_when_empty_id(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        db.save_message("", "hello", "speech")
        db.client.collection.assert_not_called()

    def test_get_conversation_returns_none_when_missing(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        mock_doc = MagicMock()
        mock_doc.exists = False
        db.client.collection.return_value.document.return_value.get.return_value = mock_doc

        result = db.get_conversation("missing")
        self.assertIsNone(result)

    def test_get_conversation_returns_dict(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.id = "conv1"
        mock_doc.to_dict.return_value = {"status": "active"}
        db.client.collection.return_value.document.return_value.get.return_value = mock_doc

        result = db.get_conversation("conv1")
        self.assertIsNotNone(result)
        self.assertEqual(result["conversation_id"], "conv1")
        self.assertEqual(result["status"], "active")

    def test_get_messages_returns_ordered_list(self):
        from firebase.db import FirestoreDB
        db = self._make_db()

        doc1, doc2 = MagicMock(), MagicMock()
        doc1.id = "m1"
        doc1.to_dict.return_value = {"text": "first", "type": "speech", "speaker": "A"}
        doc2.id = "m2"
        doc2.to_dict.return_value = {"text": "second", "type": "asl", "speaker": None}

        (db.client.collection.return_value
         .document.return_value
         .collection.return_value
         .order_by.return_value
         .stream.return_value) = iter([doc1, doc2])

        messages = db.get_messages("conv1")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["id"], "m1")
        self.assertEqual(messages[0]["text"], "first")
        self.assertEqual(messages[1]["id"], "m2")

    def test_get_messages_empty_id_returns_empty(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        result = db.get_messages("")
        self.assertEqual(result, [])
        db.client.collection.assert_not_called()

    def test_serialize_converts_timestamps(self):
        from firebase.db import FirestoreDB
        from datetime import datetime, timezone
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {"text": "hello", "created_at": ts, "speaker": None}
        result = FirestoreDB._serialize(data)
        self.assertEqual(result["created_at"], ts.isoformat())
        self.assertEqual(result["text"], "hello")

    def test_finalize_conversation_no_op_on_empty(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        db.finalize_conversation("")
        db.client.collection.assert_not_called()

    def test_list_conversations_respects_limit(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        (db.client.collection.return_value
         .order_by.return_value
         .limit.return_value
         .stream.return_value) = iter([])

        db.list_conversations(limit=10)
        db.client.collection.return_value.order_by.return_value.limit.assert_called_with(10)

    def test_list_conversations_caps_at_100(self):
        from firebase.db import FirestoreDB
        db = self._make_db()
        (db.client.collection.return_value
         .order_by.return_value
         .limit.return_value
         .stream.return_value) = iter([])

        db.list_conversations(limit=999)
        db.client.collection.return_value.order_by.return_value.limit.assert_called_with(100)


if __name__ == "__main__":
    unittest.main()
