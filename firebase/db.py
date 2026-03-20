import os
from typing import Optional
from google.cloud import firestore


class FirestoreDB:
    """
    Firestore client for saving and retrieving captioning conversations.

    Structure:
        conversations/{conversation_id}
            - status: "active" | "finalized"
            - created_at: timestamp
            - updated_at: timestamp
            messages/{message_id}  (subcollection, ordered by created_at)
                - text: string
                - type: "speech" | "asl"
                - speaker: string (optional)
                - created_at: timestamp
    """
    def __init__(self, project_id: Optional[str] = None):
        self.client = firestore.Client(project=project_id)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _serialize(data: dict) -> dict:
        """Convert Firestore Timestamp values to ISO-8601 strings for JSON responses."""
        result = {}
        for key, value in data.items():
            if hasattr(value, "isoformat"):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    @staticmethod
    def _make_display_name(text: str, max_len: int = 64) -> Optional[str]:
        """Build a short, readable conversation title from transcript text."""
        if not isinstance(text, str):
            return None
        normalized = " ".join(text.strip().split())
        if not normalized:
            return None
        if len(normalized) <= max_len:
            return normalized
        cut = normalized[:max_len].rstrip()
        return f"{cut}..."

    # ── write operations ──────────────────────────────────────────────────────

    def create_conversation(self, conversation_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation document and return its ID.

        Args:
            conversation_id: Optional explicit document ID. If omitted, Firestore
                             generates one automatically.

        Returns:
            The conversation_id of the created document.
        """
        if conversation_id:
            conv_ref = self.client.collection("conversations").document(conversation_id)
        else:
            conv_ref = self.client.collection("conversations").document()

        payload = {
            "status": "active",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        if user_id:
            payload["user_id"] = user_id

        conv_ref.set(payload, merge=True)
        return conv_ref.id

    def save_message(
        self,
        conversation_id: str,
        text: str,
        source: str,
        speaker: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Save a transcript message to Firestore under a conversation.

        Creates the conversation document (with created_at / status) on first write
        so callers do not need to call create_conversation separately.

        Args:
            conversation_id: Firestore document ID for the conversation.
            text: Transcript text.
            source: Either "speech" or "asl".
            speaker: Optional speaker label from diarization.
        """
        if not conversation_id:
            return
        conv_ref = self.client.collection("conversations").document(conversation_id)
        # Initialize status/created_at on first message; always bump updated_at.
        payload = {
            "status": "active",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        if user_id:
            payload["user_id"] = user_id
        conv_ref.set(payload, merge=True)
        msg_ref = conv_ref.collection("messages").document()
        payload = {
            "text": text,
            "type": source,
            "speaker": speaker,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        msg_ref.set(payload)

    def set_conversation_display_name_if_missing(self, conversation_id: str, text: str) -> Optional[str]:
        """
        Set conversation display_name from text only when missing.

        Returns:
            The existing or newly set display name, or None if unavailable.
        """
        if not conversation_id:
            return None
        display_name = self._make_display_name(text)
        if not display_name:
            return None

        conv_ref = self.client.collection("conversations").document(conversation_id)
        snap = conv_ref.get()
        existing = None
        if snap.exists:
            existing = (snap.to_dict() or {}).get("display_name")
        if isinstance(existing, str) and existing.strip():
            return existing

        conv_ref.set(
            {
                "display_name": display_name,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return display_name

    def finalize_conversation(self, conversation_id: str):
        """
        Mark a conversation as finalized.

        Args:
            conversation_id: Firestore document ID for the conversation.
        """
        if not conversation_id:
            return
        conv_ref = self.client.collection("conversations").document(conversation_id)
        conv_ref.set(
            {"status": "finalized", "updated_at": firestore.SERVER_TIMESTAMP},
            merge=True,
        )

    # ── read operations ───────────────────────────────────────────────────────

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """
        Retrieve a conversation document (without messages).

        Returns:
            dict with conversation fields plus "conversation_id", or None if not found.
        """
        if not conversation_id:
            return None
        doc = self.client.collection("conversations").document(conversation_id).get()
        if not doc.exists:
            return None
        data = self._serialize(doc.to_dict() or {})
        data["conversation_id"] = doc.id
        return data

    def get_messages(self, conversation_id: str) -> list:
        """
        Retrieve all messages for a conversation in chronological order.

        Messages are ordered by their server-assigned created_at timestamp so the
        list is a chronological record of the captioning session.

        Returns:
            List of message dicts, each with an "id" field and serialized timestamps.
        """
        if not conversation_id:
            return []
        messages_ref = (
            self.client.collection("conversations")
            .document(conversation_id)
            .collection("messages")
            .order_by("created_at")
        )
        result = []
        for doc in messages_ref.stream():
            data = self._serialize(doc.to_dict() or {})
            data["id"] = doc.id
            result.append(data)
        return result

    def list_conversations(self, limit: int = 50, user_id: Optional[str] = None) -> list:
        """
        List conversations ordered by most-recently-updated first.

        Args:
            limit: Maximum number of conversations to return (default 50, max 100).

        Returns:
            List of conversation dicts, each with a "conversation_id" field.
        """
        limit = min(limit, 100)
        query = self.client.collection("conversations")
        if user_id:
            # Requires composite index in Firestore: user_id ASC, updated_at DESC.
            query = query.where("user_id", "==", user_id)
        query = query.order_by("updated_at", direction=firestore.Query.DESCENDING).limit(limit)
        result = []
        for doc in query.stream():
            data = self._serialize(doc.to_dict() or {})
            data["conversation_id"] = doc.id
            result.append(data)
        return result
