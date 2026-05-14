import uuid
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
            - conversation_uuid: account-scoped user identifier (Firebase UID or similar)
            - owner_hint: optional string
            messages/{message_id}  (subcollection, ordered by created_at)
                - text: string
                - type: "speech" | "asl"
                - speaker: string (optional)
                - conversation_uuid: account-scoped user identifier
                - created_at: timestamp
    """

    def __init__(self, project_id: Optional[str] = None):
        self.client = firestore.Client(project=project_id)

    # -- helpers -------------------------------------------------------------

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

    def _conversation_ref(self, conversation_id: str):
        return self.client.collection("conversations").document(conversation_id)

    def _get_conversation_doc(self, conversation_id: str):
        if not conversation_id:
            return None
        doc = self._conversation_ref(conversation_id).get()
        if not doc.exists:
            return None
        return doc

    def _ensure_conversation_access(
        self,
        conversation_id: str,
        conversation_uuid: Optional[str] = None,
    ) -> dict:
        """
        Return the stored conversation document data and enforce UUID match when provided.
        """
        if not conversation_id:
            raise ValueError("conversation_id is required")
        doc = self._get_conversation_doc(conversation_id)
        if doc is None:
            raise LookupError("Conversation not found")
        data = doc.to_dict() or {}
        stored_uuid = data.get("conversation_uuid")
        if conversation_uuid and stored_uuid and stored_uuid != conversation_uuid:
            raise PermissionError("access denied")
        data["conversation_id"] = doc.id
        return data

    def _attach_uuid_if_missing(self, conversation_id: str, conversation_uuid: Optional[str]) -> Optional[str]:
        """
        Backfill conversation_uuid on older documents when we already know the accessor identifier.
        """
        if not conversation_id or not conversation_uuid:
            return conversation_uuid

        doc = self._get_conversation_doc(conversation_id)
        if doc is None:
            raise LookupError("Conversation not found")
        data = doc.to_dict() or {}
        stored_uuid = data.get("conversation_uuid")
        if stored_uuid:
            if stored_uuid != conversation_uuid:
                raise PermissionError("access denied")
            return stored_uuid

        self._conversation_ref(conversation_id).set(
            {
                "conversation_uuid": conversation_uuid,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return conversation_uuid

    # -- write operations ----------------------------------------------------

    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        conversation_uuid: Optional[str] = None,
        owner_hint: Optional[str] = None,
    ) -> dict:
        """
        Create a new conversation document and return its identifiers.

        Returns:
            dict with conversation_id and conversation_uuid.
        """
        if conversation_id:
            conv_ref = self.client.collection("conversations").document(conversation_id)
        else:
            conv_ref = self.client.collection("conversations").document()

        resolved_uuid = conversation_uuid or str(uuid.uuid4())
        payload = {
            "status": "active",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "conversation_uuid": resolved_uuid,
        }
        if isinstance(owner_hint, str) and owner_hint.strip():
            payload["owner_hint"] = owner_hint.strip()

        conv_ref.set(payload, merge=True)
        return {"conversation_id": conv_ref.id, "conversation_uuid": resolved_uuid}

    def save_transcript(self, conversation_id: str, conversation_uuid: Optional[str], segment: dict):
        """
        Save a transcript segment to Firestore under a conversation.
        """
        if not conversation_id:
            return
        if not isinstance(segment, dict):
            raise TypeError("segment must be a dict")

        conv_data = self._ensure_conversation_access(conversation_id, conversation_uuid)
        resolved_uuid = conversation_uuid or conv_data.get("conversation_uuid")
        if resolved_uuid:
            resolved_uuid = self._attach_uuid_if_missing(conversation_id, resolved_uuid)

        conv_ref = self._conversation_ref(conversation_id)
        conv_ref.set(
            {
                "status": "active",
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        msg_ref = conv_ref.collection("messages").document()
        payload = dict(segment)
        payload.setdefault("type", "speech")
        payload.setdefault("speaker", None)
        if resolved_uuid:
            payload["conversation_uuid"] = resolved_uuid
        payload["created_at"] = firestore.SERVER_TIMESTAMP
        msg_ref.set(payload)

    def save_message(
        self,
        conversation_id: str,
        text: str,
        source: str,
        speaker: Optional[str] = None,
        conversation_uuid: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Backwards-compatible wrapper for transcript writes.
        """
        segment = {
            "text": text,
            "type": source,
            "speaker": speaker,
        }
        if isinstance(metadata, dict) and metadata:
            segment.update(metadata)
        self.save_transcript(conversation_id, conversation_uuid, segment)

    def set_conversation_display_name_if_missing(
        self,
        conversation_id: str,
        text: str,
        conversation_uuid: Optional[str] = None,
    ) -> Optional[str]:
        """
        Set conversation display_name from text only when missing.
        """
        if not conversation_id:
            return None
        display_name = self._make_display_name(text)
        if not display_name:
            return None

        conv_data = self._ensure_conversation_access(conversation_id, conversation_uuid)
        stored_uuid = conv_data.get("conversation_uuid")
        resolved_uuid = conversation_uuid or stored_uuid
        if resolved_uuid:
            self._attach_uuid_if_missing(conversation_id, resolved_uuid)

        conv_ref = self._conversation_ref(conversation_id)
        existing = conv_data.get("display_name")
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

    def finalize_conversation(self, conversation_id: str, conversation_uuid: Optional[str] = None):
        """
        Mark a conversation as finalized.
        """
        if not conversation_id:
            return
        conv_data = self._ensure_conversation_access(conversation_id, conversation_uuid)
        stored_uuid = conv_data.get("conversation_uuid")
        resolved_uuid = conversation_uuid or stored_uuid
        if resolved_uuid:
            self._attach_uuid_if_missing(conversation_id, resolved_uuid)

        conv_ref = self._conversation_ref(conversation_id)
        conv_ref.set(
            {"status": "finalized", "updated_at": firestore.SERVER_TIMESTAMP},
            merge=True,
        )

    # -- read operations -----------------------------------------------------

    def get_conversation(
        self,
        conversation_id: str,
        conversation_uuid: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Retrieve a conversation document (without messages).
        """
        if not conversation_id:
            return None
        doc = self._get_conversation_doc(conversation_id)
        if doc is None:
            return None
        data = self._serialize(doc.to_dict() or {})
        stored_uuid = data.get("conversation_uuid")
        if conversation_uuid and stored_uuid and stored_uuid != conversation_uuid:
            raise PermissionError("access denied")
        data["conversation_id"] = doc.id
        return data

    def get_messages(
        self,
        conversation_id: str,
        conversation_uuid: Optional[str] = None,
    ) -> list:
        """
        Retrieve all messages for a conversation in chronological order.
        """
        if not conversation_id:
            return []
        conv_data = self._ensure_conversation_access(conversation_id, conversation_uuid)
        stored_uuid = conv_data.get("conversation_uuid")
        if stored_uuid:
            self._attach_uuid_if_missing(conversation_id, stored_uuid)

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

    def query_conversations_by_uuid(self, conversation_uuid: str, limit: int = 50) -> list:
        """
        List conversations scoped to a specific conversation_uuid.
        """
        if not conversation_uuid:
            return []
        limit = min(limit, 100)
        query = (
            self.client.collection("conversations")
            .where("conversation_uuid", "==", conversation_uuid)
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        result = []
        for doc in query.stream():
            data = self._serialize(doc.to_dict() or {})
            data["conversation_id"] = doc.id
            result.append(data)
        return result

    def list_conversations(
        self,
        limit: int = 50,
        conversation_uuid: Optional[str] = None,
    ) -> list:
        """
        List conversations ordered by most-recently-updated first.
        """
        limit = min(limit, 100)
        if conversation_uuid:
            return self.query_conversations_by_uuid(conversation_uuid, limit=limit)

        query = (
            self.client.collection("conversations")
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        result = []
        for doc in query.stream():
            data = self._serialize(doc.to_dict() or {})
            data["conversation_id"] = doc.id
            result.append(data)
        return result
