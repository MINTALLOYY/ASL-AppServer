import os
from typing import Optional
from google.cloud import firestore


class FirestoreDB:
    """
    Simple Firestore client for saving conversation messages and finalizing conversations.
    
    Structure:
        conversations/{conversation_id}
            - status: "finalized" (optional)
            - updated_at: timestamp
            messages/{message_id}
                - text: string
                - type: "speech" | "asl"
                - speaker: string (optional)
                - created_at: timestamp
    """
    def __init__(self, project_id: Optional[str] = None):
        self.client = firestore.Client(project=project_id)
#
    def save_message(self, conversation_id: str, text: str, source: str, speaker: Optional[str] = None):
        """
        Save a transcript message to Firestore under a conversation.
        
        Args:
            conversation_id: Firestore document ID for the conversation.
            text: Transcript text.
            source: Either "speech" or "asl".
            speaker: Optional speaker label from diarization.
        """
        if not conversation_id:
            return
        conv_ref = self.client.collection("conversations").document(conversation_id)
        conv_ref.set({"updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
        msg_ref = conv_ref.collection("messages").document()
        payload = {
            "text": text,
            "type": source,
            "speaker": speaker,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        msg_ref.set(payload)

    def finalize_conversation(self, conversation_id: str):
        """
        Mark a conversation as finalized.
        
        Args:
            conversation_id: Firestore document ID for the conversation.
        """
        if not conversation_id:
            return
        conv_ref = self.client.collection("conversations").document(conversation_id)
        conv_ref.set({"status": "finalized", "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
