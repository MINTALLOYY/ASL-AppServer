#!/usr/bin/env python3
"""
Backfill missing conversation_uuid fields on existing conversation documents.

Usage:
    python scripts/backfill_conversation_uuid.py [--dry-run]
"""

import argparse
import json
import uuid

from google.cloud import firestore

from firebase.db import FirestoreDB


def main():
    parser = argparse.ArgumentParser(description="Backfill missing conversation_uuid values.")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed updates without writing them.")
    args = parser.parse_args()

    db = FirestoreDB()
    conversations = db.client.collection("conversations").stream()
    updates = []

    for doc in conversations:
        data = doc.to_dict() or {}
        if data.get("conversation_uuid"):
            continue
        new_uuid = str(uuid.uuid4())
        updates.append({
            "conversation_id": doc.id,
            "conversation_uuid": new_uuid,
        })
        if not args.dry_run:
            db.client.collection("conversations").document(doc.id).set(
                {
                    "conversation_uuid": new_uuid,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

    print(json.dumps({"updated": updates, "count": len(updates)}, indent=2))


if __name__ == "__main__":
    main()
