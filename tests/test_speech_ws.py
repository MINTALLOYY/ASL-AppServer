import json
import unittest
from unittest.mock import MagicMock, patch

import config

VALID_UUID = "123e4567-e89b-42d3-a456-426614174000"


with patch("speech.chirp_stream.ChirpStreamer"), \
     patch("speech.chirp_stream.speaker_label_from_result"), \
     patch("firebase.db.FirestoreDB"):
    import app as server_app


class FakeSock:
    def __init__(self):
        self.handler = None

    def route(self, path):
        def decorator(fn):
            self.handler = fn
            return fn

        return decorator


class FakeWS:
    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def receive(self, timeout=None):
        if self._messages:
            return self._messages.pop(0)
        return None


class FakeAlt:
    def __init__(self, transcript):
        self.transcript = transcript
        self.words = []


class FakeResult:
    def __init__(self, transcript):
        self.is_final = True
        self.stability = 0.99
        self.alternatives = [FakeAlt(transcript)]


class FakeResponse:
    def __init__(self, transcript):
        self.results = [FakeResult(transcript)]


class FakeChirpStreamer:
    def __init__(self, diarization_speaker_count=2):
        self.diarization_speaker_count = diarization_speaker_count
        self.finished = False

    def responses(self):
        return iter([FakeResponse("hello world")])

    def debug_stats(self):
        return {"finished": self.finished}

    def add_audio_base64(self, value):
        self.last_audio = value

    def finish(self):
        self.finished = True


class TestSpeechWebsocket(unittest.TestCase):
    def setUp(self):
        self._orig_enforce = config.ENFORCE_CONVERSATION_UUID
        self._orig_db = server_app.db
        config.ENFORCE_CONVERSATION_UUID = False

    def tearDown(self):
        config.ENFORCE_CONVERSATION_UUID = self._orig_enforce
        server_app.db = self._orig_db

    def test_missing_uuid_is_rejected_when_enforced(self):
        config.ENFORCE_CONVERSATION_UUID = True
        from websockets.speech_ws import register_speech_ws

        sock = FakeSock()
        register_speech_ws(sock)
        ws = FakeWS()

        with server_app.app.test_request_context("/speech/ws"):
            sock.handler(ws)

        self.assertTrue(ws.sent)
        self.assertIn("missing_conversation_uuid", ws.sent[0])

    def test_provided_uuid_is_used_for_firestore_writes(self):
        from websockets.speech_ws import register_speech_ws

        fake_db = MagicMock()
        fake_db.save_transcript = MagicMock()
        fake_db.set_conversation_display_name_if_missing = MagicMock()
        server_app.db = fake_db

        sock = FakeSock()
        with patch("websockets.speech_ws.db", fake_db), \
             patch("websockets.speech_ws.ChirpStreamer", FakeChirpStreamer), \
             patch("websockets.speech_ws.speaker_label_from_result", return_value="Speaker_1"):
            register_speech_ws(sock)
            ws = FakeWS()
            with server_app.app.test_request_context(f"/speech/ws?conversation_id=conv-1&conversation_uuid={VALID_UUID}"):
                sock.handler(ws)

        fake_db.set_conversation_display_name_if_missing.assert_called()
        fake_db.save_transcript.assert_called()
        args, kwargs = fake_db.save_transcript.call_args
        self.assertEqual(kwargs["conversation_id"], "conv-1")
        self.assertEqual(kwargs["conversation_uuid"], VALID_UUID)
        self.assertEqual(kwargs["segment"]["text"], "hello world")
        self.assertEqual(kwargs["segment"]["type"], "speech")
        self.assertEqual(kwargs["segment"]["speaker"], "Speaker_1")


if __name__ == "__main__":
    unittest.main()
