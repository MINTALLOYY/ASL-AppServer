import unittest
from unittest.mock import MagicMock, patch
from google.cloud import speech_v1 as speech
from speech.chirp_stream import ChirpStreamer

class TestChirpStreamer(unittest.TestCase):

    @patch("google.cloud.speech_v1.SpeechClient")
    def test_request_generator(self, MockSpeechClient):
        # Initialize ChirpStreamer
        streamer = ChirpStreamer(language_code="en-US", sample_rate_hz=16000, diarization_speaker_count=2)

        # Mock audio data
        audio_data = b"test_audio_data"
        streamer._audio_q.put(audio_data)
        streamer._finished.set()

        # Get the generator
        generator = streamer._request_generator()

        # Request should contain audio data
        first_request = next(generator)
        self.assertIsInstance(first_request, speech.StreamingRecognizeRequest)
        self.assertEqual(first_request.audio_content, audio_data)

    @patch("google.cloud.speech_v1.SpeechClient")
    def test_responses(self, MockSpeechClient):
        # Mock the SpeechClient
        mock_client = MockSpeechClient.return_value
        mock_response = MagicMock()
        mock_client.streaming_recognize.return_value = iter([mock_response])

        # Initialize ChirpStreamer
        streamer = ChirpStreamer(language_code="en-US", sample_rate_hz=16000, diarization_speaker_count=2)

        # Mock the _request_generator
        streamer._request_generator = MagicMock(return_value=iter([]))

        # Call responses and check the result
        responses = streamer.responses()
        self.assertEqual(list(responses), [mock_response])
        mock_client.streaming_recognize.assert_called_once()
        call_args, call_kwargs = mock_client.streaming_recognize.call_args
        self.assertEqual(len(call_args), 2)
        self.assertIsInstance(call_args[0], speech.StreamingRecognitionConfig)
        self.assertIs(call_args[1], streamer._request_generator())
        self.assertEqual(call_kwargs, {})

if __name__ == "__main__":
    unittest.main()
