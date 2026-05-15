This is the server-side code for LW's 2026 TSA Software Development team.

---

## ASL Translation

This server includes a sign-language translation pipeline for short ASL word clips. It does not try to translate full sentences. Instead, it recognizes one sign at a time from MediaPipe landmark sequences and returns the most likely word.

### Supported words

The current model can translate these ASL words:

`book`, `dance`, `doctor`, `drink`, `eat`, `family`, `finish`, `give`, `go`, `help`, `how`, `meet`, `no`, `play`, `school`, `walk`, `want`, `woman`, `work`, `yes`

### How the model is trained

Training clips live under `asl/trainingandtesting/training_data/raw/<sign>/` as `.npy` video frame sequences. During training, each clip is processed with MediaPipe Holistic to extract pose and hand landmarks, the landmarks are normalized relative to the shoulders, and a 50-frame window with the strongest signing signal is selected for the classifier.

The training script compares three sklearn models on the same landmark features:

- RandomForestClassifier
- RBF SVC in a StandardScaler pipeline
- GradientBoostingClassifier

The script runs 5-fold stratified cross-validation for each candidate and keeps the highest-scoring model. After that, it performs a hard-example pass that duplicates clips that were misclassified or fell below the confidence threshold, then trains the chosen model on the augmented dataset.

### Model and runtime behavior

The deployed artifact is `asl/asl_classifier.pkl`, with labels stored in `asl/label_map.json`. At runtime, the predictor loads that saved sklearn classifier and produces a ranked list of sign predictions with confidence scores. The live websocket only emits a sign when the top prediction passes the confidence threshold.

### How we tested it

The ASL pipeline was validated in three ways:

- Offline model selection with 5-fold stratified cross-validation.
- A hard-example pass to check the low-confidence and misclassified clips during training.
- Runtime checks through `tests/test_asl_transcribe.py`, `asl/test_asl.html`, and `asl/test_asl_upload.html`, plus the webcam test scripts in `asl/trainingandtesting/`.

---

## Group Captioning

Group captioning is the live speech pipeline for multi-speaker conversations. It streams microphone audio to Google Cloud Speech, applies speaker diarization, and turns each finalized utterance into a caption tagged with a speaker label such as `Speaker_1` or `Speaker_2`.

### What it does

The captioning path is built for live conversation, not batch transcription. It listens for real-time audio chunks on the speech websocket, waits for Google Speech to finalize a result, extracts the speaker tag from the final words, and emits a `final_transcript` event containing:

- `text` — the recognized utterance
- `speaker` — the diarized speaker label when available
- `conversation_id` and `conversation_uuid` — the active session identifiers

When a conversation ID is present, each final caption is also saved to Firestore as a `speech` message so the full conversation can be replayed later.

### How the captioning pipeline works

The websocket endpoint is `ws[s]://host/speech/ws`. It defaults to `mode=captioning` and accepts a `num_speakers` query parameter, which is clamped between 2 and 6. Internally it uses Google Speech v1 with the `latest_long` model and a `SpeakerDiarizationConfig` so the backend can assign utterances to different speakers in the same conversation.

The backend also supports a separate `identifying` mode for label discovery, but the normal group captioning flow stays in `captioning` mode so the client receives plain spoken text plus the diarized speaker name.

### Implementation steps

1. Create a conversation before opening the websocket by calling `POST /conversations` and keeping the returned `conversation_id` in app state.
2. Open `ws[s]://host/speech/ws?conversation_id=<id>&mode=captioning&num_speakers=2`.
3. Stream raw 16 kHz, 16-bit mono PCM audio to the websocket as `audio_chunk` messages.
4. Listen for `final_transcript` events and append each result to the local caption list using the returned `speaker` and `text`.
5. When the session ends, call `POST /speech/finalize` with the same `conversation_id` so the conversation is marked `finalized` in Firestore.
6. For playback or history screens, load the saved transcript with `GET /conversations/<id>` or `GET /conversations/<id>/messages`.

### How we tested it

The group captioning path was validated with websocket and endpoint tests rather than only manual audio checks:

- `tests/test_speech_ws.py` verifies websocket setup, UUID handling, and Firestore transcript writes.
- `tests/test_endpoint_smoke.py` confirms the speech routes are wired and callable.
- `speech/chirp_stream.py` is exercised through the websocket path with the diarization configuration and Google Speech streaming request loop.

---

## API Reference

### Health
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Server health check |

### Conversations (Chat Saving & Retrieval)
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/conversations` | Create a new conversation |
| GET | `/conversations` | List conversations (most-recent first) |
| GET | `/conversations/<id>` | Get conversation + all messages |
| GET | `/conversations/<id>/messages` | Get messages only |

**POST /conversations** — create a conversation  
Optional body: `{"conversation_id": "my-custom-id"}`  
Returns: `{"conversation_id": "...", "status": "active"}` (HTTP 201)

**GET /conversations?limit=50** — list conversations  
Returns: `{"conversations": [{conversation_id, status, created_at, updated_at}, ...]}`

**GET /conversations/{id}** — get full conversation  
Returns conversation metadata plus a `messages` array in chronological order:
```json
{
  "conversation_id": "abc123",
  "status": "active",
  "created_at": "2024-01-01T00:00:00+00:00",
  "updated_at": "2024-01-01T00:05:00+00:00",
  "messages": [
    {"id": "m1", "text": "Hello", "type": "speech", "speaker": "Speaker_1", "created_at": "..."},
    {"id": "m2", "text": "THANK_YOU", "type": "asl",    "speaker": null,       "created_at": "..."}
  ]
}
```

**GET /conversations/{id}/messages** — messages only  
Returns: `{"conversation_id": "...", "messages": [...]}`

### Speech WebSocket
`ws[s]://host/speech/ws?conversation_id=<id>&mode=captioning&num_speakers=2`  
Streams live speech with speaker diarization; saves transcripts to Firestore automatically when `conversation_id` is provided.

### ASL WebSocket
`ws[s]://host/asl/ws?conversation_id=<id>`  
Streams ASL sign-language predictions; saves recognized words to Firestore when `conversation_id` is provided.

### ASL Transcription
`POST /asl/transcribe`  
Uploads a recorded ASL video and returns structured translation output, including the best prediction, top predictions, and frame statistics.

### ASL Diagnostics
`GET /asl/diagnostics`  
Checks whether the ASL model and MediaPipe runtime can load successfully.

### Finalize
`POST /speech/finalize` — `{"conversation_id": "..."}` marks conversation as `finalized` in Firestore.

---

## Firestore Rules

The server uses a **service-account** (Admin SDK), which bypasses Firestore security rules.  
For the Flutter app to read conversations **directly** from Firebase (without going through this server), add these rules in the Firebase console:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /conversations/{conversationId} {
      allow read, write: if request.auth != null;
      match /messages/{messageId} {
        allow read, write: if request.auth != null;
      }
    }
  }
}
```

If you want public read access during development (no auth required), replace the condition with `if true;` — **do not ship this in production**.

> **No rule changes are required** for the server itself to save/retrieve data.  
> Rules are only needed if the Flutter app reads Firestore directly (not via this API).

---

## Flutter Implementation Plan

Below is the step-by-step plan for adding conversation persistence to the Flutter app.

### 1 — Start a conversation

Before opening either WebSocket, call `POST /conversations` once to obtain a `conversation_id`.  
Persist it in local state (e.g. a `StateNotifier` / `Riverpod` provider).

```dart
Future<String> createConversation() async {
  final resp = await http.post(
    Uri.parse('$baseUrl/conversations'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({}),           // or pass a custom ID
  );
  if (resp.statusCode == 201) {
    return jsonDecode(resp.body)['conversation_id'] as String;
  }
  throw Exception('Failed to create conversation');
}
```

### 2 — Pass conversation_id to both WebSockets

**Speech WS** — append as a query parameter:
```dart
final uri = Uri.parse('$wsBaseUrl/speech/ws')
    .replace(queryParameters: {
      'conversation_id': conversationId,
      'mode': 'captioning',
      'num_speakers': '2',
    });
final channel = WebSocketChannel.connect(uri);
```

**ASL WS** — same pattern:
```dart
final uri = Uri.parse('$wsBaseUrl/asl/ws')
    .replace(queryParameters: {'conversation_id': conversationId});
final channel = WebSocketChannel.connect(uri);
```

The server saves each recognized transcript / ASL word to Firestore automatically.

### 3 — Display messages in real time

Listen to the `final_transcript` (speech) and `asl_result` (ASL) events from the WebSocket streams and append them to your local message list.  
Each item should include:

| field | source |
|-------|--------|
| `text` | WS payload |
| `speaker` | WS payload (speech) or `null` (ASL) |
| `type` | `"speech"` or `"asl"` |
| `timestamp` | `DateTime.now()` on the client |

### 4 — Finalize the session

When the user ends the session, call:
```dart
await http.post(
  Uri.parse('$baseUrl/speech/finalize'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({'conversation_id': conversationId}),
);
```

This marks the conversation as `finalized` in Firestore.

### 5 — Retrieve a saved conversation

On a history / playback screen, fetch the full conversation:
```dart
Future<Map<String, dynamic>> fetchConversation(String id) async {
  final resp = await http.get(Uri.parse('$baseUrl/conversations/$id'));
  if (resp.statusCode == 200) {
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }
  throw Exception('Not found');
}
```

The response already has messages in chronological order. Render them in a `ListView`:

```dart
final messages = (data['messages'] as List).cast<Map<String, dynamic>>();
// messages[0] is the oldest; messages.last is the newest
```

### 6 — List saved conversations

On a history screen:
```dart
final resp = await http.get(
  Uri.parse('$baseUrl/conversations').replace(queryParameters: {'limit': '20'}),
);
final convs = (jsonDecode(resp.body)['conversations'] as List)
    .cast<Map<String, dynamic>>();
```

### Data model

```dart
class ChatMessage {
  final String id;
  final String text;
  final String type;        // "speech" | "asl"
  final String? speaker;
  final DateTime createdAt;
}
```

### Summary of required HTTP calls

| Step | Method | URL |
|------|--------|-----|
| Create conversation | POST | `/conversations` |
| Finalize session | POST | `/speech/finalize` |
| List history | GET | `/conversations?limit=N` |
| Load full conversation | GET | `/conversations/<id>` |
| Load messages only | GET | `/conversations/<id>/messages` |

