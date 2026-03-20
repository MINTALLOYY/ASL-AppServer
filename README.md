This is the server-side code for LW's 2026 TSA Software Development team.

---

## API Reference

### Health
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Server health check |

### Conversations (Chat Saving & Retrieval)
All conversation save/read routes require Firebase Auth in `Authorization: Bearer <ID_TOKEN>`.

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
Requires `Authorization: Bearer <ID_TOKEN>` during WebSocket handshake.  
Streams live speech with speaker diarization; saves transcripts to Firestore automatically when `conversation_id` is provided.

### ASL WebSocket
`ws[s]://host/asl/ws?conversation_id=<id>`  
Requires `Authorization: Bearer <ID_TOKEN>` during WebSocket handshake.  
Streams ASL sign-language predictions; saves recognized words to Firestore when `conversation_id` is provided.

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
      allow read, write: if request.auth != null && resource.data.user_id == request.auth.uid;
      match /messages/{messageId} {
        allow read, write: if request.auth != null
          && get(/databases/$(database)/documents/conversations/$(conversationId)).data.user_id == request.auth.uid;
      }
    }
  }
}
```

If you want public read access during development (no auth required), replace the condition with `if true;` — **do not ship this in production**.

> **No rule changes are required** for the server itself to save/retrieve data.  
> Rules are only needed if the Flutter app reads Firestore directly (not via this API).
>
> For server-side list queries by owner (`where user_id == uid` + `order_by updated_at desc`), create a Firestore composite index for:
> - Collection: `conversations`
> - Fields: `user_id` (Ascending), `updated_at` (Descending)

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
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer $idToken',
    },
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
final channel = IOWebSocketChannel.connect(
  uri.toString(),
  headers: {'Authorization': 'Bearer $idToken'},
);
```

**ASL WS** — same pattern:
```dart
final uri = Uri.parse('$wsBaseUrl/asl/ws')
    .replace(queryParameters: {'conversation_id': conversationId});
final channel = IOWebSocketChannel.connect(
  uri.toString(),
  headers: {'Authorization': 'Bearer $idToken'},
);
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
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer $idToken',
  },
  body: jsonEncode({'conversation_id': conversationId}),
);
```

This marks the conversation as `finalized` in Firestore.

### 5 — Retrieve a saved conversation

On a history / playback screen, fetch the full conversation:
```dart
Future<Map<String, dynamic>> fetchConversation(String id) async {
  final resp = await http.get(
    Uri.parse('$baseUrl/conversations/$id'),
    headers: {'Authorization': 'Bearer $idToken'},
  );
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
  headers: {'Authorization': 'Bearer $idToken'},
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
