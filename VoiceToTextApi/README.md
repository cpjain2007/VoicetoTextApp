# VoiceToText API

Tiny Express API that accepts an uploaded audio file and:
- uses AssemblyAI for transcription and speaker-label metadata
- uses OpenAI for AI insights (summary + action items)

## 1) Setup

```bash
cd VoiceToTextApi
copy .env.example .env
```

Edit `.env`:

- `ASSEMBLYAI_API_KEY`: required for `/transcribe`.
- `OPENAI_API_KEY`: required for AI features (`/ai/insights` and `ai` field in `/transcribe` response).
- `SERVER_BEARER_TOKEN`: any token string you choose.
- `PORT`: optional (default `3001`).
- `SPEAKER_MATCH_THRESHOLD`: optional float for local speaker-profile matching.
- `OPENAI_AI_MODEL`: optional OpenAI model (default `gpt-4o-mini`).

Install packages:

```bash
npm install
```

## 2) Run locally

```bash
npm run dev
```

Health check:

- `GET http://localhost:3001/health`

Transcribe endpoint:

- `POST http://localhost:3001/transcribe`
- `multipart/form-data`
- file field name: `file`
- optional auth header: `Authorization: Bearer <SERVER_BEARER_TOKEN>`

Response:

```json
{
  "text": "your transcript",
  "detectedSpeakerName": "Speaker 1",
  "speakerConfidence": 0.98,
  "utterances": [
    { "speaker": "A", "text": "Hello", "start": 0, "end": 520 }
  ],
  "ai": {
    "summary": "Short summary",
    "actionItems": ["Item 1", "Item 2"]
  }
}
```

AI insights endpoint:

- `POST http://localhost:3001/ai/insights`
- `application/json`
- body: `{ "text": "transcript text" }`

## 3) Connect from Expo app

In `VoiceToTextApp/.env`:

```env
EXPO_PUBLIC_TRANSCRIBE_API_URL=https://your-deployed-api.com/transcribe
EXPO_PUBLIC_TRANSCRIBE_API_TOKEN=your_custom_token_here
```

For local testing on phone, use your PC's LAN IP:

```env
EXPO_PUBLIC_TRANSCRIBE_API_URL=http://192.168.x.x:3001/transcribe
EXPO_PUBLIC_TRANSCRIBE_API_TOKEN=your_custom_token_here
```

Then restart Expo (`npm start`).
