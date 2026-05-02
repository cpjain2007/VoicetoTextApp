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
- `SERVER_BEARER_TOKEN`: any token string you choose. When set, **every route** (including `GET /health`) requires `Authorization: Bearer <same value>`.
- `PORT`: optional for local only (default `3001`). Omit `PORT` from env files used with Firebase deploy (Cloud reserves `PORT`).
- `MAX_UPLOAD_MB`: optional upload cap for `/transcribe` (default `25`, max `100`).
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

- `GET http://localhost:3001/health` (same `Authorization` header as below when `SERVER_BEARER_TOKEN` is set)

Transcribe endpoints:

- **`POST /transcribe-base64`** (used by the Expo app): `application/json` body `{ "audioBase64": "<base64>", "mimeType": "audio/m4a" }` (optional `speakerName`). Same auth and size limits as multipart.
- **`POST /transcribe`**: `multipart/form-data`, field name `file` (for curl/scripts).

Both:

- max decoded size: `MAX_UPLOAD_MB` (default 25 MB); MIME must be `audio/*` or `application/octet-stream`
- auth header when `SERVER_BEARER_TOKEN` is set: `Authorization: Bearer <SERVER_BEARER_TOKEN>` (required on all routes)

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
    "actionItems": ["Item 1", "Item 2"],
    "topics": ["Scheduling", "Follow-up"]
  }
}
```

`ai` is included when `OPENAI_API_KEY` is set (otherwise omitted). The same shape is **sanitized and stored** on each **`POST /history`** entry under `ai`. If the client omits `ai` but sends `text`, the server runs the model once and backfills before save.

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

**AI tab checklist (3 steps)**

1. **Server:** Set a real `OPENAI_API_KEY` in `VoiceToTextApi/.env` (and in Firebase Functions env when deployed). Verify: `cd VoiceToTextApi && npm run verify:ai`
2. **App:** Set `EXPO_PUBLIC_TRANSCRIBE_API_URL` and optional token in `VoiceToTextApp/.env` as above. Verify: `cd VoiceToTextApp && npm run verify:env`
3. **Device:** Open the app’s **AI** tab — it backfills `/ai/insights` for recent clips missing an `ai` block; new recordings receive `ai` from `/transcribe-base64` when the key is set.

## 4) Deploy as Firebase Cloud Functions (Gen 2)

This folder is also the **Firebase Functions** codebase (see repo root `firebase.json`).

From the **repo root** (`Projects/`):

1. One-time: `npm install` then `npm run firebase:login` (browser sign-in).
2. Pick project: `npm run firebase:use -- YOUR_PROJECT_ID` or deploy script below.
3. Enable **Firestore** in the Firebase console (Build → Firestore). Speaker profiles are stored in document `apiState/speakerProfiles` (override with env `SPEAKER_FIRESTORE_DOC` as `collectionId/documentId`).
4. Set function environment (Firebase console → Functions → your function `api` → Environment variables), same keys as `.env.example`: `ASSEMBLYAI_API_KEY`, `OPENAI_API_KEY`, `SERVER_BEARER_TOKEN`, optional `SPEAKER_MATCH_THRESHOLD`, `OPENAI_AI_MODEL`, `ASSEMBLYAI_LANGUAGE_FALLBACKS`, etc. Do not set `SPEAKER_STORE_BACKEND` in the console (the function forces Firestore).
5. Deploy (PowerShell from repo root):

```powershell
.\scripts\firebase-deploy.ps1 -ProjectId YOUR_PROJECT_ID
```

Or: `npm run firebase:deploy` after `npm run firebase:use -- YOUR_PROJECT_ID`.

6. After deploy, open the **function URL** from the console (Gen 2 looks like `https://api-xxxxx-uc.a.run.app`). Point the Expo app at:

```env
EXPO_PUBLIC_TRANSCRIBE_API_URL=https://YOUR_FUNCTION_HOST/transcribe
```

7. The function is deployed with **public invoker** so Expo can reach it; **set `SERVER_BEARER_TOKEN`** so every route still requires `Authorization: Bearer …`. `minInstances: 1` is enabled to reduce cold starts (adds a small continuous cost on Blaze).

Local dev is unchanged: `SPEAKER_STORE_BACKEND=file` (default) keeps using `speaker-profiles.json`.
