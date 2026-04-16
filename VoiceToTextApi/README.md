# VoiceToText API

Tiny Express API that accepts an uploaded audio file and returns speech-to-text using OpenAI transcription.

## 1) Setup

```bash
cd VoiceToTextApi
copy .env.example .env
```

Edit `.env`:

- `OPENAI_API_KEY`: your OpenAI API key.
- `SERVER_BEARER_TOKEN`: any token string you choose.
- `PORT`: optional (default `3001`).

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
{ "text": "your transcript" }
```

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
