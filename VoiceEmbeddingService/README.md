# Voice Embedding Service

Cloud Run service for speaker embeddings. It uses SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) to turn an audio clip into a normalized speaker embedding.

The Docker image downloads the model during image build and stores it under `/models/speechbrain-spkrec-ecapa-voxceleb`. Runtime requests should not need to download model files from Hugging Face.

## Local Run

```powershell
cd VoiceEmbeddingService
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:EMBEDDING_SERVICE_TOKEN="local-dev-token"
uvicorn app:app --reload --port 8080
```

## Deploy To Cloud Run

Use `min-instances=0` to keep idle cost low.

```powershell
gcloud run deploy voicetotext-speaker-embedding `
  --source VoiceEmbeddingService `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --timeout 300 `
  --min-instances 0 `
  --set-env-vars EMBEDDING_SERVICE_TOKEN=replace_with_shared_secret
```

Then set these on the Firebase API runtime:

```powershell
SPEAKER_EMBEDDING_SERVICE_URL=https://your-cloud-run-url
SPEAKER_EMBEDDING_SERVICE_TOKEN=replace_with_shared_secret
SPEAKER_EMBEDDING_TIMEOUT_MS=10000
```

The Firebase API will use embeddings first when these variables are configured, and fall back to the existing fingerprint matcher if the service is unavailable.
