import base64
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from speechbrain.inference.speaker import EncoderClassifier


MODEL_SOURCE = os.getenv("SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
SERVICE_TOKEN = os.getenv("EMBEDDING_SERVICE_TOKEN", "").strip()

app = FastAPI(title="Voice Embedding Service")


class EmbedRequest(BaseModel):
    audioBase64: str
    mimeType: Optional[str] = "audio/m4a"


class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str


def require_auth(authorization: Optional[str]) -> None:
    if not SERVICE_TOKEN:
        return
    expected = f"Bearer {SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@lru_cache(maxsize=1)
def get_classifier() -> EncoderClassifier:
    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir="/tmp/speechbrain-spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )


def decode_to_wav(audio_base64: str) -> Path:
    try:
        audio_bytes = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid audioBase64") from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")

    temp_dir = Path(tempfile.mkdtemp(prefix="voice-embedding-"))
    input_path = temp_dir / "input_audio"
    output_path = temp_dir / "audio.wav"
    input_path.write_bytes(audio_bytes)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=400, detail="Audio conversion timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail="Audio conversion failed") from exc

    return output_path


def build_embedding(wav_path: Path) -> List[float]:
    signal, sample_rate = torchaudio.load(str(wav_path))
    if signal.numel() == 0:
        raise HTTPException(status_code=400, detail="Audio contains no samples")
    if sample_rate != 16000:
        signal = torchaudio.functional.resample(signal, sample_rate, 16000)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    classifier = get_classifier()
    with torch.no_grad():
        embedding = classifier.encode_batch(signal).squeeze()
        embedding = torch.nn.functional.normalize(embedding, dim=0)

    return [float(value) for value in embedding.cpu().tolist()]


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_SOURCE}


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    wav_path = decode_to_wav(payload.audioBase64)
    try:
        embedding = build_embedding(wav_path)
        return {"embedding": embedding, "model": MODEL_SOURCE}
    finally:
        shutil.rmtree(wav_path.parent, ignore_errors=True)
