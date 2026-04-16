const express = require("express");
const cors = require("cors");
const multer = require("multer");
const dotenv = require("dotenv");
const OpenAI = require("openai");
const ffmpegPath = require("ffmpeg-static");
const fs = require("fs/promises");
const path = require("path");
const os = require("os");
const { randomUUID } = require("crypto");
const { spawn } = require("child_process");

dotenv.config({ override: true });

const app = express();
const port = process.env.PORT || 3001;
const speakerSimilarityThreshold = Number(process.env.SPEAKER_MATCH_THRESHOLD || "0.965");
const speakerStorePath = path.join(__dirname, "speaker-profiles.json");

app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

const serverToken = process.env.SERVER_BEARER_TOKEN;
const openaiApiKey = process.env.OPENAI_API_KEY;

if (!openaiApiKey) {
  // Fail early so deployment issues are obvious.
  console.warn("Warning: OPENAI_API_KEY is missing.");
}

const client = new OpenAI({
  apiKey: openaiApiKey,
});

const dotProduct = (a, b) => a.reduce((sum, item, index) => sum + item * (b[index] || 0), 0);

const vectorMagnitude = (vector) => Math.sqrt(dotProduct(vector, vector));

const cosineSimilarity = (a, b) => {
  const denominator = vectorMagnitude(a) * vectorMagnitude(b);
  if (!denominator) {
    return 0;
  }
  return dotProduct(a, b) / denominator;
};

const parseInt16LE = (buffer, offset) => buffer.readInt16LE(offset) / 32768;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const buildVoiceSignature = (pcmBuffer) => {
  const totalSamples = Math.floor(pcmBuffer.length / 2);
  if (totalSamples < 4000) {
    throw new Error("Audio clip is too short for speaker recognition.");
  }

  let sumAbs = 0;
  let sumSquares = 0;
  let zeroCrossings = 0;
  let previousSample = 0;
  const frameSize = 320;
  const energyFrames = [];
  let frameEnergy = 0;
  let frameSamples = 0;

  for (let i = 0; i < totalSamples; i += 1) {
    const sample = parseInt16LE(pcmBuffer, i * 2);
    const absSample = Math.abs(sample);
    sumAbs += absSample;
    sumSquares += sample * sample;
    if ((sample >= 0 && previousSample < 0) || (sample < 0 && previousSample >= 0)) {
      zeroCrossings += 1;
    }
    previousSample = sample;
    frameEnergy += sample * sample;
    frameSamples += 1;
    if (frameSamples >= frameSize) {
      energyFrames.push(frameEnergy / frameSamples);
      frameEnergy = 0;
      frameSamples = 0;
    }
  }

  if (frameSamples > 0) {
    energyFrames.push(frameEnergy / frameSamples);
  }

  const meanAbs = sumAbs / totalSamples;
  const rms = Math.sqrt(sumSquares / totalSamples);
  const zcr = zeroCrossings / totalSamples;
  const meanEnergy = energyFrames.reduce((sum, item) => sum + item, 0) / energyFrames.length;
  const energyVariance =
    energyFrames.reduce((sum, item) => sum + (item - meanEnergy) ** 2, 0) / energyFrames.length;
  const dynamicRange = clamp(Math.sqrt(energyVariance) / (meanEnergy + 1e-7), 0, 10);

  return [
    clamp(meanAbs, 0, 1),
    clamp(rms, 0, 1),
    clamp(zcr, 0, 1),
    clamp(meanEnergy, 0, 1),
    clamp(dynamicRange / 10, 0, 1),
  ];
};

const convertAudioToPcm = async (inputBuffer, fileExtension) => {
  if (!ffmpegPath) {
    throw new Error("ffmpeg-static is not available.");
  }

  const tempInputPath = path.join(os.tmpdir(), `voicetotext-${randomUUID()}${fileExtension}`);
  await fs.writeFile(tempInputPath, inputBuffer);

  try {
    const pcmBuffer = await new Promise((resolve, reject) => {
      const ffmpeg = spawn(ffmpegPath, [
        "-i",
        tempInputPath,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
      ]);

      const stdoutChunks = [];
      const stderrChunks = [];

      ffmpeg.stdout.on("data", (chunk) => stdoutChunks.push(chunk));
      ffmpeg.stderr.on("data", (chunk) => stderrChunks.push(chunk));
      ffmpeg.on("error", reject);
      ffmpeg.on("close", (code) => {
        if (code !== 0) {
          reject(
            new Error(
              `ffmpeg conversion failed (${code}): ${Buffer.concat(stderrChunks).toString("utf8")}`,
            ),
          );
          return;
        }
        resolve(Buffer.concat(stdoutChunks));
      });
    });

    return pcmBuffer;
  } finally {
    await fs.unlink(tempInputPath).catch(() => undefined);
  }
};

const extensionFromMime = (mimeType) => {
  if (!mimeType) {
    return ".m4a";
  }
  if (mimeType.includes("wav")) {
    return ".wav";
  }
  if (mimeType.includes("mpeg") || mimeType.includes("mp3")) {
    return ".mp3";
  }
  if (mimeType.includes("ogg")) {
    return ".ogg";
  }
  if (mimeType.includes("webm")) {
    return ".webm";
  }
  return ".m4a";
};

const readSpeakerProfiles = async () => {
  try {
    const raw = await fs.readFile(speakerStorePath, "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed.profiles) ? parsed.profiles : [];
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
};

const writeSpeakerProfiles = async (profiles) => {
  const payload = JSON.stringify({ profiles }, null, 2);
  await fs.writeFile(speakerStorePath, payload, "utf8");
};

const updateSpeakerProfile = async (speakerName, signature) => {
  const profiles = await readSpeakerProfiles();
  const existingIndex = profiles.findIndex((item) => item.name === speakerName);
  if (existingIndex >= 0) {
    const current = profiles[existingIndex];
    const total = (current.samples || 0) + 1;
    const currentVector = Array.isArray(current.vector) ? current.vector : signature.map(() => 0);
    profiles[existingIndex] = {
      name: speakerName,
      samples: total,
      vector: currentVector.map((value, index) => (value * (total - 1) + signature[index]) / total),
    };
  } else {
    profiles.push({ name: speakerName, samples: 1, vector: signature });
  }
  await writeSpeakerProfiles(profiles);
};

const detectSpeakerName = async (signature) => {
  const profiles = await readSpeakerProfiles();
  if (profiles.length === 0) {
    return null;
  }
  let best = null;
  for (const profile of profiles) {
    if (!Array.isArray(profile.vector) || profile.vector.length !== signature.length) {
      continue;
    }
    const score = cosineSimilarity(signature, profile.vector);
    if (!best || score > best.score) {
      best = { name: profile.name, score };
    }
  }
  if (!best || best.score < speakerSimilarityThreshold) {
    return null;
  }
  return best;
};

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.get("/speakers", async (_req, res) => {
  try {
    const profiles = await readSpeakerProfiles();
    return res.json({
      speakers: profiles.map((profile) => ({
        name: profile.name,
        samples: profile.samples || 0,
      })),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not load speaker profiles.";
    return res.status(500).json({ error: message });
  }
});

app.delete("/speakers", async (_req, res) => {
  try {
    await writeSpeakerProfiles([]);
    return res.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not clear speaker profiles.";
    return res.status(500).json({ error: message });
  }
});

app.post("/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (serverToken) {
      const authHeader = req.headers.authorization || "";
      const providedToken = authHeader.startsWith("Bearer ")
        ? authHeader.slice(7)
        : "";
      if (providedToken !== serverToken) {
        return res.status(401).json({ error: "Unauthorized token." });
      }
    }

    if (!openaiApiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not configured." });
    }

    if (!req.file) {
      return res.status(400).json({ error: "Missing audio file in 'file' field." });
    }

    const extension = extensionFromMime(req.file.mimetype);
    const pcmBuffer = await convertAudioToPcm(req.file.buffer, extension);
    const voiceSignature = buildVoiceSignature(pcmBuffer);
    const manualSpeakerName =
      typeof req.body.speakerName === "string" ? req.body.speakerName.trim() : "";
    if (manualSpeakerName) {
      await updateSpeakerProfile(manualSpeakerName, voiceSignature);
    }
    const detectedSpeaker = manualSpeakerName ? null : await detectSpeakerName(voiceSignature);

    const transcript = await client.audio.transcriptions.create({
      model: "gpt-4o-mini-transcribe",
      file: await OpenAI.toFile(req.file.buffer, req.file.originalname || "audio.m4a", {
        type: req.file.mimetype || "audio/m4a",
      }),
    });

    return res.json({
      text: transcript.text || "",
      detectedSpeakerName: detectedSpeaker?.name || null,
      speakerConfidence: detectedSpeaker?.score ?? null,
    });
  } catch (error) {
    const message =
      error && typeof error === "object" && "message" in error
        ? error.message
        : "Unexpected transcription error.";
    return res.status(500).json({ error: String(message) });
  }
});

app.listen(port, () => {
  console.log(`VoiceToText API listening on port ${port}`);
});
