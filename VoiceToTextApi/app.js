const express = require("express");
const cors = require("cors");
const multer = require("multer");
const OpenAI = require("openai");
const { AssemblyAI } = require("assemblyai");
const fs = require("fs/promises");
const path = require("path");
const os = require("os");
const { randomUUID } = require("crypto");
const { spawn } = require("child_process");
const speakerStore = require("./speakerStore");
const voiceRecognition = require("./voiceRecognition");

const app = express();
const serverToken = process.env.SERVER_BEARER_TOKEN;
const openaiApiKey = process.env.OPENAI_API_KEY;
const assemblyApiKey = process.env.ASSEMBLYAI_API_KEY;
const openaiAiModel = process.env.OPENAI_AI_MODEL || "gpt-4o-mini";
const assemblySpeechModels = ["universal-3-pro", "universal-2"];
const forcedLanguageCode = process.env.ASSEMBLYAI_FORCE_LANGUAGE_CODE || "";
const languageFallbackCodes = (process.env.ASSEMBLYAI_LANGUAGE_FALLBACKS || "hi,te,bn")
  .split(",")
  .map((item) => item.trim())
  .filter(Boolean);
/** Higher = fewer wrong-speaker IDs; lower if the same person is often missed. */
const speakerSimilarityThreshold = Number(process.env.SPEAKER_MATCH_THRESHOLD || "0.97");
const speakerMatchMargin = Number(process.env.SPEAKER_MATCH_MARGIN || "0.05");
const speakerMatchRelaxedEnabled = ["true", "1", "yes"].includes(
  (process.env.SPEAKER_MATCH_RELAXED_ENABLED || "").trim().toLowerCase(),
);
/** If the runner-up is still “strong”, require at least this cosine gap (reduces confused twins / similar voices). */
const speakerMatchMinGapBetweenProfiles = Number(process.env.SPEAKER_MATCH_MIN_GAP_BETWEEN_PROFILES || "0.035");
const speakerMatchSecondStrongMin = Number(process.env.SPEAKER_MATCH_SECOND_STRONG_MIN || "0.84");
const maxUploadMb = Math.min(Math.max(Number(process.env.MAX_UPLOAD_MB || "25") || 25, 1), 100);
const maxUploadBytes = maxUploadMb * 1024 * 1024;
const assemblySpeakerIdentificationEnv = (process.env.ASSEMBLYAI_SPEAKER_IDENTIFICATION || "true")
  .trim()
  .toLowerCase();
const speakerIdentificationEnabled =
  assemblySpeakerIdentificationEnv !== "false" && assemblySpeakerIdentificationEnv !== "0";
const assemblyKnownSpeakersMax = Math.min(
  Math.max(Number(process.env.ASSEMBLYAI_SPEAKER_ID_MAX_NAMES || "20") || 20, 1),
  50,
);
const assemblySpeakerNameMaxLen = 35;
const assemblySpeakerDescriptionMaxLen = Math.min(
  Math.max(Number(process.env.ASSEMBLYAI_SPEAKER_DESCRIPTION_MAX || "220") || 220, 40),
  500,
);
const speakerRecentVectorsMax = Math.min(
  Math.max(Number(process.env.SPEAKER_RECENT_VECTORS_MAX || "5") || 5, 1),
  12,
);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const isRetryableError = (err) => {
  if (!err || typeof err !== "object") {
    return false;
  }
  const status = err.status ?? err.statusCode ?? err.response?.status ?? err.code;
  if (status === 429 || status === 502 || status === 503 || status === 504 || status === "ECONNRESET") {
    return true;
  }
  const msg = String(err.message || "").toLowerCase();
  return (
    msg.includes("timeout") ||
    msg.includes("timed out") ||
    msg.includes("fetch failed") ||
    msg.includes("network") ||
    msg.includes("econnreset") ||
    msg.includes("socket hang up")
  );
};

const withRetries = async (fn, { label = "request", maxAttempts = 3, baseDelayMs = 900 } = {}) => {
  let lastErr;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt === maxAttempts || !isRetryableError(err)) {
        throw err;
      }
      const delay = baseDelayMs * attempt;
      console.warn(`${label} failed (attempt ${attempt}/${maxAttempts}), retrying in ${delay}ms`, {
        message: err instanceof Error ? err.message : String(err),
      });
      await sleep(delay);
    }
  }
  throw lastErr;
};

const audioMimeOk = (mime) => {
  const base = (mime || "").toLowerCase().split(";")[0].trim();
  if (!base) {
    return true;
  }
  if (base === "application/octet-stream") {
    return true;
  }
  return base.startsWith("audio/");
};

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: maxUploadBytes, files: 1 },
  fileFilter: (req, file, cb) => {
    if (audioMimeOk(file.mimetype)) {
      cb(null, true);
      return;
    }
    cb(
      new Error(
        "Unsupported audio type. Use an audio/* Content-Type (or application/octet-stream for raw uploads).",
      ),
    );
  },
});

app.use(cors());
app.use(express.json({ limit: `${Math.min(maxUploadMb * 2 + 15, 55)}mb` }));

if (process.env.K_SERVICE && !serverToken) {
  console.warn("SERVER_BEARER_TOKEN is not set — deployed API has no application-level auth.");
}

if (serverToken) {
  app.use((req, res, next) => {
    const raw = (req.headers.authorization || "").trim();
    const match = /^Bearer\s+(\S+)/i.exec(raw);
    const provided = match ? match[1] : "";
    if (provided !== serverToken) {
      return res.status(401).json({ error: "Unauthorized." });
    }
    return next();
  });
}

const isCloudRun = Boolean(process.env.K_SERVICE);
if (!openaiApiKey && isCloudRun) {
  console.warn("OPENAI_API_KEY is not set on this Cloud Run revision — AI features are disabled.");
}
if (!assemblyApiKey && isCloudRun) {
  console.warn("ASSEMBLYAI_API_KEY is not set on this Cloud Run revision — /transcribe will fail.");
}

let openaiClientInstance = null;
const getOpenAIClient = () => {
  if (!openaiApiKey) {
    return null;
  }
  if (!openaiClientInstance) {
    openaiClientInstance = new OpenAI({ apiKey: openaiApiKey });
  }
  return openaiClientInstance;
};

let assemblyClientInstance = null;
const getAssemblyClient = () => {
  if (!assemblyApiKey) {
    return null;
  }
  if (!assemblyClientInstance) {
    assemblyClientInstance = new AssemblyAI({ apiKey: assemblyApiKey });
  }
  return assemblyClientInstance;
};

const transcribeWithRetry = (assemblyClient, payload) =>
  withRetries(() => assemblyClient.transcripts.transcribe(payload), { label: "AssemblyAI.transcribe" });

const buildAiInsights = async (text) => {
  if (!openaiApiKey || !text.trim()) {
    return null;
  }
  const client = getOpenAIClient();
  if (!client) {
    return null;
  }

  const response = await withRetries(
    () =>
      client.responses.create({
        model: openaiAiModel,
        input: [
          {
            role: "system",
            content:
              "You summarize transcripts into concise business notes. Return strict JSON with keys summary and actionItems. actionItems must be an array of short strings.",
          },
          {
            role: "user",
            content: `Transcript:\n${text}`,
          },
        ],
        text: {
          format: {
            type: "json_schema",
            name: "transcript_insights",
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                summary: { type: "string" },
                actionItems: { type: "array", items: { type: "string" } },
              },
              required: ["summary", "actionItems"],
            },
          },
        },
      }),
    { label: "OpenAI.buildAiInsights" },
  );

  const raw = response.output_text || "{}";
  const parsed = JSON.parse(raw);
  return {
    summary: typeof parsed.summary === "string" ? parsed.summary : "",
    actionItems: Array.isArray(parsed.actionItems)
      ? parsed.actionItems.filter((item) => typeof item === "string")
      : [],
  };
};

const buildSpeakerCorrectionSuggestion = async (latestText, recentHistory) => {
  if (!openaiApiKey || !latestText.trim()) {
    return null;
  }

  const compactHistory = Array.isArray(recentHistory)
    ? recentHistory.slice(0, 5).map((item, index) => ({
        index,
        speakerName: typeof item?.speakerName === "string" ? item.speakerName : "",
        text: typeof item?.text === "string" ? item.text.slice(0, 180) : "",
      }))
    : [];

  const client = getOpenAIClient();
  if (!client) {
    return null;
  }

  const response = await withRetries(
    () =>
      client.responses.create({
        model: openaiAiModel,
        input: [
          {
            role: "system",
            content:
              "Detect if user asks to rename the most recent speaker label. Only suggest when explicit rename intent exists in latest transcript. Return strict JSON.",
          },
          {
            role: "user",
            content: JSON.stringify({
              latestText,
              recentHistory: compactHistory,
              instruction:
                "If intent is to rename last log speaker, return shouldSuggest=true and suggestedSpeakerName with clean title case. Otherwise false.",
            }),
          },
        ],
        text: {
          format: {
            type: "json_schema",
            name: "speaker_correction",
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                shouldSuggest: { type: "boolean" },
                suggestedSpeakerName: { type: "string" },
                reason: { type: "string" },
              },
              required: ["shouldSuggest", "suggestedSpeakerName", "reason"],
            },
          },
        },
      }),
    { label: "OpenAI.buildSpeakerCorrectionSuggestion" },
  );

  const raw = response.output_text || "{}";
  const parsed = JSON.parse(raw);
  return {
    shouldSuggest: !!parsed.shouldSuggest,
    suggestedSpeakerName:
      typeof parsed.suggestedSpeakerName === "string" ? parsed.suggestedSpeakerName.trim() : "",
    reason: typeof parsed.reason === "string" ? parsed.reason.trim() : "",
  };
};

let cachedFfmpegPath;
const getFfmpegPath = () => {
  if (cachedFfmpegPath === undefined) {
    cachedFfmpegPath = require("ffmpeg-static") || null;
  }
  return cachedFfmpegPath;
};

const convertAudioToPcm = async (inputBuffer, fileExtension) => {
  const ffmpegBin = getFfmpegPath();
  if (!ffmpegBin) {
    throw new Error("ffmpeg-static is not available.");
  }

  const tempInputPath = path.join(os.tmpdir(), `voicetotext-${randomUUID()}${fileExtension}`);
  await fs.writeFile(tempInputPath, inputBuffer);

  try {
    const pcmBuffer = await new Promise((resolve, reject) => {
      const ffmpeg = spawn(ffmpegBin, [
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

const updateSpeakerProfile = async (speakerName, signature) => {
  const profiles = await speakerStore.readSpeakerProfiles();
  const sigCopy = [...signature];
  const existingIndex = profiles.findIndex((item) => item.name === speakerName);
  if (existingIndex >= 0) {
    const current = profiles[existingIndex];
    const total = (current.samples || 0) + 1;
    const currentVector =
      Array.isArray(current.vector) && current.vector.length > 0 ? current.vector : signature.map(() => 0);
    const alignedCurrent = voiceRecognition.padVoiceVector(currentVector, signature.length, 0.5);
    const prevRecent = Array.isArray(current.vectorsRecent)
      ? current.vectorsRecent.filter((v) => Array.isArray(v) && v.length > 0).map((v) => [...v])
      : [];
    const nextRecent = [...prevRecent, sigCopy].slice(-speakerRecentVectorsMax);
    profiles[existingIndex] = {
      ...current,
      name: speakerName,
      samples: total,
      vector: alignedCurrent.map((value, index) => (value * (total - 1) + signature[index]) / total),
      vectorsRecent: nextRecent,
    };
  } else {
    profiles.push({
      name: speakerName,
      samples: 1,
      vector: signature,
      vectorsRecent: [sigCopy],
    });
  }
  await speakerStore.writeSpeakerProfiles(profiles);
};

const getSpeakerMatchRelaxedThreshold = () => {
  const raw = typeof process.env.SPEAKER_MATCH_RELAXED_THRESHOLD === "string" ? process.env.SPEAKER_MATCH_RELAXED_THRESHOLD.trim() : "";
  if (raw) {
    const parsed = Number(raw);
    if (!Number.isNaN(parsed)) {
      return Math.min(Math.max(parsed, 0.75), 0.995);
    }
  }
  return Math.max(0.82, speakerSimilarityThreshold - 0.04);
};

const isAmbiguousSpeakerPair = (best, second) => {
  if (!second) {
    return false;
  }
  if (second.score < speakerMatchSecondStrongMin) {
    return false;
  }
  return best.score - second.score < speakerMatchMinGapBetweenProfiles;
};

const detectSpeakerName = async (signature) => {
  const profiles = await speakerStore.readSpeakerProfiles();
  if (profiles.length === 0) {
    return null;
  }
  const scored = [];
  for (const profile of profiles) {
    const score = voiceRecognition.bestCosineScoreAgainstProfile(signature, profile);
    if (score == null) {
      continue;
    }
    scored.push({ name: profile.name, score });
  }
  if (scored.length === 0) {
    return null;
  }
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1];
  if (isAmbiguousSpeakerPair(best, second)) {
    return null;
  }
  if (best.score >= speakerSimilarityThreshold) {
    return best;
  }
  if (!speakerMatchRelaxedEnabled) {
    return null;
  }
  const relaxed = getSpeakerMatchRelaxedThreshold();
  const marginOk = !second || best.score - second.score >= speakerMatchMargin;
  if (marginOk && best.score >= relaxed) {
    return best;
  }
  return null;
};

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.get("/speakers", async (_req, res) => {
  try {
    const profiles = await speakerStore.readSpeakerProfiles();
    return res.json({
      speakers: profiles.map((profile) => {
        const desc =
          typeof profile.speakerDescription === "string"
            ? profile.speakerDescription
            : typeof profile.description === "string"
              ? profile.description
              : "";
        return {
          name: profile.name,
          samples: profile.samples || 0,
          ...(desc.trim() ? { description: desc.trim() } : {}),
        };
      }),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not load speaker profiles.";
    return res.status(500).json({ error: message });
  }
});

app.patch("/speakers", async (req, res) => {
  try {
    const rawName = typeof req.body?.name === "string" ? req.body.name.trim() : "";
    if (!rawName) {
      return res.status(400).json({ error: "Missing name in JSON body." });
    }
    const rawDesc =
      typeof req.body?.speakerDescription === "string" ? req.body.speakerDescription : "";
    const speakerDescription = String(rawDesc).trim().slice(0, assemblySpeakerDescriptionMaxLen);

    const profiles = await speakerStore.readSpeakerProfiles();
    const key = rawName.toLowerCase();
    const idx = profiles.findIndex((item) => String(item.name || "").trim().toLowerCase() === key);
    if (idx < 0) {
      return res.status(404).json({ error: "Speaker not found." });
    }

    const current = profiles[idx];
    const next = { ...current };
    if (speakerDescription) {
      next.speakerDescription = speakerDescription;
      delete next.description;
    } else {
      delete next.speakerDescription;
      delete next.description;
    }
    profiles[idx] = next;
    await speakerStore.writeSpeakerProfiles(profiles);

    const descOut =
      typeof next.speakerDescription === "string"
        ? next.speakerDescription
        : typeof next.description === "string"
          ? next.description
          : "";
    return res.json({
      ok: true,
      speaker: {
        name: next.name,
        samples: next.samples || 0,
        ...(descOut.trim() ? { description: descOut.trim() } : {}),
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not update speaker profile.";
    return res.status(500).json({ error: message });
  }
});

app.delete("/speakers", async (_req, res) => {
  try {
    await speakerStore.writeSpeakerProfiles([]);
    return res.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not clear speaker profiles.";
    return res.status(500).json({ error: message });
  }
});

const executeTranscription = async (req, res, fileLike) => {
  try {
    if (!assemblyApiKey) {
      return res.status(500).json({ error: "ASSEMBLYAI_API_KEY is not configured." });
    }

    const assemblyClient = getAssemblyClient();
    if (!assemblyClient) {
      return res.status(500).json({ error: "ASSEMBLYAI_API_KEY is not configured." });
    }

    if (!audioMimeOk(fileLike.mimetype)) {
      return res.status(400).json({ error: "Unsupported audio type." });
    }

    console.log("Transcription started", {
      mime: fileLike.mimetype,
      size: fileLike.size,
      name: String(fileLike.originalname || "upload").slice(0, 120),
    });

    const extension = extensionFromMime(fileLike.mimetype);
    const pcmBuffer = await convertAudioToPcm(fileLike.buffer, extension);
    const voiceSignature = voiceRecognition.buildVoiceSignature(pcmBuffer);
    const speakerNameHeader =
      typeof req.headers["x-speaker-name"] === "string" ? req.headers["x-speaker-name"].trim() : "";
    const speakerNameQuery =
      typeof req.query?.speakerName === "string" ? req.query.speakerName.trim() : "";
    const manualSpeakerNameFromBody =
      typeof req.body?.speakerName === "string" ? req.body.speakerName.trim() : "";
    const manualSpeakerName = speakerNameQuery || speakerNameHeader || manualSpeakerNameFromBody;
    if (manualSpeakerName) {
      await updateSpeakerProfile(manualSpeakerName, voiceSignature);
    }
    const detectedSpeaker = manualSpeakerName ? null : await detectSpeakerName(voiceSignature);

    const profilesForIdentification = await speakerStore.readSpeakerProfiles();
    const knownSpeakerValues = voiceRecognition.prioritizeSpeakerNameInKnownList(
      voiceRecognition.buildKnownSpeakerValuesForIdentification(
        profilesForIdentification,
        manualSpeakerName,
        { maxNames: assemblyKnownSpeakersMax, nameMaxLen: assemblySpeakerNameMaxLen },
      ),
      manualSpeakerName,
    );
    const assemblySpeakers = voiceRecognition.prioritizeManualSpeakerFirstSpeakers(
      voiceRecognition.profilesToAssemblySpeakers(profilesForIdentification, manualSpeakerName, {
        maxNames: assemblyKnownSpeakersMax,
        nameMaxLen: assemblySpeakerNameMaxLen,
        descriptionMaxLen: assemblySpeakerDescriptionMaxLen,
      }),
      manualSpeakerName,
    );
    const speechUnderstandingBlock =
      voiceRecognition.buildSpeechUnderstandingSpeakerIdentificationFromSpeakers(
        assemblySpeakers,
        speakerIdentificationEnabled,
      ) ||
      voiceRecognition.buildSpeechUnderstandingSpeakerIdentification(
        knownSpeakerValues,
        speakerIdentificationEnabled,
      );

    const baseTranscriptOptions = {
      audio: fileLike.buffer,
      speaker_labels: true,
      ...(forcedLanguageCode ? { language_code: forcedLanguageCode } : { language_detection: true }),
    };

    let transcript = null;
    let lastModelError = null;
    const modelsToTry = assemblySpeechModels.length > 0 ? assemblySpeechModels : ["best"];

    for (const model of modelsToTry) {
      try {
        const candidate = await transcribeWithRetry(
          assemblyClient,
          voiceRecognition.mergeAssemblyTranscriptPayload(
            {
              ...baseTranscriptOptions,
              speech_models: [model],
            },
            speechUnderstandingBlock,
          ),
        );
        if (candidate && candidate.status === "error") {
          lastModelError = new Error(candidate.error || `AssemblyAI rejected model ${model}.`);
          continue;
        }
        transcript = candidate;
        lastModelError = null;
        break;
      } catch (error) {
        lastModelError = error;
      }
    }

    if (!transcript && lastModelError) {
      const multilingualFallbackOptions = voiceRecognition.mergeAssemblyTranscriptPayload(
        {
          audio: fileLike.buffer,
          speaker_labels: true,
          ...(forcedLanguageCode ? { language_code: forcedLanguageCode } : { language_detection: true }),
          speech_models: ["best"],
        },
        speechUnderstandingBlock,
      );
      const fallbackTranscript = await transcribeWithRetry(assemblyClient, multilingualFallbackOptions);
      if (!fallbackTranscript || fallbackTranscript.status === "error") {
        let languageSpecificTranscript = null;
        let languageSpecificError = null;
        for (const languageCode of languageFallbackCodes) {
          const candidate = await transcribeWithRetry(
            assemblyClient,
            voiceRecognition.mergeAssemblyTranscriptPayload(
              {
                audio: fileLike.buffer,
                speaker_labels: true,
                language_code: languageCode,
                speech_models: ["best"],
              },
              speechUnderstandingBlock,
            ),
          );
          if (candidate && candidate.status !== "error") {
            languageSpecificTranscript = candidate;
            languageSpecificError = null;
            break;
          }
          languageSpecificError = candidate?.error || `Fallback failed for ${languageCode}.`;
        }
        if (!languageSpecificTranscript) {
          throw new Error(
            String(
              languageSpecificError ||
                (fallbackTranscript && fallbackTranscript.error) ||
                "AssemblyAI fallback transcription failed.",
            ),
          );
        }
        transcript = languageSpecificTranscript;
      } else {
        transcript = fallbackTranscript;
      }
    }

    if (transcript.status === "error") {
      throw new Error(transcript.error || "AssemblyAI transcription failed.");
    }

    const transcriptText = transcript.text || "";
    const utterances = Array.isArray(transcript.utterances) ? transcript.utterances : [];
    const dominantAssemblySpeakerLabel = voiceRecognition.formatDominantAssemblySpeakerLabel(utterances);
    const speakerIdentificationMapping = voiceRecognition.pickSpeakerIdentificationMapping(transcript);
    const aiInsights = await buildAiInsights(transcriptText).catch(() => null);

    const speakerIdentificationCandidates =
      assemblySpeakers.length > 0 ? assemblySpeakers.map((s) => s.name) : knownSpeakerValues;

    return res.json({
      text: transcriptText,
      detectedSpeakerName: detectedSpeaker?.name || null,
      speakerConfidence: detectedSpeaker?.score ?? null,
      assemblySpeakerLabel: dominantAssemblySpeakerLabel,
      speakerIdentificationCandidates,
      speakerIdentificationMapping,
      utterances: utterances.map((item) => ({
        speaker: item.speaker || null,
        text: item.text || "",
        start: item.start || 0,
        end: item.end || 0,
      })),
      ai: aiInsights,
    });
  } catch (error) {
    const message =
      error && typeof error === "object" && "message" in error
        ? error.message
        : "Unexpected transcription error.";
    return res.status(500).json({ error: String(message) });
  }
};

app.post("/transcribe-base64", async (req, res) => {
  const raw = typeof req.body?.audioBase64 === "string" ? req.body.audioBase64.trim() : "";
  if (!raw) {
    return res.status(400).json({ error: "Missing audioBase64 in JSON body." });
  }
  let mimeFromDataUrl = null;
  let b64 = raw.replace(/\s/g, "");
  const dataUrl = /^data:([^;]+);base64,(.+)$/i.exec(b64);
  if (dataUrl) {
    mimeFromDataUrl = dataUrl[1].trim();
    b64 = dataUrl[2].replace(/\s/g, "");
  }
  let buffer;
  try {
    buffer = Buffer.from(b64, "base64");
  } catch {
    return res.status(400).json({ error: "Invalid base64 audio." });
  }
  if (!buffer.length) {
    return res.status(400).json({ error: "Decoded audio is empty." });
  }
  if (buffer.length > maxUploadBytes) {
    return res.status(400).json({ error: `Audio too large (max ${maxUploadMb} MB).` });
  }
  const mimeType =
    mimeFromDataUrl ||
    (typeof req.body?.mimeType === "string" && req.body.mimeType.trim()
      ? req.body.mimeType.trim()
      : "audio/m4a");
  if (!audioMimeOk(mimeType)) {
    return res.status(400).json({ error: "Unsupported audio type." });
  }
  const fileLike = {
    buffer,
    mimetype: mimeType,
    originalname: "recording.m4a",
    size: buffer.length,
  };
  return executeTranscription(req, res, fileLike);
});

app.post("/transcribe", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Missing audio file in 'file' field." });
  }
  const fileLike = {
    buffer: req.file.buffer,
    mimetype: req.file.mimetype,
    originalname: req.file.originalname,
    size: req.file.size,
  };
  return executeTranscription(req, res, fileLike);
});

app.post("/ai/insights", async (req, res) => {
  try {
    if (!openaiApiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not configured." });
    }
    const text = typeof req.body?.text === "string" ? req.body.text : "";
    if (!text.trim()) {
      return res.status(400).json({ error: "Missing transcript text." });
    }
    const insights = await buildAiInsights(text);
    return res.json({ ai: insights });
  } catch (error) {
    const message = error instanceof Error ? error.message : "AI insights failed.";
    return res.status(500).json({ error: message });
  }
});

app.post("/ai/speaker-correction", async (req, res) => {
  try {
    if (!openaiApiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not configured." });
    }
    const latestText = typeof req.body?.latestText === "string" ? req.body.latestText : "";
    const recentHistory = Array.isArray(req.body?.recentHistory) ? req.body.recentHistory : [];
    if (!latestText.trim()) {
      return res.status(400).json({ error: "Missing latestText." });
    }
    const suggestion = await buildSpeakerCorrectionSuggestion(latestText, recentHistory);
    return res.json({ suggestion });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Speaker correction failed.";
    return res.status(500).json({ error: message });
  }
});

app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({ error: `Audio file too large (max ${maxUploadMb} MB).` });
    }
    return res.status(400).json({ error: err.message || "Upload error." });
  }
  if (typeof err?.message === "string" && err.message.includes("Unsupported audio type")) {
    return res.status(400).json({ error: err.message });
  }
  if (typeof err?.message === "string" && err.message.includes("Unexpected end of form")) {
    console.warn("Multipart parse failed (truncated upload?)", { path: req.path, method: req.method });
    return res.status(400).json({
      error:
        "Upload was cut off before the full audio file arrived. Try again, or update the app if this persists.",
    });
  }
  return next(err);
});

app.use((err, req, res, next) => {
  if (res.headersSent) {
    return next(err);
  }
  const msg = err instanceof Error ? err.message : String(err);
  console.error("Unhandled error:", msg, err instanceof Error ? err.stack : "");
  const expose = process.env.EXPOSE_ERROR_DETAILS === "true";
  return res.status(500).json({
    error: "Internal server error.",
    ...(expose ? { detail: msg } : {}),
  });
});

module.exports = app;
