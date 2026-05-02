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
const historyStore = require("./historyStore");
const voiceRecognition = require("./voiceRecognition");

const app = express();
const serverToken =
  typeof process.env.SERVER_BEARER_TOKEN === "string" ? process.env.SERVER_BEARER_TOKEN.trim() : "";
const openaiApiKey = process.env.OPENAI_API_KEY;
const assemblyApiKey = process.env.ASSEMBLYAI_API_KEY;
const openaiAiModel = process.env.OPENAI_AI_MODEL || "gpt-4o-mini";
const googleMapsApiKey = (process.env.GOOGLE_MAPS_API_KEY || process.env.GOOGLE_DIRECTIONS_API_KEY || "").trim();
const assemblySpeechModels = ["universal-3-pro", "universal-2"];
const forcedLanguageCode = process.env.ASSEMBLYAI_FORCE_LANGUAGE_CODE || "";
const languageFallbackCodes = (process.env.ASSEMBLYAI_LANGUAGE_FALLBACKS || "hi,te,bn")
  .split(",")
  .map((item) => item.trim())
  .filter(Boolean);
/** Higher = fewer wrong-speaker IDs; lower if the same person is often missed. */
const speakerSimilarityThreshold = Number(process.env.SPEAKER_MATCH_THRESHOLD || "0.95");
const speakerMatchMargin = Number(process.env.SPEAKER_MATCH_MARGIN || "0.04");
const speakerMatchRelaxedEnabled = ["true", "1", "yes"].includes(
  (process.env.SPEAKER_MATCH_RELAXED_ENABLED || "true").trim().toLowerCase(),
);
/** If the runner-up is still “strong”, require at least this cosine gap (reduces confused twins / similar voices). */
const speakerMatchMinGapBetweenProfiles = Number(process.env.SPEAKER_MATCH_MIN_GAP_BETWEEN_PROFILES || "0.025");
const speakerMatchSecondStrongMin = Number(process.env.SPEAKER_MATCH_SECOND_STRONG_MIN || "0.84");
const speakerMatchHighConfidenceOverride = Number(process.env.SPEAKER_MATCH_HIGH_CONFIDENCE_OVERRIDE || "0.995");
const speakerEmbeddingServiceUrl = (process.env.SPEAKER_EMBEDDING_SERVICE_URL || "").trim().replace(/\/+$/, "");
const speakerEmbeddingServiceToken = (process.env.SPEAKER_EMBEDDING_SERVICE_TOKEN || "").trim();
const speakerEmbeddingMatchThreshold = Number(process.env.SPEAKER_EMBEDDING_MATCH_THRESHOLD || "0.55");
const speakerEmbeddingMatchMargin = Number(process.env.SPEAKER_EMBEDDING_MATCH_MARGIN || "0.05");
const speakerEmbeddingHighConfidenceOverride = Number(process.env.SPEAKER_EMBEDDING_HIGH_CONFIDENCE_OVERRIDE || "0.75");
const speakerEmbeddingTimeoutMs = Math.min(
  Math.max(Number(process.env.SPEAKER_EMBEDDING_TIMEOUT_MS || "18000") || 18000, 1000),
  45000,
);
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
    const provided = match ? match[1].trim() : "";
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

const fetchSpeakerEmbedding = async (audioBuffer, mimeType) => {
  if (!speakerEmbeddingServiceUrl) {
    return null;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), speakerEmbeddingTimeoutMs);
  try {
    const response = await withRetries(
      () =>
        fetch(`${speakerEmbeddingServiceUrl}/embed`, {
          method: "POST",
          signal: controller.signal,
          headers: {
            "Content-Type": "application/json",
            ...(speakerEmbeddingServiceToken ? { Authorization: `Bearer ${speakerEmbeddingServiceToken}` } : {}),
          },
          body: JSON.stringify({
            audioBase64: audioBuffer.toString("base64"),
            mimeType: mimeType || "audio/m4a",
          }),
        }),
      { label: "SpeakerEmbedding.embed", maxAttempts: 1, baseDelayMs: 600 },
    );
    const text = await response.text();
    if (!response.ok) {
      throw new Error(text || `Speaker embedding service failed (${response.status}).`);
    }
    const payload = JSON.parse(text);
    const embedding = Array.isArray(payload?.embedding) ? payload.embedding.map(Number) : [];
    if (embedding.length === 0 || embedding.some((value) => !Number.isFinite(value))) {
      throw new Error("Speaker embedding service returned an invalid embedding.");
    }
    return embedding;
  } finally {
    clearTimeout(timeout);
  }
};

const buildPersonAiSummary = async (speakerName, contextText) => {
  if (!openaiApiKey || !contextText.trim()) {
    return "";
  }
  const client = getOpenAIClient();
  if (!client) {
    return "";
  }
  const label =
    typeof speakerName === "string" && speakerName.trim() ? speakerName.trim().slice(0, 120) : "This speaker";

  const response = await withRetries(
    () =>
      client.responses.create({
        model: openaiAiModel,
        input: [
          {
            role: "system",
            content:
              "You write a short spoken briefing about ONE person using only the voice-log excerpts provided. Do not invent facts, names, or events. If the logs are thin or only technical metadata, say so briefly. Use 2–6 short sentences that sound natural when read aloud by a voice assistant—clear, neutral, and friendly. Return strict JSON with a single key narrative (string). No bullet characters or markdown.",
          },
          {
            role: "user",
            content: `App label for this person: ${label}\n\n---\n${contextText.slice(0, 12000)}`,
          },
        ],
        text: {
          format: {
            type: "json_schema",
            name: "person_voice_summary",
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                narrative: { type: "string" },
              },
              required: ["narrative"],
            },
          },
        },
      }),
    { label: "OpenAI.buildPersonAiSummary" },
  );

  const raw = response.output_text || "{}";
  const parsed = JSON.parse(raw);
  return typeof parsed.narrative === "string" ? parsed.narrative.trim() : "";
};

const buildSpeakerTodayPlan = async (speakerName, dateLabel, contextText) => {
  if (!openaiApiKey || !contextText.trim()) {
    return "";
  }
  const client = getOpenAIClient();
  if (!client) {
    return "";
  }
  const label =
    typeof speakerName === "string" && speakerName.trim() ? speakerName.trim().slice(0, 120) : "This speaker";
  const when =
    typeof dateLabel === "string" && dateLabel.trim() ? dateLabel.trim().slice(0, 120) : "the user's local today";

  const response = await withRetries(
    () =>
      client.responses.create({
        model: openaiAiModel,
        input: [
          {
            role: "system",
            content:
              "You help with planning for ONE person using voice-app logs. SECTION 1 is that person's clips from the calendar day named in the user message; SECTION 2 is other recent app-wide context (may include other speakers) to ground advice—use it only to clarify steps or dependencies, not to invent new plans. If SECTION 1 clearly has no usable schedule or tasks for that day, respond with narrative exactly: No information for today. Otherwise: in 3–7 short sentences suitable for text-to-speech, say what they appear to be doing that day and concrete steps to get it done. Do not invent addresses or times not supported by the logs. Return strict JSON with key narrative (string). No markdown.",
          },
          {
            role: "user",
            content: `Calendar day in focus (user's local timezone): ${when}\nApp label for this person: ${label}\n\n---\n${contextText.slice(0, 12000)}`,
          },
        ],
        text: {
          format: {
            type: "json_schema",
            name: "speaker_today_plan",
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                narrative: { type: "string" },
              },
              required: ["narrative"],
            },
          },
        },
      }),
    { label: "OpenAI.buildSpeakerTodayPlan" },
  );

  const raw = response.output_text || "{}";
  const parsed = JSON.parse(raw);
  return typeof parsed.narrative === "string" ? parsed.narrative.trim() : "";
};

const buildExtractDestination = async (contextText) => {
  if (!openaiApiKey || !contextText.trim()) {
    return "";
  }
  const client = getOpenAIClient();
  if (!client) {
    return "";
  }

  const response = await withRetries(
    () =>
      client.responses.create({
        model: openaiAiModel,
        input: [
          {
            role: "system",
            content:
              "From notes and transcripts, extract a single place the user wants to DRIVE or GO TO (street address, building + city, or well-known place name that maps navigation could use). If multiple, pick the clearest trip destination. If none, return an empty destination string. Return strict JSON { destination: string }.",
          },
          {
            role: "user",
            content: contextText.slice(0, 12000),
          },
        ],
        text: {
          format: {
            type: "json_schema",
            name: "extract_destination",
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                destination: { type: "string" },
              },
              required: ["destination"],
            },
          },
        },
      }),
    { label: "OpenAI.buildExtractDestination" },
  );

  const raw = response.output_text || "{}";
  const parsed = JSON.parse(raw);
  return typeof parsed.destination === "string" ? parsed.destination.trim().slice(0, 500) : "";
};

const fetchDrivingDurationWithTraffic = async (originLat, originLng, destination) => {
  if (!googleMapsApiKey) {
    throw new Error("GOOGLE_MAPS_API_KEY is not configured on the server.");
  }
  const dest = typeof destination === "string" ? destination.trim() : "";
  if (!dest) {
    throw new Error("Missing destination.");
  }
  const lat = Number(originLat);
  const lng = Number(originLng);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    throw new Error("Invalid origin coordinates.");
  }

  const params = new URLSearchParams({
    origin: `${lat},${lng}`,
    destination: dest,
    mode: "driving",
    departure_time: String(Math.floor(Date.now() / 1000)),
    traffic_model: "best_guess",
    key: googleMapsApiKey,
  });
  const url = `https://maps.googleapis.com/maps/api/directions/json?${params.toString()}`;
  const res = await fetch(url);
  const data = await res.json();
  if (data.status !== "OK" || !Array.isArray(data.routes) || data.routes.length === 0) {
    const msg =
      typeof data.error_message === "string" && data.error_message.trim()
        ? data.error_message
        : typeof data.status === "string"
          ? data.status
          : "Directions request failed.";
    throw new Error(msg);
  }
  const leg = data.routes[0].legs[0];
  const baseSec = Number(leg?.duration?.value) || 0;
  const trafficSecRaw = leg?.duration_in_traffic?.value;
  const trafficSec = Number.isFinite(Number(trafficSecRaw)) ? Number(trafficSecRaw) : baseSec;
  const summary =
    (typeof leg?.duration_in_traffic?.text === "string" && leg.duration_in_traffic.text.trim()
      ? leg.duration_in_traffic.text
      : null) ||
    (typeof leg?.duration?.text === "string" ? leg.duration.text : "") ||
    "";

  return {
    durationSeconds: baseSec,
    durationInTrafficSeconds: trafficSec,
    summaryText: summary,
    baselineMinutes: Math.max(1, Math.round(baseSec / 60)),
    trafficMinutes: Math.max(1, Math.round(trafficSec / 60)),
  };
};

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
              "You summarize transcripts into concise notes. Return strict JSON with keys summary, actionItems, topics, and followUpQuestions. summary is one short paragraph. actionItems is concrete follow-ups (short strings). topics is 3-8 short topical tags (Title Case). followUpQuestions is 0-3 short, natural questions the user could be asked aloud to supply missing but important details only when clearly needed (examples: full street address for a trip, appointment time, person's full name). Each question must stand alone and sound natural when spoken by a voice assistant. Use an empty array if the transcript is complete enough. If transcript is empty or noise, use empty strings and empty arrays.",
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
                topics: { type: "array", items: { type: "string" } },
                followUpQuestions: { type: "array", items: { type: "string" } },
              },
              required: ["summary", "actionItems", "topics", "followUpQuestions"],
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
    topics: Array.isArray(parsed.topics) ? parsed.topics.filter((item) => typeof item === "string") : [],
    followUpQuestions: Array.isArray(parsed.followUpQuestions)
      ? parsed.followUpQuestions.filter((item) => typeof item === "string")
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

const cleanEnrollmentSource = (value) => {
  const source = typeof value === "string" ? value.trim() : "";
  const allowed = new Set(["speaker_name_input", "unknown_speaker_prompt", "speaker_conflict_prompt"]);
  return allowed.has(source) ? source : "speaker_name_input";
};

const speakerNameKey = (value) => String(value || "").trim().toLowerCase();

const buildEnrollmentSample = (speakerName, signature, metadata = {}, speakerEmbedding = null) => {
  const createdAtMs = Date.now();
  const historyClientId =
    typeof metadata.historyClientId === "string" ? metadata.historyClientId.trim().slice(0, 120) : "";
  return {
    sampleId: randomUUID(),
    speakerName,
    source: cleanEnrollmentSource(metadata.source),
    createdAtMs,
    createdAtIso: new Date(createdAtMs).toISOString(),
    ...(historyClientId ? { historyClientId } : {}),
    vector: [...signature],
    ...(Array.isArray(speakerEmbedding) && speakerEmbedding.length > 0 ? { embeddingVector: [...speakerEmbedding] } : {}),
  };
};

const averageVectors = (vectors) => {
  const valid = vectors.filter((vector) => Array.isArray(vector) && vector.length > 0);
  if (valid.length === 0) {
    return null;
  }
  const dim = valid[0].length;
  const aligned = valid.map((vector) => voiceRecognition.padVoiceVector(vector, dim, 0));
  return Array.from({ length: dim }, (_, index) =>
    aligned.reduce((sum, vector) => sum + vector[index], 0) / aligned.length,
  );
};

const updateSpeakerProfile = async (speakerName, signature, metadata = {}, speakerEmbedding = null) => {
  const profiles = await readSpeakerProfilesMerged();
  const sigCopy = [...signature];
  const embeddingCopy = Array.isArray(speakerEmbedding) && speakerEmbedding.length > 0 ? [...speakerEmbedding] : null;
  const displaySpeakerName = speakerName.trim();
  const existingIndex = profiles.findIndex((item) => speakerNameKey(item.name) === speakerNameKey(displaySpeakerName));
  let savedSpeakerName = displaySpeakerName;
  if (existingIndex >= 0) {
    const current = profiles[existingIndex];
    const storedSpeakerName = typeof current.name === "string" && current.name.trim() ? current.name.trim() : displaySpeakerName;
    savedSpeakerName = storedSpeakerName;
    const enrollmentSample = buildEnrollmentSample(storedSpeakerName, sigCopy, metadata, embeddingCopy);
    const total = (current.samples || 0) + 1;
    const currentVector =
      Array.isArray(current.vector) && current.vector.length > 0 ? current.vector : signature.map(() => 0);
    const alignedCurrent = voiceRecognition.padVoiceVector(currentVector, signature.length, 0.5);
    const prevRecent = Array.isArray(current.vectorsRecent)
      ? current.vectorsRecent.filter((v) => Array.isArray(v) && v.length > 0).map((v) => [...v])
      : [];
    const nextRecent = [...prevRecent, sigCopy].slice(-speakerRecentVectorsMax);
    const enrollmentSamples = Array.isArray(current.enrollmentSamples)
      ? [...current.enrollmentSamples, enrollmentSample]
      : [enrollmentSample];
    const embeddingVectors = enrollmentSamples
      .map((sample) => sample?.embeddingVector)
      .filter((vector) => Array.isArray(vector) && vector.length > 0);
    const embeddingVector = averageVectors(embeddingVectors);
    const embeddingRecent = embeddingVectors.slice(-speakerRecentVectorsMax);
    profiles[existingIndex] = {
      ...current,
      name: storedSpeakerName,
      samples: total,
      vector: alignedCurrent.map((value, index) => (value * (total - 1) + signature[index]) / total),
      vectorsRecent: nextRecent,
      enrollmentSamples,
      ...(embeddingVector ? { embeddingVector, embeddingRecent } : {}),
    };
  } else {
    const enrollmentSample = buildEnrollmentSample(displaySpeakerName, sigCopy, metadata, embeddingCopy);
    profiles.push({
      name: displaySpeakerName,
      samples: 1,
      vector: signature,
      vectorsRecent: [sigCopy],
      enrollmentSamples: [enrollmentSample],
      ...(embeddingCopy ? { embeddingVector: embeddingCopy, embeddingRecent: [embeddingCopy] } : {}),
    });
  }
  await speakerStore.writeSpeakerProfiles(profiles);
  return savedSpeakerName;
};

/** Max embedding floats returned per vector in GET /speakers (full ECAPA ~192). */
const publicSpeakerEmbeddingMaxLength = Math.min(
  Math.max(Number(process.env.PUBLIC_SPEAKER_EMBEDDING_MAX_LENGTH || "1024") || 1024, 32),
  2048,
);

const roundNumericArray = (value, decimals, maxLength) => {
  if (!Array.isArray(value) || value.length === 0) {
    return { numbers: null, sourceLength: 0, truncated: false };
  }
  const sourceLength = value.length;
  const capped = typeof maxLength === "number" && maxLength > 0 ? value.slice(0, maxLength) : value;
  const numbers = capped.map((item) => {
    const n = Number(item);
    if (!Number.isFinite(n)) {
      return 0;
    }
    return Number(n.toFixed(decimals));
  });
  return { numbers, sourceLength, truncated: sourceLength > numbers.length };
};

const publicEnrollmentSamples = (profile) =>
  Array.isArray(profile?.enrollmentSamples)
    ? profile.enrollmentSamples
        .filter((sample) => sample && typeof sample === "object" && typeof sample.sampleId === "string")
        .map((sample) => {
          const fp = roundNumericArray(sample.vector, 6, 32);
          const emb = roundNumericArray(sample.embeddingVector, 6, publicSpeakerEmbeddingMaxLength);
          return {
            sampleId: sample.sampleId,
            source: typeof sample.source === "string" ? sample.source : "unknown",
            createdAtMs: typeof sample.createdAtMs === "number" ? sample.createdAtMs : null,
            createdAtIso: typeof sample.createdAtIso === "string" ? sample.createdAtIso : null,
            historyClientId: typeof sample.historyClientId === "string" ? sample.historyClientId : null,
            hasFingerprint: Array.isArray(sample.vector) && sample.vector.length > 0,
            hasEmbedding: Array.isArray(sample.embeddingVector) && sample.embeddingVector.length > 0,
            voiceFingerprintDimensions: fp.sourceLength || null,
            voiceFingerprint: fp.numbers,
            voiceFingerprintTruncated: fp.truncated,
            embeddingDimensions: emb.sourceLength || null,
            embeddingVector: emb.numbers,
            embeddingTruncated: emb.truncated,
          };
        })
    : [];

const publicProfileVoiceRollups = (profile) => {
  const aggFp = roundNumericArray(profile?.vector, 6, 32);
  const recentFp = Array.isArray(profile?.vectorsRecent)
    ? profile.vectorsRecent.map((v) => roundNumericArray(v, 6, 32)).filter((item) => item.numbers && item.numbers.length)
    : [];
  const aggEmb = roundNumericArray(profile?.embeddingVector, 6, publicSpeakerEmbeddingMaxLength);
  const recentEmb = Array.isArray(profile?.embeddingRecent)
    ? profile.embeddingRecent
        .map((v) => roundNumericArray(v, 6, publicSpeakerEmbeddingMaxLength))
        .filter((item) => item.numbers && item.numbers.length)
    : [];
  return {
    profileVoiceFingerprint: aggFp.numbers,
    profileVoiceFingerprintDimensions: aggFp.sourceLength || null,
    profileVoiceFingerprintTruncated: aggFp.truncated,
    profileVoiceFingerprintsRecent: recentFp.map((item) => ({
      values: item.numbers,
      dimensions: item.sourceLength,
      truncated: item.truncated,
    })),
    profileSpeakerEmbedding: aggEmb.numbers
      ? {
          values: aggEmb.numbers,
          dimensions: aggEmb.sourceLength,
          truncated: aggEmb.truncated,
        }
      : null,
    profileSpeakerEmbeddingsRecent: recentEmb.map((item) => ({
      values: item.numbers,
      dimensions: item.sourceLength,
      truncated: item.truncated,
    })),
  };
};

const rebuildProfileFromEnrollmentSamples = (profile, enrollmentSamples) => {
  const validSamples = enrollmentSamples.filter((sample) => Array.isArray(sample?.vector) && sample.vector.length > 0);
  if (validSamples.length === 0) {
    return null;
  }
  const dim = validSamples[0].vector.length;
  const vectors = validSamples.map((sample) => voiceRecognition.padVoiceVector(sample.vector, dim, 0.5));
  const vector = Array.from({ length: dim }, (_, index) =>
    vectors.reduce((sum, item) => sum + item[index], 0) / vectors.length,
  );
  const embeddingVectors = validSamples
    .map((sample) => sample?.embeddingVector)
    .filter((embedding) => Array.isArray(embedding) && embedding.length > 0);
  const embeddingVector = averageVectors(embeddingVectors);
  const rebuilt = {
    ...profile,
    samples: validSamples.length,
    vector,
    vectorsRecent: vectors.slice(-speakerRecentVectorsMax),
    enrollmentSamples: validSamples,
  };
  if (embeddingVector) {
    rebuilt.embeddingVector = embeddingVector;
    rebuilt.embeddingRecent = embeddingVectors.slice(-speakerRecentVectorsMax);
  } else {
    delete rebuilt.embeddingVector;
    delete rebuilt.embeddingRecent;
  }
  return rebuilt;
};

const mergeEnrollmentSamples = (a = [], b = []) => {
  const byId = new Map();
  for (const sample of [...a, ...b]) {
    if (!sample || typeof sample !== "object") {
      continue;
    }
    const id = typeof sample.sampleId === "string" && sample.sampleId.trim() ? sample.sampleId : randomUUID();
    byId.set(id, { ...sample, sampleId: id });
  }
  return [...byId.values()].sort((left, right) => {
    const leftTime = typeof left.createdAtMs === "number" ? left.createdAtMs : 0;
    const rightTime = typeof right.createdAtMs === "number" ? right.createdAtMs : 0;
    return leftTime - rightTime;
  });
};

const mergeSpeakerProfilePair = (base, incoming) => {
  const baseSamples = Math.max(Number(base.samples) || 0, 0);
  const incomingSamples = Math.max(Number(incoming.samples) || 0, 0);
  const totalSamples = baseSamples + incomingSamples || 1;
  const dim = Math.max(
    Array.isArray(base.vector) ? base.vector.length : 0,
    Array.isArray(incoming.vector) ? incoming.vector.length : 0,
  );
  const baseVector = dim > 0 ? voiceRecognition.padVoiceVector(base.vector || [], dim, 0.5) : [];
  const incomingVector = dim > 0 ? voiceRecognition.padVoiceVector(incoming.vector || [], dim, 0.5) : [];
  const enrollmentSamples = mergeEnrollmentSamples(base.enrollmentSamples, incoming.enrollmentSamples);
  const vectorsRecent = [
    ...(Array.isArray(base.vectorsRecent) ? base.vectorsRecent : []),
    ...(Array.isArray(incoming.vectorsRecent) ? incoming.vectorsRecent : []),
  ]
    .filter((vector) => Array.isArray(vector) && vector.length > 0)
    .slice(-speakerRecentVectorsMax);
  const embeddingVectors = enrollmentSamples
    .map((sample) => sample?.embeddingVector)
    .filter((embedding) => Array.isArray(embedding) && embedding.length > 0);
  if (embeddingVectors.length === 0) {
    if (Array.isArray(base.embeddingVector) && base.embeddingVector.length > 0) {
      embeddingVectors.push(base.embeddingVector);
    }
    if (Array.isArray(incoming.embeddingVector) && incoming.embeddingVector.length > 0) {
      embeddingVectors.push(incoming.embeddingVector);
    }
  }
  const embeddingVector = averageVectors(embeddingVectors);
  const embeddingRecent = [
    ...(Array.isArray(base.embeddingRecent) ? base.embeddingRecent : []),
    ...(Array.isArray(incoming.embeddingRecent) ? incoming.embeddingRecent : []),
    ...embeddingVectors,
  ]
    .filter((embedding) => Array.isArray(embedding) && embedding.length > 0)
    .slice(-speakerRecentVectorsMax);
  const speakerDescription =
    typeof base.speakerDescription === "string" && base.speakerDescription.trim()
      ? base.speakerDescription
      : typeof incoming.speakerDescription === "string" && incoming.speakerDescription.trim()
        ? incoming.speakerDescription
        : undefined;
  const description =
    typeof base.description === "string" && base.description.trim()
      ? base.description
      : typeof incoming.description === "string" && incoming.description.trim()
        ? incoming.description
        : undefined;

  return {
    ...base,
    name: typeof base.name === "string" && base.name.trim() ? base.name.trim() : incoming.name,
    samples: totalSamples,
    ...(dim > 0
      ? {
          vector: baseVector.map(
            (value, index) => (value * baseSamples + incomingVector[index] * incomingSamples) / totalSamples,
          ),
        }
      : {}),
    ...(vectorsRecent.length > 0 ? { vectorsRecent } : {}),
    ...(enrollmentSamples.length > 0 ? { enrollmentSamples } : {}),
    ...(embeddingVector ? { embeddingVector } : {}),
    ...(embeddingRecent.length > 0 ? { embeddingRecent } : {}),
    ...(speakerDescription ? { speakerDescription } : {}),
    ...(description ? { description } : {}),
  };
};

const mergeSpeakerProfilesByName = (profiles) => {
  const merged = [];
  const byName = new Map();
  let changed = false;

  for (const profile of Array.isArray(profiles) ? profiles : []) {
    const key = speakerNameKey(profile?.name);
    if (!key) {
      changed = true;
      continue;
    }

    const cleanProfile = {
      ...profile,
      name: String(profile.name).trim(),
    };
    const existingIndex = byName.get(key);
    if (existingIndex === undefined) {
      byName.set(key, merged.length);
      merged.push(cleanProfile);
      changed = changed || cleanProfile.name !== profile.name;
      continue;
    }

    merged[existingIndex] = mergeSpeakerProfilePair(merged[existingIndex], cleanProfile);
    changed = true;
  }

  return { profiles: merged, changed };
};

const readSpeakerProfilesMerged = async () => {
  const profiles = await speakerStore.readSpeakerProfiles();
  const merged = mergeSpeakerProfilesByName(profiles);
  if (merged.changed) {
    await speakerStore.writeSpeakerProfiles(merged.profiles);
  }
  return merged.profiles;
};

const getSpeakerMatchRelaxedThreshold = () => {
  const raw = typeof process.env.SPEAKER_MATCH_RELAXED_THRESHOLD === "string" ? process.env.SPEAKER_MATCH_RELAXED_THRESHOLD.trim() : "";
  if (raw) {
    const parsed = Number(raw);
    if (!Number.isNaN(parsed)) {
      return Math.min(Math.max(parsed, 0.75), 0.995);
    }
  }
  return Math.max(0.86, speakerSimilarityThreshold - 0.08);
};

const isAmbiguousSpeakerPair = (best, second) => {
  if (!second) {
    return false;
  }
  if (best.score >= speakerMatchHighConfidenceOverride && best.score > second.score) {
    return false;
  }
  if (second.score < speakerMatchSecondStrongMin) {
    return false;
  }
  return best.score - second.score < speakerMatchMinGapBetweenProfiles;
};

const collectEmbeddingCandidates = (profile, dim) => {
  const candidates = [];
  const seen = new Set();
  const add = (embedding, metadata = {}) => {
    if (!Array.isArray(embedding) || embedding.length === 0) {
      return;
    }
    const vector = voiceRecognition.padVoiceVector(embedding, dim, 0);
    const key = vector.join(",");
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    candidates.push({ vector, ...metadata });
  };

  if (Array.isArray(profile?.enrollmentSamples)) {
    for (const sample of profile.enrollmentSamples) {
      add(sample?.embeddingVector, {
        sampleId: typeof sample?.sampleId === "string" ? sample.sampleId : null,
        sampleSource: typeof sample?.source === "string" ? sample.source : null,
        sampleCreatedAtIso: typeof sample?.createdAtIso === "string" ? sample.createdAtIso : null,
      });
    }
  }
  add(profile?.embeddingVector, { sampleSource: "embedding_profile" });
  if (Array.isArray(profile?.embeddingRecent)) {
    for (const embedding of profile.embeddingRecent) {
      add(embedding, { sampleSource: "embedding_recent" });
    }
  }
  return candidates;
};

const detectSpeakerNameByEmbedding = async (speakerEmbedding) => {
  if (!Array.isArray(speakerEmbedding) || speakerEmbedding.length === 0) {
    return null;
  }
  const profiles = await readSpeakerProfilesMerged();
  const scored = [];
  const dim = speakerEmbedding.length;
  for (const profile of profiles) {
    const candidates = collectEmbeddingCandidates(profile, dim);
    for (const candidate of candidates) {
      const score = voiceRecognition.cosineSimilarity(speakerEmbedding, candidate.vector);
      if (!Number.isFinite(score)) {
        continue;
      }
      const current = scored.find((item) => item.name === profile.name);
      if (!current || score > current.score) {
        const next = {
          name: profile.name,
          score,
          sampleId: candidate.sampleId || null,
          sampleSource: candidate.sampleSource || "embedding",
          sampleCreatedAtIso: candidate.sampleCreatedAtIso || null,
          recognitionEngine: "embedding",
        };
        if (current) {
          Object.assign(current, next);
        } else {
          scored.push(next);
        }
      }
    }
  }
  if (scored.length === 0) {
    return null;
  }

  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1];
  console.log("Speaker embedding candidates", {
    best: best
      ? {
          name: best.name,
          score: Number(best.score.toFixed(4)),
          sampleId: best.sampleId,
          sampleSource: best.sampleSource,
        }
      : null,
    second: second
      ? {
          name: second.name,
          score: Number(second.score.toFixed(4)),
          sampleId: second.sampleId,
          sampleSource: second.sampleSource,
        }
      : null,
    threshold: speakerEmbeddingMatchThreshold,
    margin: speakerEmbeddingMatchMargin,
    highConfidenceOverride: speakerEmbeddingHighConfidenceOverride,
  });
  if (second && best.score < speakerEmbeddingHighConfidenceOverride && best.score - second.score < speakerEmbeddingMatchMargin) {
    return null;
  }
  return best.score >= speakerEmbeddingMatchThreshold ? best : null;
};

const detectSpeakerName = async (signature) => {
  const profiles = await readSpeakerProfilesMerged();
  if (profiles.length === 0) {
    return null;
  }
  const scored = [];
  for (const profile of profiles) {
    const match = voiceRecognition.bestSpeakerMatchAgainstProfile(signature, profile);
    if (!match || match.score == null) {
      continue;
    }
    scored.push({
      name: profile.name,
      score: match.score,
      sampleId: match.sampleId,
      sampleSource: match.sampleSource,
      sampleCreatedAtIso: match.sampleCreatedAtIso,
      recognitionEngine: "fingerprint",
    });
  }
  if (scored.length === 0) {
    return null;
  }
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1];
  console.log("Speaker match candidates", {
    best: best
      ? {
          name: best.name,
          score: Number(best.score.toFixed(4)),
          sampleId: best.sampleId,
          sampleSource: best.sampleSource,
        }
      : null,
    second: second
      ? {
          name: second.name,
          score: Number(second.score.toFixed(4)),
          sampleId: second.sampleId,
          sampleSource: second.sampleSource,
        }
      : null,
    threshold: speakerSimilarityThreshold,
    relaxedEnabled: speakerMatchRelaxedEnabled,
    relaxedThreshold: getSpeakerMatchRelaxedThreshold(),
    highConfidenceOverride: speakerMatchHighConfidenceOverride,
  });
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
    const profiles = await readSpeakerProfilesMerged();
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
          ...publicProfileVoiceRollups(profile),
          enrollmentSamples: publicEnrollmentSamples(profile),
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

    const profiles = await readSpeakerProfilesMerged();
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

app.delete("/speakers/:name", async (req, res) => {
  try {
    const rawName = typeof req.params?.name === "string" ? decodeURIComponent(req.params.name).trim() : "";
    if (!rawName) {
      return res.status(400).json({ error: "Missing speaker name." });
    }
    const profiles = await readSpeakerProfilesMerged();
    const key = rawName.toLowerCase();
    const remaining = profiles.filter((item) => String(item.name || "").trim().toLowerCase() !== key);
    if (remaining.length === profiles.length) {
      return res.status(404).json({ error: "Speaker not found." });
    }
    await speakerStore.writeSpeakerProfiles(remaining);
    return res.json({ ok: true, deletedSpeakerName: rawName });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not delete speaker profile.";
    return res.status(500).json({ error: message });
  }
});

app.delete("/speakers/:name/samples/:sampleId", async (req, res) => {
  try {
    const rawName = typeof req.params?.name === "string" ? decodeURIComponent(req.params.name).trim() : "";
    const sampleId =
      typeof req.params?.sampleId === "string" ? decodeURIComponent(req.params.sampleId).trim() : "";
    if (!rawName || !sampleId) {
      return res.status(400).json({ error: "Missing speaker name or sample id." });
    }

    const profiles = await readSpeakerProfilesMerged();
    const key = rawName.toLowerCase();
    const idx = profiles.findIndex((item) => String(item.name || "").trim().toLowerCase() === key);
    if (idx < 0) {
      return res.status(404).json({ error: "Speaker not found." });
    }

    const current = profiles[idx];
    const enrollmentSamples = Array.isArray(current.enrollmentSamples) ? current.enrollmentSamples : [];
    const remainingSamples = enrollmentSamples.filter((sample) => sample?.sampleId !== sampleId);
    if (remainingSamples.length === enrollmentSamples.length) {
      return res.status(404).json({ error: "Voice sample not found." });
    }

    const rebuilt = rebuildProfileFromEnrollmentSamples(current, remainingSamples);
    if (rebuilt) {
      profiles[idx] = rebuilt;
    } else {
      profiles.splice(idx, 1);
    }
    await speakerStore.writeSpeakerProfiles(profiles);
    return res.json({
      ok: true,
      deletedSpeakerName: current.name,
      deletedSampleId: sampleId,
      remainingSamples: rebuilt?.samples || 0,
      speakerDeleted: !rebuilt,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not delete voice sample.";
    return res.status(500).json({ error: message });
  }
});

app.get("/history", async (req, res) => {
  try {
    const history = await historyStore.listHistoryEntries(req.query?.limit);
    return res.json({ history });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not load transcript history.";
    return res.status(500).json({ error: message });
  }
});

app.post("/history", async (req, res) => {
  try {
    let payload = req.body && typeof req.body === "object" ? { ...req.body } : {};
    const text = typeof payload.text === "string" ? payload.text.trim() : "";
    const clientAi = payload.ai;
    const hasUsableClientAi =
      clientAi &&
      typeof clientAi === "object" &&
      ((typeof clientAi.summary === "string" && clientAi.summary.trim()) ||
        (Array.isArray(clientAi.actionItems) &&
          clientAi.actionItems.some((item) => typeof item === "string" && item.trim())) ||
        (Array.isArray(clientAi.topics) &&
          clientAi.topics.some((item) => typeof item === "string" && item.trim())) ||
        (Array.isArray(clientAi.followUpQuestions) &&
          clientAi.followUpQuestions.some((item) => typeof item === "string" && item.trim())));
    if (!hasUsableClientAi && text && openaiApiKey) {
      try {
        const generated = await buildAiInsights(text);
        if (generated) {
          payload = { ...payload, ai: generated };
        }
      } catch (insightError) {
        console.warn("OpenAI history backfill failed", {
          message: insightError instanceof Error ? insightError.message : String(insightError),
        });
      }
    }
    const saved = await historyStore.appendHistoryEntry(payload);
    return res.status(201).json({ ok: true, history: saved });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not save transcript history.";
    return res.status(500).json({ error: message });
  }
});

app.post("/history/clear-ai", async (_req, res) => {
  try {
    const result = await historyStore.clearAiFromAllHistory();
    return res.json({ ok: true, ...result });
  } catch (error) {
    console.error("history/clear-ai failed", error);
    const message = error instanceof Error ? error.message : "Could not clear AI fields from transcript history.";
    return res.status(500).json({ error: message });
  }
});

const executeTranscription = async (req, res, fileLike) => {
  const startedAtMs = Date.now();
  let lastTimingMs = startedAtMs;
  const timings = [];
  const recordTiming = (step) => {
    const now = Date.now();
    timings.push({ step, ms: now - lastTimingMs });
    lastTimingMs = now;
  };
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
    recordTiming("audio_fingerprint");
    let speakerEmbedding = null;
    let embeddingFetchError = null;
    if (speakerEmbeddingServiceUrl) {
      try {
        speakerEmbedding = await fetchSpeakerEmbedding(fileLike.buffer, fileLike.mimetype);
      } catch (err) {
        embeddingFetchError = err instanceof Error ? err.message : String(err);
        console.warn("Speaker embedding unavailable; falling back to fingerprint matching", {
          message: embeddingFetchError,
        });
      }
    }
    recordTiming("speaker_embedding");
    const speakerNameHeader =
      typeof req.headers["x-speaker-name"] === "string" ? req.headers["x-speaker-name"].trim() : "";
    const speakerNameQuery =
      typeof req.query?.speakerName === "string" ? req.query.speakerName.trim() : "";
    const manualSpeakerNameFromBody =
      typeof req.body?.speakerName === "string" ? req.body.speakerName.trim() : "";
    const manualSpeakerName = speakerNameQuery || speakerNameHeader || manualSpeakerNameFromBody;
    let enrolledSpeakerName = null;
    if (manualSpeakerName) {
      enrolledSpeakerName = await updateSpeakerProfile(
        manualSpeakerName,
        voiceSignature,
        {
          source: req.body?.enrollmentSource || req.headers["x-enrollment-source"],
          historyClientId: req.body?.historyClientId || req.headers["x-history-client-id"],
        },
        speakerEmbedding,
      );
    }
    const detectedSpeaker = manualSpeakerName
      ? null
      : (speakerEmbedding ? await detectSpeakerNameByEmbedding(speakerEmbedding) : null) ||
        (await detectSpeakerName(voiceSignature));
    recordTiming("speaker_matching");

    const profilesForIdentification = await readSpeakerProfilesMerged();
    const knownSpeakerValues = voiceRecognition.prioritizeSpeakerNameInKnownList(
      voiceRecognition.buildKnownSpeakerValuesForIdentification(
        profilesForIdentification,
        enrolledSpeakerName || manualSpeakerName,
        { maxNames: assemblyKnownSpeakersMax, nameMaxLen: assemblySpeakerNameMaxLen },
      ),
      enrolledSpeakerName || manualSpeakerName,
    );
    const assemblySpeakers = voiceRecognition.prioritizeManualSpeakerFirstSpeakers(
      voiceRecognition.profilesToAssemblySpeakers(profilesForIdentification, enrolledSpeakerName || manualSpeakerName, {
        maxNames: assemblyKnownSpeakersMax,
        nameMaxLen: assemblySpeakerNameMaxLen,
        descriptionMaxLen: assemblySpeakerDescriptionMaxLen,
      }),
      enrolledSpeakerName || manualSpeakerName,
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
    recordTiming("speaker_identification_prep");

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
    recordTiming("assembly_transcription");

    const transcriptText = transcript.text || "";
    const utterances = Array.isArray(transcript.utterances) ? transcript.utterances : [];
    const dominantAssemblySpeakerLabel = voiceRecognition.formatDominantAssemblySpeakerLabel(utterances);
    const speakerIdentificationMapping = voiceRecognition.pickSpeakerIdentificationMapping(transcript);

    const speakerIdentificationCandidates =
      assemblySpeakers.length > 0 ? assemblySpeakers.map((s) => s.name) : knownSpeakerValues;
    recordTiming("response_preparation");
    console.log("Transcription timing", {
      totalMs: Date.now() - startedAtMs,
      steps: timings,
      embeddingTimeoutMs: speakerEmbeddingTimeoutMs,
      speakerEmbeddingEnabled: !!speakerEmbeddingServiceUrl,
      speakerEmbeddingAvailable: !!speakerEmbedding,
      recognitionEngine: detectedSpeaker?.recognitionEngine || "fingerprint",
    });

    const recognitionAttempted = !manualSpeakerName;
    const speakerRecognitionEngineForDiagnostics = manualSpeakerName
      ? null
      : detectedSpeaker
        ? detectedSpeaker.recognitionEngine
        : "none";
    const speakerMatchUsedEmbedding =
      recognitionAttempted && detectedSpeaker?.recognitionEngine === "embedding";
    const speakerMatchUsedFingerprint =
      recognitionAttempted && detectedSpeaker?.recognitionEngine === "fingerprint";
    const transcriptionDiagnostics = {
      embeddingServiceConfigured: !!speakerEmbeddingServiceUrl,
      embeddingFetchAttempted: !!speakerEmbeddingServiceUrl,
      embeddingFetchSucceeded: !!speakerEmbedding,
      embeddingDimensions: Array.isArray(speakerEmbedding) ? speakerEmbedding.length : null,
      embeddingError: embeddingFetchError
        ? String(embeddingFetchError).replace(/\s+/g, " ").trim().slice(0, 320)
        : null,
      embeddingTimeoutMs: speakerEmbeddingTimeoutMs,
      speakerRecognitionEngine: speakerRecognitionEngineForDiagnostics,
      speakerMatchUsedEmbedding,
      speakerMatchUsedFingerprint,
      recognitionAttempted,
      profileEnrollmentAttempted: !!manualSpeakerName,
      fingerprintVectorSavedToProfile: !!manualSpeakerName,
      embeddingVectorSavedToProfile: !!(manualSpeakerName && speakerEmbedding),
      timingTotalMs: Date.now() - startedAtMs,
      timingSteps: timings.map((item) => ({ step: item.step, ms: item.ms })),
    };

    let aiInsights = null;
    if (openaiApiKey && transcriptText.trim()) {
      try {
        aiInsights = await buildAiInsights(transcriptText);
      } catch (insightError) {
        console.warn("OpenAI transcript insights failed", {
          message: insightError instanceof Error ? insightError.message : String(insightError),
        });
      }
    }

    return res.json({
      text: transcriptText,
      enrolledSpeakerName,
      detectedSpeakerName: detectedSpeaker?.name || null,
      speakerConfidence: detectedSpeaker?.score ?? null,
      detectedSpeakerSampleId: detectedSpeaker?.sampleId || null,
      detectedSpeakerSampleSource: detectedSpeaker?.sampleSource || null,
      detectedSpeakerSampleCreatedAtIso: detectedSpeaker?.sampleCreatedAtIso || null,
      speakerRecognitionEngine: manualSpeakerName ? null : detectedSpeaker?.recognitionEngine || "none",
      speakerEmbeddingEnabled: !!speakerEmbeddingServiceUrl,
      speakerEmbeddingAvailable: !!speakerEmbedding,
      assemblySpeakerLabel: dominantAssemblySpeakerLabel,
      speakerIdentificationCandidates,
      speakerIdentificationMapping,
      transcriptionDiagnostics,
      ...(aiInsights ? { ai: aiInsights } : {}),
      utterances: utterances.map((item) => ({
        speaker: item.speaker || null,
        text: item.text || "",
        start: item.start || 0,
        end: item.end || 0,
      })),
    });
  } catch (error) {
    console.warn("Transcription failed timing", {
      totalMs: Date.now() - startedAtMs,
      steps: timings,
    });
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

app.post("/ai/person-summary", async (req, res) => {
  try {
    if (!openaiApiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not configured." });
    }
    const speakerName = typeof req.body?.speakerName === "string" ? req.body.speakerName : "";
    const text = typeof req.body?.text === "string" ? req.body.text : "";
    if (!text.trim()) {
      return res.status(400).json({ error: "Missing text bundle for summary." });
    }
    const narrative = await buildPersonAiSummary(speakerName, text);
    if (!narrative) {
      return res.status(500).json({ error: "Could not generate person summary." });
    }
    return res.json({ narrative });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Person summary failed.";
    return res.status(500).json({ error: message });
  }
});

app.post("/ai/speaker-today-plan", async (req, res) => {
  try {
    if (!openaiApiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not configured." });
    }
    const speakerName = typeof req.body?.speakerName === "string" ? req.body.speakerName : "";
    const dateLabel = typeof req.body?.dateLabel === "string" ? req.body.dateLabel : "";
    const text = typeof req.body?.text === "string" ? req.body.text : "";
    if (!text.trim()) {
      return res.status(400).json({ error: "Missing text bundle for today plan." });
    }
    const narrative = await buildSpeakerTodayPlan(speakerName, dateLabel, text);
    if (!narrative) {
      return res.status(500).json({ error: "Could not generate today's plan." });
    }
    return res.json({ narrative });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Today plan failed.";
    return res.status(500).json({ error: message });
  }
});

app.post("/ai/extract-destination", async (req, res) => {
  try {
    if (!openaiApiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not configured." });
    }
    const text = typeof req.body?.text === "string" ? req.body.text : "";
    if (!text.trim()) {
      return res.status(400).json({ error: "Missing text to extract destination from." });
    }
    const destination = await buildExtractDestination(text);
    return res.json({ destination });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Destination extraction failed.";
    return res.status(500).json({ error: message });
  }
});

app.post("/traffic/duration", async (req, res) => {
  try {
    const originLat = req.body?.originLat;
    const originLng = req.body?.originLng;
    const destination = typeof req.body?.destination === "string" ? req.body.destination : "";
    const result = await fetchDrivingDurationWithTraffic(originLat, originLng, destination);
    return res.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Traffic lookup failed.";
    const status = message.includes("not configured") ? 501 : 400;
    return res.status(status).json({ error: message });
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
