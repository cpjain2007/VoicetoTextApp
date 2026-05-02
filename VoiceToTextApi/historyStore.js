const fs = require("fs/promises");
const path = require("path");

const localHistoryPath = () => path.join(__dirname, "transcript-history.json");

const getFirestore = () => {
  const { getFirestore } = require("firebase-admin/firestore");
  return getFirestore();
};

const historyCollection = () =>
  getFirestore().collection((process.env.HISTORY_FIRESTORE_COLLECTION || "transcriptHistory").trim());

const cleanString = (value, maxLen) => {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim().slice(0, maxLen);
};

const cleanNullableString = (value, maxLen) => {
  const cleaned = cleanString(value, maxLen);
  return cleaned || null;
};

const cleanBoolean = (value) => value === true;

const cleanNumber = (value) => (typeof value === "number" && Number.isFinite(value) ? value : null);

const cleanTimingSteps = (value) => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.slice(0, 25).map((item) => ({
    step: cleanString(item?.step, 48),
    ms: cleanNumber(item?.ms) ?? 0,
  }));
};

const cleanRecognitionEngine = (value) => {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  return value === "embedding" || value === "fingerprint" || value === "none" ? value : null;
};

const cleanTranscriptionDiagnostics = (value) => {
  if (!value || typeof value !== "object") {
    return null;
  }
  return {
    embeddingServiceConfigured: value.embeddingServiceConfigured === true,
    embeddingFetchAttempted: value.embeddingFetchAttempted === true,
    embeddingFetchSucceeded: value.embeddingFetchSucceeded === true,
    embeddingDimensions: cleanNumber(value.embeddingDimensions),
    embeddingError: cleanNullableString(value.embeddingError, 320),
    embeddingTimeoutMs: cleanNumber(value.embeddingTimeoutMs),
    speakerRecognitionEngine: cleanRecognitionEngine(value.speakerRecognitionEngine),
    speakerMatchUsedEmbedding:
      value.speakerMatchUsedEmbedding === true ? true : value.speakerMatchUsedEmbedding === false ? false : null,
    speakerMatchUsedFingerprint:
      value.speakerMatchUsedFingerprint === true
        ? true
        : value.speakerMatchUsedFingerprint === false
          ? false
          : null,
    recognitionAttempted: value.recognitionAttempted === true,
    profileEnrollmentAttempted: value.profileEnrollmentAttempted === true,
    fingerprintVectorSavedToProfile: value.fingerprintVectorSavedToProfile === true,
    embeddingVectorSavedToProfile: value.embeddingVectorSavedToProfile === true,
    timingTotalMs: cleanNumber(value.timingTotalMs),
    timingSteps: cleanTimingSteps(value.timingSteps),
  };
};

const cleanHistoryDiagnosticsPayload = (value) => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const phase = value.phase === "enrollment" ? "enrollment" : "transcribe";
  const base = cleanTranscriptionDiagnostics(value);
  if (!base) {
    return null;
  }
  return { phase, ...base };
};

const cleanAttributionSource = (value) => {
  const allowed = new Set([
    "speaker_name_input",
    "voice_match",
    "unknown_speaker_prompt",
    "speaker_conflict_prompt",
    "unknown",
  ]);
  return allowed.has(value) ? value : "unknown";
};

const sanitizeHistoryEntry = (entry) => {
  const now = Date.now();
  const createdAtMs = cleanNumber(entry?.createdAtMs) || now;
  const clientId = cleanString(entry?.id, 80) || `${createdAtMs}`;
  const speakerName = cleanString(entry?.speakerName, 120) || "Unknown speaker";
  const text = cleanString(entry?.text, 20000);
  const createdAt = cleanString(entry?.createdAt, 80) || new Date(createdAtMs).toISOString();

  return {
    clientId,
    speakerName,
    text,
    createdAt,
    createdAtMs,
    createdAtIso: new Date(createdAtMs).toISOString(),
    speakerAttributionSource: cleanAttributionSource(entry?.speakerAttributionSource),
    speakerNameInput: cleanNullableString(entry?.speakerNameInput, 120),
    promptedSpeakerName: cleanNullableString(entry?.promptedSpeakerName, 120),
    detectedSpeakerName: cleanNullableString(entry?.detectedSpeakerName, 120),
    speakerConfidence: cleanNumber(entry?.speakerConfidence),
    matchedEnrollmentSampleId: cleanNullableString(entry?.matchedEnrollmentSampleId, 120),
    matchedEnrollmentSampleSource: cleanNullableString(entry?.matchedEnrollmentSampleSource, 120),
    matchedEnrollmentSampleCreatedAtIso: cleanNullableString(entry?.matchedEnrollmentSampleCreatedAtIso, 120),
    assemblySpeakerLabel: cleanNullableString(entry?.assemblySpeakerLabel, 120),
    wasSpeakerNameInputProvided: cleanBoolean(entry?.wasSpeakerNameInputProvided),
    wasUnknownSpeakerPromptShown: cleanBoolean(entry?.wasUnknownSpeakerPromptShown),
    wasVoiceMatchUsed: cleanBoolean(entry?.wasVoiceMatchUsed),
    wasConflictPromptShown: cleanBoolean(entry?.wasConflictPromptShown),
    wasVoiceProfileEnrolled: cleanBoolean(entry?.wasVoiceProfileEnrolled),
    firstPassRecognitionEngine: cleanRecognitionEngine(entry?.firstPassRecognitionEngine),
    transcriptionDiagnosticsInitial: cleanHistoryDiagnosticsPayload({
      ...entry?.transcriptionDiagnosticsInitial,
      phase: "transcribe",
    }),
    transcriptionDiagnosticsEnrollment: cleanHistoryDiagnosticsPayload({
      ...entry?.transcriptionDiagnosticsEnrollment,
      phase: "enrollment",
    }),
  };
};

async function appendToFile(entry) {
  let entries = [];
  try {
    const raw = await fs.readFile(localHistoryPath(), "utf8");
    const parsed = JSON.parse(raw);
    entries = Array.isArray(parsed.entries) ? parsed.entries : [];
  } catch (error) {
    if (!error || typeof error !== "object" || error.code !== "ENOENT") {
      throw error;
    }
  }
  entries.unshift(entry);
  await fs.writeFile(localHistoryPath(), JSON.stringify({ entries }, null, 2), "utf8");
  return { id: entry.clientId, ...entry };
}

async function appendToFirestore(entry) {
  const docRef = await historyCollection().add({
    ...entry,
    savedAt: new Date().toISOString(),
  });
  return { id: docRef.id, ...entry };
}

async function listFromFile(limit) {
  try {
    const raw = await fs.readFile(localHistoryPath(), "utf8");
    const parsed = JSON.parse(raw);
    const entries = Array.isArray(parsed.entries) ? parsed.entries : [];
    return entries.slice(0, limit);
  } catch (error) {
    if (error && typeof error === "object" && error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

async function listFromFirestore(limit) {
  const snap = await historyCollection().orderBy("createdAtMs", "desc").limit(limit).get();
  return snap.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
}

const useFirestore = () => process.env.SPEAKER_STORE_BACKEND === "firestore";

module.exports = {
  sanitizeHistoryEntry,
  appendHistoryEntry: async (entry) => {
    const cleanEntry = sanitizeHistoryEntry(entry);
    return useFirestore() ? appendToFirestore(cleanEntry) : appendToFile(cleanEntry);
  },
  listHistoryEntries: async (rawLimit) => {
    const limit = Math.min(Math.max(Number(rawLimit) || 50, 1), 200);
    return useFirestore() ? listFromFirestore(limit) : listFromFile(limit);
  },
};
