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

const cleanStringList = (value, maxItems, itemMaxLen) => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .slice(0, maxItems)
    .map((item) => cleanString(item, itemMaxLen))
    .filter((item) => item.length > 0);
};

/** Normalized AI payload persisted on each history entry (from transcribe or POST /history). */
const cleanAiInsights = (value) => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const summary = cleanString(value.summary, 4000);
  const actionItems = cleanStringList(value.actionItems, 24, 400);
  const topics = cleanStringList(value.topics, 16, 64);
  const followUpQuestions = cleanStringList(value.followUpQuestions, 5, 280);
  if (!summary && actionItems.length === 0 && topics.length === 0 && followUpQuestions.length === 0) {
    return null;
  }
  return {
    summary,
    actionItems,
    topics,
    ...(followUpQuestions.length > 0 ? { followUpQuestions } : {}),
  };
};

const sanitizeHistoryEntry = (entry) => {
  const now = Date.now();
  const createdAtMs = cleanNumber(entry?.createdAtMs) || now;
  const clientId = cleanString(entry?.id, 80) || `${createdAtMs}`;
  const speakerName = cleanString(entry?.speakerName, 120) || "Unknown speaker";
  const text = cleanString(entry?.text, 20000);
  const createdAt = cleanString(entry?.createdAt, 80) || new Date(createdAtMs).toISOString();

  const aiPayload = cleanAiInsights(entry?.ai);
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
    answeredVoiceFollowUp: cleanNullableString(entry?.answeredVoiceFollowUp, 500),
    firstPassRecognitionEngine: cleanRecognitionEngine(entry?.firstPassRecognitionEngine),
    transcriptionDiagnosticsInitial: cleanHistoryDiagnosticsPayload({
      ...entry?.transcriptionDiagnosticsInitial,
      phase: "transcribe",
    }),
    transcriptionDiagnosticsEnrollment: cleanHistoryDiagnosticsPayload({
      ...entry?.transcriptionDiagnosticsEnrollment,
      phase: "enrollment",
    }),
    ...(aiPayload ? { ai: aiPayload } : {}),
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

/** Remove AI payloads and voice follow-up tags from every stored history row (file or Firestore). */
async function clearAiFromFile() {
  let entries = [];
  try {
    const raw = await fs.readFile(localHistoryPath(), "utf8");
    const parsed = JSON.parse(raw);
    entries = Array.isArray(parsed.entries) ? parsed.entries : [];
  } catch (error) {
    if (error && typeof error === "object" && error.code === "ENOENT") {
      return { cleared: 0, updated: 0, mode: "file" };
    }
    throw error;
  }
  let updated = 0;
  const stripped = entries.map((entry) => {
    if (!entry || typeof entry !== "object") {
      return entry;
    }
    const hadAi = Object.prototype.hasOwnProperty.call(entry, "ai") && entry.ai != null;
    const hadFollowUp =
      Object.prototype.hasOwnProperty.call(entry, "answeredVoiceFollowUp") &&
      entry.answeredVoiceFollowUp != null &&
      String(entry.answeredVoiceFollowUp).trim() !== "";
    if (hadAi || hadFollowUp) {
      updated += 1;
    }
    const next = { ...entry };
    delete next.ai;
    delete next.answeredVoiceFollowUp;
    return next;
  });
  await fs.writeFile(localHistoryPath(), JSON.stringify({ entries: stripped }, null, 2), "utf8");
  return { cleared: stripped.length, updated, mode: "file" };
}

async function clearAiFromFirestore() {
  const { FieldValue } = require("firebase-admin/firestore");
  const db = getFirestore();
  const snap = await historyCollection().get();
  let batch = db.batch();
  let ops = 0;
  let updated = 0;
  for (const doc of snap.docs) {
    const data = doc.data() || {};
    const hadAi = data.ai != null;
    const hadFollowUp =
      data.answeredVoiceFollowUp != null && String(data.answeredVoiceFollowUp).trim() !== "";
    if (hadAi || hadFollowUp) {
      updated += 1;
    }
    batch.update(doc.ref, {
      ai: FieldValue.delete(),
      answeredVoiceFollowUp: FieldValue.delete(),
    });
    ops += 1;
    if (ops >= 500) {
      await batch.commit();
      batch = db.batch();
      ops = 0;
    }
  }
  if (ops > 0) {
    await batch.commit();
  }
  return { cleared: snap.size, updated, mode: "firestore" };
}

const useFirestore = () => process.env.SPEAKER_STORE_BACKEND === "firestore";

/**
 * @typedef {{ kind: "firestore", ref: import("firebase-admin/firestore").DocumentReference, docId: string, entry: Record<string, unknown> }} FirestoreHistoryPointer
 * @typedef {{ kind: "file", entries: unknown[], index: number, entry: Record<string, unknown> }} FileHistoryPointer
 */

/**
 * Locate a history row by Firestore document id or clientId (or file clientId).
 * @param {string} externalId
 * @returns {Promise<(FirestoreHistoryPointer|FileHistoryPointer)|null>}
 */
async function resolveHistoryEntryPointer(externalId) {
  const id = typeof externalId === "string" ? externalId.trim() : "";
  if (!id) {
    return null;
  }
  if (useFirestore()) {
    const col = historyCollection();
    const byDoc = await col.doc(id).get();
    if (byDoc.exists) {
      return { kind: "firestore", ref: byDoc.ref, docId: byDoc.id, entry: byDoc.data() || {} };
    }
    const qSnap = await col.where("clientId", "==", id).limit(1).get();
    if (qSnap.empty) {
      return null;
    }
    const d = qSnap.docs[0];
    return { kind: "firestore", ref: d.ref, docId: d.id, entry: d.data() || {} };
  }
  let entries = [];
  try {
    const raw = await fs.readFile(localHistoryPath(), "utf8");
    const parsed = JSON.parse(raw);
    entries = Array.isArray(parsed.entries) ? parsed.entries : [];
  } catch (error) {
    if (error && typeof error === "object" && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
  const idx = entries.findIndex((e) => e && typeof e === "object" && (e.clientId === id || e.id === id));
  if (idx < 0) {
    return null;
  }
  return { kind: "file", entries, index: idx, entry: entries[idx] };
}

/**
 * @param {FirestoreHistoryPointer|FileHistoryPointer} pointer
 * @param {object} aiRaw
 */
async function applyHistoryEntryAi(pointer, aiRaw) {
  const aiPayload = cleanAiInsights(aiRaw);
  if (pointer.kind === "firestore") {
    const { FieldValue } = require("firebase-admin/firestore");
    if (aiPayload) {
      await pointer.ref.update({ ai: aiPayload });
    } else {
      await pointer.ref.update({ ai: FieldValue.delete() });
    }
    return {
      id: pointer.docId,
      clientId: typeof pointer.entry?.clientId === "string" ? pointer.entry.clientId : null,
      ai: aiPayload,
    };
  }
  const nextEntry = { ...pointer.entry };
  if (aiPayload) {
    nextEntry.ai = aiPayload;
  } else {
    delete nextEntry.ai;
  }
  pointer.entries[pointer.index] = nextEntry;
  await fs.writeFile(localHistoryPath(), JSON.stringify({ entries: pointer.entries }, null, 2), "utf8");
  const cid = typeof nextEntry.clientId === "string" ? nextEntry.clientId : null;
  return { id: cid || String(pointer.index), clientId: cid, ai: aiPayload };
}

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
  clearAiFromAllHistory: async () => (useFirestore() ? clearAiFromFirestore() : clearAiFromFile()),
  resolveHistoryEntryPointer,
  applyHistoryEntryAi,
};
