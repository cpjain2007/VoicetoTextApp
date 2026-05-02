import { StatusBar } from "expo-status-bar";
import Ionicons from "@expo/vector-icons/Ionicons";
import { useEffect, useMemo, useRef, useState } from "react";
import { Audio } from "expo-av";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { EncodingType, readAsStringAsync } from "expo-file-system/legacy";
import {
  ActivityIndicator,
  Alert,
  Modal,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

type TranscriptionDiagnosticsPayload = {
  embeddingServiceConfigured: boolean;
  embeddingFetchAttempted: boolean;
  embeddingFetchSucceeded: boolean;
  embeddingDimensions: number | null;
  embeddingError: string | null;
  embeddingTimeoutMs: number | null;
  speakerRecognitionEngine: "embedding" | "fingerprint" | "none" | null;
  speakerMatchUsedEmbedding: boolean | null;
  speakerMatchUsedFingerprint: boolean | null;
  recognitionAttempted: boolean;
  profileEnrollmentAttempted: boolean;
  fingerprintVectorSavedToProfile: boolean;
  embeddingVectorSavedToProfile: boolean;
  timingTotalMs: number | null;
  timingSteps: Array<{ step: string; ms: number }>;
};

type TranscriptionDiagnosticsPhase = TranscriptionDiagnosticsPayload & {
  phase: "transcribe" | "enrollment";
};

type TranscriptLogItem = {
  id: string;
  speakerName: string;
  text: string;
  createdAt: string;
  createdAtMs: number;
  cloudHistoryId?: string;
  speakerAttributionSource?: SpeakerAttributionSource;
  speakerNameInput?: string | null;
  promptedSpeakerName?: string | null;
  detectedSpeakerName?: string | null;
  speakerConfidence?: number | null;
  matchedEnrollmentSampleId?: string | null;
  matchedEnrollmentSampleSource?: string | null;
  matchedEnrollmentSampleCreatedAtIso?: string | null;
  assemblySpeakerLabel?: string | null;
  wasSpeakerNameInputProvided?: boolean;
  wasUnknownSpeakerPromptShown?: boolean;
  wasVoiceMatchUsed?: boolean;
  wasConflictPromptShown?: boolean;
  wasVoiceProfileEnrolled?: boolean;
  /**
   * First transcribe API response `speakerRecognitionEngine`: which matcher picked the enrolled speaker (if any).
   * Persisted for history clarity; also derivable from transcriptionDiagnosticsInitial when missing.
   */
  firstPassRecognitionEngine?: "embedding" | "fingerprint" | "none";
  /** Server-side diagnostics for the first /transcribe-base64 call (no manual name). */
  transcriptionDiagnosticsInitial?: TranscriptionDiagnosticsPhase | null;
  /** Present when unknown-speaker flow re-uploaded audio with a name (enrollment). */
  transcriptionDiagnosticsEnrollment?: TranscriptionDiagnosticsPhase | null;
};

type TranscriptionResult = {
  text: string;
  enrolledSpeakerName: string | null;
  detectedSpeakerName: string | null;
  speakerConfidence: number | null;
  detectedSpeakerSampleId: string | null;
  detectedSpeakerSampleSource: string | null;
  detectedSpeakerSampleCreatedAtIso: string | null;
  assemblySpeakerLabel: string | null;
  speakerRecognitionEngine?: string | null;
  speakerEmbeddingEnabled?: boolean;
  speakerEmbeddingAvailable?: boolean;
  transcriptionDiagnostics?: TranscriptionDiagnosticsPayload | null;
};

type NumericVectorPayload = {
  values: number[];
  dimensions?: number | null;
  truncated?: boolean;
};

type SpeakerProfile = {
  name: string;
  samples: number;
  /** Optional hint for AssemblyAI Speaker Identification (`speakerDescription` in API store). */
  description?: string;
  enrollmentSamples?: EnrollmentSample[];
  /** Rollup 12-D fingerprint averaged across enrollments (API). */
  profileVoiceFingerprint?: number[] | null;
  profileVoiceFingerprintDimensions?: number | null;
  profileVoiceFingerprintTruncated?: boolean;
  profileVoiceFingerprintsRecent?: NumericVectorPayload[];
  profileSpeakerEmbedding?: NumericVectorPayload | null;
  profileSpeakerEmbeddingsRecent?: NumericVectorPayload[];
};

type EnrollmentSample = {
  sampleId: string;
  source?: string | null;
  createdAtMs?: number | null;
  createdAtIso?: string | null;
  historyClientId?: string | null;
  hasFingerprint?: boolean;
  hasEmbedding?: boolean;
  voiceFingerprint?: number[] | null;
  voiceFingerprintDimensions?: number | null;
  voiceFingerprintTruncated?: boolean;
  embeddingVector?: number[] | null;
  embeddingDimensions?: number | null;
  embeddingTruncated?: boolean;
};

type SpeakerCorrectionSuggestion = {
  shouldSuggest: boolean;
  suggestedSpeakerName: string;
  reason: string;
};

type SpeakerAttributionSource =
  | "speaker_name_input"
  | "voice_match"
  | "unknown_speaker_prompt"
  | "speaker_conflict_prompt"
  | "unknown";

type AppTab = "record" | "history" | "speakers";

const HISTORY_STORAGE_KEY = "voicetotext.history.v1";
const UNKNOWN_SPEAKER_LABEL = "Unknown speaker";
const UNKNOWN_SPEAKER_NAME_MAX_CHARS = 80;

/** Matches default `ASSEMBLYAI_SPEAKER_DESCRIPTION_MAX` on the API. */
const SPEAKER_DESCRIPTION_MAX_CHARS = 220;

const normalizeSpeakerKey = (value: string) => value.trim().toLowerCase();

const isUnknownSpeakerLabel = (value: string) =>
  normalizeSpeakerKey(value) === normalizeSpeakerKey(UNKNOWN_SPEAKER_LABEL);

const formatAttributionSource = (source?: SpeakerAttributionSource) => {
  switch (source) {
    case "speaker_name_input":
      return "Typed speaker name";
    case "voice_match":
      return "Voice match";
    case "unknown_speaker_prompt":
      return "Prompt after unknown speaker";
    case "speaker_conflict_prompt":
      return "Conflict prompt";
    case "unknown":
    default:
      return "Unknown source";
  }
};

const formatSampleSource = (source?: string | null) => {
  if (source === "speaker_name_input") {
    return "Typed name";
  }
  if (source === "unknown_speaker_prompt") {
    return "Unknown prompt";
  }
  if (source === "speaker_conflict_prompt") {
    return "Conflict prompt";
  }
  return "Unknown source";
};

const formatSampleTime = (sample: EnrollmentSample) => {
  const timestamp = typeof sample.createdAtMs === "number" ? sample.createdAtMs : Date.parse(sample.createdAtIso || "");
  if (!Number.isFinite(timestamp)) {
    return "Unknown time";
  }
  return new Date(timestamp).toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const formatVectorCsv = (values?: number[] | null) =>
  Array.isArray(values) && values.length > 0 ? values.map((n) => String(n)).join(", ") : "";

const VectorBlock = ({
  title,
  subtitle,
  csv,
  emptyLabel,
  scrollMaxHeight = 132,
}: {
  title: string;
  subtitle?: string;
  csv: string;
  emptyLabel: string;
  scrollMaxHeight?: number;
}) => (
  <View style={styles.vectorBlock}>
    <Text style={styles.vectorBlockTitle}>{title}</Text>
    {subtitle ? <Text style={styles.vectorBlockSubtitle}>{subtitle}</Text> : null}
    {csv ? (
      <ScrollView
        style={[styles.vectorScroll, { maxHeight: scrollMaxHeight }]}
        nestedScrollEnabled
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.vectorMono} selectable>
          {csv}
        </Text>
      </ScrollView>
    ) : (
      <Text style={styles.vectorBlockEmpty}>{emptyLabel}</Text>
    )}
  </View>
);

const formatHistoryDateGroup = (timestampMs: number) => {
  if (!Number.isFinite(timestampMs)) {
    return "Unknown date";
  }
  return new Date(timestampMs).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "2-digit",
    year: "numeric",
  });
};

const readRecognitionEngine = (
  value: unknown,
): TranscriptionDiagnosticsPayload["speakerRecognitionEngine"] => {
  if (value === "embedding" || value === "fingerprint" || value === "none") {
    return value;
  }
  if (value === null || value === undefined) {
    return null;
  }
  return null;
};

const parseTranscriptionDiagnostics = (raw: unknown): TranscriptionDiagnosticsPayload | null => {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const o = raw as Record<string, unknown>;
  const stepsRaw = o.timingSteps;
  const timingSteps = Array.isArray(stepsRaw)
    ? stepsRaw.slice(0, 30).map((s) => {
        if (!s || typeof s !== "object") {
          return { step: "", ms: 0 };
        }
        const st = s as Record<string, unknown>;
        return {
          step: typeof st.step === "string" ? st.step : "",
          ms: typeof st.ms === "number" && Number.isFinite(st.ms) ? st.ms : 0,
        };
      })
    : [];
  return {
    embeddingServiceConfigured: o.embeddingServiceConfigured === true,
    embeddingFetchAttempted: o.embeddingFetchAttempted === true,
    embeddingFetchSucceeded: o.embeddingFetchSucceeded === true,
    embeddingDimensions:
      typeof o.embeddingDimensions === "number" && Number.isFinite(o.embeddingDimensions)
        ? o.embeddingDimensions
        : null,
    embeddingError:
      typeof o.embeddingError === "string" && o.embeddingError.trim()
        ? o.embeddingError.trim().slice(0, 400)
        : null,
    embeddingTimeoutMs:
      typeof o.embeddingTimeoutMs === "number" && Number.isFinite(o.embeddingTimeoutMs)
        ? o.embeddingTimeoutMs
        : null,
    speakerRecognitionEngine: readRecognitionEngine(o.speakerRecognitionEngine),
    speakerMatchUsedEmbedding:
      o.speakerMatchUsedEmbedding === true ? true : o.speakerMatchUsedEmbedding === false ? false : null,
    speakerMatchUsedFingerprint:
      o.speakerMatchUsedFingerprint === true ? true : o.speakerMatchUsedFingerprint === false ? false : null,
    recognitionAttempted: o.recognitionAttempted === true,
    profileEnrollmentAttempted: o.profileEnrollmentAttempted === true,
    fingerprintVectorSavedToProfile: o.fingerprintVectorSavedToProfile === true,
    embeddingVectorSavedToProfile: o.embeddingVectorSavedToProfile === true,
    timingTotalMs:
      typeof o.timingTotalMs === "number" && Number.isFinite(o.timingTotalMs) ? o.timingTotalMs : null,
    timingSteps,
  };
};

const readDiagnosticsPhase = (raw: unknown): TranscriptionDiagnosticsPhase | undefined => {
  const base = parseTranscriptionDiagnostics(raw);
  if (!base) {
    return undefined;
  }
  const phase =
    raw && typeof raw === "object" && (raw as Record<string, unknown>).phase === "enrollment"
      ? "enrollment"
      : "transcribe";
  return { ...base, phase };
};

const formatHistoryDiagnosticsText = (item: TranscriptLogItem) => {
  const formatOne = (title: string, d: TranscriptionDiagnosticsPhase) => {
    const emb = !d.embeddingFetchAttempted
      ? "not attempted (service off)"
      : d.embeddingFetchSucceeded
        ? `ok (${d.embeddingDimensions ?? "?"} dims)`
        : d.embeddingError
          ? `failed: ${d.embeddingError}`
          : "failed";
    const engine =
      d.profileEnrollmentAttempted && !d.recognitionAttempted
        ? "skipped (enrollment only)"
        : d.speakerRecognitionEngine ?? "—";
    const steps =
      d.timingSteps.length > 0 ? d.timingSteps.map((s) => `${s.step} ${s.ms}ms`).join(", ") : "—";
    return [
      title,
      `  Embedding service configured: ${d.embeddingServiceConfigured ? "yes" : "no"}`,
      `  Embedding client timeout: ${d.embeddingTimeoutMs ?? "—"} ms`,
      `  Embedding fetch: ${emb}`,
      `  Recognition attempted: ${d.recognitionAttempted ? "yes" : "no"}`,
      `  Match engine: ${engine}`,
      `  Used embedding for match: ${d.speakerMatchUsedEmbedding === true ? "yes" : d.speakerMatchUsedEmbedding === false ? "no" : "—"}`,
      `  Used fingerprint for match: ${d.speakerMatchUsedFingerprint === true ? "yes" : d.speakerMatchUsedFingerprint === false ? "no" : "—"}`,
      `  Saved 12-D fingerprint to profile: ${d.fingerprintVectorSavedToProfile ? "yes" : "no"}`,
      `  Saved embedding vector to profile: ${d.embeddingVectorSavedToProfile ? "yes" : "no"}`,
      `  Server total time: ${d.timingTotalMs ?? "—"} ms`,
      `  Steps: ${steps}`,
    ].join("\n");
  };
  const blocks: string[] = [];
  if (item.transcriptionDiagnosticsInitial) {
    blocks.push(formatOne("Pass 1 — listen / detect", item.transcriptionDiagnosticsInitial));
  }
  if (item.transcriptionDiagnosticsEnrollment) {
    blocks.push(formatOne("Pass 2 — name + enroll", item.transcriptionDiagnosticsEnrollment));
  }
  return blocks.join("\n\n");
};

/** Maps `recordTiming(step)` keys from VoiceToTextApi `executeTranscription` to code-path descriptions. */
const SERVER_TIMING_STEP_PATH: Record<string, string> = {
  audio_fingerprint:
    "app.js → convertAudioToPcm + voiceRecognition.buildVoiceSignature (12-D fingerprint from PCM)",
  speaker_embedding:
    "app.js → fetchSpeakerEmbedding (neural vector) when SPEAKER_EMBEDDING_SERVICE_URL is set; errors fall back to fingerprint matching",
  speaker_matching:
    "app.js → detectSpeakerNameByEmbedding if embedding present, else detectSpeakerName (fingerprint); skipped when a manual speaker name is supplied (enroll-only request)",
  speaker_identification_prep:
    "app.js → readSpeakerProfilesMerged + build AssemblyAI speaker hints (speech_understanding / known names)",
  assembly_transcription:
    "app.js → transcribeWithRetry (AssemblyAI with speaker_labels + language detection or forced language)",
  response_preparation:
    "app.js → dominant speaker label + utterances + transcriptionDiagnostics payload",
};

const explainServerRecognitionBranch = (d: TranscriptionDiagnosticsPhase): string[] => {
  const lines: string[] = [];
  if (d.profileEnrollmentAttempted && !d.recognitionAttempted) {
    lines.push(
      "Outcome: enrollment-only — manual speaker name was sent, so `executeTranscription` updated the profile and did not run speaker matching.",
    );
    if (d.fingerprintVectorSavedToProfile) {
      lines.push("  └ Saved new 12-D fingerprint sample on the speaker profile.");
    }
    if (d.embeddingVectorSavedToProfile) {
      lines.push("  └ Saved neural embedding on the profile (embedding fetch succeeded earlier).");
    } else if (d.embeddingFetchAttempted && !d.embeddingFetchSucceeded) {
      lines.push("  └ Embedding not stored (embedding fetch failed); fingerprint enrollment still applied.");
    }
    return lines;
  }
  if (!d.recognitionAttempted) {
    lines.push("Outcome: recognition not attempted (unexpected state — check diagnostics).");
    return lines;
  }
  if (d.speakerRecognitionEngine === "embedding") {
    lines.push(
      "Outcome: match via embedding — `detectSpeakerNameByEmbedding` scored enrolled profiles against this clip’s vector.",
    );
  } else if (d.speakerRecognitionEngine === "fingerprint") {
    lines.push(
      "Outcome: match via fingerprint — embedding unset or unused; `detectSpeakerName` compared 12-D signatures.",
    );
  } else if (d.speakerRecognitionEngine === "none" || d.speakerRecognitionEngine === null) {
    lines.push(
      "Outcome: no match — neither embedding nor fingerprint produced a confident enrolled speaker; UI may prompt for a name.",
    );
  } else {
    lines.push(`Outcome: engine=${d.speakerRecognitionEngine ?? "—"}.`);
  }
  if (d.speakerMatchUsedEmbedding === true) {
    lines.push("  └ Confirmed: embedding path contributed to the match decision.");
  }
  if (d.speakerMatchUsedFingerprint === true) {
    lines.push("  └ Confirmed: fingerprint path contributed to the match decision.");
  }
  return lines;
};

const formatServerPassLogicalPath = (title: string, d: TranscriptionDiagnosticsPhase): string[] => {
  const out: string[] = [`${title} (server: executeTranscription in app.js)`];
  if (d.timingSteps.length > 0) {
    d.timingSteps.forEach((s, i) => {
      const path = SERVER_TIMING_STEP_PATH[s.step] || "(see API timing label)";
      out.push(`  ${i + 1}. ${s.step} (+${s.ms} ms) — ${path}`);
    });
  } else {
    out.push("  (No timing steps in payload — older server build or truncated response.)");
  }
  explainServerRecognitionBranch(d).forEach((line) => out.push(`  ${line}`));
  return out;
};

/** Resolve which matcher decided pass-1 speaker (stored field wins, else diagnostics). */
const resolvePass1RecognitionEngine = (
  item: TranscriptLogItem,
): "embedding" | "fingerprint" | "none" | undefined => {
  const direct = item.firstPassRecognitionEngine;
  if (direct === "embedding" || direct === "fingerprint" || direct === "none") {
    return direct;
  }
  const e = item.transcriptionDiagnosticsInitial?.speakerRecognitionEngine;
  if (e === "embedding" || e === "fingerprint" || e === "none") {
    return e;
  }
  return undefined;
};

/** Precise embedding vs fingerprint breakdown for the History tab. */
const formatHistoryVoiceMatchDetail = (item: TranscriptLogItem): string => {
  const d1 = item.transcriptionDiagnosticsInitial;
  const d2 = item.transcriptionDiagnosticsEnrollment;
  const engine = resolvePass1RecognitionEngine(item);

  const lines: string[] = [];

  lines.push("Pass 1 — listen (request had no manual speaker name)");
  if (engine === "embedding") {
    lines.push(
      "  • Enrolled speaker decided by: neural embedding match (clip vector vs stored ECAPA-style embeddings).",
    );
  } else if (engine === "fingerprint") {
    lines.push(
      "  • Enrolled speaker decided by: 12-D audio fingerprint match (PCM stats vs stored fingerprints).",
    );
  } else if (engine === "none") {
    lines.push("  • Enrolled speaker decided by: no match — neither path exceeded the confidence threshold.");
  } else {
    lines.push("  • Enrolled speaker decided by: unknown (older entry or diagnostics missing).");
  }

  if (typeof item.speakerConfidence === "number" && Number.isFinite(item.speakerConfidence)) {
    lines.push(
      `  • Pass-1 match score (server): ${(item.speakerConfidence * 100).toFixed(1)}% (embedding vs fingerprint use different internal thresholds).`,
    );
  }

  if (d1) {
    const embOk = d1.embeddingFetchSucceeded === true;
    const embTry = d1.embeddingFetchAttempted === true;
    lines.push(
      `  • Clip neural embedding computed: ${embOk ? `yes — ${d1.embeddingDimensions ?? "?"} dims` : embTry ? `no — ${d1.embeddingError?.trim() || "fetch failed"}` : "n/a (SPEAKER_EMBEDDING_SERVICE_URL not set)"}.`,
    );
    lines.push(
      `  • Server match flags — used_embedding=${d1.speakerMatchUsedEmbedding === true ? "yes" : d1.speakerMatchUsedEmbedding === false ? "no" : "—"}, used_fingerprint=${d1.speakerMatchUsedFingerprint === true ? "yes" : d1.speakerMatchUsedFingerprint === false ? "no" : "—"}.`,
    );
    if (engine === "fingerprint" && embOk) {
      lines.push(
        "  • Interpretation: embedding ran but did not select the speaker; fingerprint matcher supplied the hit.",
      );
    }
    if (engine === "fingerprint" && !embTry) {
      lines.push("  • Interpretation: embedding service off — matching used fingerprints only.");
    }
    if (engine === "embedding" && d1.speakerMatchUsedFingerprint === true) {
      lines.push(
        "  • Note: fingerprint flag may also be true when both signals were evaluated; winning engine is embedding.",
      );
    }
  } else {
    lines.push("  • Pass-1 diagnostic payload not stored — only the summary line above may apply.");
  }

  if (d2) {
    lines.push("Pass 2 — enroll (request included manual speaker name)");
    lines.push(
      `  • Speaker ID matching on this request: ${d2.recognitionAttempted ? "attempted" : "skipped"} (server enroll path when name is supplied).`,
    );
    lines.push(
      `  • Saved to profile — 12-D fingerprint enrollment: ${d2.fingerprintVectorSavedToProfile ? "yes" : "no"}.`,
    );
    lines.push(
      `  • Saved to profile — neural embedding enrollment: ${d2.embeddingVectorSavedToProfile ? "yes" : "no"}${d2.embeddingVectorSavedToProfile || !d2.embeddingFetchAttempted ? "" : " (clip had no usable embedding)"}.`,
    );
  }

  return lines.join("\n");
};

const readFirstPassRecognitionEngineFromPayload = (item: Record<string, unknown>) => {
  const v = item.firstPassRecognitionEngine;
  if (v === "embedding" || v === "fingerprint" || v === "none") {
    return v;
  }
  return undefined;
};

const firstPassEngineFromTranscriptionResult = (
  r: TranscriptionResult,
): TranscriptLogItem["firstPassRecognitionEngine"] | undefined => {
  const v = r.speakerRecognitionEngine;
  if (v === "embedding" || v === "fingerprint" || v === "none") {
    return v;
  }
  if (v === null) {
    return "none";
  }
  return undefined;
};

/** Narrative: app decisions (App.tsx) + server pipeline steps for each pass. */
const formatHistoryLogicalFlow = (item: TranscriptLogItem): string => {
  const lines: string[] = [];
  lines.push("App (App.tsx)");
  lines.push("  1. Stop recording → transcribeAudio(uri, \"\") — first POST /transcribe-base64 without manual speaker name.");
  if (item.wasConflictPromptShown) {
    lines.push(
      "  2. Conflict — resolveSpeakerNameConflict: choose between a typed name and the server’s enrolled-speaker guess.",
    );
  }
  if (item.wasUnknownSpeakerPromptShown) {
    lines.push(
      "  Unknown-speaker flow — openUnknownSpeakerNamePrompt: first pass still produced the “Unknown speaker” label.",
    );
    if (item.promptedSpeakerName) {
      lines.push(
        `  └ User named “${item.promptedSpeakerName}” → second transcribeAudio(..., name) for enrollment (same clip).`,
      );
    } else {
      lines.push("  └ User skipped — no second POST; transcript kept as unknown without enrolling.");
    }
  } else if (item.wasVoiceMatchUsed) {
    const eng = resolvePass1RecognitionEngine(item);
    const detail =
      eng === "embedding"
        ? " — winner: neural embedding (speaker vector)"
        : eng === "fingerprint"
          ? " — winner: 12-D audio fingerprint"
          : eng === "none"
            ? ""
            : "";
    lines.push(
      `  After pass 1 — enrolled speaker from voice matching${detail}; unknown-speaker modal was not needed.`,
    );
  } else {
    lines.push(
      "  After pass 1 — unknown-speaker modal not shown; final name comes from Source / attribution below (not a voice match to an enrollment).",
    );
  }
  lines.push(
    `  Final label: ${formatAttributionSource(item.speakerAttributionSource)} → “${item.speakerName}”.`,
  );
  if (item.assemblySpeakerLabel) {
    lines.push(
      `  AssemblyAI diarization speaker id (dominant): ${item.assemblySpeakerLabel} — utterance labels, separate from enrolled name.`,
    );
  }

  if (item.transcriptionDiagnosticsInitial) {
    lines.push("");
    lines.push(...formatServerPassLogicalPath("Pass 1 — listen / match / transcribe", item.transcriptionDiagnosticsInitial));
  }
  if (item.transcriptionDiagnosticsEnrollment) {
    lines.push("");
    lines.push(...formatServerPassLogicalPath("Pass 2 — enroll named speaker", item.transcriptionDiagnosticsEnrollment));
  }
  if (!item.transcriptionDiagnosticsInitial && !item.transcriptionDiagnosticsEnrollment) {
    lines.push("");
    lines.push(
      "(No transcriptionDiagnostics on this entry — only the app steps above; reconnect or re-record to capture server timing.)",
    );
  }

  return lines.join("\n");
};

const readString = (item: Record<string, unknown>, key: string) =>
  typeof item[key] === "string" ? item[key] : "";

const readNullableString = (item: Record<string, unknown>, key: string) => {
  const value = readString(item, key).trim();
  return value || null;
};

const readBoolean = (item: Record<string, unknown>, key: string) => item[key] === true;

const readNullableNumber = (item: Record<string, unknown>, key: string) =>
  typeof item[key] === "number" && Number.isFinite(item[key]) ? item[key] : null;

const readAttributionSource = (value: unknown): SpeakerAttributionSource | undefined => {
  if (
    value === "speaker_name_input" ||
    value === "voice_match" ||
    value === "unknown_speaker_prompt" ||
    value === "speaker_conflict_prompt" ||
    value === "unknown"
  ) {
    return value;
  }
  return undefined;
};

const normalizeHistoryItem = (value: unknown): TranscriptLogItem | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const item = value as Record<string, unknown>;
  const createdAtMs = readNullableNumber(item, "createdAtMs") ?? Date.now();
  const clientId = readString(item, "clientId").trim();
  const rawId = readString(item, "id").trim();
  const id = clientId || rawId || `${createdAtMs}`;
  const cloudHistoryId = clientId && rawId && rawId !== clientId ? rawId : readString(item, "cloudHistoryId").trim();

  const transcriptionDiagnosticsInitial = readDiagnosticsPhase(item.transcriptionDiagnosticsInitial);
  const transcriptionDiagnosticsEnrollment = readDiagnosticsPhase(item.transcriptionDiagnosticsEnrollment);
  const fromPayload = readFirstPassRecognitionEngineFromPayload(item);
  const fromDiag = transcriptionDiagnosticsInitial?.speakerRecognitionEngine;
  const firstPassRecognitionEngine =
    fromPayload ??
    (fromDiag === "embedding" || fromDiag === "fingerprint" || fromDiag === "none" ? fromDiag : undefined);

  return {
    id,
    speakerName: readString(item, "speakerName").trim() || UNKNOWN_SPEAKER_LABEL,
    text: readString(item, "text"),
    createdAt: readString(item, "createdAt") || new Date(createdAtMs).toLocaleTimeString(),
    createdAtMs,
    ...(cloudHistoryId ? { cloudHistoryId } : {}),
    speakerAttributionSource: readAttributionSource(item.speakerAttributionSource),
    speakerNameInput: readNullableString(item, "speakerNameInput"),
    promptedSpeakerName: readNullableString(item, "promptedSpeakerName"),
    detectedSpeakerName: readNullableString(item, "detectedSpeakerName"),
    speakerConfidence: readNullableNumber(item, "speakerConfidence"),
    matchedEnrollmentSampleId: readNullableString(item, "matchedEnrollmentSampleId"),
    matchedEnrollmentSampleSource: readNullableString(item, "matchedEnrollmentSampleSource"),
    matchedEnrollmentSampleCreatedAtIso: readNullableString(item, "matchedEnrollmentSampleCreatedAtIso"),
    assemblySpeakerLabel: readNullableString(item, "assemblySpeakerLabel"),
    wasSpeakerNameInputProvided: readBoolean(item, "wasSpeakerNameInputProvided"),
    wasUnknownSpeakerPromptShown: readBoolean(item, "wasUnknownSpeakerPromptShown"),
    wasVoiceMatchUsed: readBoolean(item, "wasVoiceMatchUsed"),
    wasConflictPromptShown: readBoolean(item, "wasConflictPromptShown"),
    wasVoiceProfileEnrolled: readBoolean(item, "wasVoiceProfileEnrolled"),
    ...(firstPassRecognitionEngine ? { firstPassRecognitionEngine } : {}),
    transcriptionDiagnosticsInitial,
    transcriptionDiagnosticsEnrollment,
  };
};

const mergeHistoryEntries = (localEntries: TranscriptLogItem[], cloudEntries: TranscriptLogItem[]) => {
  const merged = new Map<string, TranscriptLogItem>();
  const keyFor = (item: TranscriptLogItem) =>
    item.cloudHistoryId || item.id || `${item.createdAtMs}:${item.speakerName}:${item.text.slice(0, 60)}`;

  [...localEntries, ...cloudEntries].forEach((item) => {
    const key = keyFor(item);
    const existing = merged.get(key);
    merged.set(key, existing ? { ...existing, ...item } : item);
  });

  return [...merged.values()].sort((a, b) => b.createdAtMs - a.createdAtMs);
};

const resolveSpeakerNameConflict = (
  manualName: string,
  result: TranscriptionResult,
  minConfidence: number,
): Promise<string> => {
  const manual = manualName.trim();
  const detected = result.detectedSpeakerName?.trim() || "";
  const confidence = result.speakerConfidence;
  const hasConfidentMatch =
    !!detected &&
    typeof confidence === "number" &&
    confidence >= minConfidence &&
    normalizeSpeakerKey(manual) !== normalizeSpeakerKey(detected);

  if (!hasConfidentMatch) {
    return Promise.resolve(manual);
  }

  const pct = (confidence * 100).toFixed(0);
  const diarization = result.assemblySpeakerLabel
    ? `\n\nAssemblyAI diarization label: ${result.assemblySpeakerLabel}`
    : "";

  return new Promise((resolve) => {
    const finish = (choice: string) => resolve(choice.trim());

    Alert.alert(
      "Check speaker name",
      `You entered "${manual}". Voice fingerprint match suggests "${detected}" (${pct}% confidence).${diarization}\n\nWhich name should we use for this transcript?`,
      [
        { text: `Use "${detected}"`, onPress: () => finish(detected) },
        { text: `Keep "${manual}"`, onPress: () => finish(manual) },
        { text: "Cancel", style: "cancel", onPress: () => finish(manual) },
      ],
      { cancelable: true, onDismiss: () => finish(manual) },
    );
  });
};

export default function App() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [history, setHistory] = useState<TranscriptLogItem[]>([]);
  const [activeTab, setActiveTab] = useState<AppTab>("record");
  const [expandedHistoryIds, setExpandedHistoryIds] = useState<string[]>([]);
  const [expandedHistorySpeakers, setExpandedHistorySpeakers] = useState<string[]>([]);
  const [expandedHistoryDateGroups, setExpandedHistoryDateGroups] = useState<string[]>([]);
  const [expandedSpeakerNames, setExpandedSpeakerNames] = useState<string[]>([]);
  const [lastSpeakerName, setLastSpeakerName] = useState(UNKNOWN_SPEAKER_LABEL);
  const [speakers, setSpeakers] = useState<SpeakerProfile[]>([]);
  const [isLoadingSpeakers, setIsLoadingSpeakers] = useState(false);
  const [speakerHintModal, setSpeakerHintModal] = useState<{ name: string; draft: string } | null>(null);
  const [isSavingSpeakerHint, setIsSavingSpeakerHint] = useState(false);
  const [hasLoadedHistory, setHasLoadedHistory] = useState(false);
  const [statusText, setStatusText] = useState("Tap the mic — we’ll handle the rest.");
  const [errorText, setErrorText] = useState<string | null>(null);
  /** Should match API `SPEAKER_MATCH_THRESHOLD` (same default 0.97) so conflict prompts align with server gating. */
  const speakerAutoAssignMinConfidence = Number(process.env.EXPO_PUBLIC_SPEAKER_MIN_CONFIDENCE || "0.97");

  const unknownSpeakerResolveRef = useRef<((value: string) => void) | null>(null);
  const lastAutoExpandedHistoryIdRef = useRef<string | null>(null);
  const [unknownSpeakerModalVisible, setUnknownSpeakerModalVisible] = useState(false);
  const [unknownSpeakerDraft, setUnknownSpeakerDraft] = useState("");
  const enrollTargetNameRef = useRef<string | null>(null);
  const [enrollTargetName, setEnrollTargetName] = useState<string | null>(null);
  const [enrollSpeakerModalVisible, setEnrollSpeakerModalVisible] = useState(false);
  const [enrollNameDraft, setEnrollNameDraft] = useState("");

  const closeUnknownSpeakerPrompt = (value: string) => {
    setUnknownSpeakerModalVisible(false);
    setUnknownSpeakerDraft("");
    const resolve = unknownSpeakerResolveRef.current;
    unknownSpeakerResolveRef.current = null;
    resolve?.(value);
  };

  const openUnknownSpeakerNamePrompt = () =>
    new Promise<string>((resolve) => {
      unknownSpeakerResolveRef.current = resolve;
      setUnknownSpeakerDraft("");
      setUnknownSpeakerModalVisible(true);
    });

  const getApiBaseUrl = () => {
    const apiUrl = process.env.EXPO_PUBLIC_TRANSCRIBE_API_URL;
    if (!apiUrl) {
      throw new Error("Missing EXPO_PUBLIC_TRANSCRIBE_API_URL.");
    }
    return apiUrl.replace(/\/transcribe\/?$/, "");
  };

  const canShowTranscript = useMemo(() => transcript.length > 0, [transcript]);
  const isRecording = recording !== null;
  const isBusy =
    isRecording || isUploading || unknownSpeakerModalVisible || enrollSpeakerModalVisible;

  const createTimeLabel = (timestampMs: number) =>
    new Date(timestampMs).toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

  const transcribeAudio = async (
    uri: string,
    currentSpeakerName: string,
    options: { enrollmentSource?: SpeakerAttributionSource; historyClientId?: string } = {},
  ) => {
    const apiUrl = process.env.EXPO_PUBLIC_TRANSCRIBE_API_URL;
    const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
    if (!apiUrl) {
      throw new Error("Missing EXPO_PUBLIC_TRANSCRIBE_API_URL.");
    }

    const trimmedSpeakerName = currentSpeakerName.trim();
    const baseUrl = apiUrl.replace(/\/transcribe\/?$/i, "");

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
      ...(trimmedSpeakerName ? { "x-speaker-name": trimmedSpeakerName } : {}),
      ...(trimmedSpeakerName && options.enrollmentSource
        ? { "x-enrollment-source": options.enrollmentSource }
        : {}),
      ...(options.historyClientId ? { "x-history-client-id": options.historyClientId } : {}),
    };

    let audioBase64: string;
    if (Platform.OS === "web") {
      const fileResponse = await fetch(uri);
      if (!fileResponse.ok) {
        throw new Error(`Could not read recording (${fileResponse.status}).`);
      }
      const blob = await fileResponse.blob();
      audioBase64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const r = reader.result;
          if (typeof r !== "string") {
            reject(new Error("Could not read recording as base64."));
            return;
          }
          const parts = r.split(",");
          resolve(parts.length > 1 ? parts[1] : r);
        };
        reader.onerror = () => reject(reader.error || new Error("FileReader failed."));
        reader.readAsDataURL(blob);
      });
    } else {
      audioBase64 = await readAsStringAsync(uri, { encoding: EncodingType.Base64 });
    }

    const requestUrl = trimmedSpeakerName
      ? `${baseUrl}/transcribe-base64?speakerName=${encodeURIComponent(trimmedSpeakerName)}`
      : `${baseUrl}/transcribe-base64`;

    const response = await fetch(requestUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({
        audioBase64,
        mimeType: "audio/m4a",
        ...(trimmedSpeakerName ? { speakerName: trimmedSpeakerName } : {}),
        ...(trimmedSpeakerName && options.enrollmentSource
          ? { enrollmentSource: options.enrollmentSource }
          : {}),
        ...(options.historyClientId ? { historyClientId: options.historyClientId } : {}),
      }),
    });

    const responseBody = await response.text();
    if (!response.ok) {
      throw new Error(responseBody || `Transcription failed (${response.status}).`);
    }

    const data = JSON.parse(responseBody) as {
      text?: string;
      transcript?: string;
      enrolledSpeakerName?: string | null;
      detectedSpeakerName?: string | null;
      speakerConfidence?: number | null;
      detectedSpeakerSampleId?: string | null;
      detectedSpeakerSampleSource?: string | null;
      detectedSpeakerSampleCreatedAtIso?: string | null;
      assemblySpeakerLabel?: string | null;
      speakerIdentificationCandidates?: string[];
      speakerIdentificationMapping?: Record<string, unknown> | null;
      speakerRecognitionEngine?: string | null;
      speakerEmbeddingEnabled?: boolean;
      speakerEmbeddingAvailable?: boolean;
      transcriptionDiagnostics?: unknown;
    };
    return {
      text: data.text?.trim() || data.transcript?.trim() || "",
      enrolledSpeakerName: data.enrolledSpeakerName?.trim() || null,
      detectedSpeakerName: data.detectedSpeakerName?.trim() || null,
      speakerConfidence: typeof data.speakerConfidence === "number" ? data.speakerConfidence : null,
      detectedSpeakerSampleId:
        typeof data.detectedSpeakerSampleId === "string" && data.detectedSpeakerSampleId.trim()
          ? data.detectedSpeakerSampleId.trim()
          : null,
      detectedSpeakerSampleSource:
        typeof data.detectedSpeakerSampleSource === "string" && data.detectedSpeakerSampleSource.trim()
          ? data.detectedSpeakerSampleSource.trim()
          : null,
      detectedSpeakerSampleCreatedAtIso:
        typeof data.detectedSpeakerSampleCreatedAtIso === "string" &&
        data.detectedSpeakerSampleCreatedAtIso.trim()
          ? data.detectedSpeakerSampleCreatedAtIso.trim()
          : null,
      assemblySpeakerLabel:
        typeof data.assemblySpeakerLabel === "string" && data.assemblySpeakerLabel.trim()
          ? data.assemblySpeakerLabel.trim()
          : null,
      speakerRecognitionEngine:
        typeof data.speakerRecognitionEngine === "string"
          ? data.speakerRecognitionEngine
          : data.speakerRecognitionEngine === null
            ? null
            : undefined,
      speakerEmbeddingEnabled: data.speakerEmbeddingEnabled === true,
      speakerEmbeddingAvailable: data.speakerEmbeddingAvailable === true,
      transcriptionDiagnostics: parseTranscriptionDiagnostics(data.transcriptionDiagnostics),
    } satisfies TranscriptionResult;
  };

  const saveHistoryEntryToCloud = async (entry: TranscriptLogItem) => {
    try {
      const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
      const response = await fetch(`${getApiBaseUrl()}/history`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify(entry),
      });
      if (!response.ok) {
        const bodyText = await response.text();
        throw new Error(bodyText || `Cloud history save failed (${response.status}).`);
      }
      const data = (await response.json()) as { history?: { id?: string } };
      const cloudHistoryId = typeof data.history?.id === "string" ? data.history.id : null;
      if (cloudHistoryId) {
        setHistory((current) =>
          current.map((item) => (item.id === entry.id ? { ...item, cloudHistoryId } : item)),
        );
      }
    } catch (error) {
      console.warn("Cloud history save failed", error);
    }
  };

  const fetchCloudHistory = async () => {
    try {
      const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
      const response = await fetch(`${getApiBaseUrl()}/history?limit=100`, {
        headers: apiToken ? { Authorization: `Bearer ${apiToken}` } : undefined,
      });
      if (!response.ok) {
        const bodyText = await response.text();
        throw new Error(bodyText || `Cloud history load failed (${response.status}).`);
      }
      const data = (await response.json()) as { history?: unknown[] };
      return Array.isArray(data.history)
        ? data.history.map(normalizeHistoryItem).filter((item): item is TranscriptLogItem => item !== null)
        : [];
    } catch (error) {
      console.warn("Cloud history load failed", error);
      return [];
    }
  };

  const fetchSpeakers = async (suppressError = false) => {
    try {
      setIsLoadingSpeakers(true);
      const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
      const response = await fetch(`${getApiBaseUrl()}/speakers`, {
        headers: apiToken ? { Authorization: `Bearer ${apiToken}` } : undefined,
      });
      if (!response.ok) {
        throw new Error("Could not load speaker profiles.");
      }
      const data = (await response.json()) as { speakers?: SpeakerProfile[] };
      setSpeakers(Array.isArray(data.speakers) ? data.speakers : []);
    } catch (error) {
      if (!suppressError) {
        const message = error instanceof Error ? error.message : "Could not load speaker profiles.";
        setErrorText(message);
      }
    } finally {
      setIsLoadingSpeakers(false);
    }
  };

  const openSpeakerHintModal = (speaker: SpeakerProfile) => {
    setSpeakerHintModal({
      name: speaker.name,
      draft: speaker.description?.trim() ?? "",
    });
  };

  const persistSpeakerHintToApi = async (name: string, speakerDescription: string) => {
    const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
    const response = await fetch(`${getApiBaseUrl()}/speakers`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
      },
      body: JSON.stringify({ name, speakerDescription }),
    });
    const bodyText = await response.text();
    if (!response.ok) {
      throw new Error(bodyText || `Could not update hint (${response.status}).`);
    }
  };

  const saveSpeakerHint = async () => {
    if (!speakerHintModal) {
      return;
    }
    const name = speakerHintModal.name;
    const speakerDescription = speakerHintModal.draft.trim().slice(0, SPEAKER_DESCRIPTION_MAX_CHARS);
    try {
      setIsSavingSpeakerHint(true);
      setErrorText(null);
      await persistSpeakerHintToApi(name, speakerDescription);
      setSpeakerHintModal(null);
      setStatusText(speakerDescription ? `Saved hint for ${name}.` : `Cleared hint for ${name}.`);
      await fetchSpeakers(true);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not save speaker hint.";
      setErrorText(message);
    } finally {
      setIsSavingSpeakerHint(false);
    }
  };

  const clearSpeakerHint = async () => {
    if (!speakerHintModal) {
      return;
    }
    const name = speakerHintModal.name;
    try {
      setIsSavingSpeakerHint(true);
      setErrorText(null);
      await persistSpeakerHintToApi(name, "");
      setSpeakerHintModal(null);
      setStatusText(`Cleared hint for ${name}.`);
      await fetchSpeakers(true);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not clear speaker hint.";
      setErrorText(message);
    } finally {
      setIsSavingSpeakerHint(false);
    }
  };

  const clearSpeakerProfiles = async () => {
    try {
      const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
      const response = await fetch(`${getApiBaseUrl()}/speakers`, {
        method: "DELETE",
        headers: apiToken ? { Authorization: `Bearer ${apiToken}` } : undefined,
      });
      if (!response.ok) {
        throw new Error("Could not reset speaker profiles.");
      }
      setSpeakers([]);
      setStatusText("Speaker profiles reset.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not reset speaker profiles.";
      setErrorText(message);
    }
  };

  const maybeSuggestSpeakerCorrection = async (
    latestText: string,
    recentHistory: TranscriptLogItem[],
  ) => {
    try {
      const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
      const response = await fetch(`${getApiBaseUrl()}/ai/speaker-correction`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({
          latestText,
          recentHistory: recentHistory.slice(0, 5).map((item) => ({
            speakerName: item.speakerName,
            text: item.text,
          })),
        }),
      });
      if (!response.ok) {
        return;
      }
      const data = (await response.json()) as { suggestion?: SpeakerCorrectionSuggestion };
      const suggestion = data.suggestion;
      if (!suggestion?.shouldSuggest || !suggestion.suggestedSpeakerName) {
        return;
      }
      const nextSpeakerName = suggestion.suggestedSpeakerName.trim();
      if (!nextSpeakerName || !recentHistory[0] || recentHistory[0].speakerName === nextSpeakerName) {
        return;
      }

      Alert.alert(
        "Rename Last Speaker?",
        `AI detected a rename request.\n\nChange "${recentHistory[0].speakerName}" to "${nextSpeakerName}" for the latest log?`,
        [
          { text: "Keep Current", style: "cancel" },
          {
            text: "Confirm",
            onPress: () => {
              setHistory((current) =>
                current.map((item, index) =>
                  index === 0
                    ? {
                        ...item,
                        speakerName: nextSpeakerName,
                      }
                    : item,
                ),
              );
              setLastSpeakerName(nextSpeakerName);
              setStatusText(`Updated last speaker to ${nextSpeakerName}.`);
            },
          },
        ],
      );
    } catch {
      // Non-blocking enhancement; ignore suggestion failures.
    }
  };

  const confirmDeleteHistoryItem = (historyId: string) => {
    Alert.alert("Delete This Log?", "This history entry will be removed.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: () => {
          setHistory((current) => current.filter((item) => item.id !== historyId));
        },
      },
    ]);
  };

  const clearTranscript = () => {
    setTranscript("");
  };

  const toggleHistoryItemExpanded = (historyId: string) => {
    setExpandedHistoryIds((current) =>
      current.includes(historyId)
        ? current.filter((id) => id !== historyId)
        : [...current, historyId],
    );
  };

  const toggleHistorySpeakerExpanded = (speakerName: string) => {
    setExpandedHistorySpeakers((current) =>
      current.includes(speakerName)
        ? current.filter((name) => name !== speakerName)
        : [...current, speakerName],
    );
  };

  const toggleHistoryDateExpanded = (dateGroupKey: string) => {
    setExpandedHistoryDateGroups((current) =>
      current.includes(dateGroupKey)
        ? current.filter((key) => key !== dateGroupKey)
        : [...current, dateGroupKey],
    );
  };

  const toggleSpeakerSamplesExpanded = (speakerName: string) => {
    setExpandedSpeakerNames((current) =>
      current.includes(speakerName)
        ? current.filter((name) => name !== speakerName)
        : [...current, speakerName],
    );
  };

  const deleteVoiceSample = (speakerName: string, sampleId: string) => {
    Alert.alert(
      "Delete Voice Sample?",
      `Remove this enrollment sample from ${speakerName}? The speaker match profile will be rebuilt from remaining samples.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: () => {
            void (async () => {
              try {
                setErrorText(null);
                const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
                const response = await fetch(
                  `${getApiBaseUrl()}/speakers/${encodeURIComponent(speakerName)}/samples/${encodeURIComponent(sampleId)}`,
                  {
                    method: "DELETE",
                    headers: apiToken ? { Authorization: `Bearer ${apiToken}` } : undefined,
                  },
                );
                const bodyText = await response.text();
                if (!response.ok) {
                  throw new Error(bodyText || `Could not delete sample (${response.status}).`);
                }
                setStatusText(`Deleted voice sample for ${speakerName}.`);
                await fetchSpeakers(true);
              } catch (error) {
                const message = error instanceof Error ? error.message : "Could not delete voice sample.";
                setErrorText(message);
              }
            })();
          },
        },
      ],
    );
  };

  useEffect(() => {
    fetchSpeakers();
  }, []);

  useEffect(() => {
    const loadHistory = async () => {
      let localHistory: TranscriptLogItem[] = [];
      try {
        const raw = await AsyncStorage.getItem(HISTORY_STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (Array.isArray(parsed)) {
            localHistory = parsed
              .map(normalizeHistoryItem)
              .filter((item): item is TranscriptLogItem => item !== null);
          }
        }
      } catch {
        // Keep app usable even if local history cannot be read.
      }

      try {
        const cloudHistory = await fetchCloudHistory();
        setHistory(mergeHistoryEntries(localHistory, cloudHistory));
      } catch {
        setHistory(localHistory);
      } finally {
        setHasLoadedHistory(true);
      }
    };

    void loadHistory();
  }, []);

  useEffect(() => {
    if (!hasLoadedHistory) {
      return;
    }
    void AsyncStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history)).catch(() => {
      // Non-blocking persistence failure.
    });
  }, [history, hasLoadedHistory]);

  const handleRecordPress = async (opts?: { fromEnrollModal?: boolean }) => {
    if (isUploading) {
      return;
    }

    try {
      setErrorText(null);

      if (recording) {
        setStatusText("Processing audio...");
        setIsUploading(true);
        await recording.stopAndUnloadAsync();
        const uri = recording.getURI();
        setRecording(null);

        if (!uri) {
          throw new Error("No recording file found.");
        }

        const enrollName = enrollTargetNameRef.current;
        enrollTargetNameRef.current = null;
        setEnrollTargetName(null);

        if (enrollName) {
          const createdAtMs = Date.now();
          const historyClientId = `${createdAtMs}`;
          let transcriptionDiagnosticsEnrollment: TranscriptionDiagnosticsPhase | undefined;
          let voiceEnrollmentRequestFailed = false;
          let result: TranscriptionResult;
          try {
            result = await transcribeAudio(uri, enrollName, {
              enrollmentSource: "speaker_name_input",
              historyClientId,
            });
          } catch (enrollErr) {
            voiceEnrollmentRequestFailed = true;
            const enrollMessage =
              enrollErr instanceof Error
                ? enrollErr.message
                : "Voice profile could not be saved on the server.";
            setErrorText(enrollMessage);
            setStatusText("Enrollment failed — try again.");
            setIsUploading(false);
            return;
          }
          if (result.transcriptionDiagnostics) {
            transcriptionDiagnosticsEnrollment = readDiagnosticsPhase({
              ...result.transcriptionDiagnostics,
              phase: "enrollment",
            });
          }
          const text = result.text;
          let normalizedSpeakerName = result.enrolledSpeakerName || enrollName;
          let speakerAttributionSource: SpeakerAttributionSource = "speaker_name_input";
          let wasConflictPromptShown = false;
          let wasVoiceMatchUsed = false;
          const hasServerDetectedSpeaker = !!result.detectedSpeakerName?.trim();
          const hasConfidentDetectedForConflict =
            hasServerDetectedSpeaker &&
            result.speakerConfidence !== null &&
            result.speakerConfidence >= speakerAutoAssignMinConfidence;

          if (hasConfidentDetectedForConflict && result.detectedSpeakerName) {
            wasConflictPromptShown = true;
            const confirmed = await resolveSpeakerNameConflict(
              enrollName,
              result,
              speakerAutoAssignMinConfidence,
            );
            normalizedSpeakerName = confirmed || normalizedSpeakerName;
            speakerAttributionSource =
              normalizeSpeakerKey(confirmed) === normalizeSpeakerKey(result.detectedSpeakerName)
                ? "voice_match"
                : "speaker_conflict_prompt";
            wasVoiceMatchUsed = speakerAttributionSource === "voice_match";
          }

          setTranscript(text);
          setLastSpeakerName(normalizedSpeakerName);
          const newHistoryEntry: TranscriptLogItem = {
            id: historyClientId,
            speakerName: normalizedSpeakerName,
            text,
            createdAt: createTimeLabel(createdAtMs),
            createdAtMs,
            speakerAttributionSource,
            speakerNameInput: enrollName,
            promptedSpeakerName: null,
            detectedSpeakerName: result.detectedSpeakerName,
            speakerConfidence: result.speakerConfidence,
            matchedEnrollmentSampleId: result.detectedSpeakerSampleId,
            matchedEnrollmentSampleSource: result.detectedSpeakerSampleSource,
            matchedEnrollmentSampleCreatedAtIso: result.detectedSpeakerSampleCreatedAtIso,
            assemblySpeakerLabel: result.assemblySpeakerLabel,
            wasSpeakerNameInputProvided: true,
            wasUnknownSpeakerPromptShown: false,
            wasVoiceMatchUsed,
            wasConflictPromptShown,
            wasVoiceProfileEnrolled: !voiceEnrollmentRequestFailed,
            ...(transcriptionDiagnosticsEnrollment
              ? { transcriptionDiagnosticsEnrollment }
              : {}),
          };
          setHistory((current) => [{ ...newHistoryEntry }, ...current]);
          void saveHistoryEntryToCloud(newHistoryEntry);
          void fetchSpeakers(true);
          if (!voiceEnrollmentRequestFailed) {
            setStatusText(
              text
                ? `Voice saved for ${normalizedSpeakerName}. Transcript ready.`
                : `Voice saved for ${normalizedSpeakerName} — record again for speech text.`,
            );
          }
          setIsUploading(false);
          return;
        }

        const createdAtMs = Date.now();
        const historyClientId = `${createdAtMs}`;
        const result = await transcribeAudio(uri, "", {
          historyClientId,
        });
        let transcriptionDiagnosticsInitial: TranscriptionDiagnosticsPhase | undefined;
        if (result.transcriptionDiagnostics) {
          transcriptionDiagnosticsInitial = readDiagnosticsPhase({
            ...result.transcriptionDiagnostics,
            phase: "transcribe",
          });
        }
        let transcriptionDiagnosticsEnrollment: TranscriptionDiagnosticsPhase | undefined;
        const text = result.text;
        const manualSpeakerName = "";
        const typedSpeakerName = "";
        let shouldRefreshSpeakers = !!manualSpeakerName;
        let voiceEnrollmentRequestFailed = false;
        /** Set when unknown-speaker flow successfully re-uploaded audio with a name (server enrollment). */
        let enrolledVoiceAsName: string | null = null;
        let promptedSpeakerName: string | null = null;
        let wasUnknownSpeakerPromptShown = false;
        let wasConflictPromptShown = false;
        let wasVoiceMatchUsed = false;
        const hasServerDetectedSpeaker = !!result.detectedSpeakerName?.trim();
        const hasConfidentDetectedForConflict =
          hasServerDetectedSpeaker &&
          result.speakerConfidence !== null &&
          result.speakerConfidence >= speakerAutoAssignMinConfidence;

        let normalizedSpeakerName =
          manualSpeakerName ||
          (hasServerDetectedSpeaker ? result.detectedSpeakerName : null) ||
          UNKNOWN_SPEAKER_LABEL;
        let speakerAttributionSource: SpeakerAttributionSource = manualSpeakerName
          ? "speaker_name_input"
          : hasServerDetectedSpeaker
            ? "voice_match"
            : "unknown";
        if (!manualSpeakerName && hasServerDetectedSpeaker) {
          wasVoiceMatchUsed = true;
        }

        if (manualSpeakerName && hasConfidentDetectedForConflict && result.detectedSpeakerName) {
          wasConflictPromptShown = true;
          const confirmed = await resolveSpeakerNameConflict(
            manualSpeakerName,
            result,
            speakerAutoAssignMinConfidence,
          );
          normalizedSpeakerName = confirmed || normalizedSpeakerName;
          speakerAttributionSource =
            normalizeSpeakerKey(confirmed) === normalizeSpeakerKey(result.detectedSpeakerName)
              ? "voice_match"
              : "speaker_conflict_prompt";
          wasVoiceMatchUsed = speakerAttributionSource === "voice_match";
        }

        if (isUnknownSpeakerLabel(normalizedSpeakerName)) {
          setIsUploading(false);
          wasUnknownSpeakerPromptShown = true;
          const entered = await openUnknownSpeakerNamePrompt();
          const trimmedEntered = entered.trim();
          promptedSpeakerName = trimmedEntered || null;
          normalizedSpeakerName = trimmedEntered || UNKNOWN_SPEAKER_LABEL;
          if (trimmedEntered) {
            speakerAttributionSource = "unknown_speaker_prompt";
            try {
              setIsUploading(true);
              setStatusText("Saving voice profile…");
              const enrollmentResult = await transcribeAudio(uri, trimmedEntered, {
                enrollmentSource: "unknown_speaker_prompt",
                historyClientId,
              });
              if (enrollmentResult.transcriptionDiagnostics) {
                transcriptionDiagnosticsEnrollment = readDiagnosticsPhase({
                  ...enrollmentResult.transcriptionDiagnostics,
                  phase: "enrollment",
                });
              }
              shouldRefreshSpeakers = true;
              enrolledVoiceAsName = enrollmentResult.enrolledSpeakerName || trimmedEntered;
              normalizedSpeakerName = enrolledVoiceAsName;
            } catch (enrollError) {
              voiceEnrollmentRequestFailed = true;
              const enrollMessage =
                enrollError instanceof Error
                  ? enrollError.message
                  : "Voice profile could not be saved on the server.";
              setErrorText(enrollMessage);
              setStatusText("Transcript saved locally. Enrollment failed — try recording again.");
            } finally {
              setIsUploading(false);
            }
          }
        }

        setTranscript(text);
        setLastSpeakerName(normalizedSpeakerName);
        const pass1RecognitionEngine = firstPassEngineFromTranscriptionResult(result);
        const newHistoryEntry: TranscriptLogItem = {
          id: historyClientId,
          speakerName: normalizedSpeakerName,
          text,
          createdAt: createTimeLabel(createdAtMs),
          createdAtMs,
          speakerAttributionSource,
          speakerNameInput: typedSpeakerName || null,
          promptedSpeakerName,
          detectedSpeakerName: result.detectedSpeakerName,
          speakerConfidence: result.speakerConfidence,
          matchedEnrollmentSampleId: result.detectedSpeakerSampleId,
          matchedEnrollmentSampleSource: result.detectedSpeakerSampleSource,
          matchedEnrollmentSampleCreatedAtIso: result.detectedSpeakerSampleCreatedAtIso,
          assemblySpeakerLabel: result.assemblySpeakerLabel,
          wasSpeakerNameInputProvided: !!typedSpeakerName,
          wasUnknownSpeakerPromptShown,
          wasVoiceMatchUsed,
          wasConflictPromptShown,
          wasVoiceProfileEnrolled: !!enrolledVoiceAsName || (!!typedSpeakerName && !voiceEnrollmentRequestFailed),
          ...(pass1RecognitionEngine ? { firstPassRecognitionEngine: pass1RecognitionEngine } : {}),
          ...(transcriptionDiagnosticsInitial ? { transcriptionDiagnosticsInitial } : {}),
          ...(transcriptionDiagnosticsEnrollment ? { transcriptionDiagnosticsEnrollment } : {}),
        };
        setHistory((current) => [
          {
            ...newHistoryEntry,
          },
          ...current,
        ]);
        void saveHistoryEntryToCloud(newHistoryEntry);
        if (shouldRefreshSpeakers) {
          void fetchSpeakers(true);
        }
        if (!voiceEnrollmentRequestFailed) {
          if (enrolledVoiceAsName) {
            setStatusText(
              text
                ? `Transcription complete. Voice saved for ${enrolledVoiceAsName} — see Known Speakers below.`
                : `No speech detected. Voice still saved for ${enrolledVoiceAsName} for recognition next time.`,
            );
          } else {
            setStatusText(text ? "Transcription complete." : "No speech detected.");
          }
        }
        setIsUploading(false);
        return;
      }

      if (!opts?.fromEnrollModal) {
        enrollTargetNameRef.current = null;
        setEnrollTargetName(null);
      }

      const permission = await Audio.requestPermissionsAsync();
      if (!permission.granted) {
        setErrorText("Microphone permission is required to record your voice.");
        setStatusText("Permission denied.");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      setTranscript("");
      const recordingFor = enrollTargetNameRef.current;
      setStatusText(
        recordingFor ? `Recording voice sample for ${recordingFor}…` : "Listening…",
      );
      const { recording: nextRecording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY,
      );
      setRecording(nextRecording);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Something went wrong while recording/transcribing.";
      setStatusText("Could not transcribe.");
      setErrorText(message);
      setRecording(null);
      enrollTargetNameRef.current = null;
      setEnrollTargetName(null);
    } finally {
      setIsUploading(false);
    }
  };

  const openEnrollSpeakerFlow = () => {
    if (isUploading || isRecording) {
      return;
    }
    setEnrollNameDraft("");
    setEnrollSpeakerModalVisible(true);
  };

  const commitEnrollAndStartRecording = () => {
    const name = enrollNameDraft.trim();
    if (!name) {
      return;
    }
    if (isUploading || isRecording) {
      return;
    }
    setEnrollSpeakerModalVisible(false);
    setEnrollNameDraft("");
    enrollTargetNameRef.current = name;
    setEnrollTargetName(name);
    void handleRecordPress({ fromEnrollModal: true });
  };

  const latestHistoryItem = useMemo(() => {
    const latest = history.reduce<TranscriptLogItem | null>(
      (currentLatest, item) =>
        !currentLatest || (item.createdAtMs || 0) > (currentLatest.createdAtMs || 0)
          ? item
          : currentLatest,
      null,
    );
    return latest;
  }, [history]);

  useEffect(() => {
    if (!latestHistoryItem || lastAutoExpandedHistoryIdRef.current === latestHistoryItem.id) {
      return;
    }
    const speakerName = latestHistoryItem.speakerName?.trim() || UNKNOWN_SPEAKER_LABEL;
    const dateLabel = formatHistoryDateGroup(latestHistoryItem.createdAtMs);
    const dateGroupKey = `${speakerName}:${dateLabel}`;

    lastAutoExpandedHistoryIdRef.current = latestHistoryItem.id;
    setExpandedHistorySpeakers([speakerName]);
    setExpandedHistoryDateGroups([dateGroupKey]);
    setExpandedHistoryIds([latestHistoryItem.id]);
  }, [latestHistoryItem]);

  const groupedHistory = useMemo(() => {
    const groups = new Map<string, Map<string, TranscriptLogItem[]>>();
    history.forEach((item) => {
      const speakerName = item.speakerName?.trim() || UNKNOWN_SPEAKER_LABEL;
      const dateLabel = formatHistoryDateGroup(item.createdAtMs);
      const dateGroups = groups.get(speakerName) || new Map<string, TranscriptLogItem[]>();
      const existing = dateGroups.get(dateLabel) || [];
      dateGroups.set(dateLabel, [...existing, item]);
      groups.set(speakerName, dateGroups);
    });
    return Array.from(groups.entries())
      .map(([speaker, dateGroups]) => [
        speaker,
        Array.from(dateGroups.entries())
          .map(([dateLabel, items]) => [
            dateLabel,
            [...items].sort((a, b) => (b.createdAtMs || 0) - (a.createdAtMs || 0)),
          ] as const)
          .sort(([, aItems], [, bItems]) => (bItems[0]?.createdAtMs || 0) - (aItems[0]?.createdAtMs || 0)),
      ] as const)
      .sort(([, aDates], [, bDates]) => (bDates[0]?.[1][0]?.createdAtMs || 0) - (aDates[0]?.[1][0]?.createdAtMs || 0));
  }, [history]);

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="light" />
      <Modal
        visible={speakerHintModal !== null}
        transparent
        animationType="fade"
        onRequestClose={() => (isSavingSpeakerHint ? undefined : setSpeakerHintModal(null))}
      >
        <View style={styles.modalBackdrop}>
          <Pressable
            style={StyleSheet.absoluteFillObject}
            onPress={() => (isSavingSpeakerHint ? undefined : setSpeakerHintModal(null))}
          />
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>AssemblyAI hint</Text>
            <Text style={styles.modalSubtitle}>
              {speakerHintModal ? `Optional context for “${speakerHintModal.name}” (voice ID).` : ""}
            </Text>
            <TextInput
              style={styles.modalInput}
              placeholder="e.g. British accent, soft voice, usually discusses budgets…"
              placeholderTextColor="#6b769e"
              value={speakerHintModal?.draft ?? ""}
              onChangeText={(text) =>
                setSpeakerHintModal((current) =>
                  current ? { ...current, draft: text.slice(0, SPEAKER_DESCRIPTION_MAX_CHARS) } : current,
                )
              }
              multiline
              editable={!isSavingSpeakerHint}
              maxLength={SPEAKER_DESCRIPTION_MAX_CHARS}
            />
            <Text style={styles.modalCharCount}>
              {(speakerHintModal?.draft.length ?? 0)}/{SPEAKER_DESCRIPTION_MAX_CHARS}
            </Text>
            <Pressable
              style={styles.clearHintButton}
              disabled={isSavingSpeakerHint}
              onPress={() => void clearSpeakerHint()}
            >
              <Text style={styles.clearHintButtonText}>Clear hint</Text>
            </Pressable>
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonSecondary]}
                disabled={isSavingSpeakerHint}
                onPress={() => setSpeakerHintModal(null)}
              >
                <Text style={styles.modalButtonSecondaryText}>Cancel</Text>
              </Pressable>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                disabled={isSavingSpeakerHint}
                onPress={() => void saveSpeakerHint()}
              >
                {isSavingSpeakerHint ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.modalButtonPrimaryText}>Save</Text>
                )}
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <Modal
        visible={enrollSpeakerModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => {
          setEnrollSpeakerModalVisible(false);
          setEnrollNameDraft("");
        }}
      >
        <View style={styles.modalBackdrop}>
          <Pressable
            style={StyleSheet.absoluteFillObject}
            onPress={() => {
              setEnrollSpeakerModalVisible(false);
              setEnrollNameDraft("");
            }}
          />
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Enroll a speaker</Text>
            <Text style={styles.modalSubtitle}>
              Add a voice sample for someone already on your mind — no need to wait for a failed match. We’ll record a
              clip, save their fingerprint (and embedding when available), and transcribe what they said.
            </Text>
            <TextInput
              style={[styles.modalInput, styles.modalInputSingleLine]}
              placeholder="Their name"
              placeholderTextColor="#6b769e"
              value={enrollNameDraft}
              onChangeText={(text) => setEnrollNameDraft(text.slice(0, UNKNOWN_SPEAKER_NAME_MAX_CHARS))}
              maxLength={UNKNOWN_SPEAKER_NAME_MAX_CHARS}
              autoCapitalize="words"
              autoFocus
            />
            <Text style={styles.modalCharCount}>
              {enrollNameDraft.length}/{UNKNOWN_SPEAKER_NAME_MAX_CHARS}
            </Text>
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonSecondary]}
                onPress={() => {
                  setEnrollSpeakerModalVisible(false);
                  setEnrollNameDraft("");
                }}
              >
                <Text style={styles.modalButtonSecondaryText}>Cancel</Text>
              </Pressable>
              <Pressable
                style={[
                  styles.modalButton,
                  styles.modalButtonPrimary,
                  !enrollNameDraft.trim() && styles.modalButtonDisabled,
                ]}
                disabled={!enrollNameDraft.trim()}
                onPress={commitEnrollAndStartRecording}
              >
                <Text style={styles.modalButtonPrimaryText}>Record sample</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <Modal
        visible={unknownSpeakerModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => closeUnknownSpeakerPrompt("")}
      >
        <View style={styles.modalBackdrop}>
          <Pressable style={StyleSheet.absoluteFillObject} onPress={() => closeUnknownSpeakerPrompt("")} />
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Who was speaking?</Text>
            <Text style={styles.modalSubtitle}>
              No enrolled voice matched this clip. Enter their name to label this transcript and send the same
              recording to the server to enroll their voice for next time. Tap Skip to keep the Unknown speaker label
              without enrolling.
            </Text>
            <TextInput
              style={[styles.modalInput, styles.modalInputSingleLine]}
              placeholder="e.g. Alice"
              placeholderTextColor="#6b769e"
              value={unknownSpeakerDraft}
              onChangeText={(text) => setUnknownSpeakerDraft(text.slice(0, UNKNOWN_SPEAKER_NAME_MAX_CHARS))}
              maxLength={UNKNOWN_SPEAKER_NAME_MAX_CHARS}
              autoCapitalize="words"
              autoFocus
            />
            <Text style={styles.modalCharCount}>
              {unknownSpeakerDraft.length}/{UNKNOWN_SPEAKER_NAME_MAX_CHARS}
            </Text>
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonSecondary]}
                onPress={() => closeUnknownSpeakerPrompt("")}
              >
                <Text style={styles.modalButtonSecondaryText}>Skip</Text>
              </Pressable>
              <Pressable
                style={[
                  styles.modalButton,
                  styles.modalButtonPrimary,
                  !unknownSpeakerDraft.trim() && styles.modalButtonDisabled,
                ]}
                disabled={!unknownSpeakerDraft.trim()}
                onPress={() => closeUnknownSpeakerPrompt(unknownSpeakerDraft.trim())}
              >
                <Text style={styles.modalButtonPrimaryText}>Save</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <View style={styles.card}>
        <Text style={styles.kicker}>Just Speak</Text>
        <Text style={styles.subtitle}>{statusText}</Text>

        <View style={styles.tabBar}>
          <Pressable
            style={[styles.tabButton, activeTab === "record" && styles.tabButtonActive]}
            onPress={() => setActiveTab("record")}
          >
            <Text style={[styles.tabButtonText, activeTab === "record" && styles.tabButtonTextActive]}>
              Record
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tabButton, activeTab === "history" && styles.tabButtonActive]}
            onPress={() => setActiveTab("history")}
          >
            <Text style={[styles.tabButtonText, activeTab === "history" && styles.tabButtonTextActive]}>
              History
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tabButton, activeTab === "speakers" && styles.tabButtonActive]}
            onPress={() => setActiveTab("speakers")}
          >
            <Text style={[styles.tabButtonText, activeTab === "speakers" && styles.tabButtonTextActive]}>
              Speakers
            </Text>
          </Pressable>
        </View>

        {errorText ? <Text style={styles.errorText}>{errorText}</Text> : null}

        {activeTab === "record" ? (
          <View style={styles.tabContent}>
            <Text style={styles.tabIntro}>
              Tap the mic to transcribe with auto speaker ID — or the badge to add a new voice on purpose.
            </Text>

            {enrollTargetName ? (
              <View style={styles.enrollChip}>
                <Ionicons
                  name={isRecording ? "ellipse" : "person"}
                  size={14}
                  color="#a8b9ff"
                />
                <Text style={styles.enrollChipText}>
                  {isRecording ? `Recording sample for ${enrollTargetName}` : `Enroll: ${enrollTargetName}`}
                </Text>
              </View>
            ) : null}

            <View style={styles.recordHeroRow}>
              <View style={styles.heroControlCol}>
                <Pressable
                  style={[styles.enrollOrb, isBusy && styles.controlDisabled]}
                  disabled={isBusy}
                  onPress={openEnrollSpeakerFlow}
                  accessibilityRole="button"
                  accessibilityLabel="Enroll speaker"
                >
                  <Ionicons name="person-add-sharp" size={26} color="#dbe2ff" />
                </Pressable>
              </View>

              <Pressable
                style={[
                  styles.fabPrimary,
                  isRecording && styles.fabRecording,
                  isUploading && styles.fabUploading,
                ]}
                onPress={() => void handleRecordPress()}
                disabled={isUploading}
                accessibilityRole="button"
                accessibilityLabel={isRecording ? "Stop recording" : "Start recording"}
              >
                {isUploading ? (
                  <ActivityIndicator color="#fff" size="large" />
                ) : (
                  <Ionicons
                    name={isRecording ? "stop" : "mic"}
                    size={38}
                    color={isRecording ? "#1a1528" : "#ffffff"}
                  />
                )}
              </Pressable>

              <View style={styles.heroControlCol} />
            </View>
            <Text style={styles.heroCaption}>
              {isUploading
                ? "Working…"
                : isRecording
                  ? "Tap the mic to stop"
                  : "Tap the center to listen"}
            </Text>

            <View style={styles.transcriptPanel}>
              <View style={styles.transcriptPanelHeader}>
                <Text style={styles.transcriptPanelTitle}>Transcript</Text>
                <Pressable
                  style={[
                    styles.clearTranscriptButton,
                    (!canShowTranscript || isBusy) && styles.clearTranscriptButtonDisabled,
                  ]}
                  disabled={!canShowTranscript || isBusy}
                  onPress={clearTranscript}
                >
                  <Text style={styles.clearTranscriptButtonText}>Clear</Text>
                </Pressable>
              </View>
              <Text style={styles.speakerLabel}>Speaker: {lastSpeakerName}</Text>
              <ScrollView style={styles.transcriptScroll} contentContainerStyle={styles.transcriptContent}>
                <Text style={styles.transcriptText}>
                  {canShowTranscript
                    ? transcript
                    : "Your spoken words will appear here once recognition starts."}
                </Text>
              </ScrollView>
            </View>
          </View>
        ) : null}

        {activeTab === "history" ? (
          <View style={[styles.historyPanel, styles.tabPanel]}>
            <View style={styles.historyHeaderRow}>
              <Text style={styles.panelTitle}>Log History</Text>
              <Text style={styles.historyCountText}>
                {history.length} {history.length === 1 ? "entry" : "entries"}
              </Text>
            </View>
            <ScrollView style={styles.historyScroll} contentContainerStyle={styles.historyContent}>
              {history.length === 0 ? (
                <Text style={styles.historyEmptyText}>No transcriptions logged yet.</Text>
              ) : (
                groupedHistory.map(([speakerName, dateGroups]) => (
                  <View key={speakerName} style={styles.historyGroup}>
                    <Pressable
                      style={styles.historyGroupTitleRow}
                      onPress={() => toggleHistorySpeakerExpanded(speakerName)}
                    >
                      <Text style={styles.historyGroupTitle}>
                        {speakerName} ({dateGroups.reduce((sum, [, items]) => sum + items.length, 0)})
                      </Text>
                      <Text style={styles.historyExpandHint}>
                        {expandedHistorySpeakers.includes(speakerName) ? "Hide" : "View"}
                      </Text>
                    </Pressable>
                    {expandedHistorySpeakers.includes(speakerName)
                      ? dateGroups.map(([dateLabel, items]) => {
                          const dateGroupKey = `${speakerName}:${dateLabel}`;
                          const isDateExpanded = expandedHistoryDateGroups.includes(dateGroupKey);

                          return (
                            <View key={dateGroupKey} style={styles.historyDateGroup}>
                              <Pressable
                                style={styles.historyDateGroupTitleRow}
                                onPress={() => toggleHistoryDateExpanded(dateGroupKey)}
                              >
                                <Text style={styles.historyDateGroupTitle}>
                                  {dateLabel} ({items.length})
                                </Text>
                                <Text style={styles.historyExpandHint}>
                                  {isDateExpanded ? "Hide" : "View"}
                                </Text>
                              </Pressable>
                              {isDateExpanded
                                ? items.map((item) => (
                                    <View key={item.id} style={styles.historyItem}>
                                      <Pressable
                                        style={styles.historySpeakerRow}
                                        onPress={() => toggleHistoryItemExpanded(item.id)}
                                      >
                                        <Text style={styles.historySpeaker}>{item.createdAt}</Text>
                                        <Text style={styles.historyExpandHint}>
                                          {expandedHistoryIds.includes(item.id) ? "Hide" : "View"}
                                        </Text>
                                      </Pressable>
                                      {expandedHistoryIds.includes(item.id) ? (
                                        <View style={styles.historyDetailsBox}>
                                          <View style={styles.historyMetaRow}>
                                            <Text style={styles.historyTimestamp}>{item.createdAt}</Text>
                                            <Pressable
                                              style={styles.deleteLogButton}
                                              onPress={() => confirmDeleteHistoryItem(item.id)}
                                            >
                                              <Text style={styles.deleteLogButtonText}>Delete</Text>
                                            </Pressable>
                                          </View>
                                          <Text style={styles.historyAttribution}>
                                            Source: {formatAttributionSource(item.speakerAttributionSource)}
                                            {typeof item.speakerConfidence === "number"
                                              ? ` • Confidence: ${(item.speakerConfidence * 100).toFixed(0)}%`
                                              : ""}
                                          </Text>
                                          {item.speakerNameInput ? (
                                            <Text style={styles.historyAttribution}>
                                              Typed name: {item.speakerNameInput}
                                            </Text>
                                          ) : null}
                                          {item.promptedSpeakerName ? (
                                            <Text style={styles.historyAttribution}>
                                              Prompted name: {item.promptedSpeakerName}
                                            </Text>
                                          ) : null}
                                          <Text style={styles.historyAttribution}>
                                            Matched sample:{" "}
                                            {item.matchedEnrollmentSampleId ||
                                              (item.wasSpeakerNameInputProvided ? "User input" : "None")}
                                          </Text>
                                          <Text style={styles.historySectionLabel}>Voice match</Text>
                                          <Text style={styles.historyVoiceMatchDetail} selectable>
                                            {formatHistoryVoiceMatchDetail(item)}
                                          </Text>
                                          <Text style={styles.historySectionLabel}>Logical path</Text>
                                          <Text style={styles.historyLogicalFlow} selectable>
                                            {formatHistoryLogicalFlow(item)}
                                          </Text>
                                          {item.transcriptionDiagnosticsInitial ||
                                          item.transcriptionDiagnosticsEnrollment ? (
                                            <>
                                              <Text style={styles.historySectionLabel}>Server diagnostics</Text>
                                              <Text style={styles.historyDiagnostics}>
                                                {formatHistoryDiagnosticsText(item)}
                                              </Text>
                                            </>
                                          ) : null}
                                          <Text style={styles.historyText}>{item.text || "(No speech detected)"}</Text>
                                        </View>
                                      ) : null}
                                    </View>
                                  ))
                                : null}
                            </View>
                          );
                        })
                      : null}
                  </View>
                ))
              )}
            </ScrollView>
          </View>
        ) : null}

        {activeTab === "speakers" ? (
          <ScrollView style={styles.speakerTabScroll} contentContainerStyle={styles.speakerTabContent}>
            <View style={styles.speakerActionsRow}>
              <Pressable style={styles.smallActionButton} onPress={() => fetchSpeakers()}>
                <Text style={styles.smallActionButtonText}>Refresh Speakers</Text>
              </Pressable>
            </View>

            <View style={styles.speakerListPanel}>
              <Text style={styles.panelTitle}>Known Speakers</Text>
              <Text style={styles.speakerListHelp}>
                Show Records lists each enrollment. Fingerprint = 12-D audio stats; embedding = ECAPA speaker
                vector when saved.
              </Text>
              {isLoadingSpeakers ? (
                <Text style={styles.historyEmptyText}>Loading speaker profiles...</Text>
              ) : speakers.length === 0 ? (
                <Text style={styles.historyEmptyText}>No enrolled speakers yet.</Text>
              ) : (
                speakers.map((speaker) => {
                  const sampleCount = speaker.enrollmentSamples?.length ?? 0;
                  const isExpanded = expandedSpeakerNames.includes(speaker.name);

                  return (
                    <View key={speaker.name} style={styles.speakerListRow}>
                      <View style={styles.speakerListHeaderRow}>
                        <View style={styles.speakerListRowMain}>
                          <Text style={styles.speakerListName}>
                            {speaker.name} ({speaker.samples} samples)
                          </Text>
                          {speaker.description ? (
                            <Text style={styles.speakerListHint} numberOfLines={2}>
                              {speaker.description}
                            </Text>
                          ) : (
                            <Text style={styles.speakerListHintPlaceholder}>No hint added</Text>
                          )}
                        </View>
                        <View style={styles.speakerListActions}>
                          <Pressable onPress={() => openSpeakerHintModal(speaker)}>
                            <Text style={styles.speakerListEdit}>Edit Hint</Text>
                          </Pressable>
                          <Pressable
                            disabled={sampleCount === 0}
                            onPress={() => toggleSpeakerSamplesExpanded(speaker.name)}
                          >
                            <Text
                              style={[
                                styles.speakerListEdit,
                                sampleCount === 0 && styles.speakerListActionDisabled,
                              ]}
                            >
                              {sampleCount === 0
                                ? "No Records"
                                : isExpanded
                                  ? "Hide Records"
                                  : `Show ${sampleCount} Records`}
                            </Text>
                          </Pressable>
                        </View>
                      </View>

                      {sampleCount > 0 && isExpanded ? (
                        <View style={styles.sampleList}>
                          <View style={styles.profileVectorsSection}>
                            <Text style={styles.profileVectorsHeading}>Profile-level vectors</Text>
                            <Text style={styles.profileVectorsHint}>
                              Averaged values are rolled up from your saved samples on the server. Use for debugging,
                              not for sharing (they identify a voice).
                            </Text>
                            <VectorBlock
                              title={`Averaged fingerprint (${speaker.profileVoiceFingerprintDimensions ?? 12}-D roll-up)`}
                              subtitle={
                                speaker.profileVoiceFingerprintTruncated ? "Truncated in API response" : undefined
                              }
                              csv={formatVectorCsv(speaker.profileVoiceFingerprint)}
                              emptyLabel="No aggregate fingerprint on profile."
                              scrollMaxHeight={120}
                            />
                            {speaker.profileVoiceFingerprintsRecent?.length ? (
                              <Text style={styles.recentVectorsLabel}>Recent roll-up fingerprints (last sessions)</Text>
                            ) : null}
                            {speaker.profileVoiceFingerprintsRecent?.map((row, idx) => (
                              <VectorBlock
                                key={`pfp-${speaker.name}-${idx}`}
                                title={`Rolling fingerprint #${idx + 1} (${row.dimensions ?? row.values.length}-D)`}
                                subtitle={row.truncated ? "Truncated in API response" : undefined}
                                csv={formatVectorCsv(row.values)}
                                emptyLabel=""
                                scrollMaxHeight={96}
                              />
                            ))}
                            {speaker.profileSpeakerEmbedding ? (
                              <VectorBlock
                                title={`Averaged speaker embedding (${speaker.profileSpeakerEmbedding.dimensions ?? speaker.profileSpeakerEmbedding.values.length}-D, neural)`}
                                subtitle={
                                  speaker.profileSpeakerEmbedding.truncated
                                    ? `Showing first ${speaker.profileSpeakerEmbedding.values.length} values; full vector is longer.`
                                    : undefined
                                }
                                csv={formatVectorCsv(speaker.profileSpeakerEmbedding.values)}
                                emptyLabel="No aggregate embedding."
                                scrollMaxHeight={168}
                              />
                            ) : (
                              <Text style={styles.vectorBlockEmpty}>No aggregate speaker embedding on profile yet.</Text>
                            )}
                            {speaker.profileSpeakerEmbeddingsRecent?.length ? (
                              <Text style={styles.recentVectorsLabel}>Recent embedding snapshots</Text>
                            ) : null}
                            {speaker.profileSpeakerEmbeddingsRecent?.map((row, idx) => (
                              <VectorBlock
                                key={`pem-${speaker.name}-${idx}`}
                                title={`Embedding snapshot #${idx + 1} (${row.dimensions ?? row.values.length}-D)`}
                                subtitle={
                                  row.truncated
                                    ? `Showing first ${row.values.length} values; full vector is longer.`
                                    : undefined
                                }
                                csv={formatVectorCsv(row.values)}
                                emptyLabel=""
                                scrollMaxHeight={140}
                              />
                            ))}
                          </View>

                          {speaker.enrollmentSamples?.map((sample) => (
                            <View key={sample.sampleId} style={styles.sampleRow}>
                              <View style={styles.sampleRowMain}>
                                <Text style={styles.sampleTitle}>{formatSampleTime(sample)}</Text>
                                <Text style={styles.sampleMeta}>
                                  {formatSampleSource(sample.source)} • sample {sample.sampleId.slice(0, 8)}…
                                </Text>
                                <VectorBlock
                                  title={`Fingerprint (${sample.voiceFingerprintDimensions ?? 12}-D audio stats)`}
                                  subtitle={
                                    sample.voiceFingerprintTruncated ? "Truncated in API response" : undefined
                                  }
                                  csv={formatVectorCsv(sample.voiceFingerprint)}
                                  emptyLabel={
                                    sample.hasFingerprint === false
                                      ? "No fingerprint stored for this sample."
                                      : "No fingerprint returned (older server data?)."
                                  }
                                  scrollMaxHeight={88}
                                />
                                <VectorBlock
                                  title={`Speaker embedding (${sample.embeddingDimensions ?? "?"}-D, neural)`}
                                  subtitle={
                                    sample.embeddingTruncated
                                      ? `Showing first ${sample.embeddingVector?.length ?? 0} values; full vector is longer.`
                                      : !sample.embeddingVector?.length && sample.hasEmbedding
                                        ? "Embedding exists but was not returned (refresh after app update)."
                                        : undefined
                                  }
                                  csv={formatVectorCsv(sample.embeddingVector)}
                                  emptyLabel={
                                    sample.hasEmbedding === false
                                      ? "No embedding stored for this sample (record with embedding service healthy + named save)."
                                      : "No embedding returned."
                                  }
                                  scrollMaxHeight={112}
                                />
                              </View>
                              <Pressable
                                style={styles.sampleDeleteButton}
                                onPress={() => deleteVoiceSample(speaker.name, sample.sampleId)}
                              >
                                <Text style={styles.sampleDeleteText}>Delete</Text>
                              </Pressable>
                            </View>
                          ))}
                        </View>
                      ) : null}
                    </View>
                  );
                })
              )}
            </View>
          </ScrollView>
        ) : null}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#070b16",
    justifyContent: "flex-start",
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  card: {
    borderRadius: 24,
    backgroundColor: "#12182e",
    padding: 22,
    borderWidth: 1,
    borderColor: "rgba(110, 132, 255, 0.22)",
    shadowColor: "#3d4fd9",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.35,
    shadowRadius: 28,
    elevation: 12,
  },
  kicker: {
    color: "#f4f6ff",
    fontSize: 32,
    fontWeight: "800",
    letterSpacing: -0.6,
    lineHeight: 38,
    marginBottom: 6,
  },
  subtitle: {
    color: "#98a8d4",
    marginTop: 6,
    marginBottom: 16,
    fontSize: 14,
    lineHeight: 20,
  },
  tabBar: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 14,
    borderRadius: 14,
    backgroundColor: "#0f152b",
    borderWidth: 1,
    borderColor: "#202c55",
    padding: 4,
  },
  tabButton: {
    flex: 1,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 9,
  },
  tabButtonActive: {
    backgroundColor: "#5f6fff",
    shadowColor: "#5f6fff",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.35,
    shadowRadius: 6,
    elevation: 4,
  },
  tabButtonText: {
    color: "#9ba8d8",
    fontSize: 12,
    fontWeight: "700",
  },
  tabButtonTextActive: {
    color: "#ffffff",
  },
  tabContent: {
    marginTop: 2,
  },
  speakerTabScroll: {
    marginTop: 2,
    maxHeight: 560,
  },
  speakerTabContent: {
    paddingBottom: 18,
  },
  tabIntro: {
    color: "#8898cc",
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 14,
    textAlign: "center",
  },
  tabPanel: {
    marginTop: 2,
  },
  speakerActionsRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 10,
  },
  smallActionButton: {
    flex: 1,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#4e5fa9",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 8,
  },
  smallActionButtonText: {
    color: "#d7deff",
    fontSize: 12,
    fontWeight: "600",
  },
  recordHeroRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 8,
    marginBottom: 6,
  },
  heroControlCol: {
    width: 68,
    alignItems: "center",
    justifyContent: "center",
  },
  enrollOrb: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: "rgba(42, 58, 120, 0.85)",
    borderWidth: 1,
    borderColor: "rgba(124, 148, 255, 0.45)",
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#4d62ff",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  fabPrimary: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: "#5f6fff",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
    shadowColor: "#6f7cff",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.55,
    shadowRadius: 16,
    elevation: 10,
  },
  fabRecording: {
    backgroundColor: "#ff5c7a",
    borderColor: "rgba(255,255,255,0.25)",
    shadowColor: "#ff3d6a",
  },
  fabUploading: {
    backgroundColor: "#4a5588",
    opacity: 0.9,
  },
  controlDisabled: {
    opacity: 0.45,
  },
  heroCaption: {
    textAlign: "center",
    color: "#7a88ba",
    fontSize: 12,
    fontWeight: "600",
    marginBottom: 8,
    letterSpacing: 0.2,
  },
  enrollChip: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "center",
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: "rgba(80, 102, 200, 0.2)",
    borderWidth: 1,
    borderColor: "rgba(120, 142, 255, 0.35)",
    marginBottom: 12,
  },
  enrollChipText: {
    color: "#c3ceff",
    fontSize: 12,
    fontWeight: "700",
  },
  historyToggleButton: {
    marginTop: 2,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#4e5fa9",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
  },
  historyToggleButtonText: {
    color: "#d7deff",
    fontSize: 14,
    fontWeight: "600",
  },
  errorText: {
    color: "#ff98a6",
    fontSize: 13,
    marginTop: 6,
    marginBottom: 10,
  },
  transcriptPanel: {
    marginTop: 14,
    borderRadius: 14,
    backgroundColor: "#0f152b",
    borderWidth: 1,
    borderColor: "#202c55",
    minHeight: 220,
  },
  transcriptPanelHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginHorizontal: 14,
    marginTop: 14,
  },
  transcriptPanelTitle: {
    color: "#aeb8df",
    fontSize: 13,
    fontWeight: "600",
  },
  clearTranscriptButton: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#5a6bb5",
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "#1a2548",
  },
  clearTranscriptButtonDisabled: {
    opacity: 0.4,
  },
  clearTranscriptButtonText: {
    color: "#c9d4ff",
    fontSize: 12,
    fontWeight: "600",
  },
  panelTitle: {
    color: "#aeb8df",
    fontSize: 13,
    fontWeight: "600",
    marginHorizontal: 14,
    marginTop: 14,
  },
  speakerLabel: {
    color: "#93a4de",
    fontSize: 12,
    marginHorizontal: 14,
    marginTop: 8,
  },
  transcriptScroll: {
    marginTop: 8,
    maxHeight: 210,
  },
  transcriptContent: {
    paddingHorizontal: 14,
    paddingBottom: 16,
  },
  transcriptText: {
    color: "#eaf0ff",
    fontSize: 16,
    lineHeight: 24,
  },
  historyPanel: {
    marginTop: 14,
    borderRadius: 14,
    backgroundColor: "#0f152b",
    borderWidth: 1,
    borderColor: "#202c55",
    minHeight: 360,
    maxHeight: 560,
  },
  historyScroll: {
    marginTop: 8,
  },
  historyContent: {
    paddingHorizontal: 12,
    paddingBottom: 16,
  },
  historyHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginRight: 14,
  },
  historyCountText: {
    color: "#8fa0d8",
    fontSize: 12,
    marginTop: 14,
  },
  historyItem: {
    marginBottom: 10,
    borderRadius: 12,
    backgroundColor: "#131c39",
    borderWidth: 1,
    borderColor: "#253264",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  historyGroup: {
    marginBottom: 10,
  },
  historyGroupTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 6,
  },
  historyGroupTitle: {
    color: "#b9c7f7",
    fontSize: 12,
    fontWeight: "700",
    marginLeft: 2,
  },
  historyDateGroup: {
    marginBottom: 8,
    paddingLeft: 8,
  },
  historyDateGroupTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 6,
  },
  historyDateGroupTitle: {
    color: "#8ea1df",
    fontSize: 11,
    fontWeight: "700",
    marginLeft: 2,
  },
  historyMetaRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
    gap: 10,
  },
  historySpeakerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  historyTimestamp: {
    color: "#8ea1df",
    fontSize: 11,
  },
  historySpeaker: {
    color: "#d5deff",
    fontSize: 12,
    fontWeight: "600",
  },
  historyExpandHint: {
    color: "#94a5de",
    fontSize: 11,
    fontWeight: "600",
  },
  historyDetailsBox: {
    marginTop: 8,
    borderTopWidth: 1,
    borderTopColor: "#2a3768",
    paddingTop: 8,
  },
  historyText: {
    color: "#eaf0ff",
    fontSize: 14,
    lineHeight: 20,
  },
  historyAttribution: {
    color: "#8ea1df",
    fontSize: 11,
    lineHeight: 16,
    marginBottom: 4,
  },
  historySectionLabel: {
    color: "#aebfff",
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    marginTop: 6,
    marginBottom: 4,
  },
  historyVoiceMatchDetail: {
    color: "#e2eaff",
    fontSize: 11,
    lineHeight: 17,
    marginBottom: 4,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    backgroundColor: "#0f1628",
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: "#2a5080",
  },
  historyLogicalFlow: {
    color: "#c5d5ff",
    fontSize: 11,
    lineHeight: 17,
    marginTop: 8,
    marginBottom: 6,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    backgroundColor: "#12182e",
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: "#2c3a6e",
  },
  historyDiagnostics: {
    color: "#b8ccff",
    fontSize: 10,
    lineHeight: 14,
    marginTop: 6,
    marginBottom: 8,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    backgroundColor: "#0a0f22",
    borderRadius: 8,
    padding: 8,
    borderWidth: 1,
    borderColor: "#202c55",
  },
  deleteLogButton: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#7b3f53",
    backgroundColor: "#2b1520",
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  deleteLogButtonText: {
    color: "#ff9bae",
    fontSize: 11,
    fontWeight: "700",
  },
  historyEmptyText: {
    color: "#9ba8d8",
    fontSize: 14,
  },
  speakerListPanel: {
    marginTop: 14,
    borderRadius: 14,
    backgroundColor: "#0f152b",
    borderWidth: 1,
    borderColor: "#202c55",
    paddingBottom: 12,
  },
  speakerListHelp: {
    color: "#8b98c9",
    fontSize: 11,
    lineHeight: 16,
    marginHorizontal: 14,
    marginTop: 4,
    marginBottom: 6,
  },
  speakerListRow: {
    marginHorizontal: 10,
    marginTop: 8,
    paddingHorizontal: 10,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: "#131c39",
    borderWidth: 1,
    borderColor: "#253264",
    gap: 10,
  },
  speakerListHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  speakerListRowMain: {
    flex: 1,
    minWidth: 0,
  },
  speakerListActions: {
    alignItems: "flex-end",
    gap: 8,
  },
  speakerListName: {
    color: "#e8edff",
    fontSize: 13,
    fontWeight: "600",
  },
  speakerListHint: {
    color: "#9aaad8",
    fontSize: 12,
    marginTop: 4,
    lineHeight: 17,
  },
  speakerListHintPlaceholder: {
    color: "#6b769e",
    fontSize: 12,
    marginTop: 4,
    fontStyle: "italic",
  },
  speakerListEdit: {
    color: "#8ea1ff",
    fontSize: 12,
    fontWeight: "700",
  },
  speakerListActionDisabled: {
    color: "#5f6b93",
  },
  sampleList: {
    marginTop: 10,
    borderTopWidth: 1,
    borderTopColor: "#253264",
    paddingTop: 8,
    gap: 8,
  },
  profileVectorsSection: {
    marginBottom: 4,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#253264",
  },
  profileVectorsHeading: {
    color: "#e8ecff",
    fontSize: 12,
    fontWeight: "700",
  },
  profileVectorsHint: {
    color: "#7580ab",
    fontSize: 10,
    marginTop: 4,
    marginBottom: 6,
    lineHeight: 14,
  },
  recentVectorsLabel: {
    color: "#8ea1df",
    fontSize: 11,
    fontWeight: "600",
    marginTop: 10,
  },
  vectorBlock: {
    marginTop: 8,
  },
  vectorBlockTitle: {
    color: "#8ea1ff",
    fontSize: 11,
    fontWeight: "600",
  },
  vectorBlockSubtitle: {
    color: "#c9a227",
    fontSize: 10,
    marginTop: 2,
  },
  vectorBlockEmpty: {
    color: "#5c6a8e",
    fontSize: 10,
    fontStyle: "italic",
    marginTop: 4,
  },
  vectorScroll: {
    marginTop: 4,
  },
  vectorMono: {
    color: "#dce4ff",
    fontSize: 9,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    lineHeight: 13,
  },
  sampleRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 10,
  },
  sampleRowMain: {
    flex: 1,
    minWidth: 0,
  },
  sampleTitle: {
    color: "#d5deff",
    fontSize: 12,
    fontWeight: "600",
  },
  sampleMeta: {
    color: "#8393c8",
    fontSize: 11,
    marginTop: 2,
  },
  sampleDeleteButton: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#7b3f53",
    backgroundColor: "#2b1520",
    paddingHorizontal: 9,
    paddingVertical: 5,
  },
  sampleDeleteText: {
    color: "#ff9bae",
    fontSize: 11,
    fontWeight: "700",
  },
  modalBackdrop: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 20,
    backgroundColor: "rgba(6, 8, 18, 0.72)",
  },
  modalCard: {
    borderRadius: 18,
    backgroundColor: "#1a2344",
    borderWidth: 1,
    borderColor: "#2c3b72",
    padding: 18,
    zIndex: 1,
  },
  modalTitle: {
    color: "#f0f4ff",
    fontSize: 17,
    fontWeight: "700",
  },
  modalSubtitle: {
    color: "#9aaad8",
    fontSize: 12,
    marginTop: 6,
    lineHeight: 17,
  },
  modalInputSingleLine: {
    minHeight: 48,
    maxHeight: 48,
    paddingVertical: 12,
  },
  modalButtonDisabled: {
    opacity: 0.45,
  },
  modalInput: {
    marginTop: 14,
    minHeight: 100,
    maxHeight: Platform.OS === "web" ? 180 : 140,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#3c4d8d",
    backgroundColor: "#101834",
    color: "#f0f4ff",
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    textAlignVertical: "top",
  },
  modalCharCount: {
    color: "#6b769e",
    fontSize: 11,
    marginTop: 6,
    textAlign: "right",
  },
  clearHintButton: {
    alignSelf: "flex-start",
    marginTop: 10,
    paddingVertical: 6,
    paddingHorizontal: 4,
  },
  clearHintButtonText: {
    color: "#ff9bae",
    fontSize: 13,
    fontWeight: "600",
    textDecorationLine: "underline",
  },
  modalActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 10,
    marginTop: 16,
  },
  modalButton: {
    borderRadius: 10,
    paddingHorizontal: 18,
    paddingVertical: 10,
    minWidth: 96,
    alignItems: "center",
    justifyContent: "center",
  },
  modalButtonSecondary: {
    borderWidth: 1,
    borderColor: "#4e5fa9",
    backgroundColor: "transparent",
  },
  modalButtonSecondaryText: {
    color: "#c8d2ff",
    fontSize: 14,
    fontWeight: "600",
  },
  modalButtonPrimary: {
    backgroundColor: "#6a7cff",
  },
  modalButtonPrimaryText: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "700",
  },
});
