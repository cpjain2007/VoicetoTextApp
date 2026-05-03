import { StatusBar } from "expo-status-bar";
import Ionicons from "@expo/vector-icons/Ionicons";
import { LinearGradient } from "expo-linear-gradient";
import * as Location from "expo-location";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Audio } from "expo-av";
import * as Speech from "expo-speech";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { EncodingType, readAsStringAsync } from "expo-file-system/legacy";
import {
  ActivityIndicator,
  Alert,
  AppState,
  AppStateStatus,
  Image,
  Linking,
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

import {
  ensureContactsPermission,
  isContactsApiAvailable,
  searchContactsWithPhones,
  type ContactPhoneRow,
} from "./contactLookup";
import { buildWebSearchQueryForEntry, suggestContactSearchFromTranscript, autoContactSearchQuery } from "./lookupHelpers";
import { openGoogleSearch } from "./webSearch";

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

/** Typed next step from the model (for future automation); use confidence + fallback for safe UX. */
type TranscriptStructuredAction = {
  type: "GENERIC" | "CALL_CONTACT" | "CREATE_TASK" | "ADD_SUBTASK" | "CHECK_CALENDAR";
  label: string;
  detail?: string;
  confidence: number;
  fallback: string;
};

type CalendarIntentFallback = {
  type: string;
  message: string;
};

type CalendarEventParameters = {
  title: string;
  location: string;
  start_time: string | null;
  end_time: string | null;
  notes: string;
  participants: string[];
};

/** Single calendar event intent from the model (matches API `calendarIntent`). */
type TranscriptCalendarIntent = {
  action: "create_event";
  parameters: CalendarEventParameters;
  confidence: number;
  fallback: CalendarIntentFallback;
};

type TranscriptAiInsights = {
  summary: string;
  actionItems: string[];
  topics: string[];
  /** Short questions the app may read aloud so the user can answer by voice (e.g. missing address). */
  followUpQuestions?: string[];
  /** Structured actions with confidence and fallback hints (from API when model supports them). */
  actions?: TranscriptStructuredAction[];
  /** When the transcript implies one concrete calendar event (e.g. visit at a time and place). */
  calendarIntent?: TranscriptCalendarIntent;
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
  /** AI summary, action items, and topics (from transcribe or cloud history). */
  ai?: TranscriptAiInsights | null;
  /** When set, this clip was recorded as a spoken answer to an AI voice follow-up question. */
  answeredVoiceFollowUp?: string | null;
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
  /** From /transcribe-base64 when the API has OPENAI_API_KEY. */
  ai?: TranscriptAiInsights | null;
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

type AppTab = "record" | "history" | "speakers" | "ai";

type AiSpeakerRollup = {
  speakerName: string;
  entryCount: number;
  topics: Array<{ display: string; count: number }>;
  actionItems: string[];
  summaries: Array<{ text: string; createdAt: string }>;
};

type AiInsightsOverview = {
  totalEntries: number;
  entriesWithAi: number;
  allTopics: Array<{ display: string; count: number }>;
  actionItems: Array<{ text: string; speaker: string; createdAt: string }>;
  bySpeaker: AiSpeakerRollup[];
  voiceFollowUps: Array<{ question: string; speaker: string; createdAt: string; historyId: string }>;
};

const HISTORY_STORAGE_KEY = "voicetotext.history.v1";
const UNKNOWN_SPEAKER_LABEL = "Unknown speaker";
const UNKNOWN_SPEAKER_NAME_MAX_CHARS = 80;

/** Matches default `ASSEMBLYAI_SPEAKER_DESCRIPTION_MAX` on the API. */
const SPEAKER_DESCRIPTION_MAX_CHARS = 220;

/** Origin for `/history`, `/transcribe-base64`, etc. No trailing slash; strips optional `…/transcribe`. */
const normalizeTranscribeApiBase = (apiUrl: string) =>
  apiUrl.trim().replace(/\/transcribe\/?$/i, "").replace(/\/+$/, "");

const expoTranscribeBearerToken = () => {
  const raw = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
  if (typeof raw !== "string") {
    return undefined;
  }
  const t = raw.trim();
  return t.length > 0 ? t : undefined;
};

/** E.164-style tel URI and a readable display label. */
const phoneMatchToDisplayAndTel = (raw: string): { display: string; tel: string } | null => {
  const trimmed = raw.trim();
  const d = trimmed.replace(/\D/g, "");
  if (d.length < 10 || d.length > 15) {
    return null;
  }
  if (d.length === 10) {
    return {
      display: `(${d.slice(0, 3)}) ${d.slice(3, 6)}-${d.slice(6)}`,
      tel: `tel:+1${d}`,
    };
  }
  if (d.length === 11 && d.startsWith("1")) {
    const n = d.slice(1);
    return {
      display: `(${n.slice(0, 3)}) ${n.slice(3, 6)}-${n.slice(6)}`,
      tel: `tel:+${d}`,
    };
  }
  return { display: trimmed, tel: `tel:+${d}` };
};

const extractPhoneNumbersFromText = (text: string): Array<{ display: string; tel: string }> => {
  const out: Array<{ display: string; tel: string }> = [];
  const seen = new Set<string>();
  if (!text?.trim()) {
    return out;
  }
  const intlRe = /\+[1-9]\d{7,14}\b/g;
  const usRe =
    /\b(?:\+1[-.\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b/g;
  const rawMatches: string[] = [];
  for (const re of [intlRe, usRe]) {
    const r = new RegExp(re.source, "g");
    let m: RegExpExecArray | null;
    while ((m = r.exec(text)) !== null) {
      rawMatches.push(m[0]);
    }
  }
  for (const raw of rawMatches) {
    const parsed = phoneMatchToDisplayAndTel(raw);
    if (!parsed || seen.has(parsed.tel)) {
      continue;
    }
    seen.add(parsed.tel);
    out.push(parsed);
  }
  return out;
};

const collectPhonesFromHistoryItem = (item: TranscriptLogItem): Array<{ display: string; tel: string }> => {
  const parts: string[] = [item.text];
  if (item.ai?.summary) {
    parts.push(item.ai.summary);
  }
  if (item.ai?.actionItems?.length) {
    parts.push(item.ai.actionItems.join("\n"));
  }
  if (item.ai?.actions?.length) {
    for (const a of item.ai.actions) {
      parts.push([a.label, a.detail, a.fallback].filter(Boolean).join("\n"));
    }
  }
  if (item.ai?.calendarIntent) {
    const ci = item.ai.calendarIntent;
    parts.push(
      [ci.parameters.title, ci.parameters.location, ci.parameters.notes, ci.parameters.participants.join(", ")].join(
        "\n",
      ),
    );
  }
  if (item.ai?.followUpQuestions?.length) {
    parts.push(item.ai.followUpQuestions.join("\n"));
  }
  if (item.answeredVoiceFollowUp) {
    parts.push(item.answeredVoiceFollowUp);
  }
  return extractPhoneNumbersFromText(parts.join("\n"));
};

const confirmAndPlacePhoneCall = (displayNumber: string, telUri: string) => {
  if (Platform.OS === "web") {
    Alert.alert("Not available", "Phone calls are not supported in this web view.");
    return;
  }
  Alert.alert("Place call?", `Call ${displayNumber}?`, [
    { text: "Cancel", style: "cancel" },
    {
      text: "Call",
      onPress: () => {
        void Linking.openURL(telUri).catch(() => {
          Alert.alert("Could not start call", "Your device may not support dialing from this app. Try copying the number.");
        });
      },
    },
  ]);
};

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

const pickFirstFollowUpQuestion = (ai: TranscriptAiInsights | null | undefined) => {
  const list = ai?.followUpQuestions;
  if (!Array.isArray(list)) {
    return null;
  }
  const q = list.map((s) => s.trim()).find((s) => s.length > 0);
  return q || null;
};

const parseTranscriptCalendarIntent = (raw: unknown): TranscriptCalendarIntent | undefined => {
  if (!raw || typeof raw !== "object") {
    return undefined;
  }
  const o = raw as Record<string, unknown>;
  if (o.action !== "create_event") {
    return undefined;
  }
  const p = o.parameters && typeof o.parameters === "object" ? (o.parameters as Record<string, unknown>) : {};
  const str = (v: unknown) => (typeof v === "string" ? v.trim() : "");
  const nullableTime = (v: unknown): string | null => {
    if (v === null || v === undefined) {
      return null;
    }
    if (typeof v === "string") {
      const t = v.trim();
      return t.length ? t : null;
    }
    return null;
  };
  const participants = Array.isArray(p.participants)
    ? p.participants
        .filter((item): item is string => typeof item === "string")
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, 50)
    : [];
  const notes =
    p.notes !== undefined && p.notes !== null && typeof p.notes === "string" ? p.notes.trim() : "";
  const parameters: CalendarEventParameters = {
    title: str(p.title),
    location: str(p.location),
    start_time: nullableTime(p.start_time),
    end_time: nullableTime(p.end_time),
    notes,
    participants,
  };
  const fbRaw = o.fallback && typeof o.fallback === "object" ? (o.fallback as Record<string, unknown>) : {};
  const fbType = typeof fbRaw.type === "string" ? fbRaw.type.trim().slice(0, 48) : "none";
  const message = typeof fbRaw.message === "string" ? fbRaw.message.trim() : "";
  let confidence = Number(o.confidence);
  if (!Number.isFinite(confidence)) {
    confidence = 0;
  }
  confidence = Math.min(1, Math.max(0, confidence));
  const hasSignal =
    parameters.title.length > 0 ||
    !!parameters.start_time ||
    parameters.location.length > 0 ||
    parameters.notes.length > 0 ||
    parameters.participants.length > 0;
  if (!hasSignal) {
    return undefined;
  }
  return {
    action: "create_event",
    parameters,
    confidence,
    fallback: { type: fbType || "none", message },
  };
};

const parseTranscriptAi = (raw: unknown): TranscriptAiInsights | undefined => {
  if (!raw || typeof raw !== "object") {
    return undefined;
  }
  const o = raw as Record<string, unknown>;
  const summary = typeof o.summary === "string" ? o.summary.trim() : "";
  const actionItems = Array.isArray(o.actionItems)
    ? o.actionItems
        .filter((item): item is string => typeof item === "string")
        .map((s) => s.trim())
        .filter(Boolean)
    : [];
  const topics = Array.isArray(o.topics)
    ? o.topics
        .filter((item): item is string => typeof item === "string")
        .map((s) => s.trim())
        .filter(Boolean)
    : [];
  const followUpQuestions = Array.isArray(o.followUpQuestions)
    ? o.followUpQuestions
        .filter((item): item is string => typeof item === "string")
        .map((s) => s.trim())
        .filter(Boolean)
    : [];
  const allowedActionTypes = new Set<TranscriptStructuredAction["type"]>([
    "GENERIC",
    "CALL_CONTACT",
    "CREATE_TASK",
    "ADD_SUBTASK",
    "CHECK_CALENDAR",
  ]);
  const actions: TranscriptStructuredAction[] | undefined = (() => {
    if (!Array.isArray(o.actions)) {
      return undefined;
    }
    const rows = o.actions
      .filter((row): row is Record<string, unknown> => !!row && typeof row === "object")
      .map((row) => {
        const rawType =
          typeof row.type === "string" ? row.type.trim().toUpperCase() : "";
        const type = allowedActionTypes.has(rawType as TranscriptStructuredAction["type"])
          ? (rawType as TranscriptStructuredAction["type"])
          : "GENERIC";
        const label = typeof row.label === "string" ? row.label.trim() : "";
        const detailRaw = typeof row.detail === "string" ? row.detail.trim() : "";
        const fallbackRaw = typeof row.fallback === "string" ? row.fallback.trim() : "";
        let confidence = Number(row.confidence);
        if (!Number.isFinite(confidence)) {
          confidence = 0;
        }
        confidence = Math.min(1, Math.max(0, confidence));
        if (!label) {
          return null;
        }
        const fallback =
          fallbackRaw || "Show this suggestion for manual follow-up.";
        const out: TranscriptStructuredAction = {
          type,
          label,
          confidence,
          fallback,
        };
        if (detailRaw) {
          out.detail = detailRaw;
        }
        return out;
      })
      .filter((x): x is TranscriptStructuredAction => x !== null);
    return rows.length > 0 ? rows : undefined;
  })();
  const calendarIntent = parseTranscriptCalendarIntent(o.calendarIntent);
  if (
    !summary &&
    actionItems.length === 0 &&
    topics.length === 0 &&
    followUpQuestions.length === 0 &&
    !actions?.length &&
    !calendarIntent
  ) {
    return undefined;
  }
  return {
    summary,
    actionItems,
    topics,
    ...(followUpQuestions.length > 0 ? { followUpQuestions } : {}),
    ...(actions?.length ? { actions } : {}),
    ...(calendarIntent ? { calendarIntent } : {}),
  };
};

/** Stable key order for reviewing the same structured object the API parses from the model. */
const formatHistoryAiModelOutputJson = (ai: TranscriptAiInsights) =>
  JSON.stringify(
    {
      calendarIntent: ai.calendarIntent ?? null,
      actions: ai.actions ?? [],
      actionItems: ai.actionItems ?? [],
      topics: ai.topics ?? [],
      followUpQuestions: ai.followUpQuestions ?? [],
      summary: ai.summary ?? "",
    },
    null,
    2,
  );

const transcriptHasAiPayload = (item: TranscriptLogItem) => {
  const ai = item.ai;
  if (!ai) {
    return false;
  }
  return (
    !!ai.summary.trim() ||
    (ai.actionItems?.length ?? 0) > 0 ||
    (ai.topics?.length ?? 0) > 0 ||
    (ai.followUpQuestions?.length ?? 0) > 0 ||
    (ai.actions?.length ?? 0) > 0 ||
    !!ai.calendarIntent
  );
};

const normalizeTopicKey = (raw: string) => raw.trim().toLowerCase().replace(/\s+/g, " ");

const mergeTopicCounts = (
  target: Map<string, { display: string; count: number }>,
  topics: string[] | undefined,
) => {
  for (const t of topics ?? []) {
    const key = normalizeTopicKey(t);
    if (!key) {
      continue;
    }
    const prev = target.get(key);
    if (prev) {
      prev.count += 1;
    } else {
      target.set(key, { display: t.trim(), count: 1 });
    }
  }
};

const buildAiInsightsOverview = (history: TranscriptLogItem[]): AiInsightsOverview => {
  const sorted = [...history].sort((a, b) => b.createdAtMs - a.createdAtMs);
  const withAi = sorted.filter(transcriptHasAiPayload);

  const globalTopics = new Map<string, { display: string; count: number }>();
  for (const item of withAi) {
    mergeTopicCounts(globalTopics, item.ai?.topics);
  }
  const allTopics = [...globalTopics.values()].sort((a, b) => b.count - a.count);

  const actionRows: AiInsightsOverview["actionItems"] = [];
  const seenAction = new Set<string>();
  for (const item of sorted) {
    if (!transcriptHasAiPayload(item)) {
      continue;
    }
    for (const line of item.ai?.actionItems ?? []) {
      const key = line.trim().toLowerCase();
      if (!key || seenAction.has(key)) {
        continue;
      }
      seenAction.add(key);
      actionRows.push({
        text: line.trim(),
        speaker: item.speakerName?.trim() || UNKNOWN_SPEAKER_LABEL,
        createdAt: item.createdAt,
      });
      if (actionRows.length >= 48) {
        break;
      }
    }
    if (actionRows.length >= 48) {
      break;
    }
  }

  const voiceFollowUps: AiInsightsOverview["voiceFollowUps"] = [];
  for (const item of sorted) {
    if (!transcriptHasAiPayload(item)) {
      continue;
    }
    const speaker = item.speakerName?.trim() || UNKNOWN_SPEAKER_LABEL;
    for (const q of item.ai?.followUpQuestions ?? []) {
      const question = q.trim();
      if (!question) {
        continue;
      }
      voiceFollowUps.push({
        question,
        speaker,
        createdAt: item.createdAt,
        historyId: item.id,
      });
      if (voiceFollowUps.length >= 24) {
        break;
      }
    }
    if (voiceFollowUps.length >= 24) {
      break;
    }
  }

  const bySpeakerMap = new Map<string, TranscriptLogItem[]>();
  for (const item of withAi) {
    const name = item.speakerName?.trim() || UNKNOWN_SPEAKER_LABEL;
    const list = bySpeakerMap.get(name) ?? [];
    list.push(item);
    bySpeakerMap.set(name, list);
  }

  const bySpeaker: AiSpeakerRollup[] = [...bySpeakerMap.entries()]
    .map(([speakerName, items]) => {
      items.sort((a, b) => b.createdAtMs - a.createdAtMs);
      const topics = new Map<string, { display: string; count: number }>();
      const actions: string[] = [];
      const seenLocal = new Set<string>();
      for (const it of items) {
        mergeTopicCounts(topics, it.ai?.topics);
        for (const a of it.ai?.actionItems ?? []) {
          const k = a.trim().toLowerCase();
          if (k && !seenLocal.has(k)) {
            seenLocal.add(k);
            actions.push(a.trim());
          }
        }
      }
      const topicList = [...topics.values()].sort((a, b) => b.count - a.count);
      const summaries = items
        .filter((it) => it.ai?.summary?.trim())
        .slice(0, 5)
        .map((it) => ({ text: it.ai!.summary.trim(), createdAt: it.createdAt }));

      return {
        speakerName,
        entryCount: items.length,
        topics: topicList,
        actionItems: actions.slice(0, 20),
        summaries,
      };
    })
    .sort((a, b) => b.entryCount - a.entryCount);

  return {
    totalEntries: history.length,
    entriesWithAi: withAi.length,
    allTopics,
    actionItems: actionRows,
    bySpeaker,
    voiceFollowUps,
  };
};

const MAX_PERSON_SUMMARY_BUNDLE_CHARS = 11000;
const MAX_TODAY_PLAN_BUNDLE_CHARS = 11000;
const TRAFFIC_POLL_INTERVAL_MS = 4 * 60 * 1000;

const getLocalDayBoundsMs = () => {
  const now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
  const end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59, 999);
  const label = now.toLocaleDateString(undefined, {
    weekday: "long",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
  return { startMs: start.getTime(), endMs: end.getTime(), label };
};

const formatLogRowForPlanBundle = (row: TranscriptLogItem) => {
  const lines: string[] = [`[${row.createdAt}] ${row.speakerName}:`];
  if (row.text?.trim()) {
    const t = row.text.trim();
    lines.push(t.slice(0, 900) + (t.length > 900 ? "…" : ""));
  }
  if (row.ai?.summary?.trim()) {
    const s = row.ai.summary.trim();
    lines.push(`AI summary: ${s.slice(0, 450)}${s.length > 450 ? "…" : ""}`);
  }
  const topics = row.ai?.topics?.filter((x) => x.trim()) ?? [];
  if (topics.length) {
    lines.push(`Topics: ${topics.join(", ")}`);
  }
  const acts = row.ai?.actionItems?.filter((x) => x.trim()) ?? [];
  if (acts.length) {
    lines.push(`Tasks: ${acts.join("; ")}`);
  }
  return lines.join("\n");
};

/** Today's clips for the speaker + other recent log context for POST /ai/speaker-today-plan. */
const buildTodayPlanBundle = (
  speakerName: string,
  history: TranscriptLogItem[],
): { bundle: string | null; dateLabel: string } => {
  const { startMs, endMs, label } = getLocalDayBoundsMs();
  const key = normalizeSpeakerKey(speakerName);
  if (!key) {
    return { bundle: null, dateLabel: label };
  }
  const sameSpeaker = history.filter(
    (h) => normalizeSpeakerKey((h.speakerName || "").trim() || UNKNOWN_SPEAKER_LABEL) === key,
  );
  const todayRows = sameSpeaker
    .filter((h) => h.createdAtMs >= startMs && h.createdAtMs <= endMs)
    .sort((a, b) => b.createdAtMs - a.createdAtMs);
  if (todayRows.length === 0) {
    return { bundle: null, dateLabel: label };
  }
  const todayIds = new Set(todayRows.map((r) => r.id));
  const contextRows = history.filter((h) => !todayIds.has(h.id)).slice(0, 30);
  const s1 = todayRows.map((r) => formatLogRowForPlanBundle(r)).join("\n\n—\n\n");
  const s2 = contextRows.map((r) => formatLogRowForPlanBundle(r)).join("\n\n—\n\n");
  let bundle = `SECTION 1 — ${speakerName} on ${label} (today, user's device calendar):\n\n${s1}\n\nSECTION 2 — Other recent app-wide voice logs:\n\n${s2.trim() || "(none)"}`;
  if (bundle.length > MAX_TODAY_PLAN_BUNDLE_CHARS) {
    bundle = `${bundle.slice(0, MAX_TODAY_PLAN_BUNDLE_CHARS)}\n…`;
  }
  return { bundle, dateLabel: label };
};

const buildRecentLogsForDestinationExtract = (history: TranscriptLogItem[]) => {
  return [...history]
    .sort((a, b) => b.createdAtMs - a.createdAtMs)
    .slice(0, 20)
    .map((h) => {
      const tx = h.text?.trim() || "";
      const ai = h.ai?.summary?.trim() || "";
      return `${h.speakerName} (${h.createdAt})\nSaid: ${tx.slice(0, 1200)}${tx.length > 1200 ? "…" : ""}\nAI notes: ${ai.slice(0, 400)}`;
    })
    .join("\n\n---\n\n");
};

/** ~60 days of logs for POST /ai/week-tasks (upcoming week planning). */
const MS_DAY = 24 * 60 * 60 * 1000;

const getUpcomingWeekWindowLabel = () => {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const end = new Date(start.getTime() + 6 * MS_DAY);
  end.setHours(23, 59, 59, 999);
  const opts: Intl.DateTimeFormatOptions = { weekday: "short", month: "short", day: "numeric", year: "numeric" };
  return `${start.toLocaleDateString(undefined, opts)} through ${end.toLocaleDateString(undefined, opts)} (local, 7 days inclusive)`;
};

const buildWeekTasksBundle = (history: TranscriptLogItem[]): string | null => {
  if (history.length === 0) {
    return null;
  }
  const sorted = [...history].sort((a, b) => b.createdAtMs - a.createdAtMs);
  const cutoff = Date.now() - 60 * MS_DAY;
  const rows = sorted.filter((h) => h.createdAtMs >= cutoff).slice(0, 80);
  if (rows.length === 0) {
    return null;
  }
  const blocks = rows.map((h) => {
    const dateStr = new Date(h.createdAtMs).toLocaleString();
    const lines: string[] = [`Speaker: ${h.speakerName} | Log time: ${h.createdAt} | ${dateStr}`];
    if (h.text?.trim()) {
      const t = h.text.trim();
      lines.push(`Transcript: ${t.slice(0, 1500)}${t.length > 1500 ? "…" : ""}`);
    }
    const acts = h.ai?.actionItems?.filter((x) => x.trim()) ?? [];
    if (acts.length) {
      lines.push(`AI action items: ${acts.join(" | ")}`);
    }
    if (h.ai?.summary?.trim()) {
      const s = h.ai.summary.trim();
      lines.push(`AI summary: ${s.slice(0, 500)}${s.length > 500 ? "…" : ""}`);
    }
    const topics = h.ai?.topics?.filter((x) => x.trim()) ?? [];
    if (topics.length) {
      lines.push(`Topics: ${topics.join(", ")}`);
    }
    return lines.join("\n");
  });
  return blocks.join("\n\n—\n\n");
};

/** Concatenate saved clips for one speaker for POST /ai/person-summary. */
const buildPersonSummaryBundle = (speakerName: string, history: TranscriptLogItem[]): string | null => {
  const key = normalizeSpeakerKey(speakerName);
  if (!key) {
    return null;
  }
  const rows = history
    .filter((h) => normalizeSpeakerKey((h.speakerName || "").trim() || UNKNOWN_SPEAKER_LABEL) === key)
    .sort((a, b) => b.createdAtMs - a.createdAtMs)
    .slice(0, 48);
  if (rows.length === 0) {
    return null;
  }
  const chunks: string[] = [];
  for (const row of rows) {
    const lines: string[] = [`Session: ${row.createdAt}`];
    const t = row.text?.trim() ?? "";
    if (t) {
      lines.push(`Transcript: ${t.slice(0, 2500)}${t.length > 2500 ? "…" : ""}`);
    }
    const summ = row.ai?.summary?.trim() ?? "";
    if (summ) {
      lines.push(`Saved AI summary: ${summ.slice(0, 600)}${summ.length > 600 ? "…" : ""}`);
    }
    const topics = row.ai?.topics?.filter((x) => x.trim()) ?? [];
    if (topics.length > 0) {
      lines.push(`Topics tagged: ${topics.join(", ")}`);
    }
    const acts = row.ai?.actionItems?.filter((x) => x.trim()) ?? [];
    if (acts.length > 0) {
      lines.push(`Tasks noted: ${acts.join("; ")}`);
    }
    if (row.answeredVoiceFollowUp?.trim()) {
      lines.push(`They answered an AI follow-up about: ${row.answeredVoiceFollowUp.trim().slice(0, 220)}`);
    }
    chunks.push(lines.join("\n"));
  }
  let bundle = chunks.join("\n\n———\n\n");
  if (bundle.length > MAX_PERSON_SUMMARY_BUNDLE_CHARS) {
    bundle = `${bundle.slice(0, MAX_PERSON_SUMMARY_BUNDLE_CHARS)}\n…`;
  }
  return bundle;
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

  const ai = parseTranscriptAi(item.ai);

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
    answeredVoiceFollowUp: readNullableString(item, "answeredVoiceFollowUp"),
    ...(firstPassRecognitionEngine ? { firstPassRecognitionEngine } : {}),
    transcriptionDiagnosticsInitial,
    transcriptionDiagnosticsEnrollment,
    ...(ai ? { ai } : {}),
  };
};

const stripAiFromLogItem = (item: TranscriptLogItem): TranscriptLogItem => {
  const next = { ...item };
  delete next.ai;
  delete next.answeredVoiceFollowUp;
  return next;
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
  const [historyAiRegeneratingId, setHistoryAiRegeneratingId] = useState<string | null>(null);
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
  const [voiceFollowUp, setVoiceFollowUp] = useState<{ question: string } | null>(null);
  const voiceFollowUpRef = useRef<{ question: string } | null>(null);
  const updateVoiceFollowUp = useCallback((next: { question: string } | null) => {
    voiceFollowUpRef.current = next;
    setVoiceFollowUp(next);
  }, []);
  const [personSummaryModal, setPersonSummaryModal] = useState<{ speakerName: string; narrative: string } | null>(
    null,
  );
  const [todayPlanModal, setTodayPlanModal] = useState<{ speakerName: string; narrative: string } | null>(null);
  const [weekTasksReportModal, setWeekTasksReportModal] = useState<string | null>(null);
  const [weekTasksLoading, setWeekTasksLoading] = useState(false);
  const [speakerAiLoading, setSpeakerAiLoading] = useState<{ kind: "summary" | "today"; key: string } | null>(null);
  const [trafficDestinationDraft, setTrafficDestinationDraft] = useState("");
  const [trafficWatchActive, setTrafficWatchActive] = useState(false);
  const [trafficStatusLine, setTrafficStatusLine] = useState<string | null>(null);
  const [trafficFetchBusy, setTrafficFetchBusy] = useState(false);
  const [contactLookupVisible, setContactLookupVisible] = useState(false);
  const [contactLookupQuery, setContactLookupQuery] = useState("");
  const [contactLookupResults, setContactLookupResults] = useState<ContactPhoneRow[]>([]);
  const [contactLookupLoading, setContactLookupLoading] = useState(false);
  const [webSearchVisible, setWebSearchVisible] = useState(false);
  const [webSearchDraft, setWebSearchDraft] = useState("");
  const trafficPollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const trafficPrevTrafficMinutesRef = useRef<number | null>(null);
  const trafficAppStateRef = useRef<AppStateStatus>(AppState.currentState);
  const historyRef = useRef(history);
  const skipContactLookupFetchRef = useRef(false);

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
    return normalizeTranscribeApiBase(apiUrl);
  };

  const openContactLookupFlow = useCallback(async (initialQuery: string) => {
    if (Platform.OS === "web") {
      Alert.alert("Contacts", "Search your contacts in the iOS or Android app.");
      return;
    }
    const avail = await isContactsApiAvailable();
    if (!avail) {
      Alert.alert("Contacts", "Address book is not available on this device.");
      return;
    }
    const granted = await ensureContactsPermission();
    if (!granted) {
      Alert.alert("Permission needed", "Allow contact access to look someone up by name or nickname.");
      return;
    }
    setContactLookupQuery(initialQuery.trim());
    setContactLookupResults([]);
    setContactLookupVisible(true);
  }, []);

  const openWebSearchFlow = useCallback((prefill: string) => {
    setWebSearchDraft(prefill.trim());
    setWebSearchVisible(true);
  }, []);

  const runAutoContactLookupAfterTranscript = useCallback(async (rawText: string) => {
    const text = rawText.trim();
    if (!text || Platform.OS === "web") {
      return;
    }
    const q = autoContactSearchQuery(text);
    if (!q || q.length < 2) {
      return;
    }
    const avail = await isContactsApiAvailable();
    if (!avail) {
      return;
    }
    const granted = await ensureContactsPermission();
    if (!granted) {
      return;
    }
    let rows: ContactPhoneRow[];
    try {
      rows = await searchContactsWithPhones(q);
    } catch {
      return;
    }
    if (rows.length === 0) {
      return;
    }
    if (rows.length === 1) {
      const r = rows[0];
      confirmAndPlacePhoneCall(`${r.displayName} · ${r.display}`, r.tel);
      return;
    }
    skipContactLookupFetchRef.current = true;
    setContactLookupQuery(q);
    setContactLookupResults(rows);
    setContactLookupLoading(false);
    setContactLookupVisible(true);
  }, []);

  useEffect(() => {
    if (!contactLookupVisible) {
      return;
    }
    if (skipContactLookupFetchRef.current) {
      skipContactLookupFetchRef.current = false;
      return;
    }
    let cancelled = false;
    const t = setTimeout(() => {
      void (async () => {
        setContactLookupLoading(true);
        try {
          const rows = await searchContactsWithPhones(contactLookupQuery);
          if (!cancelled) {
            setContactLookupResults(rows);
          }
        } catch {
          if (!cancelled) {
            setContactLookupResults([]);
            Alert.alert("Contact search failed", "Could not read contacts.");
          }
        } finally {
          if (!cancelled) {
            setContactLookupLoading(false);
          }
        }
      })();
    }, 320);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [contactLookupVisible, contactLookupQuery]);

  const canShowTranscript = useMemo(() => transcript.length > 0, [transcript]);
  const aiInsightsOverview = useMemo(() => buildAiInsightsOverview(history), [history]);

  const speakActiveFollowUp = useCallback(() => {
    if (!voiceFollowUp?.question) {
      return;
    }
    Speech.stop();
    Speech.speak(voiceFollowUp.question, {
      language: "en-US",
      rate: Platform.OS === "ios" ? 0.92 : 1,
    });
  }, [voiceFollowUp?.question]);

  const isRecording = recording !== null;

  useEffect(() => {
    historyRef.current = history;
  }, [history]);

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
    const apiToken = expoTranscribeBearerToken();
    if (!apiUrl) {
      throw new Error("Missing EXPO_PUBLIC_TRANSCRIBE_API_URL.");
    }

    const trimmedSpeakerName = currentSpeakerName.trim();
    const baseUrl = normalizeTranscribeApiBase(apiUrl);

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
      ai?: unknown;
    };
    const ai = parseTranscriptAi(data.ai);
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
      ...(ai ? { ai } : {}),
    } satisfies TranscriptionResult;
  };

  const saveHistoryEntryToCloud = async (entry: TranscriptLogItem) => {
    try {
      const apiToken = expoTranscribeBearerToken();
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

  const refreshHistoryEntryAi = async (item: TranscriptLogItem) => {
    const apiToken = expoTranscribeBearerToken();
    if (!item.text?.trim()) {
      Alert.alert("Nothing to summarize", "This entry has no transcript text.");
      return;
    }
    setHistoryAiRegeneratingId(item.id);
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiToken) {
      headers.Authorization = `Bearer ${apiToken}`;
    }
    const base = getApiBaseUrl();
    const tryIds = Array.from(
      new Set(
        [item.cloudHistoryId, item.id].filter((x): x is string => typeof x === "string" && x.trim().length > 0),
      ),
    );
    const applyAi = (raw: unknown) => {
      const ai = parseTranscriptAi(raw);
      if (!ai) {
        return false;
      }
      setHistory((cur) => cur.map((h) => (h.id === item.id ? { ...h, ai } : h)));
      return true;
    };
    try {
      for (const externalId of tryIds) {
        const res = await fetch(`${base}/history/regenerate-ai`, {
          method: "POST",
          headers,
          body: JSON.stringify({ id: externalId }),
        });
        if (res.ok) {
          const data = (await res.json()) as { history?: { ai?: unknown } };
          if (applyAi(data.history?.ai)) {
            return;
          }
          continue;
        }
        if (res.status !== 404) {
          const msg = await res.text();
          throw new Error(msg || `Regenerate failed (${res.status}).`);
        }
      }
      const res2 = await fetch(`${base}/ai/insights`, {
        method: "POST",
        headers,
        body: JSON.stringify({ text: item.text }),
      });
      if (!res2.ok) {
        const msg = await res2.text();
        throw new Error(msg || `AI insights failed (${res2.status}).`);
      }
      const data2 = (await res2.json()) as { ai?: unknown };
      if (!applyAi(data2.ai)) {
        throw new Error("No AI content returned.");
      }
    } catch (e) {
      Alert.alert("AI refresh failed", e instanceof Error ? e.message : "Unknown error.");
    } finally {
      setHistoryAiRegeneratingId(null);
    }
  };

  const fetchCloudHistory = async () => {
    try {
      const apiToken = expoTranscribeBearerToken();
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

  const confirmClearAllAiHistory = () => {
    Alert.alert(
      "Clear all AI notes?",
      "Removes AI summaries, tasks, topics, follow-ups, and voice follow-up tags from the server and this device. Transcripts and speaker data stay.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Clear",
          style: "destructive",
          onPress: () => {
            void (async () => {
              let serverErr: string | null = null;
              try {
                setErrorText(null);
                const apiToken = expoTranscribeBearerToken();
                const response = await fetch(`${getApiBaseUrl()}/history/clear-ai`, {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
                  },
                  body: JSON.stringify({}),
                });
                if (!response.ok) {
                  const raw = await response.text();
                  let detail = raw || `HTTP ${response.status}`;
                  try {
                    const j = JSON.parse(raw) as { error?: string };
                    if (typeof j.error === "string" && j.error.trim()) {
                      detail = j.error.trim();
                    }
                  } catch {
                    /* use raw text */
                  }
                  serverErr =
                    response.status === 401
                      ? `${detail} (check EXPO_PUBLIC_TRANSCRIBE_API_TOKEN matches SERVER_BEARER_TOKEN)`
                      : response.status === 404
                        ? `${detail} (redeploy API so POST /history/clear-ai exists)`
                        : detail;
                }
              } catch (err) {
                serverErr = err instanceof Error ? err.message : "Network error";
              }
              Speech.stop();
              updateVoiceFollowUp(null);
              setHistory((h) => h.map(stripAiFromLogItem));
              if (serverErr) {
                setErrorText(serverErr);
                Alert.alert(
                  "Server did not clear AI data",
                  `Your device log no longer shows AI notes. Server: ${serverErr}`,
                );
              }
            })();
          },
        },
      ],
    );
  };

  const generatePersonVoiceSummary = useCallback(
    async (speakerName: string) => {
      const bundle = buildPersonSummaryBundle(speakerName, history);
      if (!bundle) {
        Alert.alert(
          "No clips for this person",
          `There are no saved log entries for “${speakerName}” yet. Record with this speaker (or merge from the server) first.`,
        );
        return;
      }
      const loadKey = normalizeSpeakerKey(speakerName);
      setSpeakerAiLoading({ kind: "summary", key: loadKey });
      setErrorText(null);
      try {
        const apiToken = expoTranscribeBearerToken();
        const response = await fetch(`${getApiBaseUrl()}/ai/person-summary`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
          },
          body: JSON.stringify({ speakerName, text: bundle }),
        });
        const raw = await response.text();
        if (!response.ok) {
          let detail = raw || `HTTP ${response.status}`;
          try {
            const j = JSON.parse(raw) as { error?: string };
            if (typeof j.error === "string" && j.error.trim()) {
              detail = j.error.trim();
            }
          } catch {
            /* use raw text */
          }
          throw new Error(detail);
        }
        const data = JSON.parse(raw) as { narrative?: string };
        const narrative = typeof data.narrative === "string" ? data.narrative.trim() : "";
        if (!narrative) {
          throw new Error("Empty summary from server.");
        }
        Speech.stop();
        setPersonSummaryModal({ speakerName, narrative });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Could not generate summary.";
        setErrorText(msg);
        Alert.alert("Voice summary failed", msg);
      } finally {
        setSpeakerAiLoading(null);
      }
    },
    [history],
  );

  const generateTodayPlanForSpeaker = useCallback(
    async (speakerName: string) => {
      const { bundle, dateLabel } = buildTodayPlanBundle(speakerName, history);
      if (!bundle) {
        Speech.stop();
        setTodayPlanModal({ speakerName, narrative: "No information for today." });
        return;
      }
      const loadKey = normalizeSpeakerKey(speakerName);
      setSpeakerAiLoading({ kind: "today", key: loadKey });
      setErrorText(null);
      try {
        const apiToken = expoTranscribeBearerToken();
        const response = await fetch(`${getApiBaseUrl()}/ai/speaker-today-plan`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
          },
          body: JSON.stringify({ speakerName, dateLabel, text: bundle }),
        });
        const raw = await response.text();
        if (!response.ok) {
          let detail = raw || `HTTP ${response.status}`;
          try {
            const j = JSON.parse(raw) as { error?: string };
            if (typeof j.error === "string" && j.error.trim()) {
              detail = j.error.trim();
            }
          } catch {
            /* use raw text */
          }
          throw new Error(detail);
        }
        const data = JSON.parse(raw) as { narrative?: string };
        const narrative = typeof data.narrative === "string" ? data.narrative.trim() : "";
        if (!narrative) {
          throw new Error("Empty plan from server.");
        }
        Speech.stop();
        setTodayPlanModal({ speakerName, narrative });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Could not generate today's plan.";
        setErrorText(msg);
        Alert.alert("Today's plan failed", msg);
      } finally {
        setSpeakerAiLoading(null);
      }
    },
    [history],
  );

  const detectDestinationFromLogs = useCallback(async () => {
    const text = buildRecentLogsForDestinationExtract(history);
    if (!text.trim()) {
      Alert.alert("No logs", "Record something first so we can look for a place to go.");
      return;
    }
    setTrafficFetchBusy(true);
    setErrorText(null);
    try {
      const apiToken = expoTranscribeBearerToken();
      const response = await fetch(`${getApiBaseUrl()}/ai/extract-destination`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({ text }),
      });
      const raw = await response.text();
      if (!response.ok) {
        let detail = raw || `HTTP ${response.status}`;
        try {
          const j = JSON.parse(raw) as { error?: string };
          if (typeof j.error === "string" && j.error.trim()) {
            detail = j.error.trim();
          }
        } catch {
          /* */
        }
        throw new Error(detail);
      }
      const data = JSON.parse(raw) as { destination?: string };
      const dest = typeof data.destination === "string" ? data.destination.trim() : "";
      if (!dest) {
        Alert.alert("No destination found", "Try typing an address or place name, or mention where you're going in a new clip.");
        return;
      }
      setTrafficDestinationDraft(dest);
      setTrafficStatusLine(`Using: ${dest}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Could not detect destination.";
      setErrorText(msg);
      Alert.alert("Extract failed", msg);
    } finally {
      setTrafficFetchBusy(false);
    }
  }, [history]);

  const runOneTrafficCheck = useCallback(async () => {
    const dest = trafficDestinationDraft.trim();
    if (!dest) {
      return;
    }
    if (Platform.OS === "web") {
      setTrafficStatusLine("Traffic watch needs a dev build on device (location + Maps).");
      return;
    }
    setTrafficFetchBusy(true);
    try {
      const perm = await Location.requestForegroundPermissionsAsync();
      if (perm.status !== "granted") {
        throw new Error("Location permission is needed for drive times.");
      }
      const pos = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
      const apiToken = expoTranscribeBearerToken();
      const response = await fetch(`${getApiBaseUrl()}/traffic/duration`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({
          originLat: pos.coords.latitude,
          originLng: pos.coords.longitude,
          destination: dest,
        }),
      });
      const raw = await response.text();
      if (!response.ok) {
        let detail = raw || `HTTP ${response.status}`;
        try {
          const j = JSON.parse(raw) as { error?: string };
          if (typeof j.error === "string" && j.error.trim()) {
            detail = j.error.trim();
          }
        } catch {
          /* */
        }
        throw new Error(detail);
      }
      const data = JSON.parse(raw) as { trafficMinutes?: number; baselineMinutes?: number; summaryText?: string };
      const tm = typeof data.trafficMinutes === "number" ? data.trafficMinutes : null;
      const summary = typeof data.summaryText === "string" ? data.summaryText : "";
      if (tm === null) {
        throw new Error("Bad traffic response.");
      }
      const line = summary ? `${summary} (~${tm} min with traffic)` : `About ${tm} minutes with current traffic.`;
      setTrafficStatusLine(line);

      const prev = trafficPrevTrafficMinutesRef.current;
      trafficPrevTrafficMinutesRef.current = tm;
      if (prev !== null && tm > prev * 1.2 && tm - prev >= 3) {
        const msg = `Heavier traffic now: about ${tm} minutes, was about ${prev}.`;
        Speech.stop();
        Speech.speak(msg, { language: "en-US", rate: Platform.OS === "ios" ? 0.92 : 1 });
        Alert.alert("Traffic update", msg);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Traffic check failed.";
      setTrafficStatusLine(msg);
      if (trafficWatchActive) {
        setErrorText(msg);
      }
    } finally {
      setTrafficFetchBusy(false);
    }
  }, [trafficDestinationDraft, trafficWatchActive]);

  useEffect(() => {
    const sub = AppState.addEventListener("change", (next) => {
      trafficAppStateRef.current = next;
    });
    return () => sub.remove();
  }, []);

  useEffect(() => {
    if (!trafficWatchActive || !trafficDestinationDraft.trim()) {
      if (trafficPollTimerRef.current) {
        clearInterval(trafficPollTimerRef.current);
        trafficPollTimerRef.current = null;
      }
      trafficPrevTrafficMinutesRef.current = null;
      return;
    }
    void runOneTrafficCheck();
    trafficPollTimerRef.current = setInterval(() => {
      if (trafficAppStateRef.current === "active") {
        void runOneTrafficCheck();
      }
    }, TRAFFIC_POLL_INTERVAL_MS);
    return () => {
      if (trafficPollTimerRef.current) {
        clearInterval(trafficPollTimerRef.current);
        trafficPollTimerRef.current = null;
      }
    };
  }, [trafficWatchActive, trafficDestinationDraft, runOneTrafficCheck]);

  const loadWeekTasksReport = useCallback(async () => {
    const bundle = buildWeekTasksBundle(history);
    if (!bundle?.trim()) {
      Alert.alert("No logs", "Save some voice clips first.");
      return;
    }
    const windowLabel = getUpcomingWeekWindowLabel();
    setWeekTasksLoading(true);
    setErrorText(null);
    try {
      const apiToken = expoTranscribeBearerToken();
      const response = await fetch(`${getApiBaseUrl()}/ai/week-tasks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({ windowLabel, text: bundle }),
      });
      const raw = await response.text();
      if (!response.ok) {
        let detail = raw || `HTTP ${response.status}`;
        try {
          const j = JSON.parse(raw) as { error?: string };
          if (typeof j.error === "string" && j.error.trim()) {
            detail = j.error.trim();
          }
        } catch {
          /* */
        }
        throw new Error(detail);
      }
      const data = JSON.parse(raw) as { report?: string };
      const report = typeof data.report === "string" ? data.report.trim() : "";
      if (!report) {
        throw new Error("Empty report from server.");
      }
      setWeekTasksReportModal(report);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Could not build week tasks.";
      setErrorText(msg);
      Alert.alert("Week tasks", msg);
    } finally {
      setWeekTasksLoading(false);
    }
  }, [history]);

  const fetchSpeakers = async (suppressError = false) => {
    try {
      setIsLoadingSpeakers(true);
      const apiToken = expoTranscribeBearerToken();
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
    const apiToken = expoTranscribeBearerToken();
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
      const apiToken = expoTranscribeBearerToken();
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
      const apiToken = expoTranscribeBearerToken();
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
    setLastSpeakerName(UNKNOWN_SPEAKER_LABEL);
    setStatusText("Tap the mic — we’ll handle the rest.");
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
                const apiToken = expoTranscribeBearerToken();
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

        const answeredVoiceFollowUpTo = voiceFollowUpRef.current?.question?.trim() || null;

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
          void runAutoContactLookupAfterTranscript(text);
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
            ...(answeredVoiceFollowUpTo ? { answeredVoiceFollowUp: answeredVoiceFollowUpTo } : {}),
            ...(result.ai ? { ai: result.ai } : {}),
            ...(transcriptionDiagnosticsEnrollment
              ? { transcriptionDiagnosticsEnrollment }
              : {}),
          };
          setHistory((current) => [{ ...newHistoryEntry }, ...current]);
          void saveHistoryEntryToCloud(newHistoryEntry);
          void fetchSpeakers(true);
          updateVoiceFollowUp(null);
          const nextFq = pickFirstFollowUpQuestion(result.ai);
          if (nextFq) {
            updateVoiceFollowUp({ question: nextFq });
          }
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
        let historyAiInsight = result.ai;
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
              historyAiInsight = enrollmentResult.ai ?? historyAiInsight;
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
        void runAutoContactLookupAfterTranscript(text);
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
          ...(historyAiInsight ? { ai: historyAiInsight } : {}),
          ...(answeredVoiceFollowUpTo ? { answeredVoiceFollowUp: answeredVoiceFollowUpTo } : {}),
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
        updateVoiceFollowUp(null);
        const nextFq = pickFirstFollowUpQuestion(historyAiInsight);
        if (nextFq) {
          updateVoiceFollowUp({ question: nextFq });
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
    <View style={styles.appRoot}>
    <SafeAreaView style={styles.screen}>
      <StatusBar style="dark" />
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
              placeholderTextColor="#94a3b8"
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
              placeholderTextColor="#94a3b8"
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
              placeholderTextColor="#94a3b8"
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
      <Modal
        visible={personSummaryModal !== null}
        transparent
        animationType="fade"
        onRequestClose={() => {
          Speech.stop();
          setPersonSummaryModal(null);
        }}
      >
        <View style={styles.modalBackdrop}>
          <Pressable
            style={StyleSheet.absoluteFillObject}
            onPress={() => {
              Speech.stop();
              setPersonSummaryModal(null);
            }}
          />
          <View style={[styles.modalCard, styles.personSummaryModalCard]}>
            <Text style={styles.modalTitle}>AI voice summary</Text>
            <Text style={styles.modalSubtitle}>
              {personSummaryModal ? `About “${personSummaryModal.speakerName}” from your saved logs.` : ""}
            </Text>
            <ScrollView style={styles.personSummaryScroll} nestedScrollEnabled keyboardShouldPersistTaps="handled">
              <Text style={styles.personSummaryBody} selectable>
                {personSummaryModal?.narrative ?? ""}
              </Text>
            </ScrollView>
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonSecondary]}
                onPress={() => {
                  Speech.stop();
                  setPersonSummaryModal(null);
                }}
              >
                <Text style={styles.modalButtonSecondaryText}>Close</Text>
              </Pressable>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={() => {
                  if (!personSummaryModal?.narrative) {
                    return;
                  }
                  Speech.stop();
                  Speech.speak(personSummaryModal.narrative, {
                    language: "en-US",
                    rate: Platform.OS === "ios" ? 0.92 : 1,
                  });
                }}
              >
                <Text style={styles.modalButtonPrimaryText}>Play aloud</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <Modal
        visible={todayPlanModal !== null}
        transparent
        animationType="fade"
        onRequestClose={() => {
          Speech.stop();
          setTodayPlanModal(null);
        }}
      >
        <View style={styles.modalBackdrop}>
          <Pressable
            style={StyleSheet.absoluteFillObject}
            onPress={() => {
              Speech.stop();
              setTodayPlanModal(null);
            }}
          />
          <View style={[styles.modalCard, styles.personSummaryModalCard]}>
            <Text style={styles.modalTitle}>Today’s plan</Text>
            <Text style={styles.modalSubtitle}>
              {todayPlanModal ? `For “${todayPlanModal.speakerName}” (${getLocalDayBoundsMs().label}).` : ""}
            </Text>
            <ScrollView style={styles.personSummaryScroll} nestedScrollEnabled keyboardShouldPersistTaps="handled">
              <Text style={styles.personSummaryBody} selectable>
                {todayPlanModal?.narrative ?? ""}
              </Text>
            </ScrollView>
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonSecondary]}
                onPress={() => {
                  Speech.stop();
                  setTodayPlanModal(null);
                }}
              >
                <Text style={styles.modalButtonSecondaryText}>Close</Text>
              </Pressable>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={() => {
                  if (!todayPlanModal?.narrative) {
                    return;
                  }
                  Speech.stop();
                  Speech.speak(todayPlanModal.narrative, {
                    language: "en-US",
                    rate: Platform.OS === "ios" ? 0.92 : 1,
                  });
                }}
              >
                <Text style={styles.modalButtonPrimaryText}>Play aloud</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <Modal
        visible={weekTasksReportModal !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setWeekTasksReportModal(null)}
      >
        <View style={styles.modalBackdrop}>
          <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setWeekTasksReportModal(null)} />
          <View style={[styles.modalCard, styles.personSummaryModalCard]}>
            <Text style={styles.modalTitle}>Next week · by person</Text>
            <Text style={styles.modalSubtitle}>{getUpcomingWeekWindowLabel()}</Text>
            <ScrollView style={styles.personSummaryScroll} nestedScrollEnabled keyboardShouldPersistTaps="handled">
              <Text style={styles.personSummaryBody} selectable>
                {weekTasksReportModal ?? ""}
              </Text>
            </ScrollView>
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={() => setWeekTasksReportModal(null)}
              >
                <Text style={styles.modalButtonPrimaryText}>Close</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <Modal
        visible={contactLookupVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setContactLookupVisible(false)}
      >
        <View style={styles.modalBackdrop}>
          <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setContactLookupVisible(false)} />
          <View style={[styles.modalCard, styles.contactLookupModalCard]}>
            <Text style={styles.modalTitle}>Contacts</Text>
            <Text style={styles.modalSubtitle}>
              We also search automatically after lines like “call …” when contact access is allowed. Or type a name here.
              Tap a number to call — you confirm before the dialer opens.
            </Text>
            <TextInput
              style={styles.modalInput}
              placeholder="Name, nickname, initials…"
              placeholderTextColor="#94a3b8"
              value={contactLookupQuery}
              onChangeText={setContactLookupQuery}
              autoCorrect={false}
              autoCapitalize="none"
            />
            {contactLookupLoading ? (
              <ActivityIndicator style={{ marginVertical: 12 }} color="#0f766e" />
            ) : (
              <ScrollView
                style={styles.contactLookupScroll}
                nestedScrollEnabled
                keyboardShouldPersistTaps="handled"
              >
                {contactLookupResults.length === 0 ? (
                  <Text style={styles.contactLookupEmpty}>
                    {contactLookupQuery.trim() ? "No matching contacts with phone numbers." : "Type to search."}
                  </Text>
                ) : (
                  contactLookupResults.map((row) => (
                    <Pressable
                      key={`${row.contactId}-${row.tel}`}
                      style={styles.contactLookupRow}
                      onPress={() =>
                        confirmAndPlacePhoneCall(`${row.displayName} · ${row.display}`, row.tel)
                      }
                      accessibilityRole="button"
                      accessibilityLabel={`Call ${row.displayName}`}
                    >
                      <View style={styles.contactLookupRowText}>
                        <Text style={styles.contactLookupName}>{row.displayName}</Text>
                        <Text style={styles.contactLookupMeta}>
                          {row.phoneLabel} · {row.display}
                        </Text>
                      </View>
                      <Ionicons name="call-outline" size={20} color="#0f766e" />
                    </Pressable>
                  ))
                )}
              </ScrollView>
            )}
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={() => setContactLookupVisible(false)}
              >
                <Text style={styles.modalButtonPrimaryText}>Close</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <Modal
        visible={webSearchVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setWebSearchVisible(false)}
      >
        <View style={styles.modalBackdrop}>
          <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setWebSearchVisible(false)} />
          <View style={[styles.modalCard, styles.contactLookupModalCard]}>
            <Text style={styles.modalTitle}>Search the web</Text>
            <Text style={styles.modalSubtitle}>Opens your browser (e.g. Google). Edit the query if you like.</Text>
            <TextInput
              style={[styles.modalInput, styles.webSearchInputMultiline]}
              placeholder="What to search…"
              placeholderTextColor="#94a3b8"
              value={webSearchDraft}
              onChangeText={setWebSearchDraft}
              multiline
            />
            <View style={styles.modalActions}>
              <Pressable
                style={[styles.modalButton, styles.modalButtonSecondary]}
                onPress={() => setWebSearchVisible(false)}
              >
                <Text style={styles.modalButtonSecondaryText}>Cancel</Text>
              </Pressable>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={() => {
                  openGoogleSearch(webSearchDraft);
                  setWebSearchVisible(false);
                }}
              >
                <Text style={styles.modalButtonPrimaryText}>Search</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <View style={styles.card}>
        <View style={styles.cardHeaderRow}>
          <View style={styles.cardHeaderMain}>
            <Text style={styles.kicker}>Just Speak</Text>
            <LinearGradient
              colors={["transparent", "rgba(249, 115, 22, 0.65)", "transparent"]}
              start={{ x: 0, y: 0.5 }}
              end={{ x: 1, y: 0.5 }}
              style={styles.headlineAccent}
            />
          </View>
          <Image
            source={require("./assets/family-hero.png")}
            style={styles.familyMascot}
            resizeMode="cover"
            accessibilityIgnoresInvertColors
            accessibilityLabel="Family illustration"
          />
        </View>
        <Text style={styles.subtitle}>{statusText}</Text>

        <View style={styles.tabBar}>
          <Pressable
            style={[styles.tabButton, activeTab === "record" && styles.tabButtonActive]}
            onPress={() => setActiveTab("record")}
          >
            <View style={styles.tabButtonInner}>
              <Ionicons
                name="mic"
                size={16}
                color={activeTab === "record" ? "#ffffff" : "#475569"}
              />
              <Text style={[styles.tabButtonText, activeTab === "record" && styles.tabButtonTextActive]}>
                Record
              </Text>
            </View>
          </Pressable>
          <Pressable
            style={[styles.tabButton, activeTab === "history" && styles.tabButtonActive]}
            onPress={() => setActiveTab("history")}
          >
            <View style={styles.tabButtonInner}>
              <Ionicons
                name="time-outline"
                size={17}
                color={activeTab === "history" ? "#ffffff" : "#475569"}
              />
              <Text style={[styles.tabButtonText, activeTab === "history" && styles.tabButtonTextActive]}>
                History
              </Text>
            </View>
          </Pressable>
          <Pressable
            style={[styles.tabButton, activeTab === "speakers" && styles.tabButtonActive]}
            onPress={() => setActiveTab("speakers")}
          >
            <View style={styles.tabButtonInner}>
              <Ionicons
                name="people-outline"
                size={17}
                color={activeTab === "speakers" ? "#ffffff" : "#475569"}
              />
              <Text style={[styles.tabButtonText, activeTab === "speakers" && styles.tabButtonTextActive]}>
                Speakers
              </Text>
            </View>
          </Pressable>
          <Pressable
            style={[styles.tabButton, activeTab === "ai" && styles.tabButtonActive]}
            onPress={() => setActiveTab("ai")}
          >
            <View style={styles.tabButtonInner}>
              <Ionicons
                name="sparkles-outline"
                size={17}
                color={activeTab === "ai" ? "#ffffff" : "#475569"}
              />
              <Text style={[styles.tabButtonText, activeTab === "ai" && styles.tabButtonTextActive]}>AI</Text>
            </View>
          </Pressable>
        </View>

        {errorText ? <Text style={styles.errorText}>{errorText}</Text> : null}

        {activeTab === "record" ? (
          <View style={styles.tabContent}>
            <Text style={styles.tabIntro}>
              Tap the mic to transcribe with auto speaker ID — or the badge to add a new voice on purpose.
            </Text>

            {voiceFollowUp && !enrollTargetName ? (
              <View style={styles.voiceFollowUpBanner}>
                <Text style={styles.voiceFollowUpLabel}>AI follow-up (speak your answer with the mic below)</Text>
                <Text style={styles.voiceFollowUpQuestion}>{voiceFollowUp.question}</Text>
                <View style={styles.voiceFollowUpActions}>
                  <Pressable
                    style={styles.voiceFollowUpButtonSecondary}
                    onPress={speakActiveFollowUp}
                    accessibilityRole="button"
                    accessibilityLabel="Play question aloud"
                  >
                    <Ionicons name="volume-high-outline" size={18} color="#0f766e" />
                    <Text style={styles.voiceFollowUpButtonSecondaryText}>Play question</Text>
                  </Pressable>
                  <Pressable
                    style={styles.voiceFollowUpButtonGhost}
                    onPress={() => {
                      Speech.stop();
                      updateVoiceFollowUp(null);
                    }}
                    accessibilityRole="button"
                    accessibilityLabel="Dismiss follow-up"
                  >
                    <Text style={styles.voiceFollowUpButtonGhostText}>Dismiss</Text>
                  </Pressable>
                </View>
              </View>
            ) : null}

            {enrollTargetName ? (
              <View style={styles.enrollChip}>
                <Ionicons
                  name={isRecording ? "ellipse" : "person"}
                  size={14}
                  color="#0f766e"
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
                  <Ionicons name="person-add-sharp" size={26} color="#0f766e" />
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
                    color="#ffffff"
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
              {canShowTranscript ? (
                <Text style={styles.speakerLabel}>Speaker: {lastSpeakerName}</Text>
              ) : null}
              <ScrollView style={styles.transcriptScroll} contentContainerStyle={styles.transcriptContent}>
                <Text style={styles.transcriptText}>
                  {canShowTranscript
                    ? transcript
                    : "Your spoken words will appear here once recognition starts."}
                </Text>
              </ScrollView>
              {canShowTranscript ? (
                <View style={styles.recordDiscoverRow}>
                  <Pressable
                    style={styles.historyDiscoverButton}
                    onPress={() => {
                      void openContactLookupFlow(suggestContactSearchFromTranscript(transcript));
                    }}
                    accessibilityRole="button"
                    accessibilityLabel="Find in contacts from transcript"
                  >
                    <Ionicons name="people-outline" size={16} color="#0f766e" />
                    <Text style={styles.historyDiscoverButtonText}>Find in contacts</Text>
                  </Pressable>
                  <Pressable
                    style={styles.historyDiscoverButton}
                    onPress={() => openWebSearchFlow(buildWebSearchQueryForEntry(transcript, null))}
                    accessibilityRole="button"
                    accessibilityLabel="Search the web from transcript"
                  >
                    <Ionicons name="globe-outline" size={16} color="#0369a1" />
                    <Text style={styles.historyDiscoverButtonText}>Search web</Text>
                  </Pressable>
                </View>
              ) : null}
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
                                            <View style={styles.historyEntryActions}>
                                              <Pressable
                                                style={styles.refreshAiButton}
                                                onPress={() => {
                                                  void refreshHistoryEntryAi(item);
                                                }}
                                                disabled={historyAiRegeneratingId === item.id || !item.text?.trim()}
                                                accessibilityRole="button"
                                                accessibilityLabel="Refresh AI insights for this entry"
                                              >
                                                <Text style={styles.refreshAiButtonText}>
                                                  {historyAiRegeneratingId === item.id ? "AI…" : "Refresh AI"}
                                                </Text>
                                              </Pressable>
                                              <Pressable
                                                style={styles.deleteLogButton}
                                                onPress={() => confirmDeleteHistoryItem(item.id)}
                                                disabled={historyAiRegeneratingId === item.id}
                                                accessibilityRole="button"
                                                accessibilityLabel="Delete this log"
                                              >
                                                <Text style={styles.deleteLogButtonText}>Delete</Text>
                                              </Pressable>
                                            </View>
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
                                          {item.ai &&
                                          (item.ai.summary ||
                                            (item.ai.actionItems?.length ?? 0) > 0 ||
                                            (item.ai.actions?.length ?? 0) > 0 ||
                                            item.ai.calendarIntent ||
                                            (item.ai.topics?.length ?? 0) > 0 ||
                                            (item.ai.followUpQuestions?.length ?? 0) > 0) ? (
                                            <View style={styles.historyAiBlock}>
                                              <Text style={[styles.historySectionLabel, styles.historyAiBlockTitle]}>
                                                AI insights
                                              </Text>
                                              {(item.ai.actionItems?.length ?? 0) > 0 ? (
                                                <>
                                                  <Text style={[styles.historySectionLabel, styles.historyAiSubLabel]}>
                                                    Action items
                                                  </Text>
                                                  {item.ai.actionItems.map((line, idx) => (
                                                    <Text key={`ai-act-${item.id}-${idx}`} style={styles.historyAiBullet}>
                                                      • {line}
                                                    </Text>
                                                  ))}
                                                </>
                                              ) : null}
                                              {(item.ai.actions?.length ?? 0) > 0 ? (
                                                <>
                                                  <Text style={[styles.historySectionLabel, styles.historyAiSubLabel]}>
                                                    Action + confidence + fallback
                                                  </Text>
                                                  {(item.ai.actions ?? []).map((act, idx) => (
                                                    <View key={`ai-act-struct-${item.id}-${idx}`} style={styles.historyStructuredAction}>
                                                      <Text style={styles.historyStructuredActionLine} selectable>
                                                        {act.type.replace(/_/g, " ")} · {act.label} · {Math.round(
                                                          act.confidence * 100,
                                                        )}
                                                        % confident
                                                      </Text>
                                                      {act.detail ? (
                                                        <Text style={styles.historyStructuredActionDetail} selectable>
                                                          Detail: {act.detail}
                                                        </Text>
                                                      ) : null}
                                                      <Text style={styles.historyStructuredActionFallback} selectable>
                                                        Fallback: {act.fallback}
                                                      </Text>
                                                    </View>
                                                  ))}
                                                </>
                                              ) : null}
                                              {item.ai.calendarIntent ? (
                                                <>
                                                  <Text style={[styles.historySectionLabel, styles.historyAiSubLabel]}>
                                                    Calendar · create_event
                                                  </Text>
                                                  <Text style={styles.historyAiJson} selectable>
                                                    {JSON.stringify(item.ai.calendarIntent, null, 2)}
                                                  </Text>
                                                </>
                                              ) : null}
                                              {(item.ai.topics?.length ?? 0) > 0 ? (
                                                <>
                                                  <Text style={[styles.historySectionLabel, styles.historyAiSubLabel]}>
                                                    Topics
                                                  </Text>
                                                  <Text style={styles.historyAiTopicsLine}>
                                                    {item.ai.topics.join(" · ")}
                                                  </Text>
                                                </>
                                              ) : null}
                                              {(item.ai.followUpQuestions?.length ?? 0) > 0 ? (
                                                <>
                                                  <Text style={[styles.historySectionLabel, styles.historyAiSubLabel]}>
                                                    Follow-up questions
                                                  </Text>
                                                  {(item.ai.followUpQuestions ?? []).map((q, idx) => (
                                                    <View key={`ai-fq-${item.id}-${idx}`} style={styles.historyFollowUpItem}>
                                                      <Text style={styles.historyFollowUpItemText}>{q}</Text>
                                                      <View style={styles.historyFollowUpItemActions}>
                                                        <Pressable
                                                          style={styles.historyFollowUpTinyButton}
                                                          onPress={() => {
                                                            Speech.stop();
                                                            Speech.speak(q, {
                                                              language: "en-US",
                                                              rate: Platform.OS === "ios" ? 0.92 : 1,
                                                            });
                                                          }}
                                                          accessibilityRole="button"
                                                          accessibilityLabel="Play question"
                                                        >
                                                          <Ionicons name="volume-high-outline" size={14} color="#0f766e" />
                                                        </Pressable>
                                                        <Pressable
                                                          style={styles.historyFollowUpAnswerLink}
                                                          onPress={() => {
                                                            Speech.stop();
                                                            updateVoiceFollowUp({ question: q });
                                                            setActiveTab("record");
                                                          }}
                                                          accessibilityRole="button"
                                                          accessibilityLabel="Answer with voice"
                                                        >
                                                          <Text style={styles.historyFollowUpAnswerLinkText}>Answer</Text>
                                                        </Pressable>
                                                      </View>
                                                    </View>
                                                  ))}
                                                </>
                                              ) : null}
                                              {item.ai.summary?.trim() ? (
                                                <>
                                                  <Text style={[styles.historySectionLabel, styles.historyAiSubLabel]}>
                                                    Summary
                                                  </Text>
                                                  <Text style={styles.historyAiSummarySecondary}>{item.ai.summary}</Text>
                                                </>
                                              ) : null}
                                              <Text style={[styles.historySectionLabel, styles.historyAiSubLabel]}>
                                                Model output (JSON)
                                              </Text>
                                              <Text style={styles.historyAiJson} selectable>
                                                {formatHistoryAiModelOutputJson(item.ai)}
                                              </Text>
                                            </View>
                                          ) : null}
                                          {item.answeredVoiceFollowUp ? (
                                            <View style={styles.historyAnsweredFollowUp}>
                                              <Text style={styles.historySectionLabel}>Answered AI question</Text>
                                              <Text style={styles.historyAnsweredFollowUpText} selectable>
                                                {item.answeredVoiceFollowUp}
                                              </Text>
                                            </View>
                                          ) : null}
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
                                          {(() => {
                                            const phones = collectPhonesFromHistoryItem(item);
                                            if (phones.length === 0) {
                                              return null;
                                            }
                                            return (
                                              <View style={styles.historyPhoneBlock}>
                                                <Text style={styles.historySectionLabel}>Phone numbers</Text>
                                                <Text style={styles.historyPhoneHint}>
                                                  Tap a number — you’ll be asked before the dialer opens.
                                                </Text>
                                                <View style={styles.historyPhoneChips}>
                                                  {phones.map((p) => (
                                                    <Pressable
                                                      key={p.tel}
                                                      style={styles.historyPhoneChip}
                                                      onPress={() => confirmAndPlacePhoneCall(p.display, p.tel)}
                                                      accessibilityRole="button"
                                                      accessibilityLabel={`Call ${p.display}`}
                                                    >
                                                      <Ionicons name="call-outline" size={16} color="#0369a1" />
                                                      <Text style={styles.historyPhoneChipText}>{p.display}</Text>
                                                    </Pressable>
                                                  ))}
                                                </View>
                                              </View>
                                            );
                                          })()}
                                          {item.text?.trim() ? (
                                            <View style={styles.historyDiscoverRow}>
                                              <Text style={styles.historySectionLabel}>Look up</Text>
                                              <View style={styles.historyDiscoverActions}>
                                                <Pressable
                                                  style={styles.historyDiscoverButton}
                                                  onPress={() => {
                                                    void openContactLookupFlow(
                                                      suggestContactSearchFromTranscript(item.text),
                                                    );
                                                  }}
                                                  accessibilityRole="button"
                                                  accessibilityLabel="Search contacts for this entry"
                                                >
                                                  <Ionicons name="people-outline" size={16} color="#0f766e" />
                                                  <Text style={styles.historyDiscoverButtonText}>Contacts</Text>
                                                </Pressable>
                                                <Pressable
                                                  style={styles.historyDiscoverButton}
                                                  onPress={() =>
                                                    openWebSearchFlow(
                                                      buildWebSearchQueryForEntry(item.text, item.ai?.summary),
                                                    )
                                                  }
                                                  accessibilityRole="button"
                                                  accessibilityLabel="Search the web for this entry"
                                                >
                                                  <Ionicons name="globe-outline" size={16} color="#0369a1" />
                                                  <Text style={styles.historyDiscoverButtonText}>Web</Text>
                                                </Pressable>
                                              </View>
                                            </View>
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
                vector when saved. Calendar = spoken today plan from today’s clips; sparkles = recap from all their
                clips.
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
                          <View style={styles.speakerListActionsInner}>
                            <Pressable
                              style={styles.personSummaryIconButton}
                              onPress={() => void generateTodayPlanForSpeaker(speaker.name)}
                              disabled={
                                speakerAiLoading?.kind === "today" &&
                                speakerAiLoading.key === normalizeSpeakerKey(speaker.name)
                              }
                              accessibilityRole="button"
                              accessibilityLabel={`Today's AI plan for ${speaker.name}`}
                            >
                              {speakerAiLoading?.kind === "today" &&
                              speakerAiLoading.key === normalizeSpeakerKey(speaker.name) ? (
                                <ActivityIndicator size="small" color="#0f766e" />
                              ) : (
                                <Ionicons name="calendar-outline" size={22} color="#0f766e" />
                              )}
                            </Pressable>
                            <Pressable
                              style={styles.personSummaryIconButton}
                              onPress={() => void generatePersonVoiceSummary(speaker.name)}
                              disabled={
                                speakerAiLoading?.kind === "summary" &&
                                speakerAiLoading.key === normalizeSpeakerKey(speaker.name)
                              }
                              accessibilityRole="button"
                              accessibilityLabel={`AI voice summary for ${speaker.name}`}
                            >
                              {speakerAiLoading?.kind === "summary" &&
                              speakerAiLoading.key === normalizeSpeakerKey(speaker.name) ? (
                                <ActivityIndicator size="small" color="#0f766e" />
                              ) : (
                                <Ionicons name="sparkles" size={22} color="#0f766e" />
                              )}
                            </Pressable>
                            <View style={styles.speakerListActionLinks}>
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

        {activeTab === "ai" ? (
          <ScrollView
            style={styles.aiTabScroll}
            contentContainerStyle={styles.aiTabContent}
            keyboardShouldPersistTaps="handled"
          >
            <Text style={styles.tabIntro}>
              Summaries, tasks, and topics from clips that already have AI data from transcription or history save
              (needs API with OpenAI at capture time).
            </Text>
            <View style={styles.aiOverviewPanel}>
              <Text style={styles.panelTitle}>Overview</Text>
              <Text style={styles.aiOverviewLine}>
                {aiInsightsOverview.entriesWithAi} of {aiInsightsOverview.totalEntries} log entries include AI notes.
              </Text>
              {history.length > 0 ? (
                <View style={styles.aiOverviewButtonCol}>
                  <Pressable
                    style={[styles.aiClearAiButton, styles.aiOverviewPanelTightButton]}
                    onPress={confirmClearAllAiHistory}
                    accessibilityRole="button"
                    accessibilityLabel="Clear all AI notes from log"
                  >
                    <Text style={styles.aiClearAiButtonText}>Clear all AI notes…</Text>
                  </Pressable>
                  <Pressable
                    style={[styles.aiWeekTasksButton, weekTasksLoading && styles.controlDisabled]}
                    disabled={weekTasksLoading}
                    onPress={() => void loadWeekTasksReport()}
                    accessibilityRole="button"
                    accessibilityLabel="Show next seven days tasks by person"
                  >
                    {weekTasksLoading ? (
                      <ActivityIndicator color="#0f766e" />
                    ) : (
                      <Text style={styles.aiWeekTasksButtonText}>Next 7 days · tasks by person</Text>
                    )}
                  </Pressable>
                </View>
              ) : null}
              {aiInsightsOverview.totalEntries === 0 ? (
                <Text style={[styles.historyEmptyText, styles.aiPanelInset]}>
                  Record something first — insights appear after history is saved.
                </Text>
              ) : aiInsightsOverview.entriesWithAi === 0 ? (
                <View style={styles.aiPanelInset}>
                  <Text style={styles.historyEmptyText}>
                    Nothing to show yet. This usually means either you have no saved clips, or no clip stored an AI block.
                  </Text>
                  <Text style={styles.aiHelpBullet}>
                    • The server needs OPENAI_API_KEY so new recordings can store an AI block with the transcript.
                  </Text>
                  <Text style={styles.aiHelpBullet}>
                    • In History, expand an entry — AI insights show action items first, then structured JSON from the model.
                  </Text>
                </View>
              ) : null}
            </View>

            <View style={styles.aiSectionPanel}>
              <Text style={styles.aiSectionHeading}>Drive-time watch</Text>
              <Text style={styles.aiSectionSub}>
                Re-checks about every 4 minutes while the app is open. If time in traffic jumps roughly 20%+ (and at
                least 3 minutes), you get a spoken alert. Set GOOGLE_MAPS_API_KEY on the API (Directions enabled).
              </Text>
              <TextInput
                style={styles.trafficDestinationInput}
                placeholder="Destination address or place"
                placeholderTextColor="#94a3b8"
                value={trafficDestinationDraft}
                onChangeText={setTrafficDestinationDraft}
                editable={!trafficWatchActive}
              />
              <View style={styles.trafficButtonRow}>
                <Pressable
                  style={[styles.trafficSecondaryButton, trafficFetchBusy && styles.controlDisabled]}
                  disabled={trafficFetchBusy}
                  onPress={() => void detectDestinationFromLogs()}
                >
                  <Text style={styles.trafficSecondaryButtonText}>Detect from logs</Text>
                </Pressable>
                <Pressable
                  style={[styles.trafficSecondaryButton, (trafficFetchBusy || !trafficDestinationDraft.trim()) && styles.controlDisabled]}
                  disabled={trafficFetchBusy || !trafficDestinationDraft.trim()}
                  onPress={() => void runOneTrafficCheck()}
                >
                  <Text style={styles.trafficSecondaryButtonText}>Check now</Text>
                </Pressable>
              </View>
              <Pressable
                style={[styles.trafficPrimaryButton, trafficFetchBusy && styles.controlDisabled]}
                disabled={trafficFetchBusy}
                onPress={() => {
                  setTrafficWatchActive((active) => {
                    if (active) {
                      Speech.stop();
                    }
                    return !active;
                  });
                }}
              >
                <Text style={styles.trafficPrimaryButtonText}>
                  {trafficWatchActive ? "Stop traffic updates" : "Start traffic updates"}
                </Text>
              </Pressable>
              {trafficStatusLine ? (
                <Text style={styles.trafficStatusText} selectable>
                  {trafficStatusLine}
                </Text>
              ) : null}
            </View>

            {aiInsightsOverview.allTopics.length > 0 ? (
              <View style={styles.aiSectionPanel}>
                <Text style={styles.aiSectionHeading}>Topics across everyone</Text>
                <Text style={styles.aiSectionSub}>
                  How often each theme appeared (from AI tags on your clips).
                </Text>
                <View style={styles.aiChipWrap}>
                  {aiInsightsOverview.allTopics.map((t, i) => (
                    <View key={`all-topic-${i}-${t.display}`} style={styles.aiChip}>
                      <Text style={styles.aiChipText}>
                        {t.display}
                        <Text style={styles.aiChipCount}> ({t.count})</Text>
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
            ) : null}

            {aiInsightsOverview.actionItems.length > 0 ? (
              <View style={styles.aiSectionPanel}>
                <Text style={styles.aiSectionHeading}>Action items</Text>
                <Text style={styles.aiSectionSub}>Recent tasks deduplicated — who and when.</Text>
                {aiInsightsOverview.actionItems.map((row, idx) => (
                  <View key={`${row.text}-${idx}`} style={styles.aiActionRow}>
                    <Text style={styles.aiActionBullet}>• {row.text}</Text>
                    <Text style={styles.aiActionMeta}>
                      {row.speaker} · {row.createdAt}
                    </Text>
                  </View>
                ))}
              </View>
            ) : null}

            {aiInsightsOverview.voiceFollowUps.length > 0 ? (
              <View style={styles.aiSectionPanel}>
                <Text style={styles.aiSectionHeading}>Voice follow-ups</Text>
                <Text style={styles.aiSectionSub}>
                  Questions the model wants you to answer — use Record to speak back; we tag that clip with the question.
                </Text>
                {aiInsightsOverview.voiceFollowUps.map((row, idx) => (
                  <View key={`vf-${row.historyId}-${idx}`} style={styles.voiceFollowUpRow}>
                    <Text style={styles.voiceFollowUpRowQuestion}>{row.question}</Text>
                    <Text style={styles.voiceFollowUpRowMeta}>
                      {row.speaker} · {row.createdAt}
                    </Text>
                    <View style={styles.voiceFollowUpRowActions}>
                      <Pressable
                        style={styles.voiceFollowUpRowButton}
                        onPress={() => {
                          Speech.stop();
                          Speech.speak(row.question, {
                            language: "en-US",
                            rate: Platform.OS === "ios" ? 0.92 : 1,
                          });
                        }}
                        accessibilityRole="button"
                        accessibilityLabel="Hear question"
                      >
                        <Ionicons name="volume-high-outline" size={16} color="#0f766e" />
                        <Text style={styles.voiceFollowUpRowButtonText}>Hear</Text>
                      </Pressable>
                      <Pressable
                        style={styles.voiceFollowUpRowButtonPrimary}
                        onPress={() => {
                          Speech.stop();
                          updateVoiceFollowUp({ question: row.question });
                          setActiveTab("record");
                        }}
                        accessibilityRole="button"
                        accessibilityLabel="Answer by voice"
                      >
                        <Text style={styles.voiceFollowUpRowButtonPrimaryText}>Answer</Text>
                      </Pressable>
                    </View>
                  </View>
                ))}
              </View>
            ) : null}

            {aiInsightsOverview.bySpeaker.length > 0 ? (
              <View style={styles.aiSectionPanel}>
                <Text style={styles.aiSectionHeading}>By speaker</Text>
                <Text style={styles.aiSectionSub}>
                  Per-person AI: calendar = today’s plan from today’s clips (other logs as context); sparkles = briefing
                  from all their clips.
                </Text>
                {aiInsightsOverview.bySpeaker.map((rollup) => (
                  <View key={rollup.speakerName} style={styles.aiSpeakerCard}>
                    <View style={styles.aiSpeakerCardHeader}>
                      <View style={styles.aiSpeakerCardTitleBlock}>
                        <Text style={styles.aiSpeakerCardTitle}>{rollup.speakerName}</Text>
                        <Text style={styles.aiSpeakerCardCount}>{rollup.entryCount} with AI</Text>
                      </View>
                      <View style={styles.speakerCardIconRow}>
                        <Pressable
                          style={styles.personSummaryIconButton}
                          onPress={() => void generateTodayPlanForSpeaker(rollup.speakerName)}
                          disabled={
                            speakerAiLoading?.kind === "today" &&
                            speakerAiLoading.key === normalizeSpeakerKey(rollup.speakerName)
                          }
                          accessibilityRole="button"
                          accessibilityLabel={`Today's plan for ${rollup.speakerName}`}
                        >
                          {speakerAiLoading?.kind === "today" &&
                          speakerAiLoading.key === normalizeSpeakerKey(rollup.speakerName) ? (
                            <ActivityIndicator size="small" color="#0f766e" />
                          ) : (
                            <Ionicons name="calendar-outline" size={22} color="#0f766e" />
                          )}
                        </Pressable>
                        <Pressable
                          style={styles.personSummaryIconButton}
                          onPress={() => void generatePersonVoiceSummary(rollup.speakerName)}
                          disabled={
                            speakerAiLoading?.kind === "summary" &&
                            speakerAiLoading.key === normalizeSpeakerKey(rollup.speakerName)
                          }
                          accessibilityRole="button"
                          accessibilityLabel={`Generate voice summary for ${rollup.speakerName}`}
                        >
                          {speakerAiLoading?.kind === "summary" &&
                          speakerAiLoading.key === normalizeSpeakerKey(rollup.speakerName) ? (
                            <ActivityIndicator size="small" color="#0f766e" />
                          ) : (
                            <Ionicons name="sparkles" size={22} color="#0f766e" />
                          )}
                        </Pressable>
                      </View>
                    </View>
                    {rollup.topics.length > 0 ? (
                      <View style={styles.aiChipWrap}>
                        {rollup.topics.map((t, i) => (
                          <View key={`${rollup.speakerName}-topic-${i}`} style={styles.aiChipMuted}>
                            <Text style={styles.aiChipTextMuted}>
                              {t.display} <Text style={styles.aiChipCountMuted}>({t.count})</Text>
                            </Text>
                          </View>
                        ))}
                      </View>
                    ) : null}
                    {rollup.actionItems.length > 0 ? (
                      <>
                        <Text style={styles.aiMiniHeading}>Their action items</Text>
                        {rollup.actionItems.map((line, idx) => (
                          <Text key={`${rollup.speakerName}-act-${idx}`} style={styles.aiBulletMuted}>
                            • {line}
                          </Text>
                        ))}
                      </>
                    ) : null}
                    {rollup.summaries.length > 0 ? (
                      <>
                        <Text style={styles.aiMiniHeading}>Recent summaries</Text>
                        {rollup.summaries.map((s, idx) => (
                          <View key={`${rollup.speakerName}-sum-${idx}`} style={styles.aiSummaryBlock}>
                            <Text style={styles.aiSummaryMeta}>{s.createdAt}</Text>
                            <Text style={styles.aiSummaryBody}>{s.text}</Text>
                          </View>
                        ))}
                      </>
                    ) : null}
                  </View>
                ))}
              </View>
            ) : null}
          </ScrollView>
        ) : null}
      </View>
    </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  appRoot: {
    flex: 1,
    backgroundColor: "#f5f3ef",
  },
  screen: {
    flex: 1,
    backgroundColor: "transparent",
    justifyContent: "flex-start",
    paddingHorizontal: 18,
    paddingVertical: 16,
  },
  cardHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 2,
  },
  cardHeaderMain: {
    flex: 1,
    minWidth: 0,
    alignItems: "center",
  },
  familyMascot: {
    width: 76,
    height: 76,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: "rgba(20, 184, 166, 0.3)",
    backgroundColor: "#fffefb",
  },
  card: {
    borderRadius: 26,
    backgroundColor: "rgba(255, 255, 255, 0.88)",
    paddingVertical: 22,
    paddingHorizontal: 20,
    borderWidth: 1.5,
    borderColor: "rgba(253, 186, 116, 0.45)",
    overflow: "hidden",
    shadowColor: "#0d9488",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 24,
    elevation: 8,
  },
  kicker: {
    color: "#0f172a",
    fontSize: 32,
    fontWeight: "800",
    letterSpacing: -0.8,
    lineHeight: 38,
    marginBottom: 4,
    textAlign: "center",
    alignSelf: "stretch",
  },
  headlineAccent: {
    alignSelf: "center",
    width: 112,
    height: 3,
    borderRadius: 2,
    marginBottom: 12,
    marginTop: 2,
  },
  subtitle: {
    color: "#334155",
    marginTop: 0,
    marginBottom: 18,
    fontSize: 14,
    lineHeight: 20,
    textAlign: "center",
    fontWeight: "500",
    opacity: 1,
  },
  tabBar: {
    flexDirection: "row",
    gap: 6,
    marginBottom: 16,
    borderRadius: 16,
    backgroundColor: "rgba(255, 255, 255, 0.75)",
    borderWidth: 1,
    borderColor: "rgba(20, 184, 166, 0.25)",
    padding: 5,
  },
  tabButton: {
    flex: 1,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
  },
  tabButtonInner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  tabButtonActive: {
    backgroundColor: "#14b8a6",
    borderWidth: 1,
    borderColor: "rgba(204, 251, 241, 0.8)",
    shadowColor: "#0d9488",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.22,
    shadowRadius: 8,
    elevation: 4,
  },
  tabButtonText: {
    color: "#475569",
    fontSize: 11,
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
  aiTabScroll: {
    marginTop: 2,
    minHeight: 300,
    maxHeight: 560,
  },
  aiTabContent: {
    paddingBottom: 24,
    flexGrow: 1,
  },
  aiOverviewPanel: {
    marginTop: 4,
    borderRadius: 18,
    backgroundColor: "#eef6f4",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.28)",
    paddingBottom: 14,
  },
  aiOverviewLine: {
    color: "#334155",
    fontSize: 13,
    lineHeight: 20,
    marginHorizontal: 14,
    marginTop: 4,
    fontWeight: "600",
  },
  aiClearAiButton: {
    alignSelf: "flex-start",
    marginHorizontal: 14,
    marginTop: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(220, 38, 38, 0.45)",
    backgroundColor: "rgba(254, 242, 242, 0.9)",
  },
  aiClearAiButtonText: {
    color: "#b91c1c",
    fontSize: 12,
    fontWeight: "800",
  },
  aiOverviewButtonCol: {
    marginTop: 10,
    marginHorizontal: 14,
    gap: 10,
  },
  aiOverviewPanelTightButton: {
    marginHorizontal: 0,
    marginTop: 0,
    alignSelf: "stretch",
  },
  aiWeekTasksButton: {
    alignSelf: "stretch",
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(13, 148, 136, 0.55)",
    backgroundColor: "rgba(236, 253, 245, 0.95)",
    alignItems: "center",
  },
  aiWeekTasksButtonText: {
    color: "#0f766e",
    fontSize: 12,
    fontWeight: "800",
    textAlign: "center",
  },
  aiSectionPanel: {
    marginTop: 14,
    borderRadius: 18,
    backgroundColor: "#eef6f4",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.28)",
    paddingVertical: 14,
    paddingHorizontal: 14,
  },
  aiPanelInset: {
    marginHorizontal: 14,
    marginTop: 8,
    marginBottom: 4,
  },
  aiHelpBullet: {
    color: "#475569",
    fontSize: 13,
    lineHeight: 20,
    marginTop: 8,
    fontWeight: "500",
  },
  aiSectionHeading: {
    color: "#0f172a",
    fontSize: 15,
    fontWeight: "800",
    marginBottom: 4,
  },
  aiSectionSub: {
    color: "#64748b",
    fontSize: 12,
    lineHeight: 17,
    marginBottom: 12,
  },
  aiChipWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  aiChip: {
    borderRadius: 999,
    backgroundColor: "rgba(20, 184, 166, 0.2)",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.35)",
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  aiChipText: {
    color: "#0f766e",
    fontSize: 12,
    fontWeight: "700",
  },
  aiChipCount: {
    color: "#115e59",
    fontWeight: "600",
  },
  aiChipMuted: {
    borderRadius: 999,
    backgroundColor: "rgba(255, 255, 255, 0.9)",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.5)",
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  aiChipTextMuted: {
    color: "#334155",
    fontSize: 11,
    fontWeight: "600",
  },
  aiChipCountMuted: {
    color: "#64748b",
    fontWeight: "600",
  },
  aiActionRow: {
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(15, 118, 110, 0.12)",
  },
  aiActionBullet: {
    color: "#0f172a",
    fontSize: 14,
    lineHeight: 21,
    fontWeight: "600",
  },
  aiActionMeta: {
    color: "#64748b",
    fontSize: 11,
    marginTop: 4,
    fontWeight: "500",
  },
  aiSpeakerCard: {
    marginTop: 12,
    padding: 14,
    borderRadius: 14,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.22)",
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  aiSpeakerCardHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 10,
    gap: 8,
  },
  aiSpeakerCardTitleBlock: {
    flex: 1,
    minWidth: 0,
  },
  personSummaryIconButton: {
    padding: 8,
    borderRadius: 12,
    backgroundColor: "rgba(204, 251, 241, 0.75)",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.35)",
  },
  speakerCardIconRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  trafficDestinationInput: {
    marginTop: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.35)",
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    color: "#1e293b",
  },
  trafficButtonRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 10,
  },
  trafficSecondaryButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
    backgroundColor: "rgba(255, 255, 255, 0.9)",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.35)",
  },
  trafficSecondaryButtonText: {
    color: "#0f766e",
    fontSize: 12,
    fontWeight: "700",
  },
  trafficPrimaryButton: {
    marginTop: 10,
    paddingVertical: 11,
    borderRadius: 12,
    backgroundColor: "#14b8a6",
    borderWidth: 1,
    borderColor: "rgba(204, 251, 241, 0.9)",
    alignItems: "center",
  },
  trafficPrimaryButtonText: {
    color: "#ffffff",
    fontSize: 13,
    fontWeight: "800",
  },
  trafficStatusText: {
    marginTop: 10,
    color: "#334155",
    fontSize: 13,
    lineHeight: 20,
    fontWeight: "600",
  },
  aiSpeakerCardTitle: {
    color: "#0f172a",
    fontSize: 16,
    fontWeight: "800",
  },
  aiSpeakerCardCount: {
    color: "#0f766e",
    fontSize: 12,
    fontWeight: "700",
  },
  aiMiniHeading: {
    color: "#047857",
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginTop: 12,
    marginBottom: 6,
  },
  aiBulletMuted: {
    color: "#475569",
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 4,
  },
  aiSummaryBlock: {
    marginTop: 8,
    padding: 10,
    borderRadius: 10,
    backgroundColor: "#f8fafc",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  aiSummaryMeta: {
    color: "#64748b",
    fontSize: 11,
    fontWeight: "600",
    marginBottom: 6,
  },
  aiSummaryBody: {
    color: "#1e293b",
    fontSize: 13,
    lineHeight: 20,
  },
  tabIntro: {
    color: "#334155",
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 16,
    textAlign: "center",
    fontWeight: "500",
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
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(20, 184, 166, 0.4)",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
    backgroundColor: "rgba(204, 251, 241, 0.5)",
  },
  smallActionButtonText: {
    color: "#115e59",
    fontSize: 12,
    fontWeight: "700",
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
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: "rgba(255, 251, 235, 0.95)",
    borderWidth: 2,
    borderColor: "rgba(251, 191, 36, 0.55)",
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#f59e0b",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 10,
    elevation: 6,
  },
  fabPrimary: {
    width: 92,
    height: 92,
    borderRadius: 46,
    backgroundColor: "#14b8a6",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 3,
    borderColor: "rgba(255, 255, 255, 0.85)",
    shadowColor: "#0d9488",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.28,
    shadowRadius: 16,
    elevation: 10,
  },
  fabRecording: {
    backgroundColor: "#f43f5e",
    borderColor: "rgba(255, 255, 255, 0.9)",
    shadowColor: "#fb7185",
  },
  fabUploading: {
    backgroundColor: "#94a3b8",
    opacity: 0.95,
  },
  controlDisabled: {
    opacity: 0.45,
  },
  heroCaption: {
    textAlign: "center",
    color: "#57534e",
    fontSize: 12,
    fontWeight: "600",
    marginBottom: 10,
    letterSpacing: 0.15,
  },
  enrollChip: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "center",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: "rgba(204, 251, 241, 0.85)",
    borderWidth: 1,
    borderColor: "rgba(20, 184, 166, 0.4)",
    marginBottom: 14,
  },
  enrollChipText: {
    color: "#115e59",
    fontSize: 12,
    fontWeight: "700",
  },
  voiceFollowUpBanner: {
    marginBottom: 14,
    padding: 14,
    borderRadius: 16,
    backgroundColor: "#ecfdf5",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.35)",
  },
  voiceFollowUpLabel: {
    color: "#047857",
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 0.4,
    marginBottom: 6,
  },
  voiceFollowUpQuestion: {
    color: "#0f172a",
    fontSize: 15,
    lineHeight: 22,
    fontWeight: "700",
    marginBottom: 12,
  },
  voiceFollowUpActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    alignItems: "center",
  },
  voiceFollowUpButtonSecondary: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.35)",
  },
  voiceFollowUpButtonSecondaryText: {
    color: "#0f766e",
    fontSize: 13,
    fontWeight: "700",
  },
  voiceFollowUpButtonGhost: {
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  voiceFollowUpButtonGhostText: {
    color: "#64748b",
    fontSize: 13,
    fontWeight: "600",
  },
  voiceFollowUpRow: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(15, 118, 110, 0.12)",
  },
  voiceFollowUpRowQuestion: {
    color: "#0f172a",
    fontSize: 14,
    lineHeight: 21,
    fontWeight: "600",
  },
  voiceFollowUpRowMeta: {
    color: "#64748b",
    fontSize: 11,
    marginTop: 4,
    fontWeight: "500",
  },
  voiceFollowUpRowActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 10,
    alignItems: "center",
  },
  voiceFollowUpRowButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 10,
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.28)",
  },
  voiceFollowUpRowButtonText: {
    color: "#0f766e",
    fontSize: 12,
    fontWeight: "700",
  },
  voiceFollowUpRowButtonPrimary: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    backgroundColor: "#14b8a6",
    borderWidth: 1,
    borderColor: "rgba(204, 251, 241, 0.9)",
  },
  voiceFollowUpRowButtonPrimaryText: {
    color: "#ffffff",
    fontSize: 12,
    fontWeight: "800",
  },
  historyFollowUpItem: {
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(100, 116, 139, 0.2)",
  },
  historyFollowUpItemText: {
    color: "#334155",
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 6,
  },
  historyFollowUpItemActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  historyFollowUpTinyButton: {
    padding: 6,
    borderRadius: 8,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.25)",
  },
  historyFollowUpAnswerLink: {
    paddingVertical: 4,
  },
  historyFollowUpAnswerLinkText: {
    color: "#0d9488",
    fontSize: 12,
    fontWeight: "800",
  },
  historyAnsweredFollowUp: {
    marginTop: 8,
    marginBottom: 4,
    padding: 10,
    borderRadius: 10,
    backgroundColor: "#f0fdfa",
    borderWidth: 1,
    borderColor: "rgba(20, 184, 166, 0.35)",
  },
  historyAnsweredFollowUpText: {
    color: "#134e4a",
    fontSize: 13,
    lineHeight: 19,
    fontWeight: "600",
  },
  historyToggleButton: {
    marginTop: 2,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(20, 184, 166, 0.55)",
    backgroundColor: "rgba(255, 255, 255, 0.55)",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
  },
  historyToggleButtonText: {
    color: "#115e59",
    fontSize: 14,
    fontWeight: "600",
  },
  errorText: {
    color: "#dc2626",
    fontSize: 13,
    marginTop: 6,
    marginBottom: 10,
  },
  transcriptPanel: {
    marginTop: 14,
    borderRadius: 18,
    backgroundColor: "#eef6f4",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.28)",
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
    color: "#0f172a",
    fontSize: 13,
    fontWeight: "700",
  },
  clearTranscriptButton: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(249, 115, 22, 0.5)",
    paddingHorizontal: 12,
    paddingVertical: 7,
    backgroundColor: "rgba(254, 243, 199, 0.65)",
  },
  clearTranscriptButtonDisabled: {
    opacity: 0.4,
  },
  clearTranscriptButtonText: {
    color: "#c2410c",
    fontSize: 12,
    fontWeight: "700",
  },
  panelTitle: {
    color: "#0f172a",
    fontSize: 13,
    fontWeight: "700",
    marginHorizontal: 14,
    marginTop: 14,
  },
  speakerLabel: {
    color: "#134e4a",
    fontSize: 12,
    fontWeight: "700",
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
    color: "#0f172a",
    fontSize: 16,
    lineHeight: 24,
  },
  historyPanel: {
    marginTop: 14,
    borderRadius: 18,
    backgroundColor: "#eef6f4",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.28)",
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
    color: "#334155",
    fontSize: 12,
    fontWeight: "600",
    marginTop: 14,
  },
  historyItem: {
    marginBottom: 10,
    borderRadius: 14,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.32)",
    paddingHorizontal: 12,
    paddingVertical: 10,
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
    elevation: 2,
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
    color: "#0f172a",
    fontSize: 13,
    fontWeight: "800",
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
    color: "#134e4a",
    fontSize: 12,
    fontWeight: "800",
    marginLeft: 2,
  },
  historyMetaRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
    gap: 10,
  },
  historyEntryActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  refreshAiButton: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#99f6e4",
    backgroundColor: "#f0fdfa",
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  refreshAiButtonText: {
    color: "#0f766e",
    fontSize: 11,
    fontWeight: "700",
  },
  historyPhoneBlock: {
    marginTop: 8,
    marginBottom: 10,
    padding: 10,
    borderRadius: 10,
    backgroundColor: "#f0f9ff",
    borderWidth: 1,
    borderColor: "#bae6fd",
  },
  historyPhoneHint: {
    fontSize: 12,
    color: "#0369a1",
    marginBottom: 8,
    lineHeight: 17,
  },
  historyPhoneChips: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  historyPhoneChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
    backgroundColor: "#e0f2fe",
    borderWidth: 1,
    borderColor: "#7dd3fc",
  },
  historyPhoneChipText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#0c4a6e",
  },
  historyDiscoverRow: {
    marginTop: 10,
    marginBottom: 8,
    padding: 10,
    borderRadius: 10,
    backgroundColor: "#f8fafc",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.2)",
  },
  historyDiscoverActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 6,
  },
  historyDiscoverButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.28)",
  },
  historyDiscoverButtonText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#0f172a",
  },
  recordDiscoverRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    paddingHorizontal: 14,
    paddingBottom: 12,
    marginTop: 4,
  },
  contactLookupModalCard: {
    maxHeight: "82%" as const,
    width: "92%",
    maxWidth: 440,
  },
  contactLookupScroll: {
    maxHeight: 280,
    marginTop: 6,
  },
  contactLookupEmpty: {
    color: "#64748b",
    fontSize: 14,
    paddingVertical: 16,
    textAlign: "center",
  },
  contactLookupRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: "#f1f5f9",
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  contactLookupRowText: {
    flex: 1,
    marginRight: 10,
  },
  contactLookupName: {
    fontSize: 15,
    fontWeight: "700",
    color: "#0f172a",
  },
  contactLookupMeta: {
    fontSize: 12,
    color: "#475569",
    marginTop: 2,
  },
  webSearchInputMultiline: {
    minHeight: 72,
    textAlignVertical: "top",
  },
  historySpeakerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  historyTimestamp: {
    color: "#475569",
    fontSize: 11,
    fontWeight: "600",
  },
  historySpeaker: {
    color: "#0f172a",
    fontSize: 13,
    fontWeight: "700",
  },
  historyExpandHint: {
    color: "#047857",
    fontSize: 12,
    fontWeight: "700",
  },
  historyDetailsBox: {
    marginTop: 8,
    borderTopWidth: 1,
    borderTopColor: "rgba(15, 118, 110, 0.22)",
    paddingTop: 8,
  },
  historyText: {
    color: "#0f172a",
    fontSize: 15,
    lineHeight: 22,
    fontWeight: "500",
  },
  historyAttribution: {
    color: "#475569",
    fontSize: 12,
    lineHeight: 17,
    marginBottom: 4,
  },
  historySectionLabel: {
    color: "#047857",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    marginTop: 6,
    marginBottom: 4,
  },
  historyAiBlock: {
    marginTop: 8,
    marginBottom: 6,
    padding: 12,
    borderRadius: 12,
    backgroundColor: "#f8fafc",
    borderWidth: 1,
    borderColor: "rgba(100, 116, 139, 0.25)",
  },
  historyAiBlockTitle: {
    marginTop: 0,
  },
  historyAiSummarySecondary: {
    color: "#64748b",
    fontSize: 12,
    lineHeight: 18,
    marginTop: 2,
    marginBottom: 4,
  },
  historyAiJson: {
    color: "#334155",
    fontSize: 11,
    lineHeight: 16,
    marginTop: 2,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
  },
  historyAiSubLabel: {
    marginTop: 10,
  },
  historyAiBullet: {
    color: "#334155",
    fontSize: 13,
    lineHeight: 20,
    marginLeft: 4,
    marginBottom: 2,
  },
  historyStructuredAction: {
    marginBottom: 10,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(148, 163, 184, 0.35)",
  },
  historyStructuredActionLine: {
    color: "#0f172a",
    fontSize: 13,
    lineHeight: 19,
    fontWeight: "700",
  },
  historyStructuredActionDetail: {
    color: "#475569",
    fontSize: 12,
    lineHeight: 17,
    marginTop: 4,
  },
  historyStructuredActionFallback: {
    color: "#64748b",
    fontSize: 11,
    lineHeight: 16,
    marginTop: 4,
    fontStyle: "italic",
  },
  historyAiTopicsLine: {
    color: "#475569",
    fontSize: 12,
    lineHeight: 18,
    fontWeight: "600",
  },
  historyVoiceMatchDetail: {
    color: "#0f172a",
    fontSize: 11,
    lineHeight: 17,
    marginBottom: 4,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    backgroundColor: "#ecfdf5",
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.35)",
  },
  historyLogicalFlow: {
    color: "#1e293b",
    fontSize: 11,
    lineHeight: 17,
    marginTop: 8,
    marginBottom: 6,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    backgroundColor: "#fffbeb",
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: "rgba(180, 83, 9, 0.35)",
  },
  historyDiagnostics: {
    color: "#1e293b",
    fontSize: 10,
    lineHeight: 15,
    marginTop: 6,
    marginBottom: 8,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    backgroundColor: "#f1f5f9",
    borderRadius: 8,
    padding: 8,
    borderWidth: 1,
    borderColor: "#cbd5e1",
  },
  deleteLogButton: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#fecaca",
    backgroundColor: "#fef2f2",
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  deleteLogButtonText: {
    color: "#b91c1c",
    fontSize: 11,
    fontWeight: "700",
  },
  historyEmptyText: {
    color: "#334155",
    fontSize: 14,
    lineHeight: 20,
    fontWeight: "500",
  },
  speakerListPanel: {
    marginTop: 14,
    borderRadius: 18,
    backgroundColor: "#eef6f4",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.28)",
    paddingBottom: 12,
  },
  speakerListHelp: {
    color: "#475569",
    fontSize: 11,
    lineHeight: 16,
    marginHorizontal: 14,
    marginTop: 4,
    marginBottom: 6,
  },
  speakerListRow: {
    marginHorizontal: 10,
    marginTop: 8,
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderRadius: 14,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(15, 118, 110, 0.32)",
    gap: 10,
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
    elevation: 2,
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
  speakerListActionsInner: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
  },
  speakerListActionLinks: {
    alignItems: "flex-end",
    gap: 8,
  },
  speakerListName: {
    color: "#1e293b",
    fontSize: 13,
    fontWeight: "600",
  },
  speakerListHint: {
    color: "#475569",
    fontSize: 12,
    marginTop: 4,
    lineHeight: 17,
  },
  speakerListHintPlaceholder: {
    color: "#64748b",
    fontSize: 12,
    marginTop: 4,
    fontStyle: "italic",
  },
  speakerListEdit: {
    color: "#047857",
    fontSize: 12,
    fontWeight: "700",
  },
  speakerListActionDisabled: {
    color: "#94a3b8",
  },
  sampleList: {
    marginTop: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(15, 118, 110, 0.22)",
    paddingTop: 8,
    gap: 8,
  },
  profileVectorsSection: {
    marginBottom: 4,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(15, 118, 110, 0.22)",
  },
  profileVectorsHeading: {
    color: "#0f172a",
    fontSize: 12,
    fontWeight: "800",
  },
  profileVectorsHint: {
    color: "#475569",
    fontSize: 10,
    marginTop: 4,
    marginBottom: 6,
    lineHeight: 14,
  },
  recentVectorsLabel: {
    color: "#134e4a",
    fontSize: 11,
    fontWeight: "700",
    marginTop: 10,
  },
  vectorBlock: {
    marginTop: 8,
  },
  vectorBlockTitle: {
    color: "#0d9488",
    fontSize: 11,
    fontWeight: "600",
  },
  vectorBlockSubtitle: {
    color: "#b45309",
    fontSize: 10,
    marginTop: 2,
  },
  vectorBlockEmpty: {
    color: "#94a3b8",
    fontSize: 10,
    fontStyle: "italic",
    marginTop: 4,
  },
  vectorScroll: {
    marginTop: 4,
  },
  vectorMono: {
    color: "#0f172a",
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
    color: "#0f172a",
    fontSize: 12,
    fontWeight: "600",
  },
  sampleMeta: {
    color: "#475569",
    fontSize: 11,
    marginTop: 2,
  },
  sampleDeleteButton: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#fecaca",
    backgroundColor: "#fef2f2",
    paddingHorizontal: 9,
    paddingVertical: 5,
  },
  sampleDeleteText: {
    color: "#b91c1c",
    fontSize: 11,
    fontWeight: "700",
  },
  modalBackdrop: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 20,
    backgroundColor: "rgba(15, 23, 42, 0.45)",
  },
  modalCard: {
    borderRadius: 22,
    backgroundColor: "#fffefb",
    borderWidth: 1,
    borderColor: "rgba(20, 184, 166, 0.35)",
    padding: 20,
    zIndex: 1,
    shadowColor: "#0d9488",
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.2,
    shadowRadius: 28,
    elevation: 16,
  },
  personSummaryModalCard: {
    maxHeight: "85%" as const,
  },
  personSummaryScroll: {
    maxHeight: 280,
    marginTop: 12,
  },
  personSummaryBody: {
    color: "#1e293b",
    fontSize: 15,
    lineHeight: 24,
    fontWeight: "500",
  },
  modalTitle: {
    color: "#134e4a",
    fontSize: 18,
    fontWeight: "800",
    letterSpacing: -0.3,
  },
  modalSubtitle: {
    color: "#64748b",
    fontSize: 12,
    marginTop: 8,
    lineHeight: 18,
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
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(20, 184, 166, 0.35)",
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    color: "#1e293b",
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 14,
    textAlignVertical: "top",
  },
  modalCharCount: {
    color: "#64748b",
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
    color: "#dc2626",
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
    borderColor: "rgba(20, 184, 166, 0.55)",
    backgroundColor: "transparent",
  },
  modalButtonSecondaryText: {
    color: "#0f766e",
    fontSize: 14,
    fontWeight: "700",
  },
  modalButtonPrimary: {
    backgroundColor: "#14b8a6",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.35)",
  },
  modalButtonPrimaryText: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "700",
  },
});
