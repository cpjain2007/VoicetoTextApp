import { StatusBar } from "expo-status-bar";
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
};

type SpeakerProfile = {
  name: string;
  samples: number;
  /** Optional hint for AssemblyAI Speaker Identification (`speakerDescription` in API store). */
  description?: string;
  enrollmentSamples?: EnrollmentSample[];
};

type EnrollmentSample = {
  sampleId: string;
  source?: string | null;
  createdAtMs?: number | null;
  createdAtIso?: string | null;
  historyClientId?: string | null;
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
  const [statusText, setStatusText] = useState("Tap the mic and start speaking.");
  const [errorText, setErrorText] = useState<string | null>(null);
  /** Should match API `SPEAKER_MATCH_THRESHOLD` (same default 0.97) so conflict prompts align with server gating. */
  const speakerAutoAssignMinConfidence = Number(process.env.EXPO_PUBLIC_SPEAKER_MIN_CONFIDENCE || "0.97");

  const unknownSpeakerResolveRef = useRef<((value: string) => void) | null>(null);
  const lastAutoExpandedHistoryIdRef = useRef<string | null>(null);
  const [unknownSpeakerModalVisible, setUnknownSpeakerModalVisible] = useState(false);
  const [unknownSpeakerDraft, setUnknownSpeakerDraft] = useState("");

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
  const isBusy = isRecording || isUploading || unknownSpeakerModalVisible;

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

  const handleRecordPress = async () => {
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

        const createdAtMs = Date.now();
        const historyClientId = `${createdAtMs}`;
        const result = await transcribeAudio(uri, "", {
          historyClientId,
        });
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
        };
        const nextHistory = [newHistoryEntry, ...history];
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
        return;
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
      setStatusText("Listening...");
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
    } finally {
      setIsUploading(false);
    }
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
        <Text style={styles.kicker}>VOICE TO TEXT</Text>
        <Text style={styles.title}>Speak. Capture. Review.</Text>
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
              Start recording to auto-detect the speaker. If no enrolled voice matches, the app will ask who was speaking.
            </Text>

            <Pressable
              style={[styles.recordButton, isBusy && styles.recordButtonActive]}
              onPress={handleRecordPress}
            >
              <Text style={styles.recordButtonText}>
                {isUploading
                  ? "Transcribing..."
                  : isRecording
                    ? "Stop Recording"
                    : "Start Recording"}
              </Text>
            </Pressable>

            <View style={styles.transcriptPanel}>
              <Text style={styles.panelTitle}>Transcript</Text>
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
                Use Show Records to view or delete voice samples. Edit Hint adds context for AssemblyAI.
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
                          {speaker.enrollmentSamples?.map((sample) => (
                            <View key={sample.sampleId} style={styles.sampleRow}>
                              <View style={styles.sampleRowMain}>
                                <Text style={styles.sampleTitle}>{formatSampleTime(sample)}</Text>
                                <Text style={styles.sampleMeta}>
                                  {formatSampleSource(sample.source)} • {sample.sampleId.slice(0, 8)}
                                </Text>
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
    backgroundColor: "#0b1020",
    justifyContent: "flex-start",
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  card: {
    borderRadius: 24,
    backgroundColor: "#141b34",
    padding: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.22,
    shadowRadius: 20,
    elevation: 8,
  },
  kicker: {
    color: "#8ea1ff",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 1.4,
    marginBottom: 8,
  },
  title: {
    color: "#f5f7ff",
    fontSize: 26,
    fontWeight: "700",
  },
  subtitle: {
    color: "#b9c1dd",
    marginTop: 8,
    marginBottom: 14,
    fontSize: 14,
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
    backgroundColor: "#6a7cff",
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
    color: "#9aaad8",
    fontSize: 12,
    lineHeight: 17,
    marginBottom: 10,
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
  recordButton: {
    backgroundColor: "#6a7cff",
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 14,
    marginBottom: 10,
  },
  recordButtonActive: {
    backgroundColor: "#ff6b7d",
  },
  recordButtonText: {
    color: "#ffffff",
    fontSize: 16,
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
  sampleRow: {
    flexDirection: "row",
    alignItems: "center",
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
