import { StatusBar } from "expo-status-bar";
import { useEffect, useMemo, useState } from "react";
import { Audio } from "expo-av";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  Alert,
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
};

type TranscriptionResult = {
  text: string;
  detectedSpeakerName: string | null;
  speakerConfidence: number | null;
};

type SpeakerProfile = {
  name: string;
  samples: number;
};

type SpeakerCorrectionSuggestion = {
  shouldSuggest: boolean;
  suggestedSpeakerName: string;
  reason: string;
};

const HISTORY_STORAGE_KEY = "voicetotext.history.v1";

export default function App() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [history, setHistory] = useState<TranscriptLogItem[]>([]);
  const [isHistoryVisible, setIsHistoryVisible] = useState(true);
  const [expandedHistoryIds, setExpandedHistoryIds] = useState<string[]>([]);
  const [speakerName, setSpeakerName] = useState("");
  const [lastSpeakerName, setLastSpeakerName] = useState("Unknown speaker");
  const [isEnrollMode, setIsEnrollMode] = useState(false);
  const [speakers, setSpeakers] = useState<SpeakerProfile[]>([]);
  const [isLoadingSpeakers, setIsLoadingSpeakers] = useState(false);
  const [hasLoadedHistory, setHasLoadedHistory] = useState(false);
  const [statusText, setStatusText] = useState("Tap the mic and start speaking.");
  const [errorText, setErrorText] = useState<string | null>(null);
  const speakerAutoAssignMinConfidence = Number(process.env.EXPO_PUBLIC_SPEAKER_MIN_CONFIDENCE || "0.975");

  const getApiBaseUrl = () => {
    const apiUrl = process.env.EXPO_PUBLIC_TRANSCRIBE_API_URL;
    if (!apiUrl) {
      throw new Error("Missing EXPO_PUBLIC_TRANSCRIBE_API_URL.");
    }
    return apiUrl.replace(/\/transcribe\/?$/, "");
  };

  const canShowTranscript = useMemo(() => transcript.length > 0, [transcript]);
  const isRecording = recording !== null;
  const isBusy = isRecording || isUploading;

  const createTimeLabel = (timestampMs: number) =>
    new Date(timestampMs).toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

  const getDayBucket = (timestampMs: number) => {
    const date = new Date(timestampMs);
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const oneDayMs = 24 * 60 * 60 * 1000;
    if (timestampMs >= todayStart) {
      return "Today";
    }
    if (timestampMs >= todayStart - oneDayMs) {
      return "Yesterday";
    }
    return date.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
    });
  };

  const transcribeAudio = async (uri: string, currentSpeakerName: string) => {
    const apiUrl = process.env.EXPO_PUBLIC_TRANSCRIBE_API_URL;
    const apiToken = process.env.EXPO_PUBLIC_TRANSCRIBE_API_TOKEN;
    if (!apiUrl) {
      throw new Error("Missing EXPO_PUBLIC_TRANSCRIBE_API_URL.");
    }

    const formData = new FormData();
    formData.append("file", {
      uri,
      name: "voice-note.m4a",
      type: "audio/m4a",
    } as any);
    const trimmedSpeakerName = currentSpeakerName.trim();

    const requestHeaders: Record<string, string> = {};
    if (apiToken) {
      requestHeaders.Authorization = `Bearer ${apiToken}`;
    }
    if (trimmedSpeakerName) {
      requestHeaders["x-speaker-name"] = trimmedSpeakerName;
    }

    const requestUrl =
      trimmedSpeakerName && apiUrl.includes("?")
        ? `${apiUrl}&speakerName=${encodeURIComponent(trimmedSpeakerName)}`
        : trimmedSpeakerName
          ? `${apiUrl}?speakerName=${encodeURIComponent(trimmedSpeakerName)}`
          : apiUrl;

    const response = await fetch(requestUrl, {
      method: "POST",
      headers: Object.keys(requestHeaders).length > 0 ? requestHeaders : undefined,
      body: formData,
    });

    if (!response.ok) {
      const details = await response.text();
      throw new Error(details || `Transcription failed (${response.status}).`);
    }

    const data = (await response.json()) as {
      text?: string;
      transcript?: string;
      detectedSpeakerName?: string | null;
      speakerConfidence?: number | null;
    };
    return {
      text: data.text?.trim() || data.transcript?.trim() || "",
      detectedSpeakerName: data.detectedSpeakerName?.trim() || null,
      speakerConfidence: typeof data.speakerConfidence === "number" ? data.speakerConfidence : null,
    } satisfies TranscriptionResult;
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

  useEffect(() => {
    fetchSpeakers();
  }, []);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const raw = await AsyncStorage.getItem(HISTORY_STORAGE_KEY);
        if (!raw) {
          return;
        }
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
          return;
        }
        const hydrated = parsed
          .filter((item) => item && typeof item === "object")
          .map((item) => ({
            id: typeof item.id === "string" ? item.id : `${Date.now()}`,
            speakerName: typeof item.speakerName === "string" ? item.speakerName : "Unknown speaker",
            text: typeof item.text === "string" ? item.text : "",
            createdAt: typeof item.createdAt === "string" ? item.createdAt : "",
            createdAtMs: typeof item.createdAtMs === "number" ? item.createdAtMs : Date.now(),
          })) as TranscriptLogItem[];
        setHistory(hydrated);
      } catch {
        // Keep app usable even if local history cannot be read.
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

        const trimmedSpeakerName = speakerName.trim();
        if (isEnrollMode && !trimmedSpeakerName) {
          throw new Error("Enter a speaker name before recording in enroll mode.");
        }

        const result = await transcribeAudio(uri, trimmedSpeakerName);
        const text = result.text;
        const manualSpeakerName = trimmedSpeakerName;
        const hasConfidentDetectedSpeaker =
          !!result.detectedSpeakerName &&
          result.speakerConfidence !== null &&
          result.speakerConfidence >= speakerAutoAssignMinConfidence;
        const normalizedSpeakerName =
          manualSpeakerName ||
          (hasConfidentDetectedSpeaker ? result.detectedSpeakerName : null) ||
          "Unknown speaker";
        setTranscript(text);
        setLastSpeakerName(normalizedSpeakerName);
        const createdAtMs = Date.now();
        const newHistoryEntry: TranscriptLogItem = {
          id: `${createdAtMs}`,
          speakerName: normalizedSpeakerName,
          text,
          createdAt: createTimeLabel(createdAtMs),
          createdAtMs,
        };
        const nextHistory = [newHistoryEntry, ...history];
        setHistory((current) => [
          {
            ...newHistoryEntry,
          },
          ...current,
        ]);
        void maybeSuggestSpeakerCorrection(text, nextHistory);
        if (manualSpeakerName) {
          void fetchSpeakers(true);
        }
        setStatusText(text ? "Transcription complete." : "No speech detected.");
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

  const groupedHistory = useMemo(() => {
    const groups = new Map<string, TranscriptLogItem[]>();
    history.forEach((item) => {
      const timestampMs = item.createdAtMs || Date.parse(item.createdAt) || Date.now();
      const bucket = getDayBucket(timestampMs);
      const existing = groups.get(bucket) || [];
      existing.push(item);
      groups.set(bucket, existing);
    });
    return Array.from(groups.entries());
  }, [history]);

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="light" />
      <View style={styles.card}>
        <Text style={styles.kicker}>VOICE TO TEXT</Text>
        <Text style={styles.title}>Speak. Capture. Review.</Text>
        <Text style={styles.subtitle}>{statusText}</Text>

        <TextInput
          style={styles.speakerInput}
          placeholder={isEnrollMode ? "Required in enroll mode (e.g. Alice)" : "Optional speaker name (e.g. Alice)"}
          placeholderTextColor="#8f98bb"
          value={speakerName}
          onChangeText={setSpeakerName}
          autoCapitalize="words"
        />

        <Pressable
          style={styles.nextSpeakerButton}
          onPress={() => setIsEnrollMode((current) => !current)}
        >
          <Text style={styles.nextSpeakerButtonText}>
            {isEnrollMode ? "Enroll Mode: ON" : "Enroll Mode: OFF"}
          </Text>
        </Pressable>

        <View style={styles.speakerActionsRow}>
          <Pressable style={styles.smallActionButton} onPress={fetchSpeakers}>
            <Text style={styles.smallActionButtonText}>Refresh Speakers</Text>
          </Pressable>
          <Pressable style={styles.smallActionButton} onPress={clearSpeakerProfiles}>
            <Text style={styles.smallActionButtonText}>Reset Speakers</Text>
          </Pressable>
        </View>

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

        <Pressable
          style={styles.historyToggleButton}
          onPress={() => setIsHistoryVisible((current) => !current)}
        >
          <Text style={styles.historyToggleButtonText}>
            {isHistoryVisible ? "Hide History" : "Show History"}
          </Text>
        </Pressable>

        {errorText ? <Text style={styles.errorText}>{errorText}</Text> : null}

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

        {isHistoryVisible ? (
          <View style={styles.historyPanel}>
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
                groupedHistory.map(([groupTitle, items]) => (
                  <View key={groupTitle} style={styles.historyGroup}>
                    <Text style={styles.historyGroupTitle}>{groupTitle}</Text>
                    {items.map((item) => (
                      <View key={item.id} style={styles.historyItem}>
                        <Pressable
                          style={styles.historySpeakerRow}
                          onPress={() => toggleHistoryItemExpanded(item.id)}
                        >
                          <Text style={styles.historySpeaker}>{item.speakerName}</Text>
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
                            <Text style={styles.historyText}>{item.text || "(No speech detected)"}</Text>
                          </View>
                        ) : null}
                      </View>
                    ))}
                  </View>
                ))
              )}
            </ScrollView>
          </View>
        ) : null}

        <View style={styles.speakerListPanel}>
          <Text style={styles.panelTitle}>Known Speakers</Text>
          {isLoadingSpeakers ? (
            <Text style={styles.historyEmptyText}>Loading speaker profiles...</Text>
          ) : speakers.length === 0 ? (
            <Text style={styles.historyEmptyText}>No enrolled speakers yet.</Text>
          ) : (
            speakers.map((speaker) => (
              <Text key={speaker.name} style={styles.speakerListItem}>
                {speaker.name} ({speaker.samples} samples)
              </Text>
            ))
          )}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#0b1020",
    justifyContent: "center",
    paddingHorizontal: 20,
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
    marginBottom: 22,
    fontSize: 14,
  },
  speakerInput: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#3c4d8d",
    color: "#f0f4ff",
    backgroundColor: "#101834",
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 10,
    fontSize: 14,
  },
  nextSpeakerButton: {
    marginBottom: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#4e5fa9",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
  },
  nextSpeakerButtonText: {
    color: "#d7deff",
    fontSize: 13,
    fontWeight: "600",
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
    minHeight: 160,
    maxHeight: 240,
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
  historyGroupTitle: {
    color: "#b9c7f7",
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 6,
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
  speakerListItem: {
    color: "#d7deff",
    fontSize: 13,
    marginHorizontal: 14,
    marginTop: 8,
  },
});
