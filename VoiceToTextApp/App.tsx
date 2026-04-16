import { StatusBar } from "expo-status-bar";
import { useMemo, useState } from "react";
import { Audio } from "expo-av";
import {
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
};

type TranscriptionResult = {
  text: string;
  detectedSpeakerName: string | null;
};

export default function App() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [history, setHistory] = useState<TranscriptLogItem[]>([]);
  const [isHistoryVisible, setIsHistoryVisible] = useState(true);
  const [speakerName, setSpeakerName] = useState("");
  const [lastSpeakerName, setLastSpeakerName] = useState("Speaker 1");
  const [activeAnonymousSpeakerNumber, setActiveAnonymousSpeakerNumber] = useState(1);
  const [statusText, setStatusText] = useState("Tap the mic and start speaking.");
  const [errorText, setErrorText] = useState<string | null>(null);

  const canShowTranscript = useMemo(() => transcript.length > 0, [transcript]);
  const isRecording = recording !== null;
  const isBusy = isRecording || isUploading;

  const createTimestampLabel = () =>
    new Date().toLocaleString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      year: "numeric",
      month: "short",
      day: "2-digit",
    });

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
    if (currentSpeakerName.trim()) {
      formData.append("speakerName", currentSpeakerName.trim());
    }

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: apiToken ? { Authorization: `Bearer ${apiToken}` } : undefined,
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
    };
    return {
      text: data.text?.trim() || data.transcript?.trim() || "",
      detectedSpeakerName: data.detectedSpeakerName?.trim() || null,
    } satisfies TranscriptionResult;
  };

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

        const result = await transcribeAudio(uri, speakerName);
        const text = result.text;
        const manualSpeakerName = speakerName.trim();
        const normalizedSpeakerName =
          manualSpeakerName ||
          result.detectedSpeakerName ||
          `Speaker ${activeAnonymousSpeakerNumber}`;
        setTranscript(text);
        setLastSpeakerName(normalizedSpeakerName);
        setHistory((current) => [
          {
            id: `${Date.now()}`,
            speakerName: normalizedSpeakerName,
            text,
            createdAt: createTimestampLabel(),
          },
          ...current,
        ]);
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

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="light" />
      <View style={styles.card}>
        <Text style={styles.kicker}>VOICE TO TEXT</Text>
        <Text style={styles.title}>Speak. Capture. Review.</Text>
        <Text style={styles.subtitle}>{statusText}</Text>

        <TextInput
          style={styles.speakerInput}
          placeholder="Enter speaker name (e.g. Alice)"
          placeholderTextColor="#8f98bb"
          value={speakerName}
          onChangeText={setSpeakerName}
          autoCapitalize="words"
        />

        <Pressable
          style={styles.nextSpeakerButton}
          onPress={() => setActiveAnonymousSpeakerNumber((current) => current + 1)}
        >
          <Text style={styles.nextSpeakerButtonText}>
            Next anonymous speaker (current: Speaker {activeAnonymousSpeakerNumber})
          </Text>
        </Pressable>

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
            <Text style={styles.panelTitle}>Log History</Text>
            <ScrollView style={styles.historyScroll} contentContainerStyle={styles.historyContent}>
              {history.length === 0 ? (
                <Text style={styles.historyEmptyText}>No transcriptions logged yet.</Text>
              ) : (
                history.map((item) => (
                  <View key={item.id} style={styles.historyItem}>
                    <Text style={styles.historyTimestamp}>{item.createdAt}</Text>
                    <Text style={styles.historySpeaker}>Speaker: {item.speakerName}</Text>
                    <Text style={styles.historyText}>{item.text || "(No speech detected)"}</Text>
                  </View>
                ))
              )}
            </ScrollView>
          </View>
        ) : null}
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
    paddingHorizontal: 14,
    paddingBottom: 16,
  },
  historyItem: {
    marginBottom: 12,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#1f2c57",
  },
  historyTimestamp: {
    color: "#93a4de",
    fontSize: 12,
    marginBottom: 4,
  },
  historySpeaker: {
    color: "#b9c7f7",
    fontSize: 12,
    marginBottom: 4,
    fontWeight: "600",
  },
  historyText: {
    color: "#eaf0ff",
    fontSize: 14,
    lineHeight: 20,
  },
  historyEmptyText: {
    color: "#9ba8d8",
    fontSize: 14,
  },
});
