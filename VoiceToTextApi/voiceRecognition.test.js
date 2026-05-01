"use strict";

/**
 * Manual / integration checks (run API with `npm install` then `npm start`):
 *
 * 1. Fingerprint gate: POST /transcribe-base64 with very short decoded audio should fail with
 *    "too short for speaker recognition" before AssemblyAI is called (if PCM path runs).
 * 2. Enrollment: POST with speakerName + valid clip updates /speakers (samples increment).
 * 3. Match: second clip without speakerName should return detectedSpeakerName when similarity
 *    clears SPEAKER_MATCH_THRESHOLD (same person, similar recording conditions). Old 5-D
 *    profiles are padded and blended on next enroll; re-enroll once if matches feel off.
 * 4. AssemblyAI Speaker ID: multi-speaker file + several enrolled names in speaker-profiles
 *    should return utterances[].speaker as names and speakerIdentificationMapping when enabled.
 * 5. Optional per-profile `speakerDescription` (or `description`) in stored JSON improves AssemblyAI
 *    `speakers[].description` hints (no app UI yet — edit speaker-profiles.json or Firestore doc).
 *
 * Automated tests below cover pure voice-recognition helpers (no network, no ffmpeg).
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const voiceRecognition = require("./voiceRecognition");

/** 16-bit LE mono PCM at 16 kHz (same format as `convertAudioToPcm` in app.js). */
function makePcmSine(samples, frequencyHz = 440, amplitude = 0.25) {
  const buf = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i += 1) {
    const sample = Math.sin((2 * Math.PI * frequencyHz * i) / 16000) * amplitude;
    const int16 = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
    buf.writeInt16LE(int16, i * 2);
  }
  return buf;
}

describe("cosineSimilarity", () => {
  it("returns 1 for identical non-zero vectors", () => {
    const v = [0.1, 0.2, 0.3, 0.4, 0.5];
    assert.equal(voiceRecognition.cosineSimilarity(v, v), 1);
  });

  it("returns 0 when either vector has zero magnitude", () => {
    assert.equal(voiceRecognition.cosineSimilarity([0, 0, 0], [1, 2, 3]), 0);
    assert.equal(voiceRecognition.cosineSimilarity([1, 2, 3], [0, 0, 0]), 0);
  });

  it("returns ~0 for orthogonal vectors", () => {
    const a = [1, 0, 0];
    const b = [0, 1, 0];
    assert.ok(Math.abs(voiceRecognition.cosineSimilarity(a, b)) < 1e-9);
  });
});

describe("buildVoiceSignature", () => {
  it("throws when PCM is shorter than MIN_SIGNATURE_SAMPLES", () => {
    const short = Buffer.alloc(voiceRecognition.MIN_SIGNATURE_SAMPLES * 2 - 4);
    assert.throws(() => voiceRecognition.buildVoiceSignature(short), /too short/i);
  });

  it("returns a VOICE_SIGNATURE_LENGTH normalized vector for valid PCM", () => {
    const pcm = makePcmSine(voiceRecognition.MIN_SIGNATURE_SAMPLES + 800);
    const sig = voiceRecognition.buildVoiceSignature(pcm);
    assert.equal(sig.length, voiceRecognition.VOICE_SIGNATURE_LENGTH);
    for (const x of sig) {
      assert.ok(x >= 0 && x <= 1, `expected clamped [0,1], got ${x}`);
    }
  });

  it("produces stable fingerprints for the same synthetic audio", () => {
    const pcm = makePcmSine(12000, 330, 0.2);
    const a = voiceRecognition.buildVoiceSignature(pcm);
    const b = voiceRecognition.buildVoiceSignature(Buffer.from(pcm));
    assert.deepEqual(a, b);
  });

  it("changes fingerprint when tone differs materially", () => {
    const low = makePcmSine(12000, 120, 0.5);
    const high = makePcmSine(12000, 3000, 0.5);
    const sLow = voiceRecognition.buildVoiceSignature(low);
    const sHigh = voiceRecognition.buildVoiceSignature(high);
    const sim = voiceRecognition.cosineSimilarity(sLow, sHigh);
    assert.ok(sim < 0.995, `expected different tones to diverge, similarity was ${sim}`);
  });
});

describe("padVoiceVector", () => {
  it("pads short legacy vectors with fill for comparison", () => {
    const legacy = [0.1, 0.2, 0.3, 0.4, 0.5];
    const padded = voiceRecognition.padVoiceVector(legacy, 12, 0.5);
    assert.equal(padded.length, 12);
    assert.deepEqual(padded.slice(0, 5), legacy);
    assert.ok(padded.slice(5).every((x) => x === 0.5));
  });

  it("truncates when stored vector is longer than query", () => {
    const long = Array.from({ length: 20 }, (_, i) => i / 20);
    const cut = voiceRecognition.padVoiceVector(long, 12, 0.5);
    assert.equal(cut.length, 12);
    assert.deepEqual(cut, long.slice(0, 12));
  });
});

describe("prioritizeSpeakerNameInKnownList", () => {
  it("moves manual name to the front when it appears later in the list", () => {
    const ordered = voiceRecognition.prioritizeSpeakerNameInKnownList(["Alice", "Bob", "Charlie"], "Charlie");
    assert.deepEqual(ordered, ["Charlie", "Alice", "Bob"]);
  });

  it("leaves list unchanged when manual name is missing or already first", () => {
    assert.deepEqual(voiceRecognition.prioritizeSpeakerNameInKnownList(["Alice", "Bob"], "Zed"), ["Alice", "Bob"]);
    assert.deepEqual(voiceRecognition.prioritizeSpeakerNameInKnownList(["Alice", "Bob"], "Alice"), [
      "Alice",
      "Bob",
    ]);
    assert.deepEqual(voiceRecognition.prioritizeSpeakerNameInKnownList(["Bob"], ""), ["Bob"]);
  });
});

describe("profilesToAssemblySpeakers", () => {
  it("builds name + description objects with custom speakerDescription", () => {
    const speakers = voiceRecognition.profilesToAssemblySpeakers(
      [{ name: "Alice", speakerDescription: "Project lead" }, { name: "Bob" }],
      null,
      { maxNames: 10, nameMaxLen: 35, descriptionMaxLen: 80 },
    );
    assert.equal(speakers.length, 2);
    assert.equal(speakers[0].name, "Alice");
    assert.equal(speakers[0].description, "Project lead");
    assert.equal(speakers[1].name, "Bob");
    assert.ok(speakers[1].description.includes("Voice-to-Text"));
  });

  it("includes manual-only enrollments with default description", () => {
    const speakers = voiceRecognition.profilesToAssemblySpeakers([], "Zed", { maxNames: 5 });
    assert.deepEqual(speakers.map((s) => s.name), ["Zed"]);
    assert.ok(speakers[0].description.length > 10);
  });
});

describe("prioritizeManualSpeakerFirstSpeakers", () => {
  it("moves manual speaker entry to the front", () => {
    const ordered = voiceRecognition.prioritizeManualSpeakerFirstSpeakers(
      [
        { name: "Alice", description: "a" },
        { name: "Bob", description: "b" },
      ],
      "Bob",
    );
    assert.equal(ordered[0].name, "Bob");
    assert.equal(ordered[1].name, "Alice");
  });
});

describe("buildSpeechUnderstandingSpeakerIdentificationFromSpeakers", () => {
  it("uses speakers array instead of known_values", () => {
    const block = voiceRecognition.buildSpeechUnderstandingSpeakerIdentificationFromSpeakers(
      [
        { name: "Ann", description: "QA" },
        { name: "Ben", description: "Dev" },
      ],
      true,
    );
    assert.ok(block.request.speaker_identification.speakers);
    assert.equal(block.request.speaker_identification.speakers.length, 2);
    assert.equal(block.request.speaker_identification.speakers[0].name, "Ann");
    assert.ok(!("known_values" in block.request.speaker_identification));
  });
});

describe("bestCosineScoreAgainstProfile", () => {
  it("returns max similarity across aggregate and recent vectors", () => {
    const sig = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const profile = {
      vector: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      vectorsRecent: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    };
    assert.equal(voiceRecognition.bestCosineScoreAgainstProfile(sig, profile), 1);
  });

  it("returns null when profile has no usable vectors", () => {
    assert.equal(voiceRecognition.bestCosineScoreAgainstProfile([0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], {}), null);
  });
});

describe("bestSpeakerMatchAgainstProfile", () => {
  it("considers aggregate profile vectors even when enrollment samples exist", () => {
    const sig = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const profile = {
      vector: sig,
      enrollmentSamples: [
        {
          sampleId: "sample-low",
          source: "speaker_name_input",
          vector: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
      ],
    };
    const match = voiceRecognition.bestSpeakerMatchAgainstProfile(sig, profile);
    assert.equal(match.score, 1);
    assert.equal(match.sampleSource, "aggregate_profile");
  });
});

describe("buildKnownSpeakerValuesForIdentification", () => {
  it("dedupes case-insensitively and includes manual name", () => {
    const profiles = [{ name: "Alice" }, { name: "  alice  " }, { name: "Bob" }];
    const out = voiceRecognition.buildKnownSpeakerValuesForIdentification(profiles, "Charlie", {
      maxNames: 20,
      nameMaxLen: 35,
    });
    assert.deepEqual(out, ["Alice", "Bob", "Charlie"]);
  });

  it("truncates each name to nameMaxLen", () => {
    const long = "x".repeat(50);
    const out = voiceRecognition.buildKnownSpeakerValuesForIdentification([{ name: long }], null, {
      maxNames: 10,
      nameMaxLen: 12,
    });
    assert.equal(out[0].length, 12);
    assert.equal(out[0], "x".repeat(12));
  });

  it("respects maxNames cap in profile order then manual", () => {
    const profiles = [{ name: "P1" }, { name: "P2" }, { name: "P3" }];
    const out = voiceRecognition.buildKnownSpeakerValuesForIdentification(profiles, "Manual", {
      maxNames: 3,
      nameMaxLen: 35,
    });
    assert.deepEqual(out, ["P1", "P2", "P3"]);
  });
});

describe("buildSpeechUnderstandingSpeakerIdentification", () => {
  it("returns null when disabled or empty known_values", () => {
    assert.equal(voiceRecognition.buildSpeechUnderstandingSpeakerIdentification(["A"], false), null);
    assert.equal(voiceRecognition.buildSpeechUnderstandingSpeakerIdentification([], true), null);
  });

  it("returns AssemblyAI-shaped object when enabled", () => {
    const block = voiceRecognition.buildSpeechUnderstandingSpeakerIdentification(["Alice", "Bob"], true);
    assert.equal(block.request.speaker_identification.speaker_type, "name");
    assert.deepEqual(block.request.speaker_identification.known_values, ["Alice", "Bob"]);
  });
});

describe("mergeAssemblyTranscriptPayload", () => {
  const su = {
    request: { speaker_identification: { speaker_type: "name", known_values: ["A"] } },
  };

  it("adds speech_understanding only for universal-3-pro / universal-2", () => {
    const u3 = voiceRecognition.mergeAssemblyTranscriptPayload(
      { audio: Buffer.from("x"), speaker_labels: true, speech_models: ["universal-3-pro"] },
      su,
    );
    assert.ok("speech_understanding" in u3);

    const u2 = voiceRecognition.mergeAssemblyTranscriptPayload(
      { audio: Buffer.from("x"), speaker_labels: true, speech_models: ["universal-2"] },
      su,
    );
    assert.ok("speech_understanding" in u2);

    const best = voiceRecognition.mergeAssemblyTranscriptPayload(
      { audio: Buffer.from("x"), speaker_labels: true, speech_models: ["best"] },
      su,
    );
    assert.ok(!("speech_understanding" in best));

    const mixed = voiceRecognition.mergeAssemblyTranscriptPayload(
      { audio: Buffer.from("x"), speaker_labels: true, speech_models: ["universal-3-pro", "best"] },
      su,
    );
    assert.ok(!("speech_understanding" in mixed));
  });

  it("does not merge when speech_models is missing", () => {
    const p = { audio: Buffer.from("x"), speaker_labels: true };
    const merged = voiceRecognition.mergeAssemblyTranscriptPayload(p, su);
    assert.equal(merged, p);
  });
});

describe("formatDominantAssemblySpeakerLabel", () => {
  it("formats single-letter diarization labels", () => {
    assert.equal(voiceRecognition.formatDominantAssemblySpeakerLabel([{ speaker: "b", text: "hi" }]), "Speaker B");
  });

  it("passes through resolved names", () => {
    assert.equal(
      voiceRecognition.formatDominantAssemblySpeakerLabel([{ speaker: "Priya", text: "hello" }]),
      "Priya",
    );
  });

  it("returns null for empty utterances", () => {
    assert.equal(voiceRecognition.formatDominantAssemblySpeakerLabel([]), null);
    assert.equal(voiceRecognition.formatDominantAssemblySpeakerLabel(null), null);
  });
});

describe("pickSpeakerIdentificationMapping", () => {
  it("returns mapping when present on transcript", () => {
    const transcript = {
      speech_understanding: {
        response: {
          speaker_identification: {
            status: "success",
            mapping: { A: "Alice", B: "Bob" },
          },
        },
      },
    };
    assert.deepEqual(voiceRecognition.pickSpeakerIdentificationMapping(transcript), {
      A: "Alice",
      B: "Bob",
    });
  });

  it("returns null when block missing", () => {
    assert.equal(voiceRecognition.pickSpeakerIdentificationMapping({ utterances: [] }), null);
  });
});
