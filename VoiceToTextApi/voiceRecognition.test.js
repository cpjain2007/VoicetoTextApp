"use strict";

/**
 * Manual / integration checks (run API with `npm install` then `npm start`):
 *
 * 1. Fingerprint gate: POST /transcribe-base64 with very short decoded audio should fail with
 *    "too short for speaker recognition" before AssemblyAI is called (if PCM path runs).
 * 2. Enrollment: POST with speakerName + valid clip updates /speakers (samples increment).
 * 3. Match: second clip without speakerName should return detectedSpeakerName when similarity
 *    clears SPEAKER_MATCH_THRESHOLD (same person, similar recording conditions).
 * 4. AssemblyAI Speaker ID: multi-speaker file + several enrolled names in speaker-profiles
 *    should return utterances[].speaker as names and speakerIdentificationMapping when enabled.
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

  it("returns a 5-dimensional normalized vector for valid PCM", () => {
    const pcm = makePcmSine(voiceRecognition.MIN_SIGNATURE_SAMPLES + 800);
    const sig = voiceRecognition.buildVoiceSignature(pcm);
    assert.equal(sig.length, 5);
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
    assert.ok(sim < 0.999, `expected different tones to diverge, similarity was ${sim}`);
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
