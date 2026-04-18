"use strict";

const dotProduct = (a, b) => a.reduce((sum, item, index) => sum + item * (b[index] || 0), 0);

const vectorMagnitude = (vector) => Math.sqrt(dotProduct(vector, vector));

const cosineSimilarity = (a, b) => {
  const denominator = vectorMagnitude(a) * vectorMagnitude(b);
  if (!denominator) {
    return 0;
  }
  return dotProduct(a, b) / denominator;
};

const parseInt16LE = (buffer, offset) => buffer.readInt16LE(offset) / 32768;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

/** Minimum PCM16 mono samples (~0.25s at 16kHz) before fingerprinting. */
const MIN_SIGNATURE_SAMPLES = 4000;

/**
 * Build a compact fingerprint from 16-bit LE mono PCM at 16 kHz (from ffmpeg).
 * @param {Buffer} pcmBuffer
 * @returns {number[]}
 */
const buildVoiceSignature = (pcmBuffer) => {
  const totalSamples = Math.floor(pcmBuffer.length / 2);
  if (totalSamples < MIN_SIGNATURE_SAMPLES) {
    throw new Error("Audio clip is too short for speaker recognition.");
  }

  let sumAbs = 0;
  let sumSquares = 0;
  let zeroCrossings = 0;
  let previousSample = 0;
  const frameSize = 320;
  const energyFrames = [];
  let frameEnergy = 0;
  let frameSamples = 0;

  for (let i = 0; i < totalSamples; i += 1) {
    const sample = parseInt16LE(pcmBuffer, i * 2);
    const absSample = Math.abs(sample);
    sumAbs += absSample;
    sumSquares += sample * sample;
    if ((sample >= 0 && previousSample < 0) || (sample < 0 && previousSample >= 0)) {
      zeroCrossings += 1;
    }
    previousSample = sample;
    frameEnergy += sample * sample;
    frameSamples += 1;
    if (frameSamples >= frameSize) {
      energyFrames.push(frameEnergy / frameSamples);
      frameEnergy = 0;
      frameSamples = 0;
    }
  }

  if (frameSamples > 0) {
    energyFrames.push(frameEnergy / frameSamples);
  }

  const meanAbs = sumAbs / totalSamples;
  const rms = Math.sqrt(sumSquares / totalSamples);
  const zcr = zeroCrossings / totalSamples;
  const meanEnergy = energyFrames.reduce((sum, item) => sum + item, 0) / energyFrames.length;
  const energyVariance =
    energyFrames.reduce((sum, item) => sum + (item - meanEnergy) ** 2, 0) / energyFrames.length;
  const dynamicRange = clamp(Math.sqrt(energyVariance) / (meanEnergy + 1e-7), 0, 10);

  return [
    clamp(meanAbs, 0, 1),
    clamp(rms, 0, 1),
    clamp(zcr, 0, 1),
    clamp(meanEnergy, 0, 1),
    clamp(dynamicRange / 10, 0, 1),
  ];
};

const truncateAssemblySpeakerName = (raw, nameMaxLen) => {
  const name = typeof raw === "string" ? raw.trim() : String(raw || "").trim();
  if (!name) {
    return "";
  }
  return name.slice(0, nameMaxLen);
};

/**
 * @param {{ name?: string }[]} profiles
 * @param {string} [manualSpeakerName]
 * @param {{ maxNames?: number; nameMaxLen?: number }} [options]
 */
const buildKnownSpeakerValuesForIdentification = (profiles, manualSpeakerName, options = {}) => {
  const nameMaxLen = options.nameMaxLen ?? 35;
  const maxNames = options.maxNames ?? 20;
  const seen = new Set();
  const out = [];
  const add = (value) => {
    const truncated = truncateAssemblySpeakerName(value, nameMaxLen);
    if (!truncated) {
      return;
    }
    const key = truncated.toLowerCase();
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    out.push(truncated);
  };
  if (Array.isArray(profiles)) {
    for (const profile of profiles) {
      add(profile?.name);
    }
  }
  add(manualSpeakerName);
  return out.slice(0, maxNames);
};

const buildSpeechUnderstandingSpeakerIdentification = (knownValues, enabled) => {
  if (!enabled || knownValues.length === 0) {
    return null;
  }
  return {
    request: {
      speaker_identification: {
        speaker_type: "name",
        known_values: knownValues,
      },
    },
  };
};

const assemblyModelsSupportingSpeakerIdentification = new Set(["universal-3-pro", "universal-2"]);

const payloadAllowsSpeakerIdentification = (payload) => {
  const models = payload?.speech_models;
  if (!Array.isArray(models) || models.length === 0) {
    return false;
  }
  for (const model of models) {
    const key = String(model || "").trim().toLowerCase();
    if (!assemblyModelsSupportingSpeakerIdentification.has(key)) {
      return false;
    }
  }
  return true;
};

const mergeAssemblyTranscriptPayload = (payload, speechUnderstanding) => {
  if (!speechUnderstanding || !payloadAllowsSpeakerIdentification(payload)) {
    return payload;
  }
  return { ...payload, speech_understanding: speechUnderstanding };
};

const formatDominantAssemblySpeakerLabel = (utterances) => {
  if (!Array.isArray(utterances) || utterances.length === 0) {
    return null;
  }
  const raw = utterances[0].speaker;
  if (raw == null || raw === "") {
    return null;
  }
  const label = String(raw).trim();
  if (!label) {
    return null;
  }
  if (/^[A-Za-z]$/.test(label)) {
    return `Speaker ${label.toUpperCase()}`;
  }
  return label;
};

const pickSpeakerIdentificationMapping = (transcript) => {
  const su = transcript?.speech_understanding;
  const resp = su && typeof su === "object" ? su.response : null;
  const block =
    resp && typeof resp === "object" && resp.speaker_identification && typeof resp.speaker_identification === "object"
      ? resp.speaker_identification
      : null;
  if (!block) {
    return null;
  }
  if ("mapping" in block && block.mapping != null) {
    return block.mapping;
  }
  return block;
};

module.exports = {
  MIN_SIGNATURE_SAMPLES,
  cosineSimilarity,
  buildVoiceSignature,
  buildKnownSpeakerValuesForIdentification,
  buildSpeechUnderstandingSpeakerIdentification,
  payloadAllowsSpeakerIdentification,
  mergeAssemblyTranscriptPayload,
  formatDominantAssemblySpeakerLabel,
  pickSpeakerIdentificationMapping,
};
