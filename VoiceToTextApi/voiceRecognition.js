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

/** Current fingerprint length (v1 profiles in the wild used 5). */
const VOICE_SIGNATURE_LENGTH = 12;

const padVoiceVector = (vec, targetLen, fill = 0.5) => {
  if (!Array.isArray(vec) || vec.length === 0) {
    return Array.from({ length: targetLen }, () => fill);
  }
  const out = vec.slice(0, targetLen);
  while (out.length < targetLen) {
    out.push(fill);
  }
  return out;
};

/**
 * Build a compact fingerprint from 16-bit LE mono PCM at 16 kHz (from ffmpeg).
 * Extends the original 5 statistics with frame dynamics, silence, peakiness,
 * temporal energy split, and lag-1 autocorrelation for better speaker separation
 * without native ML dependencies.
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
  let maxAbs = 0;
  let silentCount = 0;
  const silenceGate = 0.02;
  const frameSize = 320;
  const energyFrames = [];
  let frameEnergy = 0;
  let frameSamples = 0;
  let lag1Product = 0;
  let prevForLag = 0;

  const half = Math.floor(totalSamples / 2);
  let sumSqFirst = 0;
  let sumSqSecond = 0;

  for (let i = 0; i < totalSamples; i += 1) {
    const sample = parseInt16LE(pcmBuffer, i * 2);
    const absSample = Math.abs(sample);
    sumAbs += absSample;
    sumSquares += sample * sample;
    if (absSample > maxAbs) {
      maxAbs = absSample;
    }
    if (absSample < silenceGate) {
      silentCount += 1;
    }
    if (i > 0) {
      lag1Product += sample * prevForLag;
    }
    prevForLag = sample;
    if (i < half) {
      sumSqFirst += sample * sample;
    } else {
      sumSqSecond += sample * sample;
    }
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
  const silenceFraction = silentCount / totalSamples;
  const peakToRms = clamp(maxAbs / (rms + 1e-7), 0, 40) / 40;
  const stdFrames = Math.sqrt(energyVariance);
  const frameEnergyCv = clamp(stdFrames / (meanEnergy + 1e-7), 0, 6) / 6;

  let meanLogDelta = 0;
  if (energyFrames.length > 1) {
    let acc = 0;
    for (let fi = 1; fi < energyFrames.length; fi += 1) {
      acc += Math.abs(Math.log(energyFrames[fi] + 1e-10) - Math.log(energyFrames[fi - 1] + 1e-10));
    }
    meanLogDelta = acc / (energyFrames.length - 1);
  }
  const frameLogDeltaNorm = clamp(meanLogDelta / 4, 0, 1);

  let maxFrameEnergy = 0;
  for (const ef of energyFrames) {
    if (ef > maxFrameEnergy) {
      maxFrameEnergy = ef;
    }
  }
  const framePeakiness = clamp(maxFrameEnergy / (meanEnergy + 1e-7), 0, 15) / 15;

  const halfRatio =
    sumSqSecond > 1e-12
      ? clamp(sumSqFirst / (sumSqSecond + 1e-12), 0, 4) / 4
      : clamp(sumSqFirst, 0, 1);

  const lagNorm = lag1Product / (sumSquares + 1e-10);
  const autocorrNorm = clamp((clamp(lagNorm, -1, 1) + 1) / 2, 0, 1);

  const base = [
    clamp(meanAbs, 0, 1),
    clamp(rms, 0, 1),
    clamp(zcr, 0, 1),
    clamp(meanEnergy, 0, 1),
    clamp(dynamicRange / 10, 0, 1),
    clamp(silenceFraction, 0, 1),
    clamp(peakToRms, 0, 1),
    clamp(frameEnergyCv, 0, 1),
    clamp(frameLogDeltaNorm, 0, 1),
    clamp(framePeakiness, 0, 1),
    clamp(halfRatio, 0, 1),
    clamp(autocorrNorm, 0, 1),
  ];

  if (base.length !== VOICE_SIGNATURE_LENGTH) {
    throw new Error("Voice signature length mismatch.");
  }
  return base;
};

const truncateAssemblySpeakerName = (raw, nameMaxLen) => {
  const name = typeof raw === "string" ? raw.trim() : String(raw || "").trim();
  if (!name) {
    return "";
  }
  return name.slice(0, nameMaxLen);
};

const DEFAULT_ASSEMBLY_SPEAKER_DESCRIPTION =
  "Speaker enrolled in this Voice-to-Text application; voice profile stored in the app.";

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

/** Put `manualSpeakerName` first in `known_values` when present (helps AssemblyAI priors). */
const prioritizeSpeakerNameInKnownList = (knownValues, manualSpeakerName) => {
  const list = Array.isArray(knownValues) ? [...knownValues] : [];
  const manual = typeof manualSpeakerName === "string" ? manualSpeakerName.trim() : "";
  if (!manual) {
    return list;
  }
  const key = manual.toLowerCase();
  const idx = list.findIndex((n) => String(n).trim().toLowerCase() === key);
  if (idx <= 0) {
    return list;
  }
  const [picked] = list.splice(idx, 1);
  return [picked, ...list];
};

/**
 * @param {{ name?: string; description?: string; speakerDescription?: string }[]} profiles
 * @param {string} [manualSpeakerName]
 * @param {{ maxNames?: number; nameMaxLen?: number; descriptionMaxLen?: number }} [options]
 * @returns {{ name: string; description: string }[]}
 */
const profilesToAssemblySpeakers = (profiles, manualSpeakerName, options = {}) => {
  const nameMaxLen = options.nameMaxLen ?? 35;
  const descMaxLen = options.descriptionMaxLen ?? 220;
  const maxNames = options.maxNames ?? 20;
  const seen = new Map();

  const addOne = (rawName, rawDesc) => {
    const name = truncateAssemblySpeakerName(rawName, nameMaxLen);
    if (!name) {
      return;
    }
    const key = name.toLowerCase();
    if (seen.has(key)) {
      return;
    }
    const desc =
      typeof rawDesc === "string" && rawDesc.trim()
        ? rawDesc.trim().slice(0, descMaxLen)
        : DEFAULT_ASSEMBLY_SPEAKER_DESCRIPTION;
    seen.set(key, { name, description: desc });
  };

  if (Array.isArray(profiles)) {
    for (const profile of profiles) {
      const custom = profile?.speakerDescription ?? profile?.description;
      addOne(profile?.name, custom);
    }
  }
  addOne(manualSpeakerName, null);

  return [...seen.values()].slice(0, maxNames);
};

const prioritizeManualSpeakerFirstSpeakers = (speakers, manualSpeakerName) => {
  const list = Array.isArray(speakers) ? [...speakers] : [];
  const manual = typeof manualSpeakerName === "string" ? manualSpeakerName.trim() : "";
  if (!manual) {
    return list;
  }
  const key = manual.toLowerCase();
  const idx = list.findIndex((s) => s && String(s.name || "").trim().toLowerCase() === key);
  if (idx <= 0) {
    return list;
  }
  const [picked] = list.splice(idx, 1);
  return [picked, ...list];
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

const buildSpeechUnderstandingSpeakerIdentificationFromSpeakers = (speakers, enabled) => {
  if (!enabled || !Array.isArray(speakers) || speakers.length === 0) {
    return null;
  }
  const cleaned = speakers
    .map((item) => ({
      name: truncateAssemblySpeakerName(item?.name, 35),
      description:
        typeof item?.description === "string" && item.description.trim()
          ? item.description.trim().slice(0, 220)
          : DEFAULT_ASSEMBLY_SPEAKER_DESCRIPTION,
    }))
    .filter((item) => item.name.length > 0);
  if (cleaned.length === 0) {
    return null;
  }
  return {
    request: {
      speaker_identification: {
        speaker_type: "name",
        speakers: cleaned,
      },
    },
  };
};

/**
 * Best cosine similarity between a live signature and stored aggregate + recent snapshots.
 * @param {number[]} signature
 * @param {{ vector?: number[]; vectorsRecent?: number[][] }} profile
 */
const bestCosineScoreAgainstProfile = (signature, profile) => {
  if (!Array.isArray(signature) || signature.length === 0) {
    return null;
  }
  const dim = signature.length;
  const candidates = [];
  if (Array.isArray(profile?.vector) && profile.vector.length > 0) {
    candidates.push(padVoiceVector(profile.vector, dim, 0.5));
  }
  if (Array.isArray(profile?.vectorsRecent)) {
    for (const v of profile.vectorsRecent) {
      if (Array.isArray(v) && v.length > 0) {
        candidates.push(padVoiceVector(v, dim, 0.5));
      }
    }
  }
  if (candidates.length === 0) {
    return null;
  }
  let best = -1;
  for (const c of candidates) {
    const score = cosineSimilarity(signature, c);
    if (score > best) {
      best = score;
    }
  }
  return best;
};

const bestSpeakerMatchAgainstProfile = (signature, profile) => {
  if (!Array.isArray(signature) || signature.length === 0) {
    return null;
  }
  const dim = signature.length;
  const candidates = [];
  if (Array.isArray(profile?.enrollmentSamples)) {
    for (const sample of profile.enrollmentSamples) {
      if (Array.isArray(sample?.vector) && sample.vector.length > 0) {
        candidates.push({
          vector: padVoiceVector(sample.vector, dim, 0.5),
          sampleId: typeof sample.sampleId === "string" ? sample.sampleId : null,
          sampleSource: typeof sample.source === "string" ? sample.source : null,
          sampleCreatedAtIso: typeof sample.createdAtIso === "string" ? sample.createdAtIso : null,
        });
      }
    }
  }
  if (candidates.length === 0 && Array.isArray(profile?.vector) && profile.vector.length > 0) {
    candidates.push({ vector: padVoiceVector(profile.vector, dim, 0.5) });
  }
  if (candidates.length === 0 && Array.isArray(profile?.vectorsRecent)) {
    for (const v of profile.vectorsRecent) {
      if (Array.isArray(v) && v.length > 0) {
        candidates.push({ vector: padVoiceVector(v, dim, 0.5) });
      }
    }
  }
  if (candidates.length === 0) {
    return null;
  }
  let best = null;
  for (const candidate of candidates) {
    const score = cosineSimilarity(signature, candidate.vector);
    if (!best || score > best.score) {
      best = {
        score,
        sampleId: candidate.sampleId || null,
        sampleSource: candidate.sampleSource || null,
        sampleCreatedAtIso: candidate.sampleCreatedAtIso || null,
      };
    }
  }
  return best;
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
  VOICE_SIGNATURE_LENGTH,
  DEFAULT_ASSEMBLY_SPEAKER_DESCRIPTION,
  cosineSimilarity,
  padVoiceVector,
  buildVoiceSignature,
  buildKnownSpeakerValuesForIdentification,
  prioritizeSpeakerNameInKnownList,
  profilesToAssemblySpeakers,
  prioritizeManualSpeakerFirstSpeakers,
  buildSpeechUnderstandingSpeakerIdentification,
  buildSpeechUnderstandingSpeakerIdentificationFromSpeakers,
  bestCosineScoreAgainstProfile,
  payloadAllowsSpeakerIdentification,
  mergeAssemblyTranscriptPayload,
  formatDominantAssemblySpeakerLabel,
  pickSpeakerIdentificationMapping,
  bestSpeakerMatchAgainstProfile,
};
