const fs = require("fs/promises");
const path = require("path");

const filePath = () => path.join(__dirname, "speaker-profiles.json");

const parseFirestorePath = () => {
  const raw = (process.env.SPEAKER_FIRESTORE_DOC || "apiState/speakerProfiles").trim();
  const parts = raw.split("/").filter(Boolean);
  if (parts.length !== 2) {
    throw new Error("SPEAKER_FIRESTORE_DOC must be exactly 'collectionId/documentId'.");
  }
  return { collectionId: parts[0], docId: parts[1] };
};

async function readFromFile() {
  try {
    const raw = await fs.readFile(filePath(), "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed.profiles) ? parsed.profiles : [];
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

async function writeToFile(profiles) {
  const payload = JSON.stringify({ profiles }, null, 2);
  await fs.writeFile(filePath(), payload, "utf8");
}

function getFirestore() {
  const { getFirestore } = require("firebase-admin/firestore");
  return getFirestore();
}

function speakerDocRef() {
  const { collectionId, docId } = parseFirestorePath();
  return getFirestore().collection(collectionId).doc(docId);
}

async function readFromFirestore() {
  const snap = await speakerDocRef().get();
  if (!snap.exists) {
    return [];
  }
  const data = snap.data();
  return Array.isArray(data?.profiles) ? data.profiles : [];
}

async function writeToFirestore(profiles) {
  await speakerDocRef().set(
    { profiles, updatedAt: new Date().toISOString() },
    { merge: true },
  );
}

const useFirestore = () => process.env.SPEAKER_STORE_BACKEND === "firestore";

module.exports = {
  readSpeakerProfiles: async () => (useFirestore() ? readFromFirestore() : readFromFile()),
  writeSpeakerProfiles: async (profiles) =>
    useFirestore() ? writeToFirestore(profiles) : writeToFile(profiles),
};
