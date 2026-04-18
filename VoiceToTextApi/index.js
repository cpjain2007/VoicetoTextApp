/**
 * Firebase Cloud Functions (Gen 2) entry — deploy from repo root: firebase deploy --only functions
 * Public URL ends with function name `api`; append paths like /health, /transcribe.
 * invoker "public" is required for Expo; protect routes with SERVER_BEARER_TOKEN in app.js.
 * minInstances: 1 reduces cold starts (adds baseline cost on Blaze).
 */
const admin = require("firebase-admin");
const { onRequest } = require("firebase-functions/v2/https");

admin.initializeApp();
process.env.SPEAKER_STORE_BACKEND = "firestore";

const app = require("./app");

exports.api = onRequest(
  {
    region: "us-central1",
    memory: "2GiB",
    timeoutSeconds: 540,
    concurrency: 2,
    minInstances: 1,
    invoker: "public",
  },
  app,
);
