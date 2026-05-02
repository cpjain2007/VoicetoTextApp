/**
 * Firebase Cloud Functions (Gen 2) entry — deploy from repo root: firebase deploy --only functions
 * Public URL ends with function name `api`; append paths like /health, /transcribe.
 * invoker "public" is required for Expo; protect routes with SERVER_BEARER_TOKEN in app.js.
 * minInstances: 1 reduces cold starts (adds baseline cost on Blaze).
 *
 * Heavy deps (firebase-admin, ./app) load inside onInit() so deploy-time discovery does not
 * execute them — avoids "Cannot determine backend specification. Timeout after 10000".
 */
const { onInit } = require("firebase-functions/v2/core");
const { onRequest } = require("firebase-functions/v2/https");
const { defineSecret } = require("firebase-functions/params");

/** Must match process.env name used in app.js for Directions / traffic. Bind with deploy + Secret Manager. */
const googleMapsApiKey = defineSecret("GOOGLE_MAPS_API_KEY");

let expressApp;

onInit(() => {
  const admin = require("firebase-admin");
  admin.initializeApp();
  process.env.SPEAKER_STORE_BACKEND = "firestore";
  expressApp = require("./app");
});

exports.api = onRequest(
  {
    region: "us-central1",
    memory: "2GiB",
    timeoutSeconds: 540,
    concurrency: 2,
    minInstances: 1,
    invoker: "public",
    secrets: [googleMapsApiKey],
  },
  (req, res) => expressApp(req, res),
);
