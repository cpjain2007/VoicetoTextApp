#!/usr/bin/env node
/**
 * Verifies local API .env has a real-looking OPENAI_API_KEY (required for
 * /transcribe-base64 `ai`, POST /history backfill, and POST /ai/insights).
 */
const fs = require("fs");
const path = require("path");
const dotenv = require("dotenv");

const envPath = path.join(__dirname, "..", ".env");

if (!fs.existsSync(envPath)) {
  console.error("Missing VoiceToTextApi/.env — copy .env.example to .env and add keys.");
  process.exit(1);
}

dotenv.config({ path: envPath, override: false });

const key = String(process.env.OPENAI_API_KEY || "").trim();
const placeholders = new Set([
  "",
  "your_openai_api_key_here",
  "sk-your-key-here",
  "replace_me",
]);

const looksPlaceholder =
  placeholders.has(key) ||
  /^your_/i.test(key) ||
  (key.length < 20 && !key.startsWith("sk-"));

if (looksPlaceholder) {
  console.error("OPENAI_API_KEY in VoiceToTextApi/.env is missing or still a placeholder.");
  console.error("Set a real key from OpenAI so the app gets summaries, topics, and action items.");
  process.exit(1);
}

console.log("VoiceToTextApi: OPENAI_API_KEY is set — AI routes enabled on this server.");
