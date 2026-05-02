#!/usr/bin/env node
/**
 * Verifies Expo app .env has EXPO_PUBLIC_TRANSCRIBE_API_URL (and optional token)
 * so the client can call /transcribe-base64 and /ai/insights.
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const envPath = path.join(root, ".env");

if (!fs.existsSync(envPath)) {
  console.error("Missing VoiceToTextApp/.env — copy .env.example to .env and fill values.");
  process.exit(1);
}

const raw = fs.readFileSync(envPath, "utf8");

const getValue = (key) => {
  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    if (line.startsWith("#") || !line.trim()) {
      continue;
    }
    const idx = line.indexOf("=");
    if (idx === -1) {
      continue;
    }
    const k = line.slice(0, idx).trim();
    if (k !== key) {
      continue;
    }
    return line.slice(idx + 1).trim().replace(/^["']|["']$/g, "");
  }
  return "";
};

const url = getValue("EXPO_PUBLIC_TRANSCRIBE_API_URL");
const badUrl =
  !url ||
  /example\.com/i.test(url) ||
  /your-api/i.test(url) ||
  !/^\s*https?:\/\//i.test(url) ||
  !/\/transcribe\/?\s*$/i.test(url);

if (badUrl) {
  console.error("EXPO_PUBLIC_TRANSCRIBE_API_URL must be your API base ending in /transcribe");
  console.error('Example: https://your-host.com/transcribe  or  http://192.168.1.10:3001/transcribe');
  process.exit(1);
}

const token = getValue("EXPO_PUBLIC_TRANSCRIBE_API_TOKEN");
if (token && (/your_optional/i.test(token) || /^your_custom/i.test(token))) {
  console.warn("EXPO_PUBLIC_TRANSCRIBE_API_TOKEN still looks like a placeholder — fix if your API uses SERVER_BEARER_TOKEN.");
}

console.log("VoiceToTextApp: EXPO_PUBLIC_TRANSCRIBE_API_URL is set.");
console.log("  → App will call transcribe + /ai/insights for the AI tab when the server has OPENAI_API_KEY.");
