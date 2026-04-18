/**
 * Builds VoiceToTextApi/.env.<PROJECT_ID> from VoiceToTextApi/.env for Firebase Gen 2 deploy.
 * Omits keys reserved by Cloud Functions (e.g. PORT). Do not commit the output file.
 */
const fs = require("fs");
const path = require("path");

const RESERVED = new Set(["PORT", "K_SERVICE", "FUNCTION_TARGET"]);

const projectId = process.argv[2];
if (!projectId) {
  console.error("Usage: node scripts/sync-functions-env.js <firebase-project-id>");
  process.exit(1);
}

const apiDir = path.join(__dirname, "..", "VoiceToTextApi");
const src = path.join(apiDir, ".env");
const dst = path.join(apiDir, `.env.${projectId}`);

if (!fs.existsSync(src)) {
  console.error("Missing VoiceToTextApi/.env");
  process.exit(1);
}

const raw = fs.readFileSync(src, "utf8");
const lines = raw.split(/\r?\n/);
const out = [];
for (const line of lines) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) {
    continue;
  }
  const eq = trimmed.indexOf("=");
  if (eq <= 0) {
    continue;
  }
  const key = trimmed.slice(0, eq).trim();
  if (RESERVED.has(key)) {
    continue;
  }
  out.push(trimmed);
}

fs.writeFileSync(dst, `${out.join("\n")}\n`, "utf8");
console.log("Wrote", path.relative(path.join(__dirname, ".."), dst));
