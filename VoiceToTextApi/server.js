const dotenv = require("dotenv");

if (!process.env.SPEAKER_STORE_BACKEND) {
  process.env.SPEAKER_STORE_BACKEND = "file";
}

dotenv.config({ override: true });

if (!process.env.OPENAI_API_KEY) {
  console.warn("OPENAI_API_KEY missing in .env — AI insights / speaker-correction will be disabled.");
}
if (!process.env.ASSEMBLYAI_API_KEY) {
  console.warn("ASSEMBLYAI_API_KEY missing in .env — POST /transcribe will fail.");
}

const app = require("./app");
const port = process.env.PORT || 3001;

app.listen(port, () => {
  console.log(`VoiceToText API listening on port ${port}`);
});
