const CALL_INTENT_PATTERNS: RegExp[] = [
  /\b(?:call|phone|ring|dial|text|sms)\s+([A-Za-z][A-Za-z0-9'.-]*(?:\s+[A-Za-z][A-Za-z0-9'.-]*){0,3})(?:\s+(?:at|on)\s|\s*[.,]|\s*$|\?)/i,
  /\b(?:call|phone|ring|dial)\s+([A-Za-z][A-Za-z0-9'.-]*(?:\s+[A-Za-z][A-Za-z0-9'.-]*){0,2})\b/i,
];

/** Name after "call", "phone", etc. — used for automatic contact lookup. */
export function extractCallIntentTarget(text: string): string | null {
  const t = text.trim();
  if (!t) {
    return null;
  }
  for (const re of CALL_INTENT_PATTERNS) {
    const m = t.match(re);
    if (m?.[1]?.trim()) {
      return m[1].trim();
    }
  }
  return null;
}

/** Very short transcript that looks like a name or initials only (e.g. "UD"). */
export function extractShortNameOnlyQuery(text: string): string | null {
  const t = text.trim();
  if (t.length < 2 || t.length > 20) {
    return null;
  }
  if (!/^[A-Za-z][A-Za-z\s.'-]*$/.test(t)) {
    return null;
  }
  const words = t.split(/\s+/).filter(Boolean);
  if (words.length > 3) {
    return null;
  }
  if (words.length >= 2 && t.length > 15) {
    return null;
  }
  return t;
}

/** Prefer call-intent target; else a short name-only line (no auto-search on long sentences). */
export function autoContactSearchQuery(text: string): string | null {
  return extractCallIntentTarget(text) ?? extractShortNameOnlyQuery(text);
}

/**
 * Heuristics for turning a voice transcript into a short contact-search string (e.g. "call UD" → "UD").
 */
export function suggestContactSearchFromTranscript(text: string): string {
  const t = text.trim();
  if (!t) {
    return "";
  }
  for (const re of CALL_INTENT_PATTERNS) {
    const m = t.match(re);
    if (m?.[1]?.trim()) {
      return m[1].trim();
    }
  }
  const firstChunk = t.split(/[.!?\n]/)[0]?.trim() ?? "";
  return firstChunk.length > 48 ? firstChunk.slice(0, 48) : firstChunk;
}
/** Build a reasonable Google query from transcript + optional AI summary. */
export function buildWebSearchQueryForEntry(text: string, aiSummary?: string | null): string {
  const parts = [text.trim()];
  if (aiSummary?.trim()) {
    parts.push(aiSummary.trim());
  }
  const blob = parts.filter(Boolean).join(" — ");
  if (blob.length <= 280) {
    return blob;
  }
  return `${blob.slice(0, 277)}…`;
}
