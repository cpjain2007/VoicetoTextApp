import * as Contacts from "expo-contacts";
import { Platform } from "react-native";

export type ContactPhoneRow = {
  contactId: string;
  displayName: string;
  phoneLabel: string;
  display: string;
  tel: string;
};

function normalizeToTel(raw: string): { display: string; tel: string } | null {
  const trimmed = raw.trim();
  const d = trimmed.replace(/\D/g, "");
  if (d.length < 10 || d.length > 15) {
    return null;
  }
  if (d.length === 10) {
    return {
      display: `(${d.slice(0, 3)}) ${d.slice(3, 6)}-${d.slice(6)}`,
      tel: `tel:+1${d}`,
    };
  }
  if (d.length === 11 && d.startsWith("1")) {
    const n = d.slice(1);
    return {
      display: `(${n.slice(0, 3)}) ${n.slice(3, 6)}-${n.slice(6)}`,
      tel: `tel:+${d}`,
    };
  }
  return { display: trimmed, tel: `tel:+${d}` };
}

function buildDisplayName(c: Contacts.ExistingContact): string {
  const fromParts = [c.firstName, c.middleName, c.lastName].filter(Boolean).join(" ").trim();
  if (fromParts) {
    return fromParts;
  }
  if (typeof c.nickname === "string" && c.nickname.trim()) {
    return c.nickname.trim();
  }
  if (typeof c.name === "string" && c.name.trim()) {
    return c.name.trim();
  }
  return "Unknown";
}

function buildHaystack(c: Contacts.ExistingContact, displayName: string): string {
  const chunks = [
    displayName,
    c.name,
    c.firstName,
    c.middleName,
    c.lastName,
    c.nickname,
    c.company,
  ].filter((x): x is string => typeof x === "string" && x.trim().length > 0);
  return chunks.join(" ").toLowerCase();
}

/** Match initials (e.g. "ud" → Uma Davis) and substring / multi-token. */
export function contactMatchesQuery(haystack: string, displayName: string, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) {
    return false;
  }
  if (haystack.includes(q)) {
    return true;
  }
  const qParts = q.split(/\s+/).filter(Boolean);
  if (qParts.length > 1 && qParts.every((p) => haystack.includes(p))) {
    return true;
  }
  if (q.length >= 2 && q.length <= 8 && !/\s/.test(q)) {
    const words = displayName.toLowerCase().split(/\s+/).filter((w) => w.length > 0);
    const initials = words.map((w) => w[0]).join("");
    if (initials === q) {
      return true;
    }
    if (words.length >= 2 && q.length === 2) {
      if (words[0][0] + words[words.length - 1][0] === q) {
        return true;
      }
    }
    if (words.length === 1 && words[0].length >= 2) {
      if (words[0].slice(0, q.length) === q) {
        return true;
      }
    }
  }
  return false;
}

export async function isContactsApiAvailable(): Promise<boolean> {
  if (Platform.OS === "web") {
    return false;
  }
  try {
    return await Contacts.isAvailableAsync();
  } catch {
    return false;
  }
}

export async function ensureContactsPermission(): Promise<boolean> {
  if (Platform.OS === "web") {
    return false;
  }
  const existing = await Contacts.getPermissionsAsync();
  if (existing.granted) {
    return true;
  }
  const requested = await Contacts.requestPermissionsAsync();
  return requested.granted;
}

/**
 * case-insensitive search over name / nickname / company; returns rows with at least one dialable number.
 */
export async function searchContactsWithPhones(query: string): Promise<ContactPhoneRow[]> {
  if (Platform.OS === "web") {
    return [];
  }
  const q = query.trim();
  if (!q) {
    return [];
  }

  const { data } = await Contacts.getContactsAsync({
    pageSize: 0,
    fields: [
      Contacts.Fields.PhoneNumbers,
      Contacts.Fields.Name,
      Contacts.Fields.FirstName,
      Contacts.Fields.MiddleName,
      Contacts.Fields.LastName,
      Contacts.Fields.Nickname,
      Contacts.Fields.Company,
    ],
  });

  const rows: ContactPhoneRow[] = [];
  const seenTel = new Set<string>();

  for (const c of data) {
    if (!c.id) {
      continue;
    }
    const displayName = buildDisplayName(c);
    const haystack = buildHaystack(c, displayName);
    if (!contactMatchesQuery(haystack, displayName, q)) {
      continue;
    }
    const phones = c.phoneNumbers ?? [];
    for (const p of phones) {
      const raw = p.number || p.digits || "";
      const norm = normalizeToTel(raw);
      if (!norm) {
        continue;
      }
      if (seenTel.has(norm.tel)) {
        continue;
      }
      seenTel.add(norm.tel);
      rows.push({
        contactId: c.id,
        displayName,
        phoneLabel: p.label || "phone",
        display: norm.display,
        tel: norm.tel,
      });
    }
  }

  return rows;
}
