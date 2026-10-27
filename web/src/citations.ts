export type Part = { text: string } | { cite: string };

/** Split an answer into text and [ID] citation parts; only ids in `known` become citations. */
export function splitCitations(answer: string, known: Set<string>): Part[] {
  const parts: Part[] = [];
  let buffer = "";
  let last = 0;
  for (const m of answer.matchAll(/\[([A-Z0-9]{10})\]/g)) {
    const idx = m.index ?? 0;
    buffer += answer.slice(last, idx);
    last = idx + m[0].length;
    if (known.has(m[1])) {
      if (buffer) parts.push({ text: buffer });
      parts.push({ cite: m[1] });
      buffer = "";
    } else {
      buffer += m[0];
    }
  }
  buffer += answer.slice(last);
  if (buffer) parts.push({ text: buffer });
  return parts;
}
