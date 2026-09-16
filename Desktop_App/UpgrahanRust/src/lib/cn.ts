/** Joins class values, dropping falsy ones. Cap uses `cx` from cva for the same job. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
