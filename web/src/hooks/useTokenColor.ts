// Read a single CSS custom property from :root as a string.
//
// Rationale: Recharts wants a literal color string for its `stroke`
// prop, not a `var(--token)` expression. We could duplicate hex
// values in JS, but then the tokens.css source-of-truth drifts. This
// hook reads the value out of the live DOM one time, caches by token
// name, and returns a reliable string.
//
// NB: the token must be defined at `:root`; it's read only on mount.
// If the token changes at runtime (theme switch) the consuming
// component must remount.

import { useMemo } from "react";

export function useTokenColor(name: string, fallback: string = "#e6ecf4"): string {
  return useMemo(() => {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
    return value || fallback;
  }, [name, fallback]);
}

export function useTokenColors<K extends string>(
  map: Record<K, string>,
  fallback: string = "#e6ecf4",
): Record<K, string> {
  return useMemo(() => {
    const out = {} as Record<K, string>;
    if (typeof window === "undefined") {
      for (const k in map) out[k] = fallback;
      return out;
    }
    const root = getComputedStyle(document.documentElement);
    for (const k in map) {
      const value = root.getPropertyValue(map[k]).trim();
      out[k] = value || fallback;
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
