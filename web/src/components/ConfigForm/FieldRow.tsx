// Single labelled row within a config section. Keeps the visual
// consistent across ~50 fields without repeating the same markup
// in every section.

import type { ReactNode } from "react";

interface Props {
  label: string;
  hint?: string;
  children: ReactNode;
}

export function FieldRow({ label, hint, children }: Props) {
  return (
    <div className="field-row">
      <span className="field-row__label">{label}</span>
      {children}
      {hint ? <span className="field-row__hint">{hint}</span> : null}
    </div>
  );
}
