// Select primitive: styled native <select>. The chevron is built
// from a pair of CSS linear-gradients in app.css -- no SVG asset.

import type { SelectHTMLAttributes } from "react";

interface Option<T extends string> {
  value: T;
  label: string;
}

interface Props<T extends string>
  extends Omit<SelectHTMLAttributes<HTMLSelectElement>, "onChange" | "value"> {
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
  ariaLabel: string;
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  ...rest
}: Props<T>) {
  return (
    <select
      className="select"
      aria-label={ariaLabel}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      {...rest}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
