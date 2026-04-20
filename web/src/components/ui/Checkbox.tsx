// Minimal Checkbox primitive. Styled natively via the `.checkbox`
// class in app.css so it inherits the design-token palette without
// a second UI library.

import type { InputHTMLAttributes } from "react";

interface Props extends Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "value" | "type"> {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  hint?: string;
}

export function Checkbox({ checked, onChange, label, hint, ...rest }: Props) {
  return (
    <label className="checkbox">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        {...rest}
      />
      <span className="checkbox__box" aria-hidden />
      <span className="checkbox__label">
        {label}
        {hint ? <span className="checkbox__hint">{hint}</span> : null}
      </span>
    </label>
  );
}
