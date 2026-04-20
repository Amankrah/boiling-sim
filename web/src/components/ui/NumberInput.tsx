// Controlled numeric field with a short text label, a unit badge, and
// commit-on-blur semantics (so per-keystroke events don't spam the
// server with ControlMessages during typing).

import { useEffect, useRef, useState } from "react";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onCommit: (value: number) => void;
  ariaLabel: string;
}

export function NumberInput({
  label,
  value,
  min,
  max,
  step = 1,
  unit,
  onCommit,
  ariaLabel,
}: Props) {
  const [local, setLocal] = useState(String(value));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const activelyTyping = document.activeElement === inputRef.current;
    if (!activelyTyping && String(value) !== local) {
      setLocal(String(value));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const commit = () => {
    const n = Math.max(min, Math.min(max, Number(local) || value));
    setLocal(String(n));
    if (n !== value) onCommit(n);
  };

  return (
    <label className="num-input">
      <span className="num-input__label">{label}</span>
      <input
        ref={inputRef}
        type="number"
        aria-label={ariaLabel}
        value={local}
        min={min}
        max={max}
        step={step}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
      />
      {unit ? <span className="num-input__unit">{unit}</span> : null}
    </label>
  );
}
