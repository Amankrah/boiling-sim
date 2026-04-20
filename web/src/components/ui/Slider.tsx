// Slider primitive: styled <input type="range"> with a filled track
// overlay. All styling in app.css. The fill's width is set inline
// (it's the only dynamic style).

interface Props {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  ariaLabel: string;
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  ariaLabel,
}: Props) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="slider">
      <div className="slider__track" />
      <div
        className="slider__fill"
        style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
      />
      <input
        type="range"
        aria-label={ariaLabel}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
