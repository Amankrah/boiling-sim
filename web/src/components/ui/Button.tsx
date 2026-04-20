// Minimal button primitive. Variants map to classes in app.css so
// the same token system drives every button in the app.

import type { ButtonHTMLAttributes } from "react";

type Variant = "secondary" | "primary" | "danger" | "ghost";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  fullWidth?: boolean;
}

export function Button({
  variant = "secondary",
  fullWidth,
  className = "",
  type = "button",
  ...rest
}: Props) {
  const variantClass =
    variant === "primary"
      ? "btn--primary"
      : variant === "danger"
        ? "btn--danger"
        : variant === "ghost"
          ? "btn--ghost"
          : "";
  const widthStyle = fullWidth ? { flex: 1 } : undefined;
  return (
    <button
      type={type}
      className={`btn ${variantClass} ${className}`.trim()}
      style={widthStyle}
      {...rest}
    />
  );
}
