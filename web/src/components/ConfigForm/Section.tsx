// Collapsible section card. Defaults open. The header is a
// full-width button so keyboard users can toggle via Enter/Space.

import { useState, type ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

export function Section({ title, subtitle, children, defaultOpen = true }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`config-section ${open ? "config-section--open" : ""}`}>
      <button
        type="button"
        className="config-section__header"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>
          <span className="config-section__title">{title}</span>
          {subtitle ? (
            <>
              {"  "}
              <span className="config-section__subtitle">{subtitle}</span>
            </>
          ) : null}
        </span>
        <span className="config-section__chevron" aria-hidden>▸</span>
      </button>
      {open ? <div className="config-section__body">{children}</div> : null}
    </div>
  );
}
