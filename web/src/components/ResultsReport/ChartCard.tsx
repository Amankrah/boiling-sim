// Card wrapper for every Results-page chart. Provides:
//   * a section element tagged with `data-chart-name="<slug>"` so
//     the global "Export all charts" routine can find every chart
//     under the report root.
//   * a small "Export PNG" icon button at the top-right that
//     snapshots THIS card to PNG via the export hook.
//   * the card chrome (title, subtitle) so the chart bodies stay
//     focused on data.
//
// Existing cards on the report (MassPartitionCard etc.) are migrated
// to this wrapper in M6; new cards (HeatUpStorylineCard, etc.) use
// it from the start.

import { useRef, useState, type ReactNode } from "react";

import { exportNodeAsPng } from "../../hooks/useChartExport";

interface Props {
  /** Stable slug used as the filename inside the per-card download
   *  AND as the entry name inside the global ZIP. Keep lower-snake. */
  name: string;
  /** Card title shown to the user. */
  title: string;
  /** Optional one-line subtitle / context (e.g. "primary solute"). */
  subtitle?: string;
  /** Optional right-side hint placed BEFORE the export button (e.g.
   *  a status pill on the existing thermal card). */
  rightSlot?: ReactNode;
  children: ReactNode;
}

export function ChartCard({ name, title, subtitle, rightSlot, children }: Props) {
  const ref = useRef<HTMLElement | null>(null);
  const [busy, setBusy] = useState(false);

  const onExport = async () => {
    if (!ref.current || busy) return;
    setBusy(true);
    try {
      await exportNodeAsPng(ref.current, name);
    } catch {
      // Swallow -- export is best-effort; user can retry.
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      ref={ref}
      className="report-card"
      data-chart-name={name}
    >
      <header className="report-card__head">
        <div className="report-card__title-block">
          <span className="report-card__title">{title}</span>
          {subtitle ? (
            <span className="report-card__subtitle">{subtitle}</span>
          ) : null}
        </div>
        <div className="report-card__head-right">
          {rightSlot}
          <button
            type="button"
            className="report-card__export"
            onClick={onExport}
            disabled={busy}
            title="Export this chart as PNG"
            aria-label={`Export ${title} as PNG`}
          >
            {busy ? "…" : "PNG"}
          </button>
        </div>
      </header>
      <div className="report-card__body">{children}</div>
    </section>
  );
}
