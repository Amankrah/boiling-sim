// Global top bar: brand, connection status, WS URL, page nav tabs,
// debug toggle. Sits in the `header` area above each page's layout.
//
// M5 addition: three nav tabs (Live / Config / Results) driven by
// the `usePage` hook in App.tsx. The "Results" tab gets a "●" dot
// + subtle amber when a run is complete so the user's eye lands
// there after a timed run finishes.

import type { Page } from "../hooks/usePage";
import type { ConnectionState } from "../types/snapshot";
import { StatusIndicator } from "./StatusIndicator";

interface Props {
  wsUrl: string;
  connectionState: ConnectionState;
  frameCount: number;
  lastFrameAt: number | null;
  lastError: string | null;
  showDebug: boolean;
  onToggleDebug: () => void;
  page: Page;
  onPageChange: (p: Page) => void;
  /** When true, the Results tab gets a "new results ready" badge. */
  resultsReady?: boolean;
}

const TAB_LABELS: Record<Page, string> = {
  live: "Live",
  config: "Config",
  results: "Results",
};

export function TopBar({
  wsUrl,
  connectionState,
  frameCount,
  lastFrameAt,
  lastError,
  showDebug,
  onToggleDebug,
  page,
  onPageChange,
  resultsReady = false,
}: Props) {
  return (
    <header className="app__header">
      <span className="topbar__brand">Boiling Sim</span>
      <nav className="topbar__tabs" aria-label="dashboard sections">
        {(Object.keys(TAB_LABELS) as Page[]).map((p) => {
          const active = p === page;
          const showBadge = p === "results" && resultsReady;
          return (
            <button
              key={p}
              type="button"
              onClick={() => onPageChange(p)}
              aria-current={active ? "page" : undefined}
              className={`topbar__tab ${active ? "topbar__tab--active" : ""}`}
            >
              {TAB_LABELS[p]}
              {showBadge ? <span className="topbar__tab-badge" aria-hidden /> : null}
            </button>
          );
        })}
      </nav>
      <StatusIndicator
        state={connectionState}
        frameCount={frameCount}
        lastFrameAt={lastFrameAt}
        error={lastError}
      />
      <span className="topbar__url" title={wsUrl}>
        {wsUrl}
      </span>
      <span className="topbar__spacer" />
      <button
        type="button"
        className="btn btn--ghost"
        onClick={onToggleDebug}
        aria-pressed={showDebug}
      >
        {showDebug ? "hide debug" : "debug"}
      </button>
    </header>
  );
}
