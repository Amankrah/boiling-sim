// Slim banner that docks to the top of the scene area when the sim
// is idle at t = 0 — i.e. the producer just launched (or the user
// just hit Reset) and is waiting for an explicit Resume / Apply &
// Start Run before stepping. Mirrors RebuildBanner's pattern.

export function IdleBanner() {
  return (
    <div className="rebuild-banner" role="status" aria-live="polite">
      <span>simulation idle at t = 0 — click Resume or Apply &amp; Start Run to begin</span>
    </div>
  );
}
