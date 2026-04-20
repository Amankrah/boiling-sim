// Slim amber banner that docks to the top of the scene area whenever
// the Python producer is mid-rebuild. Replaces the earlier full-
// screen blur overlay -- the 3D scene keeps rendering underneath so
// the viewer sees the camera state that will be restored when the
// new Simulation comes up.

export function RebuildBanner() {
  return (
    <div className="rebuild-banner" role="status" aria-live="polite">
      <span className="rebuild-banner__spinner" aria-hidden />
      <span>rebuilding simulation — t will restart at 0</span>
    </div>
  );
}
