// Live-view control panel. Phase 6.6 M8 trimmed this to the
// live-editable knobs only:
//
//   - Heat flux       (LIVE, no rebuild)
//   - Pause / Resume  (LIVE)
//   - Reset           (rebuild with current cfg)
//   - Share link      (copy URL with scene + camera)
//   - Open full config  (navigate to the Configuration page)
//
// The nutrient / material / carrot-geometry selectors moved to the
// Configuration page. Users land there via the "full config" link
// or the top-bar Config tab, edit every field, then click "Apply &
// Start Run" which sends `set_config` + `start_run` in one batch.

import { useState } from "react";

import type { ShareableParams } from "../share";
import type { ControlMessage, Snapshot } from "../types/snapshot";

import { Button } from "./ui/Button";
import { Card, CardBody, CardHeader } from "./ui/Card";
import { Slider } from "./ui/Slider";

interface Props {
  snapshot: Snapshot | null;
  params: ShareableParams;
  onParamsChange: (next: ShareableParams) => void;
  sendCommand: (cmd: ControlMessage) => void;
  onCopyShareLink?: () => void;
  /** Optional navigate-to-config callback; surfaces the Configuration
   *  tab directly from this panel so the user can edit deeper knobs
   *  without hunting for the top-bar nav. */
  onOpenConfig?: () => void;
}

export function ControlPanel({
  snapshot,
  params,
  onParamsChange,
  sendCommand,
  onCopyShareLink,
  onOpenConfig,
}: Props) {
  const isPaused = snapshot?.is_paused ?? false;
  const heatFluxKw = params.heatFluxWPerM2 / 1000;
  const [copyFlash, setCopyFlash] = useState(false);

  const commitHeatFlux = (kw: number) => {
    const w = kw * 1000;
    onParamsChange({ ...params, heatFluxWPerM2: w });
    sendCommand({ type: "set_heat_flux", value: w });
  };

  const handleCopy = () => {
    if (!onCopyShareLink) return;
    onCopyShareLink();
    setCopyFlash(true);
    window.setTimeout(() => setCopyFlash(false), 1200);
  };

  return (
    <Card>
      <CardHeader title="Live controls" />
      <CardBody>
        <div className="controls">
          <section className="control-section">
            <div className="control-section__header">
              <span className="control-section__label">Heat flux</span>
              <span className="control-section__value">
                {heatFluxKw.toFixed(0)} kW/m²
              </span>
            </div>
            <Slider
              value={Math.round(heatFluxKw)}
              min={0}
              max={80}
              step={1}
              onChange={commitHeatFlux}
              ariaLabel="wall heat flux in kilowatts per square metre"
            />
            <div className="share-hint">
              Live-editable. Nutrient / material / geometry / duration live
              on the Configuration page.
            </div>
          </section>

          <section className="control-section">
            <div className="control-section__header">
              <span className="control-section__label">Run</span>
            </div>
            <div className="control-row">
              {isPaused ? (
                <Button
                  variant="primary"
                  fullWidth
                  onClick={() => sendCommand({ type: "resume" })}
                >
                  Resume
                </Button>
              ) : (
                <Button
                  fullWidth
                  onClick={() => sendCommand({ type: "pause" })}
                >
                  Pause
                </Button>
              )}
              <Button
                variant="danger"
                fullWidth
                onClick={() => sendCommand({ type: "reset" })}
              >
                Reset
              </Button>
            </div>
          </section>

          {onOpenConfig ? (
            <section className="control-section">
              <div className="control-section__header">
                <span className="control-section__label">Full config</span>
              </div>
              <Button onClick={onOpenConfig} fullWidth>
                Open Configuration →
              </Button>
              <p className="share-hint">
                Edit every pot / water / carrot / heating / grid / solver /
                boiling / nutrient / duration field, then Apply &amp; Start
                Run with a validated batch.
              </p>
            </section>
          ) : null}

          {onCopyShareLink ? (
            <section className="control-section">
              <div className="control-section__header">
                <span className="control-section__label">Share</span>
              </div>
              <Button
                onClick={handleCopy}
                className={copyFlash ? "copied" : ""}
                title="Copy a URL that reproduces this scene setup and camera. The new session starts from t=0."
              >
                {copyFlash ? "Link copied ✓" : "Copy share link"}
              </Button>
              <p className="share-hint">
                Encodes heat flux, material, carrot size, and camera — not
                sim time.
              </p>
            </section>
          ) : null}
        </div>
      </CardBody>
    </Card>
  );
}
