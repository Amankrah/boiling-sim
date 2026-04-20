// Configuration page wrapper. The actual form lives in
// ConfigForm/ConfigForm.tsx; this page is just the route + layout.
// On Apply, the form dispatches `set_config` + `start_run` and
// invokes onApplied to navigate back to the Live view.

import { ConfigForm } from "../components/ConfigForm/ConfigForm";
import type { ShareableParams } from "../share";
import type { ControlMessage, Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot | null;
  params: ShareableParams;
  sendCommand: (cmd: ControlMessage) => void;
  onDone: () => void;
}

export function ConfigPage({ snapshot, params, sendCommand, onDone }: Props) {
  // `params` stays in the Live-view share-link scheme; the Config
  // page uses its own internal ScenarioDraft state seeded from the
  // Pydantic defaults. A future iteration could seed the form from
  // the current snapshot's config echo (v3 wire format doesn't
  // carry the full config yet; only nutrient names + thermal
  // headline values).
  void params;
  return (
    <div className="config-layout">
      <ConfigForm
        snapshot={snapshot}
        sendCommand={sendCommand}
        onApplied={onDone}
      />
    </div>
  );
}
