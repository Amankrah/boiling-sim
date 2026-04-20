// Dev-guide §6.4 calls for a GLB-loaded carrot mesh with per-vertex
// colour from `carrot_surface_c`. For M5 we use a procedural capped
// cylinder positioned at the scenario's carrot centre and coloured by
// the scalar `carrot_retention` / `carrot_retention2`. Per-vertex
// shading lands alongside a real GLB asset in a later pass -- the
// wire format already carries the vectors (`carrot_surface_c*`),
// they're just empty in M1-M3.
//
// Colour mapping: high retention → vivid orange (fresh carrot),
// low retention → muted brown (cooked-through). When both solutes
// are active we average them; this shows degradation regardless of
// which mechanism dominates.

import { useMemo } from "react";
import * as THREE from "three";

import type { Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot;
  /**
   * Carrot geometry lives in the YAML scenario config (carrot.diameter_m,
   * carrot.length_m, carrot.position). The snapshot wire format does not
   * carry those yet, so we default-guess matching the Phase-4 scenarios
   * (25 mm diameter, 50 mm length, centred above the pot base). Callers
   * can override.
   */
  diameterM?: number;
  lengthM?: number;
  /** World-space centre of the carrot's axis midpoint. */
  centre?: [number, number, number];
}

/** Three-stop gradient aligned to the `--accent-warm` token:
 *  fresh (vivid amber) → cooked (muted amber) → charred (dark brown).
 *  Applied as the cylinder's meshStandardMaterial color; PBR lighting
 *  from the env map + key light does the rest. */
function retentionToColor(retentionPct: number): THREE.Color {
  const t = Math.max(0, Math.min(1, retentionPct / 100));
  const fresh = new THREE.Color("#f5a524"); // --accent-warm
  const cooked = new THREE.Color("#a46419"); // 40% darker
  const charred = new THREE.Color("#3a2614"); // dark brown
  if (t > 0.5) {
    return cooked.lerp(fresh, (t - 0.5) * 2);
  }
  return charred.lerp(cooked, t * 2);
}

export function CarrotMesh({
  snapshot,
  diameterM = 0.025,
  lengthM = 0.05,
  centre = [0, 0, 0.055], // matches configs/scenarios/default.yaml carrot.position + half length
}: Props) {
  const color = useMemo(() => {
    // Average both retentions when the second solute is active; otherwise
    // use the primary. Detection: retention2 ≈ 100 (default for disabled)
    // still nudges the colour slightly; fine for a live indicator.
    const dual = snapshot.carrot_retention2 < 99.99;
    const r = dual
      ? (snapshot.carrot_retention + snapshot.carrot_retention2) * 0.5
      : snapshot.carrot_retention;
    return retentionToColor(r);
  }, [snapshot.carrot_retention, snapshot.carrot_retention2]);

  return (
    <group position={centre} rotation={[Math.PI / 2, 0, 0]}>
      {/* Cylinder's axis is along local Y; the group rotation above maps
          local-Y → world-Z, matching the physics scenario's carrot axis. */}
      <mesh>
        <cylinderGeometry args={[diameterM / 2, diameterM / 2, lengthM, 32, 1]} />
        <meshStandardMaterial color={color} roughness={0.55} />
      </mesh>
    </group>
  );
}
