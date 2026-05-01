// Procedural-cylinder carrot renderer. v6: reads pose + count + axis
// from the snapshot (Python's auto-placement output), so configurations
// with N carrots laying flat (axis=0/1) render N cylinders correctly.
//
// Colour mapping: high retention → vivid orange (fresh carrot),
// low retention → muted brown (cooked-through). When both solutes
// are active we average them. Per-instance retention is a future
// extension (needs labelled cells); for now all instances share the
// aggregate scalar.

import { useMemo } from "react";
import * as THREE from "three";

import type { Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot;
}

/** Three-stop gradient aligned to the `--accent-warm` token:
 *  fresh (vivid amber) → cooked (muted amber) → charred (dark brown). */
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

/**
 * Map ``carrot_axis`` (Python's 0=x, 1=y, 2=z) + cylinder local-Y
 * convention into a world-space rotation tuple.
 * - axis=0 (world +x): rotate local-Y → world-X by +π/2 about world Z
 * - axis=1 (world +y): identity (local-Y already aligns with world-Y)
 * - axis=2 (world +z): rotate local-Y → world-Z by +π/2 about world X
 *
 * For axis=z, the Python anchor is the *base* of the cylinder; the
 * cylinder geometry is centred on its midpoint, so we offset the
 * group +length/2 along the axis. For axis=x/y the anchor is already
 * the centre, no offset.
 */
function poseFromAxis(
  axis: number,
  centre: [number, number, number],
  lengthM: number,
): { position: [number, number, number]; rotation: [number, number, number] } {
  if (axis === 0) {
    return { position: centre, rotation: [0, 0, Math.PI / 2] };
  }
  if (axis === 1) {
    return { position: centre, rotation: [0, 0, 0] };
  }
  // axis === 2 (z): legacy vertical, anchor is base.
  return {
    position: [centre[0], centre[1], centre[2] + lengthM / 2],
    rotation: [Math.PI / 2, 0, 0],
  };
}

export function CarrotMesh({ snapshot }: Props) {
  const color = useMemo(() => {
    const dual = snapshot.carrot_retention2 < 99.99;
    const r = dual
      ? (snapshot.carrot_retention + snapshot.carrot_retention2) * 0.5
      : snapshot.carrot_retention;
    return retentionToColor(r);
  }, [snapshot.carrot_retention, snapshot.carrot_retention2]);

  const diameterM = snapshot.carrot_diameter_m;
  const lengthM = snapshot.carrot_length_m;
  const axis = snapshot.carrot_axis;

  // Empty pool (rebuild marker frames carry carrot_count=0): render nothing.
  if (snapshot.carrot_count === 0 || snapshot.carrot_centres.length === 0) {
    return null;
  }

  return (
    <>
      {snapshot.carrot_centres.map((centre, idx) => {
        const { position, rotation } = poseFromAxis(axis, centre, lengthM);
        return (
          <group key={idx} position={position} rotation={rotation}>
            <mesh>
              <cylinderGeometry
                args={[diameterM / 2, diameterM / 2, lengthM, 32, 1]}
              />
              <meshStandardMaterial color={color} roughness={0.55} />
            </mesh>
          </group>
        );
      })}
    </>
  );
}
