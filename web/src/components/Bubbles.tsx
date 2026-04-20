// Instanced bubble rendering. We allocate at MAX_INSTANCES on mount
// (re-allocation tanks FPS), set per-instance matrices each frame from
// snapshot.bubbles, and set scale = 0 on unused slots so they don't
// show. The sphere mesh is low-poly (8x8 subdivisions) -- at hundreds
// of instances rendered 30 times per second that's still cheap.
//
// MAX_INSTANCES must be >= the solver's `cfg.boiling.max_bubbles`
// (dev-grid default 100_000). We honour that ceiling here; the
// renderer will truncate gracefully if the snapshot ever exceeds it.

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot;
  /** Hard upper bound on instanced slots. Default matches `cfg.boiling.max_bubbles`. */
  maxInstances?: number;
}

const DEFAULT_MAX_INSTANCES = 100_000;

export function Bubbles({ snapshot, maxInstances = DEFAULT_MAX_INSTANCES }: Props) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // Scratch matrix reused per instance.
  const tmp = useMemo(() => new THREE.Matrix4(), []);
  const zeroScale = useMemo(() => new THREE.Vector3(0, 0, 0), []);
  const quatIdentity = useMemo(() => new THREE.Quaternion(), []);

  useEffect(() => {
    const mesh = meshRef.current;
    if (mesh === null) return;
    // Whether or not the snapshot has moved, the pool is already
    // allocated to maxInstances. Upload one matrix per slot.
    const n = Math.min(snapshot.bubbles.length, maxInstances);
    for (let i = 0; i < n; i++) {
      const b = snapshot.bubbles[i];
      // Bubble radius is in metres; geometry is a unit sphere so scale = radius.
      const r = b.radius;
      const pos = new THREE.Vector3(b.position[0], b.position[1], b.position[2]);
      tmp.compose(pos, quatIdentity, new THREE.Vector3(r, r, r));
      mesh.setMatrixAt(i, tmp);
    }
    // Zero out the rest so stale bubbles vanish.
    for (let i = n; i < maxInstances; i++) {
      tmp.compose(new THREE.Vector3(), quatIdentity, zeroScale);
      mesh.setMatrixAt(i, tmp);
    }
    mesh.count = n; // reduce draw call work when bubble count is low
    mesh.instanceMatrix.needsUpdate = true;
  }, [snapshot, maxInstances, tmp, zeroScale, quatIdentity]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, maxInstances]}
      frustumCulled={false}
    >
      <sphereGeometry args={[1, 10, 10]} />
      <meshStandardMaterial
        color={"#e0f2fe"}
        roughness={0.1}
        metalness={0.05}
        transparent
        opacity={0.7}
        emissive={"#38bdf8"}
        emissiveIntensity={0.18}
      />
    </instancedMesh>
  );
}
