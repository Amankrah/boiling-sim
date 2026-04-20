// Sky-sphere with a top-to-bottom vertex-color gradient. Sits at a
// fixed large radius around the camera-origin; rendered with
// BackSide so the camera sees its inside. Since it's a mesh (not a
// fragment-shader-on-screen trick), camera orbit reveals a
// consistent "room" gradient rather than a fixed-to-viewport wash.
//
// Colors match tokens.css (--bg-scene-a, --bg-scene-b) but are
// duplicated here as hex literals -- Three.js material colors need
// Color objects, not CSS var strings.

import { useMemo } from "react";
import * as THREE from "three";

const TOP_COLOR = new THREE.Color("#1e2635"); // --bg-scene-a
const BOTTOM_COLOR = new THREE.Color("#0c1118"); // --bg-scene-b

interface Props {
  radius?: number;
}

export function GradientBackground({ radius = 4 }: Props) {
  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(radius, 32, 20);
    // Paint each vertex by Z height (physics Z up -> our world up).
    // maxZ at top, minZ at bottom; lerp between TOP/BOTTOM colors.
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const z = pos.getZ(i);
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const range = Math.max(maxZ - minZ, 1e-6);
    const color = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const t = (pos.getZ(i) - minZ) / range;
      color.copy(BOTTOM_COLOR).lerp(TOP_COLOR, t);
      colors[i * 3 + 0] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [radius]);

  return (
    <mesh geometry={geometry} frustumCulled={false}>
      <meshBasicMaterial
        vertexColors
        side={THREE.BackSide}
        depthWrite={false}
        // Render first so nothing else accidentally z-fights with it.
      />
    </mesh>
  );
}
