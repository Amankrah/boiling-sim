// WaterVolume -- box mesh with a ray-marching fragment shader that
// reads temperature + alpha from a `Data3DTexture` rebuilt from each
// snapshot. The box is sized to the downsampled grid extent
// (grid_ds.nx * grid_ds.dx) × (... × dy) × (... × dz) and positioned
// at the grid's world-space centre.
//
// Wire format handling: Python emits temperature in C-contiguous
// (nx, ny, nz) order (k fastest). Three.js `Data3DTexture(data, W, H, D)`
// expects `data[x + y*W + z*W*H]` = texel at (x, y, z). We transpose on
// the JS side into a pre-allocated RG buffer (R = alpha, G = normalised
// temperature) on every snapshot update. At 50x50x30 voxels x 30 Hz
// this is 2.25M writes/s -- cheap. The alternative (Python serialises
// in F-order) was rejected to keep the wire format stable across
// downstream consumers (Omniverse Kit in Phase 7 will also want natural
// numpy order).

import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { GridMeta, Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot;
}

// Shader constants. 48 steps is comfortable for 50x50x30 voxels at
// 30 FPS on an RTX 6000 Ada; drop to 32 if a weaker GPU struggles.
const RAYMARCH_STEPS = 48;

const VERT_SHADER = /* glsl */ `
out vec3 vLocalPosition;
void main() {
  vLocalPosition = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAG_SHADER = /* glsl */ `
precision highp sampler3D;
in vec3 vLocalPosition;
uniform sampler3D uVolume;
uniform vec3 uBoxSize;
uniform vec3 uCameraLocal;
uniform float uOpacity;
out vec4 fragColor;

const int STEPS = ${RAYMARCH_STEPS};

bool intersectBox(vec3 origin, vec3 dir, vec3 boxMin, vec3 boxMax,
                  out float tMin, out float tMax) {
  vec3 invDir = 1.0 / dir;
  vec3 t0s = (boxMin - origin) * invDir;
  vec3 t1s = (boxMax - origin) * invDir;
  vec3 tsmall = min(t0s, t1s);
  vec3 tlarge = max(t0s, t1s);
  tMin = max(max(tsmall.x, tsmall.y), tsmall.z);
  tMax = min(min(tlarge.x, tlarge.y), tlarge.z);
  return tMax > tMin;
}

vec3 temperatureToColor(float t01) {
  // Cool → dark blue, warm → orange/red. Cheap two-stop gradient;
  // looks right enough for the 20-100 C dynamic range of a boiling
  // pot without pulling a full viridis LUT texture.
  vec3 cold = vec3(0.05, 0.15, 0.55);
  vec3 warm = vec3(1.00, 0.35, 0.10);
  return mix(cold, warm, t01);
}

void main() {
  vec3 halfSize = uBoxSize * 0.5;
  vec3 boxMin = -halfSize;
  vec3 boxMax =  halfSize;
  vec3 rayDir = normalize(vLocalPosition - uCameraLocal);
  float tMin;
  float tMax;
  if (!intersectBox(uCameraLocal, rayDir, boxMin, boxMax, tMin, tMax)) {
    discard;
  }
  tMin = max(tMin, 0.0);
  float stepLen = (tMax - tMin) / float(STEPS);
  vec3 p = uCameraLocal + rayDir * tMin;
  vec3 stepVec = rayDir * stepLen;
  vec4 accum = vec4(0.0);

  for (int i = 0; i < STEPS; i++) {
    vec3 uvw = (p + halfSize) / uBoxSize;
    if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {
      vec2 s = texture(uVolume, uvw).rg; // R = alpha, G = normalised T
      float waterAlpha = s.r;
      float tempN = s.g;
      if (waterAlpha > 0.02) {
        vec3 color = temperatureToColor(tempN);
        // Scale opacity by step length so the final look is roughly
        // resolution-independent. Constant factor (30.0) tuned to look
        // semi-translucent at the dev-grid (~50^3 downsampled).
        float dens = uOpacity * waterAlpha * stepLen * 30.0;
        dens = clamp(dens, 0.0, 1.0);
        accum.rgb += (1.0 - accum.a) * color * dens;
        accum.a   += (1.0 - accum.a) * dens;
        if (accum.a > 0.97) break;
      }
    }
    p += stepVec;
  }
  if (accum.a < 0.01) discard;
  fragColor = accum;
}
`;

function gridCentreWorld(grid: GridMeta): THREE.Vector3 {
  return new THREE.Vector3(
    grid.origin[0] + (grid.nx * grid.dx) * 0.5,
    grid.origin[1] + (grid.ny * grid.dx) * 0.5,
    grid.origin[2] + (grid.nz * grid.dx) * 0.5,
  );
}

function boxExtent(grid: GridMeta): THREE.Vector3 {
  return new THREE.Vector3(
    grid.nx * grid.dx,
    grid.ny * grid.dx,
    grid.nz * grid.dx,
  );
}

/**
 * Transpose a C-order (nx, ny, nz) float array into the RG layout
 * Three.js `Data3DTexture` expects when addressed as
 * `(x=i, y=j, z=k)`, packing normalised temperature + alpha.
 *
 * Output size is 2 * nx * ny * nz bytes (RG8 format). The temperature
 * is rescaled to [0, 1] using (t - 20) / 80 so a sane colour gradient
 * spans 20-100 C.
 */
export function packVolumeData(
  temperature: ArrayLike<number>,
  alpha: ArrayLike<number>,
  nx: number,
  ny: number,
  nz: number,
  out: Uint8Array,
): Uint8Array {
  if (out.length !== 2 * nx * ny * nz) {
    throw new Error(
      `packVolumeData: out buffer length ${out.length} !== 2 * ${nx} * ${ny} * ${nz}`,
    );
  }
  if (temperature.length !== nx * ny * nz) {
    throw new Error(
      `packVolumeData: temperature length ${temperature.length} !== nx*ny*nz (${nx * ny * nz})`,
    );
  }
  if (alpha.length !== nx * ny * nz) {
    throw new Error(
      `packVolumeData: alpha length ${alpha.length} !== nx*ny*nz`,
    );
  }
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        // Python linear index: k fastest, i slowest.
        const src = i * ny * nz + j * nz + k;
        // Three.js texture index: x fastest, z slowest; here (x,y,z) = (i,j,k).
        const dst = 2 * (i + j * nx + k * nx * ny);
        const a = alpha[src];
        const tNorm = (temperature[src] - 20.0) / 80.0;
        out[dst + 0] = Math.max(0, Math.min(255, (a * 255) | 0));
        out[dst + 1] = Math.max(0, Math.min(255, (tNorm * 255) | 0));
      }
    }
  }
  return out;
}

export function WaterVolume({ snapshot }: Props) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();

  const extent = useMemo(() => boxExtent(snapshot.grid_ds), [snapshot.grid_ds]);
  const centre = useMemo(() => gridCentreWorld(snapshot.grid_ds), [snapshot.grid_ds]);

  // Preallocate the volume texture + its backing buffer keyed on
  // downsampled grid dimensions so we don't allocate per frame.
  const { texture, buffer } = useMemo(() => {
    const nx = snapshot.grid_ds.nx;
    const ny = snapshot.grid_ds.ny;
    const nz = snapshot.grid_ds.nz;
    const data = new Uint8Array(2 * nx * ny * nz);
    const tex = new THREE.Data3DTexture(data, nx, ny, nz);
    tex.format = THREE.RGFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    return { texture: tex, buffer: data };
  }, [snapshot.grid_ds.nx, snapshot.grid_ds.ny, snapshot.grid_ds.nz]);

  // Dispose GPU texture when the grid changes or we unmount.
  useEffect(() => {
    return () => {
      texture.dispose();
    };
  }, [texture]);

  // Uniforms object -- reused frame to frame so React doesn't recreate
  // the shader material.
  const uniforms = useMemo(
    () => ({
      uVolume: { value: texture },
      uBoxSize: { value: extent.clone() },
      uCameraLocal: { value: new THREE.Vector3() },
      uOpacity: { value: 1.0 },
    }),
    // uniform object is stable across snapshots; values update in-place.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [texture],
  );

  // Re-upload texel data whenever the snapshot changes.
  useEffect(() => {
    const nx = snapshot.grid_ds.nx;
    const ny = snapshot.grid_ds.ny;
    const nz = snapshot.grid_ds.nz;
    packVolumeData(snapshot.temperature, snapshot.alpha, nx, ny, nz, buffer);
    texture.needsUpdate = true;
    uniforms.uBoxSize.value.copy(extent);
  }, [snapshot, buffer, texture, uniforms, extent]);

  // Per-frame: update the camera-in-local-space uniform.
  useFrame(() => {
    if (meshRef.current === null) return;
    const tmp = new THREE.Vector3();
    tmp.copy(camera.position);
    meshRef.current.worldToLocal(tmp);
    uniforms.uCameraLocal.value.copy(tmp);
  });

  return (
    <mesh ref={meshRef} position={centre}>
      <boxGeometry args={[extent.x, extent.y, extent.z]} />
      <shaderMaterial
        uniforms={uniforms}
        vertexShader={VERT_SHADER}
        fragmentShader={FRAG_SHADER}
        glslVersion={THREE.GLSL3}
        transparent
        depthWrite={false}
        side={THREE.BackSide}
      />
    </mesh>
  );
}
