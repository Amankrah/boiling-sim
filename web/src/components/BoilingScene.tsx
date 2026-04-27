// BoilingScene -- Canvas + lights + environment + orbit controls
// + all subscene components. Physics-Z is vertical (matching the
// solver's world coordinates), so we flip Three.js's default Y-up
// to Z-up on the camera and let OrbitControls orbit around that
// axis.
//
// Visual stack (Phase 6.5 redesign):
//   - GradientBackground: inverted sphere with vertex-color gradient
//     from --bg-scene-a at top to --bg-scene-b at bottom. Gives the
//     camera a consistent "room" regardless of orbit angle.
//   - Environment preset="city" (non-background): supplies PBR
//     reflection samples so the pot's metallic surfaces actually
//     look metallic.
//   - Grid (drei): reference floor at z=0 that fades into the
//     distance; anchors the pot spatially.
//   - Key/fill/rim lighting: one directional, one soft fill, one
//     low rim from behind the pot.

import { Environment, Grid, OrbitControls, Stats } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useRef } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";

import type { Snapshot } from "../types/snapshot";

import { Bubbles } from "./Bubbles";
import { CarrotMesh } from "./CarrotMesh";
import { GradientBackground } from "./GradientBackground";
import { Pot } from "./Pot";
import { Stove } from "./Stove";
import { WaterVolume } from "./WaterVolume";

export interface CameraPose {
  position: [number, number, number];
  target: [number, number, number];
}

interface Props {
  snapshot: Snapshot;
  initialCamera?: CameraPose;
  onCameraChange?: (pose: CameraPose) => void;
  /** When true, render drei's Stats FPS overlay. Off by default so it
   *  doesn't clash with the app's brand in the top-left. App.tsx
   *  passes `showStats={showDebug}` so it surfaces only in debug. */
  showStats?: boolean;
}

const DEFAULT_CAMERA: CameraPose = {
  // Reframe for the freestanding electric range: cabinet center
  // sits at world (0.14, 0.12), backsplash rises behind at y ~ 0.38,
  // oven door drops to z ~ -0.35. Target the cooktop centre rather
  // than the origin so the frame shows cabinet + pot + backsplash
  // without pushing the oven door out of shot.
  position: [0.75, -0.85, 0.55],
  target: [0.14, 0.12, 0.05],
};

export function BoilingScene({
  snapshot,
  initialCamera = DEFAULT_CAMERA,
  onCameraChange,
  showStats = false,
}: Props) {
  return (
    <Canvas
      style={{ width: "100%", height: "100%" }}
      shadows="soft"
      camera={{
        position: initialCamera.position,
        up: [0, 0, 1],
        fov: 40,
        near: 0.001,
        far: 6.0,
      }}
      onCreated={({ camera }) => {
        camera.up.set(0, 0, 1);
        const t = initialCamera.target;
        camera.lookAt(new THREE.Vector3(t[0], t[1], t[2]));
        camera.updateProjectionMatrix();
      }}
    >
      {/* --- environment & background ----------------------- */}
      <GradientBackground />
      <Suspense fallback={null}>
        <Environment preset="city" background={false} blur={0.6} />
      </Suspense>

      {/* --- reference floor grid --------------------------- */}
      {/* Positioned at the stove cabinet's base (z = -0.458 m from
          Stove.tsx CAB_BASE_Z) so the grid reads visually as the
          floor the stove stands on. The simulation's z = 0 plane
          is the pot base / cooktop top, ~46 cm above the grid.
          Keeping this as a separate reference plane avoids the
          visual illusion of the stove being "sunk into the floor"
          that came from rendering the grid at simulation z = 0. */}
      <Grid
        position={[0, 0, -0.458]}
        args={[2, 2]}
        cellSize={0.02}
        cellThickness={0.6}
        cellColor={"#2c3a4d"}
        sectionSize={0.1}
        sectionThickness={1}
        sectionColor={"#475569"}
        fadeDistance={1.4}
        fadeStrength={1.1}
        infiniteGrid={false}
        rotation={[Math.PI / 2, 0, 0]}
      />

      {/* --- lighting: key + fill + rim --------------------- */}
      <ambientLight intensity={0.25} />
      {/* Key light casts shadows -- shadow-camera bounds are tight
          (+/- 0.6 m around origin) so the 1024^2 shadow map stays
          crisp for the cabinet + pot area instead of being wasted
          on the infinite grid. */}
      <directionalLight
        position={[0.9, -1.2, 1.4]}
        intensity={1.2}
        color={"#ffffff"}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-left={-0.6}
        shadow-camera-right={0.9}
        shadow-camera-top={0.9}
        shadow-camera-bottom={-0.6}
        shadow-camera-near={0.1}
        shadow-camera-far={3.5}
        shadow-bias={-0.0005}
      />
      <directionalLight
        position={[-0.6, 0.6, 0.4]}
        intensity={0.35}
        color={"#a9b4c2"}
      />
      <directionalLight
        position={[0.0, 0.9, 0.2]}
        intensity={0.25}
        color={"#f5a524"}
      />

      {/* --- scene content ---------------------------------- */}
      {/* Stove first -- sits below z=0 so the pot's base rests on
          its top plate. Emissive coil + indicator LED track
          snapshot.wall_heat_flux so the heat source visibly
          responds to the live heat-flux slider. */}
      <Stove snapshot={snapshot} />
      {/* Pot dimensions come from the v4 snapshot so the rendered
          pot scales with whatever dims are actually being
          simulated (edited via the Config page's Pot section). */}
      <Pot
        diameterM={snapshot.pot_diameter_m}
        heightM={snapshot.pot_height_m}
        wallThicknessM={snapshot.pot_wall_thickness_m}
        baseThicknessM={snapshot.pot_base_thickness_m}
      />
      <WaterVolume snapshot={snapshot} />
      <Bubbles snapshot={snapshot} />
      <CarrotMesh snapshot={snapshot} />

      <SceneOrbitControls
        initialTarget={initialCamera.target}
        onCameraChange={onCameraChange}
      />
      {showStats ? <Stats /> : null}
    </Canvas>
  );
}

/** OrbitControls wrapper that debounces `change` events into
 *  CameraPose callbacks. Lives inside the Canvas so it can use the R3F
 *  control ref directly. */
function SceneOrbitControls({
  initialTarget,
  onCameraChange,
}: {
  initialTarget: [number, number, number];
  onCameraChange?: (pose: CameraPose) => void;
}) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const debounceRef = useRef<number | null>(null);

  // Install a change listener that debounces to ~150 ms. Rapid
  // dragging coalesces into one URL write; final position gets
  // recorded reliably.
  useEffect(() => {
    const ctrls = controlsRef.current;
    if (!ctrls || !onCameraChange) return;
    const handler = () => {
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current);
      }
      debounceRef.current = window.setTimeout(() => {
        const cam = ctrls.object;
        const t = ctrls.target;
        onCameraChange({
          position: [cam.position.x, cam.position.y, cam.position.z],
          target: [t.x, t.y, t.z],
        });
      }, 150);
    };
    ctrls.addEventListener("change", handler);
    return () => {
      ctrls.removeEventListener("change", handler);
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current);
      }
    };
  }, [onCameraChange]);

  return (
    <OrbitControls
      ref={controlsRef}
      target={initialTarget}
      enableDamping
      dampingFactor={0.1}
      minDistance={0.25}
      maxDistance={2.5}
      makeDefault
    />
  );
}
