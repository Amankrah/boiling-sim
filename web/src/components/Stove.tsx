// Stove -- a freestanding 4-burner electric range.
//
// Design
// ------
// Previous draft was a flat slab with one knob on top and coil
// rings in a single recessed pan; it read as "a block" and the
// knob came out behind the pot from the default camera. This
// rewrite delivers a recognizable kitchen range:
//
//   * Rounded matte-stainless cabinet sinking below z=0 (so the
//     pot's base disc sits flush on the cooktop at z=0).
//   * Toe-kick + 4 rubber feet for a grounded silhouette.
//   * Black-enamel cooktop with 4 burners in a 2x2 grid. The
//     pot sits on the FRONT-LEFT burner (world origin); the
//     other three are cold decoration.
//   * Vertical backsplash rising behind the cooktop, with 4
//     chrome knobs protruding toward the viewer. Only knob #0
//     (aligned with the active burner) has an emissive pointer
//     tick; a small LED beside it mirrors the same heat flux.
//   * Oven door on the front face with a translucent glass
//     window and a chrome handle bar.
//   * Brand text on the backsplash upper edge.
//
// Coordinate system: physics-Z UP, camera in +X/-Y octant so the
// viewer sees the -Y face. The cabinet CENTER offsets to
// (+0.14, +0.12) so the active burner at world (0,0) lands at
// the cooktop's front-left corner. Any object whose "front face"
// should face the viewer is placed at the -Y extreme of its
// local volume.

import { useMemo } from "react";
import * as THREE from "three";

import { MeshTransmissionMaterial, RoundedBox, Text } from "@react-three/drei";

import type { Snapshot } from "../types/snapshot";

// Upper end of the heat-flux slider (50 kW/m^2 matches ControlPanel).
// Above this, the visual saturates rather than clipping.
const MAX_FLUX_W_M2 = 50_000;

// Cabinet center in world XY. The active burner sits at world
// origin; this offset places that burner at the cooktop's
// front-left position.
const CAB_X = 0.14;
const CAB_Y = 0.12;

// Cabinet dimensions (meters): W (X), D (Y), H (Z).
const CAB_W = 0.56;
const CAB_D = 0.50;
const CAB_H = 0.45;

// Cabinet TOP face Z. Must sit flush with the cooktop BOTTOM so
// the cabinet doesn't plug the drilled burner holes from inside.
// Cooktop top lives at z = 0 (pot-base level), cooktop is 8 mm
// thick, so cabinet top = -0.008.
const CAB_TOP_Z = -0.008;
const CAB_POS_Z = CAB_TOP_Z - CAB_H / 2;   // -0.233
const CAB_BASE_Z = CAB_TOP_Z - CAB_H;      // -0.458

// Burner centers in world XY. Index 0 is ACTIVE.
const BURNERS: ReadonlyArray<readonly [number, number]> = [
  [0.00, 0.00], // front-left  (active, pot sits here)
  [0.28, 0.00], // front-right (cold)
  [0.00, 0.24], // back-left   (cold)
  [0.28, 0.24], // back-right  (cold)
];

// Concentric coil ring radii + tube radius for each burner.
// Outer ring diameter ~= 0.196 m, comfortably wider than the
// 0.20 m pot's inner grip but narrow enough that the pot sits
// ON TOP of the coil like a real hob.
const COIL_RADII = [0.038, 0.058, 0.078, 0.098];
const COIL_TUBE_R = 0.0035;

// Backsplash back-of-cabinet position (rises at +Y edge).
const SPLASH_W = CAB_W;
const SPLASH_D = 0.02;
const SPLASH_H = 0.18;
const SPLASH_CENTER_Y = CAB_Y + CAB_D / 2 + SPLASH_D / 2; // 0.12 + 0.25 + 0.01 = 0.38
const SPLASH_FRONT_Y = SPLASH_CENTER_Y - SPLASH_D / 2;    // 0.37 (viewer-facing face)

// Knob body: chrome cylinder lying along world-Y so its flat
// cap faces the viewer (-Y). Base embedded in backsplash face.
const KNOB_H = 0.020;
const KNOB_R_TOP = 0.015;
const KNOB_R_BOT = 0.018;
const KNOB_X: readonly number[] = [0.00, 0.095, 0.185, 0.28];
const KNOB_Y = SPLASH_FRONT_Y - KNOB_H / 2; // center 10mm in front of splash
const KNOB_Z = 0.080;

// Oven door on the -Y face of the cabinet.
const DOOR_W = 0.52;
const DOOR_D = 0.015;
const DOOR_H = 0.28;
const DOOR_CENTER_Y = CAB_Y - CAB_D / 2 - DOOR_D / 2; // -0.1375
// 80 mm down from the cabinet top; half-height below that.
const DOOR_CENTER_Z = CAB_TOP_Z - 0.08 - DOOR_H / 2;  // -0.228

interface Props {
  snapshot: Snapshot | null;
}

export function Stove({ snapshot }: Props) {
  const heatFlux = snapshot?.wall_heat_flux ?? 0;
  const fluxFrac = Math.min(1, Math.max(0, heatFlux / MAX_FLUX_W_M2));

  // Color ramp: amber (hsl 0.10) -> red (hsl 0.02) as flux rises.
  // Lightness also climbs so at flux=0 the coil reads dark.
  const emissiveColor = useMemo(() => {
    const hue = 0.10 - 0.08 * fluxFrac;
    const light = 0.35 + 0.20 * fluxFrac;
    return new THREE.Color().setHSL(hue, 1.0, light);
  }, [fluxFrac]);

  const emissiveIntensity = 0.15 + fluxFrac * 3.0;

  // A static dark color for the cold burners/knobs so the three
  // decorative burners don't share a ref with the live color.
  const coldColor = useMemo(() => new THREE.Color("#15181d"), []);

  // Cooktop plate built as an ExtrudeGeometry so we can drill four
  // circular holes for the burner wells. A plain RoundedBox has no
  // way to expose a recessed pan; the previous draft tried to hide
  // coils "inside" an opaque plate, which just made the cooktop
  // look smooth and featureless.
  //
  // Shape lives in a rectangle with rounded corners (to match the
  // cabinet's rounded aesthetic); four Paths are pushed as holes at
  // the burner centres in cabinet-local coordinates. Extrusion
  // depth is 8 mm along +Z; we place the mesh so the top face
  // lands at z = 0 (pot base height).
  const cooktopGeometry = useMemo(() => {
    const w = CAB_W / 2;
    const d = CAB_D / 2;
    const cr = 0.012; // corner radius, matches cabinet

    const shape = new THREE.Shape();
    shape.moveTo(-w + cr, -d);
    shape.lineTo(w - cr, -d);
    shape.quadraticCurveTo(w, -d, w, -d + cr);
    shape.lineTo(w, d - cr);
    shape.quadraticCurveTo(w, d, w - cr, d);
    shape.lineTo(-w + cr, d);
    shape.quadraticCurveTo(-w, d, -w, d - cr);
    shape.lineTo(-w, -d + cr);
    shape.quadraticCurveTo(-w, -d, -w + cr, -d);

    BURNERS.forEach(([bx, by]) => {
      const hole = new THREE.Path();
      // Burner centres in cabinet-local coords.
      hole.absarc(bx - CAB_X, by - CAB_Y, 0.108, 0, Math.PI * 2, false);
      shape.holes.push(hole);
    });

    return new THREE.ExtrudeGeometry(shape, {
      depth: 0.008,
      bevelEnabled: false,
      curveSegments: 32,
    });
  }, []);

  return (
    <group>
      {/* ================================================================ */}
      {/* Cabinet body                                                      */}
      {/* ================================================================ */}
      {/* Matte-stainless dark cabinet. Top face lives at CAB_TOP_Z
          (below cooktop bottom) so the cabinet body does NOT plug
          the burner holes from inside. */}
      <RoundedBox
        args={[CAB_W, CAB_D, CAB_H]}
        radius={0.012}
        smoothness={4}
        position={[CAB_X, CAB_Y, CAB_POS_Z]}
        receiveShadow
      >
        <meshStandardMaterial
          color={"#3a3f48"}
          metalness={0.55}
          roughness={0.45}
        />
      </RoundedBox>

      {/* Toe-kick: slight dark inset at the bottom so the cabinet
          reads as sitting on feet, not floating. */}
      <mesh
        position={[CAB_X, CAB_Y, CAB_BASE_Z + 0.04]}
        receiveShadow
      >
        <boxGeometry args={[CAB_W - 0.02, CAB_D - 0.02, 0.04]} />
        <meshStandardMaterial
          color={"#15181e"}
          metalness={0.3}
          roughness={0.65}
        />
      </mesh>

      {/* Feet (4 rubber pads). */}
      {([
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
      ] as const).map(([sx, sy], i) => (
        <mesh
          key={i}
          position={[
            CAB_X + sx * (CAB_W / 2 - 0.03),
            CAB_Y + sy * (CAB_D / 2 - 0.03),
            CAB_BASE_Z - 0.010,
          ]}
          rotation={[Math.PI / 2, 0, 0]}
          receiveShadow
        >
          <cylinderGeometry args={[0.015, 0.015, 0.020, 16]} />
          <meshStandardMaterial
            color={"#0b0d12"}
            metalness={0.2}
            roughness={0.85}
          />
        </mesh>
      ))}

      {/* ================================================================ */}
      {/* Cooktop plate (black enamel, drilled for 4 burner wells)          */}
      {/* ================================================================ */}
      {/* ExtrudeGeometry with 4 circular holes. Mesh is placed so the
          shape's local z=0 lands at world z=-0.008; the +0.008 extrusion
          then brings the top face flush with z=0 (pot base level). */}
      <mesh
        position={[CAB_X, CAB_Y, -0.008]}
        geometry={cooktopGeometry}
        receiveShadow
      >
        <meshStandardMaterial
          color={"#0a0c10"}
          metalness={0.22}
          roughness={0.32}
        />
      </mesh>

      {/* ================================================================ */}
      {/* 4 burner wells                                                    */}
      {/* ================================================================ */}
      {/* Each burner is a recessed well punched through the cooktop
          (see cooktopGeometry). The well floor lands on the cabinet
          top at z = CAB_TOP_Z = -0.008. Stack, top-down:
              z =  0.000   cooktop top / chrome bezel rim
              z =  0.000   coil-ring top (flush with cooktop top)
              z = -0.003   coil-ring centre
              z = -0.006   coil-ring bottom (sitting on drip pan top)
              z = -0.006   drip-pan top
              z = -0.008   drip-pan bottom / cabinet top
          Only burner 0 lights up with heat flux; the other three stay
          cold and decorative. */}
      {BURNERS.map(([bx, by], i) => {
        const active = i === 0;
        const coilEmissive = active ? emissiveColor : coldColor;
        const coilIntensity = active ? emissiveIntensity : 0;
        return (
          <group key={i} position={[bx, by, 0]}>
            {/* Well wall -- open cylinder visible when looking into
                the hole at an angle. DoubleSide so the inner face
                renders. */}
            <mesh
              position={[0, 0, -0.004]}
              rotation={[Math.PI / 2, 0, 0]}
            >
              <cylinderGeometry
                args={[0.108, 0.108, 0.008, 64, 1, true]}
              />
              <meshStandardMaterial
                color={"#1e2026"}
                metalness={0.5}
                roughness={0.4}
                side={THREE.DoubleSide}
              />
            </mesh>

            {/* Drip pan -- dark matte disk forming the well floor. */}
            <mesh
              position={[0, 0, -0.007]}
              rotation={[Math.PI / 2, 0, 0]}
              receiveShadow
            >
              <cylinderGeometry args={[0.106, 0.106, 0.002, 64]} />
              <meshStandardMaterial
                color={"#050608"}
                roughness={0.9}
                metalness={0.08}
              />
            </mesh>

            {/* Chrome bezel -- torus at the top rim of the well so the
                cut edge of the cooktop reads as a polished trim ring. */}
            <mesh position={[0, 0, 0.0002]}>
              <torusGeometry args={[0.108, 0.0022, 16, 96]} />
              <meshStandardMaterial
                color={"#c7cfdc"}
                metalness={0.95}
                roughness={0.18}
              />
            </mesh>

            {/* Four concentric coil rings. Top of outermost ring lands
                flush with the cooktop surface (z = 0); they sit on the
                drip pan at z = -0.006. Emissive response on burner 0
                only. */}
            {COIL_RADII.map((r, ri) => {
              const boost = 1 + (COIL_RADII.length - 1 - ri) * 0.08;
              return (
                <mesh key={ri} position={[0, 0, -0.003]}>
                  <torusGeometry args={[r, COIL_TUBE_R, 16, 80]} />
                  <meshStandardMaterial
                    color={"#1a1d23"}
                    emissive={coilEmissive}
                    emissiveIntensity={active ? coilIntensity * boost : 0}
                    metalness={0.3}
                    roughness={0.55}
                  />
                </mesh>
              );
            })}

            {/* Central hub -- short cylinder holding the coil together
                at its spider. A tiny emissive hint shows the hot
                element even when the pot covers the outer rings. */}
            <mesh
              position={[0, 0, -0.005]}
              rotation={[Math.PI / 2, 0, 0]}
            >
              <cylinderGeometry args={[0.012, 0.012, 0.003, 24]} />
              <meshStandardMaterial
                color={"#15181d"}
                emissive={coilEmissive}
                emissiveIntensity={active ? coilIntensity * 0.5 : 0}
                metalness={0.4}
                roughness={0.5}
              />
            </mesh>

            {/* Three radial spider arms connecting the hub to the
                outer coil. */}
            {[0, (2 * Math.PI) / 3, (4 * Math.PI) / 3].map((theta, ai) => (
              <mesh
                key={ai}
                position={[0, 0, -0.0055]}
                rotation={[0, 0, theta]}
              >
                <boxGeometry args={[0.098, 0.003, 0.0012]} />
                <meshStandardMaterial
                  color={"#1a1d23"}
                  metalness={0.35}
                  roughness={0.6}
                />
              </mesh>
            ))}
          </group>
        );
      })}

      {/* ================================================================ */}
      {/* Backsplash                                                        */}
      {/* ================================================================ */}
      <RoundedBox
        args={[SPLASH_W, SPLASH_D, SPLASH_H]}
        radius={0.004}
        smoothness={3}
        position={[CAB_X, SPLASH_CENTER_Y, SPLASH_H / 2]}
        receiveShadow
      >
        <meshStandardMaterial
          color={"#3a3f48"}
          metalness={0.55}
          roughness={0.45}
        />
      </RoundedBox>

      {/* Subtle inset panel on the splash (darker, recessed look).
          Rotated so the plane stands vertical in world XZ and its
          normal points toward the viewer (-Y). Without the rotation
          the default plane lies flat in XY at z = SPLASH_H/2 -- which
          is exactly knob-centre height, so it read as a flat shelf
          slicing through the four knobs. */}
      <mesh
        position={[CAB_X, SPLASH_FRONT_Y - 0.0005, SPLASH_H / 2]}
        rotation={[Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[SPLASH_W - 0.02, SPLASH_H - 0.03]} />
        <meshStandardMaterial
          color={"#22262d"}
          metalness={0.3}
          roughness={0.55}
        />
      </mesh>

      {/* ================================================================ */}
      {/* 4 knobs                                                           */}
      {/* ================================================================ */}
      {KNOB_X.map((kx, i) => {
        const active = i === 0;
        const tickColor = active ? emissiveColor : new THREE.Color("#2a2d33");
        const tickIntensity = active
          ? Math.max(0.5, emissiveIntensity * 0.7)
          : 0;
        return (
          <group
            key={i}
            position={[kx, KNOB_Y, KNOB_Z]}
          >
            {/* Knob body: cylinderGeometry's axis is local-Y, which
                is world-Y at zero rotation -- the cap lands at
                -Y (toward the viewer). A gentle taper (0.015 top
                / 0.018 bottom) reads as a real control knob. */}
            <mesh castShadow>
              <cylinderGeometry
                args={[KNOB_R_TOP, KNOB_R_BOT, KNOB_H, 32]}
              />
              <meshStandardMaterial
                color={"#d8dee9"}
                metalness={0.85}
                roughness={0.22}
              />
            </mesh>
            {/* Cap disc -- slightly darker, gives the knob a two-tone look. */}
            <mesh position={[0, -KNOB_H / 2 - 0.0005, 0]}>
              <cylinderGeometry
                args={[KNOB_R_TOP, KNOB_R_TOP, 0.001, 32]}
              />
              <meshStandardMaterial
                color={"#aab3c0"}
                metalness={0.9}
                roughness={0.18}
              />
            </mesh>
            {/* Pointer tick on cap face, in the 12-o'clock position. */}
            <mesh
              position={[0, -KNOB_H / 2 - 0.0012, KNOB_R_TOP - 0.004]}
            >
              <boxGeometry args={[0.002, 0.002, 0.006]} />
              <meshStandardMaterial
                color={tickColor}
                emissive={tickColor}
                emissiveIntensity={tickIntensity}
                metalness={0.3}
                roughness={0.4}
              />
            </mesh>
          </group>
        );
      })}

      {/* Per-knob text labels on the splash face, one line under
          each knob. drei <Text> renders a crisp SDF atlas so text
          stays sharp at any zoom; rotation keeps the glyph plane
          parallel to the splash (normal -Y, toward viewer). */}
      {KNOB_X.map((kx, i) => {
        const labels: readonly string[] = ["FRONT L", "FRONT R", "REAR L", "REAR R"];
        const active = i === 0;
        return (
          <Text
            key={`knob-label-${i}`}
            position={[kx, SPLASH_FRONT_Y - 0.0015, KNOB_Z - 0.030]}
            rotation={[Math.PI / 2, 0, 0]}
            fontSize={0.008}
            color={active ? "#e7eaf0" : "#8b93a3"}
            anchorX="center"
            anchorY="middle"
            letterSpacing={0.05}
          >
            {labels[i]}
          </Text>
        );
      })}

      {/* ================================================================ */}
      {/* Active-burner LED (on splash, beside knob 0)                      */}
      {/* ================================================================ */}
      <mesh
        position={[KNOB_X[0], SPLASH_FRONT_Y - 0.001, KNOB_Z + 0.030]}
      >
        <sphereGeometry args={[0.0045, 16, 16]} />
        <meshStandardMaterial
          color={emissiveColor}
          emissive={emissiveColor}
          emissiveIntensity={Math.max(0.3, emissiveIntensity * 0.9)}
          metalness={0.2}
          roughness={0.3}
        />
      </mesh>

      {/* ================================================================ */}
      {/* Oven door + glass window + handle                                 */}
      {/* ================================================================ */}
      {/* Door slab, slightly proud of the cabinet front face. */}
      <RoundedBox
        args={[DOOR_W, DOOR_D, DOOR_H]}
        radius={0.008}
        smoothness={3}
        position={[CAB_X, DOOR_CENTER_Y, DOOR_CENTER_Z]}
        receiveShadow
      >
        <meshStandardMaterial
          color={"#2a2e36"}
          metalness={0.65}
          roughness={0.35}
        />
      </RoundedBox>

      {/* Glass window -- inset dark translucent pane. */}
      <mesh
        position={[
          CAB_X,
          DOOR_CENTER_Y - DOOR_D / 2 - 0.0015,
          DOOR_CENTER_Z,
        ]}
      >
        <boxGeometry args={[DOOR_W - 0.12, 0.003, DOOR_H - 0.10]} />
        <MeshTransmissionMaterial
          resolution={256}
          samples={4}
          thickness={0.02}
          roughness={0.18}
          transmission={0.55}
          ior={1.2}
          color={"#0a0a12"}
          backside={false}
        />
      </mesh>

      {/* Handle bar -- horizontal chrome cylinder on two standoffs,
          mounted just above the glass window. */}
      <group
        position={[
          CAB_X,
          DOOR_CENTER_Y - DOOR_D / 2 - 0.015,
          DOOR_CENTER_Z + DOOR_H / 2 - 0.030,
        ]}
      >
        {/* Two standoff stubs. */}
        {([-0.19, 0.19] as const).map((dx, i) => (
          <mesh
            key={i}
            position={[dx, 0.005, 0]}
            castShadow
          >
            <cylinderGeometry args={[0.005, 0.005, 0.015, 16]} />
            <meshStandardMaterial
              color={"#c7cfdc"}
              metalness={0.95}
              roughness={0.2}
            />
          </mesh>
        ))}
        {/* The bar itself -- cylinder along X (rotate local-Y to world-X). */}
        <mesh
          position={[0, 0.012, 0]}
          rotation={[0, 0, Math.PI / 2]}
          castShadow
        >
          <cylinderGeometry args={[0.008, 0.008, 0.42, 24]} />
          <meshStandardMaterial
            color={"#c7cfdc"}
            metalness={0.95}
            roughness={0.18}
          />
        </mesh>
      </group>

      {/* ================================================================ */}
      {/* Brand text                                                        */}
      {/* ================================================================ */}
      <Text
        position={[
          CAB_X + CAB_W / 2 - 0.025,
          SPLASH_FRONT_Y - 0.0015,
          SPLASH_H - 0.020,
        ]}
        rotation={[Math.PI / 2, 0, 0]}
        fontSize={0.020}
        color={"#f5f7fb"}
        anchorX="right"
        anchorY="middle"
        letterSpacing={0.04}
      >
        BOILINGSIM RANGE
      </Text>
      <Text
        position={[
          CAB_X - CAB_W / 2 + 0.025,
          SPLASH_FRONT_Y - 0.0015,
          SPLASH_H - 0.020,
        ]}
        rotation={[Math.PI / 2, 0, 0]}
        fontSize={0.011}
        color={"#9aa3b2"}
        anchorX="left"
        anchorY="middle"
        letterSpacing={0.08}
      >
        MODEL RT-25
      </Text>

      {/* ================================================================ */}
      {/* Warm glow light above the active burner                           */}
      {/* ================================================================ */}
      {/* Intensity tracks heat flux; at flux=0 it contributes nothing,
          so a cold stove looks cold. */}
      <pointLight
        position={[0, 0, 0.015]}
        color={emissiveColor}
        intensity={fluxFrac * 1.2}
        distance={0.35}
        decay={2}
      />
    </group>
  );
}
