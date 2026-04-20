// Pot -- a realistic stockpot with rim, opposed handles, and a
// translucent glass lid. Every part is derived from the props so
// the pot scales cleanly as the user adjusts diameter / height /
// wall thickness / base thickness on the Configuration page. The
// actual source of truth for those values is the v4 snapshot
// (`snapshot.pot_*`), threaded through BoilingScene.
//
// Coordinate system
// -----------------
// The solver runs with physics-Z vertical, so the outer `group` is
// rotated [pi/2, 0, 0] to remap local-Y (the axis of the cylinder
// primitives) to world-Z. Inside the group everything is expressed
// in local coordinates; external callers never see them.
//
//   local-Y axis = world-Z (vertical)
//   local-X, local-Z  = world-X, world-(-Y)  (horizontal plane)
//
// Group origin is positioned at z = heightM / 2 so the pot's base
// disc sits exactly at world z = 0 (cooktop level, where Pot.tsx
// has always landed).
//
// Anatomy
// -------
//   * Outer shell -- full cylinder, stainless
//   * Inner shell -- thinner cylinder inside, darker cavity
//   * Base disc   -- flat circle at the pot floor
//   * Rim lip     -- torus at the top edge, slight flare + chrome
//   * Handles     -- two opposed D-shapes on the +/- X sides
//   * Lid         -- translucent glass dome + chrome rim + knob
//                    (gated by `lidVisible`, default true)

import { useMemo } from "react";
import * as THREE from "three";

import { MeshTransmissionMaterial } from "@react-three/drei";

interface Props {
  diameterM?: number;
  heightM?: number;
  wallThicknessM?: number;
  baseThicknessM?: number;
  /** When false, the lid geometry is omitted -- useful if a future
   *  UI wants to show the pot open. Default true. */
  lidVisible?: boolean;
}

// Reference: matte-black enamelled stockpot with chrome trim
// + side handles + glass lid. Body reads dark so the chrome rim
// and handles pop as highlights (matches the photo reference the
// user shared).
const COLOR_BODY = "#1c1e24";
const COLOR_CAVITY = "#2a2c33";
const COLOR_BASE = "#0d0e12";
const COLOR_CHROME = "#e0e5ee";
const COLOR_HANDLE = "#cdd3df";

export function Pot({
  diameterM = 0.20,
  heightM = 0.12,
  wallThicknessM = 0.003,
  baseThicknessM = 0.005,
  lidVisible = true,
}: Props) {
  // --- Derived dimensions (all in local coords; Y is vertical here) ---
  const outerR = diameterM / 2;
  const innerR = Math.max(outerR - wallThicknessM, outerR * 0.8);
  const rimR = outerR * 1.015;
  const rimTubeR = Math.max(wallThicknessM * 0.6, 0.0012);

  // Inner cavity starts above the base plate and stops a hair below
  // the rim so the walls end cleanly against the flared lip.
  const cavityH = Math.max(heightM - baseThicknessM * 2, heightM * 0.2);

  // Base disc centre -- sits baseThicknessM / 2 above the pot floor
  // (the group's local -Y extreme) so the disc face is at local
  // y = -heightM/2 + baseThicknessM.
  const baseY = -heightM / 2 + baseThicknessM / 2;
  const rimY = heightM / 2;

  // Handles: chunky chrome D-loops mounted on the upper portion
  // of the pot body via visible rivet heads. The loop's PLANE is
  // tangential to the pot (it wraps alongside the pot's curve)
  // rather than radial, matching the real stockpot in the user's
  // reference where the handle looks like a thick loop hanging
  // off the side of the pot rather than a radial protrusion.
  //
  // Tube thickness is ~5 % of outerR (chunky), extension is
  // handleArcR outward from the wall, and the arc is roughly as
  // wide (along the pot wall) as it is tall, giving a nearly-round
  // loop silhouette.
  const handleArcR = outerR * 0.22;
  const handleTubeR = Math.max(outerR * 0.050, 0.004);
  const handleStubLen = handleArcR * 0.55;
  const handleCentreY = heightM * 0.20;
  // Rivet collars: decorative chrome discs hiding the seam where
  // the handle tube meets the pot wall. Larger than the tube so
  // they fully cover the entry point, slightly thicker than before
  // so the discs read as welded flanges, not stickers.
  const rivetR = handleTubeR * 2.0;
  const rivetThick = handleTubeR * 0.8;

  // Lid parameters -- all derived from outer diameter so shrinking
  // the pot scales the lid too. The reference lid is nearly FLAT
  // (very shallow dome) with a prominent chrome collar around its
  // edge and an hourglass-mushroom knob on top.
  const lidR = outerR * 1.02;
  const lidRise = lidR * 0.08;
  const knobR = outerR * 0.09;
  const knobH = outerR * 0.12;
  const lidRimTubeR = Math.max(wallThicknessM * 1.4, 0.0022);

  // Glass-dome geometry: compute the spherical-cap parameters that
  // produce a cap of radius `lidR` and rise `lidRise`.
  //
  //   sphereR^2 = lidR^2 + (sphereR - lidRise)^2
  //   => sphereR = (lidR^2 + lidRise^2) / (2 * lidRise)
  //   thetaMax   = acos((sphereR - lidRise) / sphereR)
  //
  // sphereGeometry's `thetaStart` begins at the top pole (local +Y);
  // slicing from 0 to thetaMax gives the shallow dome we want.
  const { sphereR, thetaMax } = useMemo(() => {
    const R = (lidR * lidR + lidRise * lidRise) / (2 * lidRise);
    const t = Math.acos((R - lidRise) / R);
    return { sphereR: R, thetaMax: t };
  }, [lidR, lidRise]);

  // ONE continuous smooth curve from the +Z rivet to the -Z rivet,
  // sweeping outward and rising slightly in the middle. The curve
  // replaces the old stubs + arc (which met at 90 deg and showed a
  // visible kink). Parametrization t in [0, 1] with:
  //
  //   X(t) = outExtent * sin(pi * t)      -> outward max at t = 0.5,
  //                                          X'(0) = X'(1) * (-1):
  //                                          tangent is +X at t=0 and -X
  //                                          at t=1, so the tube meets each
  //                                          rivet perpendicular to the wall.
  //   Y(t) = upExtent * sin^2(pi * t)     -> rises smoothly to `upExtent`
  //                                          at the apex; Y'(0) = Y'(1) = 0
  //                                          so there's no vertical kink at
  //                                          the rivet.
  //   Z(t) = arcR * cos(pi * t)           -> +arcR at t=0 to -arcR at t=1.
  //
  // The tangent at both endpoints is purely radial, so the tube enters
  // each rivet cleanly without any bend.
  const handleArcCurve = useMemo(() => {
    const outExtent = handleArcR + handleStubLen;
    const upExtent = handleArcR * 0.55;
    class HandleArc extends THREE.Curve<THREE.Vector3> {
      constructor() {
        super();
      }
      override getPoint(t: number, optionalTarget = new THREE.Vector3()) {
        const pit = Math.PI * t;
        const s = Math.sin(pit);
        return optionalTarget.set(
          outExtent * s,
          upExtent * s * s,
          handleArcR * Math.cos(pit),
        );
      }
    }
    return new HandleArc();
  }, [handleArcR, handleStubLen]);

  return (
    <group
      rotation={[Math.PI / 2, 0, 0]}
      position={[0, 0, heightM / 2]}
    >
      {/* =====================================================
          Outer shell -- stainless cylinder, solid (opaque so the
          scene doesn't lose depth cues). castShadow on so the pot
          drops a shadow onto the cooktop through the scene rig.
         ===================================================== */}
      <mesh castShadow receiveShadow>
        <cylinderGeometry
          args={[outerR, outerR, heightM, 96, 1, true]}
        />
        <meshStandardMaterial
          color={COLOR_BODY}
          metalness={0.6}
          roughness={0.32}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Inner shell -- narrower + darker cavity. Back-side-only so
          the visible face is the one you'd see looking down into
          the pot; also stops the outer shell's reflections from
          bleeding through double-sided material. */}
      <mesh receiveShadow>
        <cylinderGeometry
          args={[innerR, innerR, cavityH, 64, 1, true]}
        />
        <meshStandardMaterial
          color={COLOR_CAVITY}
          metalness={0.45}
          roughness={0.55}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Base disc -- closes the floor of the pot. */}
      <mesh
        position={[0, baseY, 0]}
        rotation={[Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <circleGeometry args={[outerR, 64]} />
        <meshStandardMaterial
          color={COLOR_BASE}
          metalness={0.55}
          roughness={0.55}
        />
      </mesh>

      {/* Inner-base disc -- the cavity floor the water sits on.
          Slightly narrower than the outer base so the wall
          thickness reads from above. */}
      <mesh
        position={[0, baseY + baseThicknessM / 2 + 0.0002, 0]}
        rotation={[Math.PI / 2, 0, 0]}
      >
        <circleGeometry args={[innerR, 64]} />
        <meshStandardMaterial
          color={COLOR_CAVITY}
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>

      {/* Rim lip -- torus at the top edge giving a visible flare.
          TorusGeometry's default ring plane is local XY with axis
          along local +Z. We want the ring FLAT (horizontal) around
          the pot, i.e. axis along local +Y (which becomes world +Z
          after the outer pot rotation). Rotating the mesh by
          [pi/2, 0, 0] about local X swings the torus axis from +Z
          to -Y, landing the ring in the local XZ plane -> world XY
          plane -> horizontal. Without this rotation the ring stood
          vertical and read as a "bucket handle" over the pot. */}
      <mesh
        position={[0, rimY, 0]}
        rotation={[Math.PI / 2, 0, 0]}
        castShadow
      >
        <torusGeometry args={[rimR, rimTubeR, 14, 96]} />
        <meshStandardMaterial
          color={COLOR_CHROME}
          metalness={0.9}
          roughness={0.22}
        />
      </mesh>

      {/* =====================================================
          Handles -- two opposed thick chrome D-loops on the +/- X
          sides of the pot, positioned in the upper third of the
          body. Each handle is built from five parts:

              * two small rivet discs sitting flush on the pot wall
                (the mount tabs you can see on a real stockpot)
              * two short mount stubs extending radially out
              * a half-torus arc bridging the stub tips, bulging
                further out from the wall

          The whole handle group is translated UP by handleCentreY
          so the loop's horizontal midline sits above the pot's
          mid-height, closer to the rim (matches the reference).

          Coordinate notes inside each handle group:
              +X -> radially outward from the pot
              +Y -> up along the pot's cylinder axis
              +Z -> tangential to the pot wall
          The -X side handle mirrors the +X one by rotating the
          whole group 180 deg about local Y (so +X local -> -X
          world).
         ===================================================== */}
      {([+1, -1] as const).map((side) => (
        <group
          key={side}
          position={[side * outerR, handleCentreY, 0]}
          rotation={[0, side > 0 ? 0 : Math.PI, 0]}
        >
          {/* Tangential mount: the two rivets sit side-by-side on
              the pot wall along the TANGENT direction (local Z),
              not stacked vertically. The arc then rises up and over
              between them in the tangent-vertical plane (local YZ),
              which after the outer pot rotation reads as a loop
              hugging the pot's curve -- the shape the user pointed
              at in the reference photo.

              Rivet at local (0, 0, +handleArcR): flush on the pot
              wall, disc faces +X (outward). */}
          <mesh
            position={[rivetThick / 2, 0, +handleArcR]}
            rotation={[0, 0, -Math.PI / 2]}
            castShadow
          >
            <cylinderGeometry args={[rivetR, rivetR, rivetThick, 24]} />
            <meshStandardMaterial
              color={COLOR_HANDLE}
              metalness={0.9}
              roughness={0.25}
            />
          </mesh>

          {/* Opposite rivet on the other side of the tangent pair. */}
          <mesh
            position={[rivetThick / 2, 0, -handleArcR]}
            rotation={[0, 0, -Math.PI / 2]}
            castShadow
          >
            <cylinderGeometry args={[rivetR, rivetR, rivetThick, 24]} />
            <meshStandardMaterial
              color={COLOR_HANDLE}
              metalness={0.9}
              roughness={0.25}
            />
          </mesh>

          {/* Single continuous tube starting AT the pot wall (local
              X = 0) so there is no visible gap between the pot body
              and the handle. The curve's tangent is purely radial
              at both endpoints, so the tube emerges from the pot
              wall perpendicular to the surface with no bend. The
              rivet discs above act as a chrome collar around each
              entry point, hiding the junction. */}
          <mesh position={[0, 0, 0]} castShadow>
            <tubeGeometry
              args={[handleArcCurve, 64, handleTubeR, 16, false]}
            />
            <meshStandardMaterial
              color={COLOR_HANDLE}
              metalness={0.88}
              roughness={0.22}
            />
          </mesh>
        </group>
      ))}

      {/* =====================================================
          Lid -- translucent glass dome + chrome rim + knob. Gated
          by `lidVisible` so a future toggle can hide it without
          touching Pot's callers.
         ===================================================== */}
      {lidVisible ? (
        <group position={[0, rimY + wallThicknessM, 0]}>
          {/* Chrome collar -- thick horizontal torus encircling
              the lid's base. Rotation [pi/2, 0, 0] swings the torus
              axis from local +Z to local -Y, which after the outer
              pot rotation becomes world -Z -> the ring lies in the
              world XY plane, i.e. HORIZONTAL, capping the pot top
              like the stainless collar in the reference photo.
              Without this rotation the ring stood vertical and
              read as an overhead bucket handle. */}
          <mesh rotation={[Math.PI / 2, 0, 0]} castShadow>
            <torusGeometry args={[lidR, lidRimTubeR, 20, 96]} />
            <meshStandardMaterial
              color={COLOR_CHROME}
              metalness={0.95}
              roughness={0.18}
            />
          </mesh>

          {/* Glass disc -- very shallow spherical cap (lidRise
              ~= 8 % of lidR) so it reads as a nearly flat glass
              lid rather than a dome. Position it so the cap's
              base circle sits at local y = 0 (lid origin).
              sphereGeometry is generated with its centre at the
              origin, so we translate down by (sphereR - lidRise)
              to land the cap circle at y = 0. */}
          <mesh
            position={[0, -(sphereR - lidRise), 0]}
            castShadow
          >
            <sphereGeometry
              args={[sphereR, 64, 32, 0, Math.PI * 2, 0, thetaMax]}
            />
            <MeshTransmissionMaterial
              resolution={256}
              samples={4}
              thickness={0.012}
              roughness={0.08}
              transmission={0.78}
              ior={1.3}
              color={"#e4e9f0"}
              backside={false}
              chromaticAberration={0.02}
              distortion={0.02}
            />
          </mesh>

          {/* Steam vent -- small dark spot near the knob, just a
              visual hint that the lid is a real cooking lid. */}
          <mesh position={[knobR * 2.2, lidRise * 0.5, 0]} castShadow>
            <cylinderGeometry
              args={[knobR * 0.18, knobR * 0.18, 0.002, 16]}
            />
            <meshStandardMaterial
              color={"#0a0c10"}
              metalness={0.4}
              roughness={0.5}
            />
          </mesh>

          {/* Knob -- hourglass / mushroom profile matching the
              reference (flared base -> narrow neck -> wide flat
              cap). Three stacked cylinders; total height knobH. */}
          <group position={[0, lidRise, 0]}>
            {/* Flared base -- widens from neck down to dome contact. */}
            <mesh
              position={[0, knobH * 0.15, 0]}
              castShadow
            >
              <cylinderGeometry
                args={[knobR * 0.42, knobR * 0.78, knobH * 0.3, 32]}
              />
              <meshStandardMaterial
                color={COLOR_CHROME}
                metalness={0.93}
                roughness={0.2}
              />
            </mesh>

            {/* Narrow neck -- straight cylinder. */}
            <mesh
              position={[0, knobH * 0.48, 0]}
              castShadow
            >
              <cylinderGeometry
                args={[knobR * 0.42, knobR * 0.42, knobH * 0.36, 32]}
              />
              <meshStandardMaterial
                color={COLOR_CHROME}
                metalness={0.93}
                roughness={0.2}
              />
            </mesh>

            {/* Wide flat cap -- the disc the cook grips. */}
            <mesh
              position={[0, knobH * 0.82, 0]}
              castShadow
            >
              <cylinderGeometry
                args={[knobR, knobR, knobH * 0.3, 32]}
              />
              <meshStandardMaterial
                color={COLOR_CHROME}
                metalness={0.93}
                roughness={0.2}
              />
            </mesh>
          </group>
        </group>
      ) : null}

    </group>
  );
}
