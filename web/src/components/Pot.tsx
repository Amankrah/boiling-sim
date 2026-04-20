// Pot mesh -- cylinder walls + base disc. Redesigned so the pot
// reads as an object against the new scene gradient rather than a
// ghost against a near-black void.
//
// Key changes from the M5 draft:
//   - Outer shell: opacity 0.85, lighter neutral (#c0cbd8),
//     metalness 0.6 / roughness 0.35. With the drei <Environment>
//     in BoilingScene, this now catches visible reflections.
//   - Inner shell removed -- it was a desperate double-layer trick
//     that fought the (broken) low-opacity outer. No longer needed.
//   - Base disc darkened to match `--border-strong` so the pot
//     sits cleanly on the scene's reference grid.
//
// Dimensions unchanged from the default scenario (20 cm D ×
// 12 cm H, 3 mm walls).

interface Props {
  diameterM?: number;
  heightM?: number;
  wallThicknessM?: number;
}

export function Pot({
  diameterM = 0.20,
  heightM = 0.12,
  wallThicknessM = 0.003,
}: Props) {
  const outerR = diameterM / 2;
  // Physics grid has Z vertical; cylinderGeometry's axis is local Y.
  // Rotate the whole group so local-Y → world-Z.
  return (
    <group rotation={[Math.PI / 2, 0, 0]} position={[0, 0, heightM / 2]}>
      {/* Outer shell -- near-opaque with PBR metallic response. */}
      <mesh castShadow receiveShadow>
        <cylinderGeometry
          args={[outerR, outerR, heightM, 64, 1, true]}
        />
        <meshStandardMaterial
          color={"#c0cbd8"}
          roughness={0.35}
          metalness={0.6}
          transparent
          opacity={0.85}
          side={2}
        />
      </mesh>
      {/* Base disc -- dark, grounds the pot on the reference grid. */}
      <mesh
        position={[0, -heightM / 2 + wallThicknessM / 2, 0]}
        rotation={[Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <circleGeometry args={[outerR, 64]} />
        <meshStandardMaterial
          color={"#2c3a4d"}
          roughness={0.6}
          metalness={0.5}
        />
      </mesh>
    </group>
  );
}
