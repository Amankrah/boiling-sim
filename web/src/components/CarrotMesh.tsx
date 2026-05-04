// Procedural-cylinder ingredient renderer. v8: walks
// ``snapshot.ingredients`` and renders one cylinder per instance per
// ingredient, with a per-ingredient color palette driven by the
// ingredient's ``name`` field.
//
// Colour mapping: high retention → vivid fresh color (orange for
// carrots, cream for potatoes, pale yellow for onions); low retention
// → muted/charred variant. When both solutes are active we average
// them. Per-instance retention is M3 infrastructure -- for ingredient
// 0 (the legacy carrot) we look up the per-instance vector; other
// ingredients use their aggregate `IngredientState.retention`.

import { useMemo } from "react";
import * as THREE from "three";

import type { IngredientState, Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot;
}

/** Three-stop gradient per ingredient name. Each palette has fresh /
 *  cooked / charred stops; retention picks the interpolant. Unknown
 *  names fall back to the carrot palette so user-defined ingredient
 *  names still render. */
const INGREDIENT_PALETTES: Record<
  string,
  { fresh: string; cooked: string; charred: string }
> = {
  carrot: { fresh: "#f5a524", cooked: "#a46419", charred: "#3a2614" },
  potato: { fresh: "#e8d6a5", cooked: "#b89968", charred: "#54422a" },
  onion: { fresh: "#f3e9c4", cooked: "#cfbe85", charred: "#7a6a3c" },
  celery: { fresh: "#bccf63", cooked: "#7e8d36", charred: "#3c4516" },
};

function retentionToColor(retentionPct: number, name: string): THREE.Color {
  const palette = INGREDIENT_PALETTES[name] ?? INGREDIENT_PALETTES.carrot;
  const t = Math.max(0, Math.min(1, retentionPct / 100));
  const fresh = new THREE.Color(palette.fresh);
  const cooked = new THREE.Color(palette.cooked);
  const charred = new THREE.Color(palette.charred);
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

/** Render one ingredient's instances. Per-instance retention is
 *  available for ingredient 0 (the legacy carrot) via M3's
 *  ``carrot_retention_per_instance`` vector; other ingredients get a
 *  uniform color from their aggregate retention. */
function renderIngredient(
  ing: IngredientState,
  ingIdx: number,
  snapshot: Snapshot,
): JSX.Element[] {
  const dual = ing.retention2 < 99.99;
  // Ingredient 0 has access to per-instance retention (M3); others fall
  // back to the aggregate so multi-instance potato pieces still color
  // sensibly.
  const perInstance =
    ingIdx === 0 ? snapshot.carrot_retention_per_instance : [];
  const perInstance2 =
    ingIdx === 0 ? snapshot.carrot_retention2_per_instance : [];
  const havePerInstance = perInstance.length === ing.centres.length;

  return ing.centres.map((centre, idx) => {
    const { position, rotation } = poseFromAxis(ing.axis, centre, ing.length_m);
    let r = ing.retention;
    let r2 = ing.retention2;
    if (havePerInstance) {
      r = perInstance[idx] ?? ing.retention;
      if (perInstance2.length === perInstance.length) {
        r2 = perInstance2[idx] ?? ing.retention2;
      }
    }
    const blend = dual ? (r + r2) * 0.5 : r;
    const color = retentionToColor(blend, ing.name);
    return (
      <group
        key={`ing-${ingIdx}-inst-${idx}`}
        position={position}
        rotation={rotation}
      >
        <mesh>
          <cylinderGeometry
            args={[ing.diameter_m / 2, ing.diameter_m / 2, ing.length_m, 32, 1]}
          />
          <meshStandardMaterial color={color} roughness={0.55} />
        </mesh>
      </group>
    );
  });
}

export function CarrotMesh({ snapshot }: Props) {
  // v8: walk snapshot.ingredients[]; legacy single-carrot scenarios
  // surface as a one-element list. Empty during rebuild markers --
  // render nothing.
  const ingredients = snapshot.ingredients ?? [];

  // useMemo for the JSX list keeps re-renders cheap when only the
  // retention numbers change; identity-stable centres array means
  // React keys match across frames.
  const meshes = useMemo(() => {
    const out: JSX.Element[] = [];
    ingredients.forEach((ing, k) => {
      out.push(...renderIngredient(ing, k, snapshot));
    });
    return out;
  }, [ingredients, snapshot]);

  if (meshes.length === 0) return null;
  return <>{meshes}</>;
}
