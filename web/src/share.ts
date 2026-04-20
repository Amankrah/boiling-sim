// Share-link URL encode/decode.
//
// Semantic scope (pinned in the Phase 6 plan non-goals): the URL
// carries **scenario parameters + camera pose only**. It does NOT
// carry simulation time or state. Opening a shared link starts a
// fresh simulation at t = 0 with the encoded scene; the camera pose
// reconstructs the viewing angle from the original session.
//
// URL shape:
//   ?hf=30000&mat=steel_304&cd=25&cl=50
//    &cx=0.25&cy=-0.25&cz=0.18
//    &cfx=0&cfy=0&cfz=0.06
//
// Keys:
//   hf         wall heat flux in W/m^2
//   mat        pot material ("steel_304" | "copper" | "aluminum")
//   cd, cl     carrot diameter and length in mm
//   cx,cy,cz   camera position (metres)
//   cfx,cfy,cfz  camera target / lookAt (metres)

export type Material = "steel_304" | "copper" | "aluminum";

export interface ShareableParams {
  heatFluxWPerM2: number;
  material: Material;
  carrotDiameterMm: number;
  carrotLengthMm: number;
}

export interface ShareableCamera {
  position: [number, number, number];
  target: [number, number, number];
}

export interface ShareState {
  params: ShareableParams;
  camera: ShareableCamera;
}

const MATERIAL_SET = new Set<Material>(["steel_304", "copper", "aluminum"]);

// Default-scenario fallbacks for any field missing from the URL.
export const DEFAULT_SHARE_STATE: ShareState = {
  params: {
    heatFluxWPerM2: 30_000,
    material: "steel_304",
    carrotDiameterMm: 25,
    carrotLengthMm: 50,
  },
  camera: {
    position: [0.25, -0.25, 0.18],
    target: [0, 0, 0.06],
  },
};

function num(value: string | null, fallback: number): number {
  if (value === null || value === "") return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function mat(value: string | null, fallback: Material): Material {
  if (value === null) return fallback;
  return MATERIAL_SET.has(value as Material) ? (value as Material) : fallback;
}

/** Parse a `URLSearchParams` (or the current `window.location.search`) into a ShareState. */
export function decodeShareState(
  search: URLSearchParams | string = typeof window !== "undefined" ? window.location.search : "",
  defaults: ShareState = DEFAULT_SHARE_STATE,
): ShareState {
  const p = search instanceof URLSearchParams ? search : new URLSearchParams(search);
  return {
    params: {
      heatFluxWPerM2: num(p.get("hf"), defaults.params.heatFluxWPerM2),
      material: mat(p.get("mat"), defaults.params.material),
      carrotDiameterMm: num(p.get("cd"), defaults.params.carrotDiameterMm),
      carrotLengthMm: num(p.get("cl"), defaults.params.carrotLengthMm),
    },
    camera: {
      position: [
        num(p.get("cx"), defaults.camera.position[0]),
        num(p.get("cy"), defaults.camera.position[1]),
        num(p.get("cz"), defaults.camera.position[2]),
      ],
      target: [
        num(p.get("cfx"), defaults.camera.target[0]),
        num(p.get("cfy"), defaults.camera.target[1]),
        num(p.get("cfz"), defaults.camera.target[2]),
      ],
    },
  };
}

/** Format a number compactly for URLs (4 significant digits). */
function fmt(n: number): string {
  if (Number.isInteger(n)) return String(n);
  return n.toPrecision(4).replace(/\.?0+$/, "");
}

/** Encode a ShareState into a querystring (no leading '?'). */
export function encodeShareState(state: ShareState): string {
  const p = new URLSearchParams();
  p.set("hf", String(state.params.heatFluxWPerM2));
  p.set("mat", state.params.material);
  p.set("cd", String(state.params.carrotDiameterMm));
  p.set("cl", String(state.params.carrotLengthMm));
  p.set("cx", fmt(state.camera.position[0]));
  p.set("cy", fmt(state.camera.position[1]));
  p.set("cz", fmt(state.camera.position[2]));
  p.set("cfx", fmt(state.camera.target[0]));
  p.set("cfy", fmt(state.camera.target[1]));
  p.set("cfz", fmt(state.camera.target[2]));
  return p.toString();
}

/** Compose a full share URL for the current location. */
export function buildShareUrl(state: ShareState, origin?: string, pathname?: string): string {
  const loc = typeof window !== "undefined" ? window.location : undefined;
  const o = origin ?? loc?.origin ?? "";
  const p = pathname ?? loc?.pathname ?? "/";
  return `${o}${p}?${encodeShareState(state)}`;
}

/** Update `window.location` without reloading. Preserves any
 *  non-share-state query params (notably `page` from the router) so
 *  the user's tab selection survives share-link writes. No-op
 *  outside a browser env. */
export function pushShareState(state: ShareState): void {
  if (typeof window === "undefined") return;
  const preserved = new URLSearchParams(window.location.search);
  // Strip the share-state keys so we can re-add them from `state`.
  for (const key of [
    "hf", "mat", "cd", "cl",
    "cx", "cy", "cz",
    "cfx", "cfy", "cfz",
  ]) {
    preserved.delete(key);
  }
  const shareQs = encodeShareState(state);
  const merged = preserved.toString()
    ? `${shareQs}&${preserved.toString()}`
    : shareQs;
  const url = `${window.location.pathname}?${merged}`;
  window.history.replaceState(null, "", url);
}
