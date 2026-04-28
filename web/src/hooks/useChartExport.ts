// Browser-side PNG export of report charts. Each Results-page chart
// gets a `data-chart-name="<slug>"` attribute on its outer wrapper
// (see ChartCard.tsx); this module rasterises those nodes via
// html-to-image and either downloads a single PNG or bundles every
// node found under a root into a ZIP via JSZip.
//
// Why html-to-image: it walks the DOM, inlines computed CSS, and
// embeds web fonts so SVGs in Recharts come out with our Inter
// glyphs intact instead of system serif fallback. Lightweight,
// server-free, no headless browser involved.
//
// All exports run at `pixelRatio = 2` so the PNGs are sharp at
// presentation zoom (slides, docs) without doubling file size for
// no reason.

import { toPng } from "html-to-image";
import JSZip from "jszip";

const PIXEL_RATIO = 2;

/** Chart background colour stays consistent across exports so the
 *  rasterised PNG doesn't pick up the page background by accident. */
const EXPORT_BG = "#0f1216";

/** Trigger a same-origin file download from a Blob / data URL. */
function triggerDownload(href: string, filename: string): void {
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/** Snapshot a single DOM node to a PNG and download it.
 *
 *  @param node     Any HTMLElement (typically a `<section data-chart-name=...>`).
 *  @param filename Suggested file name; the .png extension is appended.
 */
export async function exportNodeAsPng(
  node: HTMLElement,
  filename: string,
): Promise<void> {
  const dataUrl = await toPng(node, {
    pixelRatio: PIXEL_RATIO,
    backgroundColor: EXPORT_BG,
    cacheBust: true,
  });
  const safe = filename.endsWith(".png") ? filename : `${filename}.png`;
  triggerDownload(dataUrl, safe);
}

/** Find every chart-tagged node under `root` and bundle their PNG
 *  snapshots into a single ZIP, then trigger a download.
 *
 *  Each node must carry `data-chart-name="<slug>"` -- that slug
 *  becomes the file name inside the ZIP.
 *
 *  @param root  Container scanned for `[data-chart-name]` (typically
 *               the report's outer div).
 *  @param runId Used as the prefix for the ZIP file name.
 */
export async function exportAllChartsAsZip(
  root: HTMLElement,
  runId: string,
): Promise<void> {
  const nodes = root.querySelectorAll<HTMLElement>("[data-chart-name]");
  if (nodes.length === 0) return;

  const zip = new JSZip();
  // Snapshot all nodes in parallel so the ZIP build doesn't block on
  // sequential rasterisation. html-to-image internally throttles via
  // microtasks, so sub-100ms cards finish well before the user
  // notices the spinner.
  const tasks: Array<Promise<void>> = [];
  nodes.forEach((node) => {
    const slug = node.getAttribute("data-chart-name") ?? "chart";
    tasks.push(
      toPng(node, {
        pixelRatio: PIXEL_RATIO,
        backgroundColor: EXPORT_BG,
        cacheBust: true,
      }).then((dataUrl) => {
        // dataUrl is "data:image/png;base64,..." -- strip the prefix.
        const base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
        zip.file(`${slug}.png`, base64, { base64: true });
      }),
    );
  });
  await Promise.all(tasks);

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  try {
    triggerDownload(url, `${runId}_charts.zip`);
  } finally {
    // Free the blob URL on the next tick (after the click handler).
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
