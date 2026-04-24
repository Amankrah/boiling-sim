# RunPod on-demand deployment

Runbook for spinning up a public BoilingSim demo on a RunPod GPU pod, running the live dashboard for 1-8 hours, and shutting down. Target cost: **~$0.40 / hour on RTX 4090 Community Cloud**, so a typical demo slot is $1-4.

Artefacts in this folder:

- **[Dockerfile.runpod](../Dockerfile.runpod)** — unified single-container image (Rust ws-server + Python solver + nginx web + supervisord).
- **[supervisord.conf](supervisord.conf)** — three-process orchestration inside the container.
- **[nginx.runpod.conf](nginx.runpod.conf)** — nginx that proxies `/stream` + `/api` + `/health` to loopback ws-server.

---

## Prerequisites (one-time, ~30 min)

1. **RunPod account** at <https://www.runpod.io/>. Load $10 of credit — this lasts months for demo-only usage.
2. **GitHub Container Registry (ghcr.io) login** so RunPod can pull your image:
   - Create a GitHub personal-access token with `read:packages`.
   - Push the image as **public** (free) or private (requires a RunPod-side registry secret; extra setup).
3. **Local GPU build host**. You need your own NVIDIA workstation to produce the image with the Warp JIT cache baked in. Without this step the image still works, it just loses ~45 s on every pod cold-start.

---

## One-time image build + push

```bash
# On your workstation (assumed: RTX 4090 + Docker + NVIDIA Container Toolkit)
cd ~/Desktop/Dev_Projects/boiling-sim
git tag v0.1
git push --tags

# Build with GPU visible so the Warp precompile layer actually runs.
docker build \
    --pull \
    -f Dockerfile.runpod \
    -t ghcr.io/amankrah/boiling-sim:v0.1 \
    -t ghcr.io/amankrah/boiling-sim:latest \
    .

# Smoke-test locally before pushing (~2 min wall time for first snapshot).
docker run --rm -it --gpus all -p 3000:80 \
    ghcr.io/amankrah/boiling-sim:v0.1
# Browser: http://localhost:3000

# Log in to ghcr.io with a PAT, then push both tags.
echo $GHCR_TOKEN | docker login ghcr.io -u Amankrah --password-stdin
docker push ghcr.io/amankrah/boiling-sim:v0.1
docker push ghcr.io/amankrah/boiling-sim:latest
```

First build takes ~15-25 min on a 4090 (CUDA base pull + Python deps + Rust build + Warp JIT). Subsequent builds with layer caching are 2-5 min for source-only changes.

If you push to a **public** GHCR repository the image will be pullable on RunPod without further setup. If you keep it private, you additionally need to create a RunPod container-registry credential (RunPod console → Settings → Container Registry Auth → New Credential) and reference it in the template.

---

## One-time RunPod template

Go to <https://www.runpod.io/console/user/templates> → **+ New Template**:

| Field | Value |
|---|---|
| Template Name | `boilingsim-v0.1` |
| Container Image | `ghcr.io/amankrah/boiling-sim:v0.1` |
| Container Registry Credentials | *(only if the image is private)* |
| Docker Command | *(leave empty — image CMD is `supervisord -n`)* |
| Container Disk | `20 GB` |
| Volume Disk | `0 GB` *(or 10 GB if you want artefacts to survive pod restarts)* |
| Volume Mount Path | `/workspace` *(only if Volume Disk > 0)* |
| Expose HTTP Ports | `80` |
| Expose TCP Ports | *(leave empty)* |
| Environment Variables | *(leave empty — image defaults work)* |

Save. You'll reuse this template for every demo launch.

---

## Per-demo workflow (~30 min before the demo)

### T − 30 min: Deploy the pod

1. RunPod console → **Pods → + Deploy**.
2. Filter: **GPU = RTX 4090**, **Cloud Type = Community Cloud** ($0.34/hr spot, $0.44/hr on-demand). Pick any region with stock.
3. **Template**: `boilingsim-v0.1` (your saved template).
4. Click **Deploy**. Pod enters **Allocating** → **Running** in ~1-3 min.

### T − 25 min: Wait for the stack to come up

Click the pod → **Logs** tab. Watch for this sequence:

```
[supervisord] started
ws-server listening on 0.0.0.0:8080
nginx started
[precheck] nvidia-smi: ... NVIDIA GeForce RTX 4090 ...
[precheck] Warp sees 1 CUDA device(s). Ready.
[run_dashboard] === Phase 6 live dashboard producer ===
  t_sim=  1.25s  step=   43  sent=   14 ...
```

If you see `[precheck] FAIL` — the pod didn't get a GPU. Rare but happens on Community Cloud spot; just **Stop → Deploy** again on a different region.

If ws-server reports `Failed to bind 0.0.0.0:8080` — port conflict, probably another supervisord restart loop. Stop + redeploy.

### T − 20 min: Get the public URL

Pod details page → **Connect → HTTP Service [Port 80]** → opens `https://<pod-id>-80.proxy.runpod.net/`.

Copy this URL. It's SSL-terminated by RunPod's proxy, no cert work on your side.

Health check from your laptop:

```bash
curl -fsS https://<pod-id>-80.proxy.runpod.net/health
# expect: 200 OK
```

### T − 15 min: Smoke-test

Open the URL in a browser. Within 30 s you should see:
- Stove + pot rendered
- Water temp climbing toward saturation
- Bubble count rising from 0 to ~2000
- Retention curve beginning to descend

If the WebSocket fails to connect (you'll see `WebSocket handshake timed out` in browser DevTools → Console), RunPod's Community Cloud proxy has intermittent WS support. Fallback: redeploy on **Secure Cloud** (~$0.70/hr) which reliably supports WebSocket upgrades.

### T − 10 min: Share the URL

Paste into the Zoom chat / slide deck / conference portal / cover letter. The URL stays valid as long as the pod is Running.

### T + demo_duration: Shut down

RunPod console → **Pods → Stop**. Billing stops **immediately** at pod stop, not at pod terminate. You can restart the same pod later (faster, same URL) or delete it (slower first run next time).

**Don't forget this step.** A weekend of forgotten uptime = ~$20. Set a phone timer.

---

## Budget table

| Scenario | Hours | Cost |
|---|---:|---:|
| Rehearsal (3 × 1 h sessions) | 3 | $1.02 |
| Conference demo (1 day, 8 h) | 8 | $2.72 |
| Horizon Europe interview (2 × 2 h + 30 min buffer) | 5 | $1.70 |
| Reviewer live-check during review window | 4 | $1.36 |
| **Typical month** | **20** | **~$7** |

Rounding up: budget **$10-15/month** and you're covered for most paper / interview seasons.

---

## Troubleshooting

**Pod stuck at "Allocating" for > 5 min.** No GPU stock in that region on Community Cloud. Stop, redeploy on another region, or switch to Secure Cloud.

**Logs show `RuntimeError: CUDA driver mismatch`.** The pod picked up a host driver older than the image's CUDA 12.6 runtime expects. Redeploy on a newer-region pod; RunPod rolls out driver updates region-by-region.

**Browser shows the shell but the pot is grey / no bubbles.** Solver hasn't produced its first snapshot yet. Wait 30 s; if still blank, check Logs for the solver process — could be stuck in Warp JIT compile (if the precompile layer got skipped at build time).

**`Connect → HTTP Service [Port 80]` button is missing.** Template didn't expose port 80. Edit the template, add the port, stop + redeploy.

**WebSocket connects but drops every 60 s.** RunPod's proxy has a default idle timeout on Community Cloud. Upgrade to Secure Cloud or reduce snapshot cadence so traffic is never idle (the 30 Hz default is well under the timeout, so this is rare).

**I want the same URL to persist across pod restarts.** Impossible on RunPod — `<pod-id>` changes each deploy. Either:
(a) use the always-on Cloudflare Tunnel as the stable reviewer URL (see [DEPLOYMENT.md](../DEPLOYMENT.md)), or
(b) put both URLs in the paper: "live demo (provisioned on request): <runpod-demo-url>; always-on reproduction: <cloudflare-url>".

---

## Updating the image

On source changes (bug fixes, new features, new scenarios):

```bash
# Bump version
git tag v0.2
git push --tags

# Rebuild locally on your GPU host (Warp precompile layer runs)
docker build -f Dockerfile.runpod \
    -t ghcr.io/amankrah/boiling-sim:v0.2 \
    -t ghcr.io/amankrah/boiling-sim:latest .
docker push ghcr.io/amankrah/boiling-sim:v0.2
docker push ghcr.io/amankrah/boiling-sim:latest
```

Or let CI do it via [.github/workflows/build-runpod.yml](../.github/workflows/build-runpod.yml) — CI builds without the Warp precompile (no GPU on GHCR runners), so cold-start adds ~45 s but paper-tagged releases are automated.

Then update the RunPod template's Container Image field to `ghcr.io/amankrah/boiling-sim:v0.2` (or keep it at `:latest` if you're OK with Template auto-picking up new pushes).

---

## Security notes

The RunPod proxy URL is **HTTPS but publicly unauthenticated**. Anyone with the URL can open the dashboard, click **Apply & Start Run** in the Config page, and submit arbitrary `ScenarioConfig` JSON. The Phase 6.7 scientific-safety audit restricted the UI to Tier-1 experiment knobs and validated them through Pydantic — there is no RCE surface — but a determined user can still saturate the GPU with back-to-back runs and cost you money.

**Mitigations**:
- Keep demo windows short. Don't leave the pod up overnight.
- For paper-reviewer access, ship the URL in the cover letter (only the review team sees it) rather than in a public preprint abstract.
- For permanent public access, use the Cloudflare Tunnel fallback with Cloudflare Access for email-based gating.
