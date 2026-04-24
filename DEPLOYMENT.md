# Deployment guide

How to run BoilingSim somewhere people can reach it. Three independent paths, any combination of which can be active at the same time. Pick based on audience + uptime need + budget.

| Path | Audience | Cost | Uptime | Effort |
|---|---|---:|---|---|
| [A. Self-hosted `docker compose`](#a-self-hosted-docker-compose) | Researchers reproducing your work | $0 | Per-user | Already built |
| [B. RunPod on-demand GPU pod](#b-runpod-on-demand-gpu-pod) | Conference demos, reviewer interviews | ~$0.40/hr | On request | Template-driven |
| [C. Cloudflare Tunnel to your workstation](#c-cloudflare-tunnel-to-your-workstation) | Always-on paper URL, reviewer fallback | $0 (Cloudflare free) | Whenever workstation is on | 1 hour setup |

The three paths **are not alternatives** — the recommended production posture for a research paper is **B + C together**: Cloudflare Tunnel gives a stable URL for paper cover-letters, RunPod gives a scheduled full-performance demo for conferences.

---

## A. Self-hosted `docker compose`

The canonical reproduction path. Anyone with an NVIDIA GPU (≥ sm_89, ≥ 24 GB VRAM) and Docker can stand up the whole stack from source.

### Prerequisites

- **GPU**: RTX 4090 / RTX 6000 Ada / A6000 / H100 / similar. Consumer or data-center Ada/Hopper/Blackwell parts all work.
- **Driver**: NVIDIA driver ≥ 555 (CUDA 12.6 runtime compatibility).
- **NVIDIA Container Toolkit**: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>.
- **Docker + Docker Compose v2**.

### Launch

```bash
git clone https://github.com/Amankrah/boiling-sim.git
cd boiling-sim
docker compose up --build
# ... wait 15-25 min for first build on a fresh machine
# ... subsequent builds with layer caching: 2-5 min
```

Open <http://localhost:3000>. The [scripts/dashboard_precheck.sh](scripts/dashboard_precheck.sh) entrypoint fails loudly if GPU passthrough is broken (rather than silently falling back to CPU), so any startup error is self-explanatory.

### Reproducing a paper figure

[benchmarks/runs.md](benchmarks/runs.md) has the exact CLI invocation, scenario YAML, and expected headline number for every HDF5/PNG artefact under [benchmarks/](benchmarks/). Example:

```bash
# Regenerate the Phase 3 steel Rohsenow validation
docker compose exec solver python scripts/run_boiling.py \
    --config configs/scenarios/default.yaml \
    --duration 180 --dx-mm 2.0 --pressure-iters 100
# → benchmarks/phase3_boiling_steel_304.{h5,png}
# → Rohsenow ratio 1.01× (headline number from Phase 3)
```

---

## B. RunPod on-demand GPU pod

**Full runbook**: [runpod/README.md](runpod/README.md). Summary below.

### Philosophy

Don't pay for idle. Rent a GPU when you need to demo; stop the pod when you're done. A typical research-paper submission cycle needs ~20 GPU-hours total across 6 months = ~$7-10.

### One-time setup

1. Build the unified image `Dockerfile.runpod` and push to `ghcr.io/amankrah/boiling-sim:v0.1`.
2. Create a RunPod Template pointing at that image, expose port 80, 20 GB container disk.

### Per-demo workflow (30 min before the demo)

1. RunPod console → **Deploy pod** (RTX 4090, Community Cloud, `boilingsim-v0.1` template).
2. Wait ~2-3 min for pod to reach **Running** + image pull + stack warm-up.
3. Click **Connect → HTTP Service [Port 80]** → get a URL like `https://<pod-id>-80.proxy.runpod.net/`.
4. Smoke test: `curl -fsS $URL/health` → 200 OK.
5. Share the URL.

After the demo: **Stop pod**. Billing stops instantly.

### Cost model

| Usage | Cost |
|---|---:|
| RTX 4090 Community Cloud | $0.34/hr spot, $0.44/hr on-demand |
| RTX 4090 Secure Cloud | $0.70/hr (only if you need WebSocket reliability) |
| Container storage | included |
| Optional Network Volume for persistent artefacts | $0.07/GB/month |

Typical paper-cycle monthly cost: **$5-15**.

### Image update loop

Either build locally on your GPU workstation (keeps the Warp JIT cache baked in, saves ~45 s on every pod cold-start), or let CI do it via [.github/workflows/build-runpod.yml](.github/workflows/build-runpod.yml) on tag push.

```bash
# Local, with Warp precompile
docker build -f Dockerfile.runpod -t ghcr.io/amankrah/boiling-sim:v0.2 .
docker push ghcr.io/amankrah/boiling-sim:v0.2

# Or via CI
git tag v0.2 && git push --tags
# CI builds without Warp precompile; cold-start adds ~45s of JIT compile
```

---

## C. Cloudflare Tunnel to your workstation

The always-on reviewer-accessible URL. Uses your existing GPU workstation + Cloudflare's free tunnel service so nothing on your firewall needs to change.

### Prerequisites (Cloudflare path)

- Your RTX 4090 workstation is running `docker compose up` locally (Path A above) and serving on `http://localhost:3000`.
- Domain managed by Cloudflare (free tier works). If you don't have one, register at any registrar and point the NS records at Cloudflare — free tier is sufficient.

### Install + run the tunnel

```bash
# On your workstation (Linux)
curl -L --output cloudflared.deb \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# Login (opens browser for Cloudflare OAuth; pick the domain to bind).
cloudflared tunnel login

# Create a named tunnel.
cloudflared tunnel create boilingsim

# Point the tunnel at your local dashboard, persist config:
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml <<EOF
tunnel: boilingsim
credentials-file: /home/$USER/.cloudflared/<tunnel-uuid>.json
ingress:
  - hostname: boilingsim.example.com
    service: http://localhost:3000
  - service: http_status:404
EOF

# Register the DNS record (Cloudflare sets up a CNAME automatically).
cloudflared tunnel route dns boilingsim boilingsim.example.com

# Run (daemonize via systemd in production — see `cloudflared service install`).
cloudflared tunnel run boilingsim
```

Browser: <https://boilingsim.example.com>.

### Systemd unit for 24/7 uptime

```bash
sudo cloudflared --config ~/.cloudflared/config.yml service install
sudo systemctl enable --now cloudflared
```

Tunnel restarts on boot. Workstation must be on (plug it into a UPS if it's under your desk).

### Optional: Cloudflare Access (email-gated reviewer URLs)

For double-blind venues or private-reviewer access, add Cloudflare Access in front of the tunnel:

1. Cloudflare dashboard → **Zero Trust → Access → Applications → Add**.
2. Application domain: `boilingsim.example.com`.
3. Policy: allow emails matching `*@ijhmt.elsevier.com` or a specific reviewer list.

Visitors get a one-time email code instead of the dashboard. Free tier supports up to 50 users.

---

## Recommended posture for a research-paper submission

**Publish URLs to reviewers / in cover letters:**

- **Primary (paper URL, always-on)**: `https://boilingsim.example.com` — Cloudflare Tunnel to your workstation (Path C). No cost, no timed gates.
- **Demo (interviews, conference)**: `https://<pod-id>-80.proxy.runpod.net/` — spun up on demand (Path B). Typical cost: $5-15/month.
- **Reproducibility (supplementary)**: <https://github.com/Amankrah/boiling-sim> — commit hash in the paper, `docker compose up` reproducible from source (Path A).

**In the paper's Methods / Data Availability statement:**

> The simulation code and full reproduction pipeline are available at
> <https://github.com/Amankrah/boiling-sim> (commit \<hash\>) and run via
> `docker compose up --build`. A live instance is served at
> <https://boilingsim.example.com> during the review window. A scheduled
> high-performance demo is available on request via the corresponding
> author; reservation triggers provisioning of a RunPod GPU instance.

This posture scores high on reproducibility reviewers' checklists (source + docker + live instance), costs ~$10/month during the review cycle, and gives reviewers a working URL from day 1 without you needing to keep a laptop running during a conference talk.

---

## Troubleshooting paths

| Symptom | Likely cause | Fix |
|---|---|---|
| Local `docker compose up` fails with `no GPU driver` | NVIDIA Container Toolkit not installed | Install per the URL above; restart Docker daemon |
| RunPod pod "Allocating" for > 5 min | No GPU stock in that region | Stop, pick a different region |
| Browser WebSocket timeout on RunPod | Community Cloud proxy WS flake | Switch to Secure Cloud (~$0.70/hr) |
| `cloudflared` disconnects every few minutes | Workstation suspending / network drop | Systemd unit + `cloudflared service install` |
| Paper reviewer says "URL didn't work" | Workstation rebooted mid-review | Provision a RunPod pod as the primary URL for the review window |
| GPU OOM on pod boot | Another sim running + memory not freed | Restart pod; GPU state is per-pod, never shared |

More project-specific troubleshooting in [GETTING_STARTED.md](GETTING_STARTED.md).

---

## Security posture

- The RunPod proxy URL is **HTTPS but public**. The Phase 6.7 scientific-safety audit restricted the UI to Tier-1 experiment knobs; there is no RCE surface. But a user with the URL can start back-to-back runs and cost you GPU time. Keep demo windows short, don't leave pods running overnight.
- The Cloudflare Tunnel URL is also **HTTPS but public by default**. Add Cloudflare Access (free, email-gated) if you need reviewer-only.
- The self-hosted Path A binds to `localhost:3000` only — not publicly accessible unless you publish the Path C tunnel.
- The Results page exports HDF5/CSV/JSON of *whatever* the user ran. If you use Path C on your work machine, be aware that reviewers can download the artefact files. This is fine for research demos.

---

## Related docs

- [runpod/README.md](runpod/README.md) — RunPod-specific runbook with full console clickthrough
- [benchmarks/runs.md](benchmarks/runs.md) — every CLI invocation that produces a benchmark artefact, for reproducibility
- [GETTING_STARTED.md](GETTING_STARTED.md) — local-dev setup (WSL2, native Linux, pre-commit hooks)
- [benchmarks/phase6_dashboard.md](benchmarks/phase6_dashboard.md) — the Phase 6 dashboard architecture this deploys
