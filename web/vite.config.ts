import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 6 dashboard front-end. The production build is served by nginx
// from Dockerfile.web which also proxies `/stream` to the Rust
// ws-server. In dev we use the plugin's native WebSocket passthrough
// by letting the browser hit ws://localhost:8080/stream directly.
export default defineConfig({
  server: {
    port: 3000,
    strictPort: true,
    host: "0.0.0.0", // so WSL2 + Windows browsers can reach via localhost
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  plugins: [react()],
});
