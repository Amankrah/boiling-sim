import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 6 dashboard front-end. Production: nginx proxies `/stream` and `/api`
// to ws-server. Dev: Vite proxies `/api` to 127.0.0.1:8080; App.tsx uses
// ws://<host>:8080/stream (see App.tsx).
export default defineConfig({
  server: {
    port: 3000,
    strictPort: true,
    host: "0.0.0.0", // so WSL2 + Windows browsers can reach via localhost
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  plugins: [react()],
});
