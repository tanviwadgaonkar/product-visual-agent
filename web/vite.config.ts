import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The built app is served by FastAPI under /app; in development Vite proxies
// API and image requests to the backend on :8000.
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/images": "http://127.0.0.1:8000",
    },
  },
  test: { environment: "node" },
});
