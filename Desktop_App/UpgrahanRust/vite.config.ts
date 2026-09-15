import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import tailwindcss from "@tailwindcss/vite";
import Icons from "unplugin-icons/vite";

// Tauri serves the built assets from disk, so no SSR and a fixed dev port it can attach to.
export default defineConfig({
  plugins: [solid(), tailwindcss(), Icons({ compiler: "solid" })],
  resolve: {
    // tsconfig `paths` only teaches tsc about `~/`; the bundler needs telling separately.
    alias: { "~": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  clearScreen: false,
  server: {
    port: 5183,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**"] },
  },
  build: {
    target: "chrome110",
    sourcemap: process.env.TAURI_ENV_DEBUG === "true",
  },
});
