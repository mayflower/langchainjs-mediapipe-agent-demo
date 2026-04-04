import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)));

export default defineConfig({
  resolve: {
    alias: [
      {
        find: "langsmith/singletons/traceable",
        replacement: path.join(
          root,
          "src",
          "shims",
          "langsmith",
          "singletons",
          "traceable.js"
        ),
      },
      {
        find: "langsmith/run_trees",
        replacement: path.join(
          root,
          "src",
          "shims",
          "langsmith",
          "run_trees.js"
        ),
      },
      {
        find: "langsmith",
        replacement: path.join(
          root,
          "src",
          "shims",
          "langsmith",
          "index.js"
        ),
      },
    ],
  },
  define: {
    "process.env.LANGSMITH_TRACING": JSON.stringify("false"),
    "process.env.LANGCHAIN_TRACING_V2": JSON.stringify("false"),
    global: "globalThis",
  },
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
});
