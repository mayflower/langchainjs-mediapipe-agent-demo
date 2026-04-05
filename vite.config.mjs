import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { defineConfig } from "vite";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const langchainDir = path.resolve(
  path.dirname(require.resolve("langchain")),
  ".."
);

export default defineConfig({
  resolve: {
    alias: [
      {
        find: /^deepagents$/,
        replacement: path.join(
          root,
          "src",
          "vendor",
          "deepagents",
          "index.browser.js"
        ),
      },
      {
        find: /^@langchain\/wasmsh$/,
        replacement: path.join(
          root,
          "src",
          "vendor",
          "wasmsh",
          "index.js"
        ),
      },
      {
        find: /^langchain$/,
        replacement: path.join(langchainDir, "dist", "index.js"),
      },
      {
        find: /^langchain\/(.+)/,
        replacement: path.join(langchainDir, "dist", "$1"),
      },
      {
        find: /^node:async_hooks$/,
        replacement: path.join(root, "src", "shims", "async-hooks.js"),
      },
      {
        find: /^(node:)?path$/,
        replacement: "path-browserify",
      },
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
    "process.platform": JSON.stringify("browser"),
    "process.env.LANGSMITH_TRACING": JSON.stringify("false"),
    "process.env.LANGCHAIN_TRACING_V2": JSON.stringify("false"),
    "process.env": JSON.stringify({}),
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
