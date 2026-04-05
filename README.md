# Gemma Browser Deep Agent Demo

This repository is a standalone browser demo for a real deep agent running on
top of MediaPipe Tasks GenAI, Gemma 4, `deepagentsjs`, and a browser-hosted
`wasmsh` sandbox.

It does four things:

- initializes `ChatMediaPipeGenAI` in the browser
- downloads the model once and passes the same bytes into MediaPipe init
- starts a browser sandbox rooted at `/workspace`
- runs a `deepagentsjs` agent that can inspect files, edit files, and execute
  shell or Python inside that sandbox

## Requirements

- Node 20+
- `pnpm`
- a WebGPU-capable browser
- the Gemma 4 web model file `gemma-4-E2B-it-web.task`

## Setup

1. Install dependencies:

```bash
pnpm install
```

2. Copy the MediaPipe WASM assets into `public/vendor/mediapipe/tasks-genai/wasm`:

```bash
pnpm run sync:assets
```

3. Start the demo:

```bash
pnpm start
```

4. Open the local Vite URL shown by `pnpm start`.

5. Optional: refresh the vendored browser agent sources after updating the
   source branches:

```bash
pnpm run sync:vendor
```

## Runtime profile

The demo uses:

- `wasmRoot: /vendor/mediapipe/tasks-genai/wasm`
- `modelAssetPath: /models/gemma/gemma-4-E2B-it-web.task`
- `maxTokens: 12288`
- `topK: 40`
- `temperature: 0.2`
- `randomSeed: 101`

## Notes

- The app runs fully in the browser. The dev server only serves files.
- LangSmith is intentionally shimmed out for the demo.
- The repository includes `public/models/gemma/gemma-4-E2B-it-web.task`.
- `pnpm run sync:vendor` refreshes vendored browser sources from:
  - `/Users/johann/src/ml/deepagentsjs` on `feat/wasmsh-sandbox`
- `src/vendor/wasmsh/index.js` is now just a thin demo adapter over the
  published browser-safe `@mayflowergmbh/wasmsh-pyodide` entry.
- The MediaPipe adapter sources are vendored from:
  - `/Users/johann/src/ml/langchainjs-community` on `feature/mediapipe-genai-clean`
- `Dockerfile` builds a static image, and `deploy/helm/deepagents` contains
  the chart used for Kubernetes deployment.
