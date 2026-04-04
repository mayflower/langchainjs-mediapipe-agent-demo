# MediaPipe LangChain Agent Demo

This repository is a small standalone browser demo for a LangChain tool-calling
agent running on top of MediaPipe Tasks GenAI and Gemma 4.

It does three things:

- initializes `ChatMediaPipeGenAI` in the browser
- runs a real LangChain agent via `@langchain/classic/agents`
- exposes live browser-side tools for weather, time, and arithmetic

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

4. Open [http://127.0.0.1:4173](http://127.0.0.1:4173).

## Runtime profile

The demo uses:

- `wasmRoot: /vendor/mediapipe/tasks-genai/wasm`
- `modelAssetPath: /models/gemma/gemma-4-E2B-it-web.task`
- `maxTokens: 2048`
- `topK: 40`
- `temperature: 0.2`
- `randomSeed: 101`

## Notes

- The app runs fully in the browser. The dev server only serves files.
- LangSmith is intentionally shimmed out for the demo.
- The weather tool uses live Open-Meteo browser fetches.
- The repository includes `public/models/gemma/gemma-4-E2B-it-web.task`.
- `Dockerfile` builds a static image, and `deploy/helm/deepagents` contains the chart used for Kubernetes deployment.
