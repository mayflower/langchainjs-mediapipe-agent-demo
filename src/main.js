import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { createDeepAgent } from "deepagents";
import { WasmshSandbox } from "@langchain/wasmsh";
import { ChatMediaPipeGenAI } from "./vendor/langchain-community/chat_models/mediapipe_genai.js";

const initButton = document.getElementById("init-button");
const resetButton = document.getElementById("reset-button");
const sendButton = document.getElementById("send-button");
const promptInput = document.getElementById("prompt-input");
const statusNode = document.getElementById("status");
const statusLineNode = statusNode.parentElement;
const activityDetailNode = document.getElementById("activity-detail");
const workspaceMainNode = document.getElementById("workspace-main");
const transcript = document.getElementById("transcript");
const todosNode = document.getElementById("todos-panel");
const workspaceNode = document.getElementById("workspace-panel");
const sandboxNode = document.getElementById("sandbox-panel");
const debugNode = document.getElementById("debug-panel");
const downloadProgressNode = document.getElementById("download-progress");
const downloadStatusNode = document.getElementById("download-status");

const metricNodes = {
  outputTokens: document.getElementById("metric-output-tokens"),
  tokenRate: document.getElementById("metric-token-rate"),
  llmCalls: document.getElementById("metric-llm-calls"),
};

const WASM_ROOT = "/vendor/mediapipe/tasks-genai/wasm";
const SANDBOX_WORKER_URL = "/vendor/wasmsh/browser-worker.js";
const SANDBOX_ASSET_BASE_URL = "/vendor/wasmsh/assets";
const SANDBOX_LIB_HEALTHCHECK = "/vendor/wasmsh/lib/protocol.mjs";
const INIT_LONG_LOAD_MS = 20_000;
const INIT_TIMEOUT_MS = 180_000;
const MODEL_MAX_TOKENS = 12288;
const MODEL_ASSET_PATH_CANDIDATES = [
  "/models/gemma/gemma-4-E2B-it-web.task",
  "/models/gemma-4-E2B-it-web.task",
];
const WASM_ENTRYPOINT = `${WASM_ROOT}/genai_wasm_internal.js`;
const SEEDED_WORKSPACE_FILES = {
  "/workspace/README.md": [
    "# Gemma Browser Lab Workspace",
    "",
    "This sandbox is intentionally small and local-first.",
    "",
    "Suggested tasks:",
    "- inspect the files in /workspace",
    "- summarize the CSV in /workspace/data/expenses.csv",
    "- edit /workspace/src/report.py to change the output wording",
    "- create a new note under /workspace/notes/",
    "",
    "When you change files, mention the path in your final answer.",
  ].join("\n"),
  "/workspace/data/expenses.csv": [
    "month,category,amount_eur",
    "January,hosting,420",
    "January,travel,180",
    "February,hosting,420",
    "February,software,210",
    "March,hosting,420",
    "March,events,640",
  ].join("\n"),
  "/workspace/src/report.py": [
    "from pathlib import Path",
    "",
    "CSV_PATH = Path('/workspace/data/expenses.csv')",
    "",
    "def main():",
    "    lines = CSV_PATH.read_text().strip().splitlines()[1:]",
    "    total = sum(int(line.split(',')[2]) for line in lines)",
    "    print(f'Total tracked spend: {total} EUR')",
    "",
    "if __name__ == '__main__':",
    "    main()",
  ].join("\n"),
  "/workspace/notes/context.md": [
    "# Context",
    "",
    "- This is a browser-only demo.",
    "- Prefer concise answers after tool use.",
    "- Use shell or Python when calculations matter.",
  ].join("\n"),
};

let deepAgent = null;
let model = null;
let sandbox = null;
let isInitializing = false;
let isRunning = false;
let initAttemptId = 0;
let initLongLoadTimer = null;
let initTimeoutTimer = null;
let initDownloadController = null;
let isModelDownloadComplete = false;
let activeStreamCards = new Map();
let todoState = [];
let sandboxState = {
  status: "idle",
  detail: "Sandbox not initialized.",
};

const runMetrics = {
  outputText: "",
  outputTokens: 0,
  tokenRate: 0,
  llmCalls: 0,
  startedAt: 0,
  firstTokenAt: 0,
  lastTokenAt: 0,
};

function debugTimestamp() {
  return new Date().toLocaleTimeString("de-DE", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function resetDebugLog() {
  debugNode.textContent = "";
}

function appendDebugLine(text) {
  const line = `[${debugTimestamp()}] ${text}`;
  debugNode.textContent = debugNode.textContent
    ? `${debugNode.textContent}\n${line}`
    : line;
  debugNode.scrollTop = debugNode.scrollHeight;
  console.log(line);
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return "unknown size";
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }

  return `${size.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`;
}

function formatPercent(received, total) {
  if (!Number.isFinite(total) || total <= 0) {
    return "working";
  }

  return `${Math.min(100, (received / total) * 100).toFixed(1)}%`;
}

function formatTokenRate(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return "0.0";
  }

  return value.toFixed(value >= 100 ? 0 : 1);
}

function extractErrorMessage(error, fallback) {
  return error instanceof Error ? error.message : fallback;
}

function setStatus(text, mode = "idle") {
  statusNode.textContent = text;
  statusNode.dataset.mode = mode;
  statusLineNode.dataset.mode = mode;
  workspaceMainNode.setAttribute(
    "aria-busy",
    mode === "working" ? "true" : "false"
  );
  if (mode !== "working") {
    setActivityDetail("");
  }
}

let lastActivityDetail = "";
function setActivityDetail(text) {
  const next = text || "";
  if (next === lastActivityDetail) {
    return;
  }
  lastActivityDetail = next;
  activityDetailNode.textContent = next;
}

function setRunMetric(key, value) {
  metricNodes[key].textContent = value;
}

function resetRunMetrics() {
  runMetrics.outputText = "";
  runMetrics.outputTokens = 0;
  runMetrics.tokenRate = 0;
  runMetrics.llmCalls = 0;
  runMetrics.startedAt = 0;
  runMetrics.firstTokenAt = 0;
  runMetrics.lastTokenAt = 0;

  setRunMetric("outputTokens", "0");
  setRunMetric("tokenRate", "0.0");
  setRunMetric("llmCalls", "0");
}

function setDownloadProgress(received, total, statusText) {
  if (Number.isFinite(total) && total > 0) {
    downloadProgressNode.removeAttribute("indeterminate");
    downloadProgressNode.max = total;
    downloadProgressNode.value = Math.min(received, total);
  } else {
    downloadProgressNode.max = 1;
    downloadProgressNode.value = 0;
    downloadProgressNode.setAttribute("indeterminate", "true");
  }

  downloadStatusNode.textContent = statusText;
}

function resetDownloadProgress() {
  isModelDownloadComplete = false;
  setDownloadProgress(0, 1, "Not started.");
}

function clearInitTimers() {
  if (initLongLoadTimer !== null) {
    clearTimeout(initLongLoadTimer);
    initLongLoadTimer = null;
  }

  if (initTimeoutTimer !== null) {
    clearTimeout(initTimeoutTimer);
    initTimeoutTimer = null;
  }
}

function clearLongLoadTimer() {
  if (initLongLoadTimer !== null) {
    clearTimeout(initLongLoadTimer);
    initLongLoadTimer = null;
  }
}

function abortModelDownload() {
  if (initDownloadController) {
    initDownloadController.abort();
    initDownloadController = null;
  }
}

function throwIfAttemptStale(attemptId) {
  if (attemptId !== initAttemptId) {
    const error = new Error("Initialization was reset before completion.");
    error.name = "AbortError";
    throw error;
  }
}

function appendMessage(kind, title, body) {
  const card = document.createElement("article");
  card.className = `message ${kind}`;

  const heading = document.createElement("h2");
  heading.textContent = title;

  const content = document.createElement("pre");
  content.textContent = body;

  card.append(heading, content);
  transcript.appendChild(card);
  transcript.scrollTop = transcript.scrollHeight;
  return content;
}

function getOrCreateStreamCard(key, kind, title) {
  let card = activeStreamCards.get(key);
  if (card) {
    return card;
  }

  const content = appendMessage(kind, title, "");
  card = { content };
  activeStreamCards.set(key, card);
  return card;
}

function resetStreamCards() {
  activeStreamCards = new Map();
}

function getSourceKey(namespace) {
  if (!Array.isArray(namespace)) {
    return "main";
  }

  const subagentNamespace = namespace.find((segment) =>
    segment.startsWith("tools:")
  );
  return subagentNamespace ?? "main";
}

function getSourceLabel(sourceKey) {
  if (sourceKey === "main") {
    return "Agent";
  }

  return `Subagent ${sourceKey.slice(-6)}`;
}

function extractTextParts(parts) {
  if (!Array.isArray(parts)) {
    return "";
  }

  return parts
    .filter((item) => item?.type === "text" && typeof item.text === "string")
    .map((item) => item.text)
    .join("");
}

function getMessageText(message) {
  if (typeof message?.text === "string" && message.text) {
    return message.text;
  }

  if (Array.isArray(message?.contentBlocks)) {
    return extractTextParts(message.contentBlocks);
  }

  if (Array.isArray(message?.content_blocks)) {
    return extractTextParts(message.content_blocks);
  }

  if (Array.isArray(message?.content)) {
    return extractTextParts(message.content);
  }

  if (typeof message?.content === "string") {
    return message.content;
  }

  return "";
}

function isToolMessageLike(message) {
  return (
    ToolMessage.isInstance(message) ||
    message?.type === "tool" ||
    typeof message?.tool_call_id === "string"
  );
}

function getToolCallChunks(message) {
  if (Array.isArray(message?.tool_call_chunks)) {
    return message.tool_call_chunks;
  }

  if (Array.isArray(message?.toolCallChunks)) {
    return message.toolCallChunks;
  }

  return [];
}

function summarizeMessageForDebug(message) {
  if (!message || typeof message !== "object") {
    return String(message);
  }

  let contentSummary = null;
  if (typeof message.content === "string") {
    contentSummary = message.content.slice(0, 200);
  } else if (Array.isArray(message.content)) {
    contentSummary = message.content.slice(0, 3).map((item) => {
      if (typeof item === "string") {
        return item.slice(0, 120);
      }
      if (item && typeof item === "object") {
        return {
          type: item.type ?? null,
          text:
            typeof item.text === "string" ? item.text.slice(0, 120) : null,
          keys: Object.keys(item).slice(0, 8),
        };
      }
      return item;
    });
  }

  const summary = {
    ctor: message.constructor?.name ?? "Object",
    type: message.type ?? null,
    keys: Object.keys(message).slice(0, 12),
    text: getMessageText(message).slice(0, 120),
    content: contentSummary,
    toolCalls: Array.isArray(message.tool_calls) ? message.tool_calls.length : 0,
    toolCallChunks: getToolCallChunks(message).length,
    invalidToolCalls: Array.isArray(message.invalid_tool_calls)
      ? message.invalid_tool_calls.length
      : 0,
    name: message.name ?? null,
    additionalKwargs:
      message.additional_kwargs && typeof message.additional_kwargs === "object"
        ? Object.fromEntries(
            Object.entries(message.additional_kwargs).slice(0, 8)
          )
        : null,
    responseMetadata:
      message.response_metadata &&
      typeof message.response_metadata === "object"
        ? Object.fromEntries(
            Object.entries(message.response_metadata).slice(0, 8)
          )
        : null,
  };

  return JSON.stringify(summary);
}

function normalizeTodoItems(value) {
  if (!Array.isArray(value)) {
    return null;
  }

  const todos = value
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const text =
        item.content ?? item.task ?? item.title ?? item.description ?? null;
      if (typeof text !== "string" || !text.trim()) {
        return null;
      }

      return {
        content: text.trim(),
        status:
          typeof item.status === "string" && item.status.trim()
            ? item.status.trim()
            : "pending",
      };
    })
    .filter(Boolean);

  return todos.length ? todos : [];
}

function renderTodos(items = todoState) {
  todosNode.innerHTML = "";

  if (!items.length) {
    todosNode.textContent =
      "No explicit todo plan yet. The deep agent will surface one when it decides to use write_todos.";
    return;
  }

  for (const item of items) {
    const row = document.createElement("div");
    row.className = "todo-item";
    row.dataset.status = item.status.toLowerCase();

    const status = document.createElement("span");
    status.className = "todo-status";
    status.textContent = item.status;

    const text = document.createElement("span");
    text.className = "todo-text";
    text.textContent = item.content;

    row.append(status, text);
    todosNode.appendChild(row);
  }
}

function updateTodos(items) {
  const normalized = normalizeTodoItems(items);
  if (!normalized) {
    return;
  }

  todoState = normalized;
  renderTodos();
}

function setSandboxState(status, detail) {
  sandboxState = { status, detail };
  renderSandboxPanel();
}

function renderSandboxPanel() {
  const lines = [
    `State: ${sandboxState.status}`,
    `Detail: ${sandboxState.detail}`,
    `Worker URL: ${SANDBOX_WORKER_URL}`,
    `Asset Base: ${SANDBOX_ASSET_BASE_URL}`,
    `Workspace: /workspace`,
    `Sandbox ID: ${sandbox?.id ?? "not initialized"}`,
  ];

  sandboxNode.textContent = lines.join("\n");
}

async function refreshWorkspacePanel() {
  if (!sandbox) {
    workspaceNode.textContent =
      "Workspace unavailable until the browser sandbox is initialized.";
    return;
  }

  try {
    const result = await sandbox.execute(
      "find /workspace -maxdepth 3 \\( -type d -o -type f \\) | sort"
    );

    if (result.exitCode !== 0) {
      workspaceNode.textContent = result.output || "Workspace listing failed.";
      return;
    }

    workspaceNode.textContent =
      result.output.trim() || "/workspace is currently empty.";
  } catch (error) {
    workspaceNode.textContent = `Workspace refresh failed: ${extractErrorMessage(
      error,
      "Unknown error."
    )}`;
  }
}

function syncUi() {
  initButton.disabled = isInitializing || model !== null;
  resetButton.disabled =
    isRunning ||
    (!isInitializing &&
      model === null &&
      deepAgent === null &&
      statusNode.dataset.mode !== "error");

  sendButton.disabled = isInitializing || isRunning || deepAgent === null;
  promptInput.disabled = isInitializing || isRunning || deepAgent === null;
}

async function resolveStaticAsset(path, label) {
  const response = await fetch(path, { method: "HEAD" });
  if (!response.ok) {
    throw new Error(
      `${label} check failed for ${path} with ${response.status} ${response.statusText}.`
    );
  }

  const contentType =
    response.headers.get("content-type") ?? "unknown content type";
  if (contentType.toLowerCase().includes("text/html")) {
    throw new Error(
      `${label} check failed for ${path}: the server returned HTML instead of the expected asset.`
    );
  }

  return {
    path,
    contentType,
  };
}

function validateModelAsset(modelAsset) {
  const warnings = [];

  if (modelAsset.contentType.toLowerCase().includes("text/html")) {
    warnings.push("The server is returning HTML instead of a binary model file.");
  }

  if (
    Number.isFinite(modelAsset.contentLength) &&
    modelAsset.contentLength > 0 &&
    modelAsset.contentLength < 1_500_000_000
  ) {
    warnings.push(
      "The file is much smaller than the expected ~2 GB Gemma 4 E2B web model."
    );
  }

  if (warnings.length === 0) {
    return;
  }

  throw new Error(
    [
      "The resolved model file does not look like a valid Gemma 4 web asset.",
      `URL: ${modelAsset.path}`,
      `Content-Type: ${modelAsset.contentType}`,
      `Content-Length: ${formatBytes(modelAsset.contentLength)}`,
      ...warnings.map((warning) => `- ${warning}`),
    ].join("\n")
  );
}

async function resolveModelAssetPath() {
  for (const candidate of MODEL_ASSET_PATH_CANDIDATES) {
    try {
      const response = await fetch(candidate, { method: "HEAD" });
      if (response.ok) {
        const contentLength = Number(response.headers.get("content-length"));
        return {
          path: candidate,
          contentType:
            response.headers.get("content-type") ?? "unknown content type",
          contentLength,
        };
      }
    } catch {}
  }

  throw new Error(
    [
      "No Gemma web model file was found for the demo.",
      "Place gemma-4-E2B-it-web.task at one of these paths:",
      ...MODEL_ASSET_PATH_CANDIDATES.map((candidate) => `- ${candidate}`),
    ].join("\n")
  );
}

async function consumeModelDownloadProgress(stream, total, attemptId) {
  const reader = stream.getReader();
  let received = 0;

  try {
    while (true) {
      throwIfAttemptStale(attemptId);
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      if (value) {
        received += value.byteLength;
      }

      setDownloadProgress(
        received,
        total,
        `Downloaded ${formatBytes(received)} of ${formatBytes(
          total
        )} (${formatPercent(received, total)}).`
      );
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  }

  setDownloadProgress(
    total || received || 1,
    total || received || 1,
    `Model download completed (${formatBytes(total || received)}).`
  );
  isModelDownloadComplete = true;
}

async function prepareModelAssetBuffer(modelAsset, attemptId) {
  throwIfAttemptStale(attemptId);
  setStatus("Downloading model asset...", "working");
  setDownloadProgress(
    0,
    modelAsset.contentLength,
    `Starting model download from ${modelAsset.path}`
  );

  const controller = new AbortController();
  initDownloadController = controller;
  let response;

  try {
    response = await fetch(modelAsset.path, {
      signal: controller.signal,
      cache: "default",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }

    throw new Error(
      `Model download failed before inference creation: ${extractErrorMessage(
        error,
        "Unknown download failure."
      )}`
    );
  }

  if (!response.ok) {
    throw new Error(
      `Model download failed for ${modelAsset.path} with ${response.status} ${response.statusText}.`
    );
  }

  const total =
    Number(response.headers.get("content-length")) || modelAsset.contentLength;
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    setDownloadProgress(
      bytes.byteLength,
      total || bytes.byteLength,
      `Model download completed (${formatBytes(total || bytes.byteLength)}).`
    );
    initDownloadController = null;
    isModelDownloadComplete = true;
    return {
      modelAssetBuffer: bytes,
      progressPromise: Promise.resolve(),
    };
  }

  const [progressStream, modelStream] = response.body.tee();
  const progressPromise = consumeModelDownloadProgress(
    progressStream,
    total,
    attemptId
  ).finally(() => {
    initDownloadController = null;
  });

  return {
    modelAssetBuffer: modelStream.getReader(),
    progressPromise,
  };
}

function describeWebGPUPlatformHint() {
  if (typeof navigator === "undefined") {
    return "";
  }

  const uaData = navigator.userAgentData;
  const rawPlatform = (navigator.platform || "").toLowerCase();
  const ua = (navigator.userAgent || "").toLowerCase();
  const isWindows =
    uaData?.platform === "Windows" ||
    rawPlatform.startsWith("win") ||
    ua.includes("windows");
  const isLinux =
    uaData?.platform === "Linux" ||
    rawPlatform.startsWith("linux") ||
    (ua.includes("linux") && !ua.includes("android"));
  const isArm =
    uaData?.architecture === "arm" ||
    ua.includes("arm64") ||
    ua.includes("aarch64");

  if (isWindows && isArm) {
    return [
      "Windows on ARM (Snapdragon X Elite / Copilot+ PCs) does not enable WebGPU by default in Chrome or Edge.",
      "Open chrome://flags/#enable-unsafe-webgpu, set it to Enabled, and restart the browser.",
      "Then verify at chrome://gpu that the WebGPU row reports 'Hardware accelerated'.",
    ].join("\n");
  }
  if (isLinux) {
    return [
      "WebGPU on Linux is still gated behind a flag in Chrome stable.",
      "Open chrome://flags/#enable-unsafe-webgpu and set it to Enabled (on older Chrome versions also enable chrome://flags/#enable-vulkan), then restart the browser.",
      "Make sure 'Use graphics acceleration when available' is on under chrome://settings/system, and verify at chrome://gpu that the WebGPU row reports 'Hardware accelerated'.",
    ].join("\n");
  }
  return [
    "Open chrome://gpu and check the WebGPU row.",
    "If it says 'Disabled' or 'Disabled via blocklist or the command line', enable chrome://flags/#enable-unsafe-webgpu and restart the browser.",
  ].join("\n");
}

async function runRuntimeCompatibilityPreflight() {
  if (
    typeof navigator === "undefined" ||
    !("gpu" in navigator) ||
    !navigator.gpu
  ) {
    throw new Error(
      "WebGPU is not available in this browser. Use a recent Chromium-based browser with WebGPU enabled."
    );
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (error) {
    throw new Error(
      `WebGPU adapter request failed: ${extractErrorMessage(
        error,
        "Unknown adapter request failure."
      )}`
    );
  }

  if (!adapter) {
    throw new Error(
      [
        "WebGPU is exposed by the browser, but no GPU adapter could be acquired.",
        describeWebGPUPlatformHint(),
      ]
        .filter(Boolean)
        .join("\n")
    );
  }

  if (!adapter.features.has("shader-f16")) {
    throw new Error(
      [
        "Your GPU adapter does not expose the WebGPU 'shader-f16' feature, which the Gemma LLM Inference task requires for half-precision weights.",
        "This is a known limitation on some GPU/driver combinations (notably Adreno on Windows-ARM and older Mesa drivers on Linux).",
        "Try a different machine, update GPU drivers, or use a Chromium build with newer Dawn/Vulkan support.",
      ].join("\n")
    );
  }
}

function buildInitializationTimeoutMessage() {
  return [
    `Model initialization timed out after ${Math.round(INIT_TIMEOUT_MS / 1000)} seconds.`,
    "Large local Gemma model creation can take significant time and may fail on unsupported GPUs or memory-constrained devices.",
    "Use Reset State to clear the session, then try again or test on a newer Chromium browser with working WebGPU.",
  ].join("\n");
}

function updateInitializationStage(attemptId, stage) {
  if (attemptId !== initAttemptId) {
    return;
  }

  if (stage === "resolving-fileset") {
    setStatus("Resolving MediaPipe fileset...", "working");
    return;
  }

  if (stage === "creating-inference") {
    if (!isModelDownloadComplete) {
      setStatus("Streaming model into MediaPipe...", "working");
      return;
    }

    setStatus("Creating Gemma inference instance...", "working");
    if (initLongLoadTimer === null) {
      initLongLoadTimer = window.setTimeout(() => {
        if (attemptId !== initAttemptId) {
          return;
        }
        setStatus(
          "Still creating Gemma inference instance. Large local models can take significant time and may fail on unsupported GPUs or memory-constrained devices.",
          "working"
        );
      }, INIT_LONG_LOAD_MS);
    }
    return;
  }

  clearLongLoadTimer();
  setStatus("Model ready.", "ready");
}

async function initializeModelWithGuards(attemptId, progressPromise) {
  const initPromise = model.initialize((progress) => {
    updateInitializationStage(attemptId, progress.stage);
  });

  let settled = false;
  const timeoutPromise = new Promise((_, reject) => {
    progressPromise.then(
      () => {
        if (settled) {
          return;
        }
        initTimeoutTimer = window.setTimeout(() => {
          reject(new Error(buildInitializationTimeoutMessage()));
        }, INIT_TIMEOUT_MS);
      },
      reject
    );
  });

  try {
    await Promise.race([initPromise, timeoutPromise]);
  } finally {
    settled = true;
    clearInitTimers();
  }
}

async function buildSandbox() {
  setSandboxState("starting", "Booting browser sandbox worker.");
  const worker = new Worker(SANDBOX_WORKER_URL, {
    name: "wasmsh-browser",
  });

  try {
    const instance = await WasmshSandbox.createBrowserWorker({
      assetBaseUrl: SANDBOX_ASSET_BASE_URL,
      worker,
      workingDirectory: "/workspace",
      initialFiles: SEEDED_WORKSPACE_FILES,
    });

    setSandboxState("ready", "Sandbox worker is running.");
    return instance;
  } catch (error) {
    worker.terminate();
    throw error;
  }
}

async function resetDemoState(options = {}) {
  const { announce = false } = options;
  initAttemptId += 1;
  clearInitTimers();
  abortModelDownload();

  if (sandbox) {
    await sandbox.stop().catch(() => {});
  }

  model = null;
  sandbox = null;
  deepAgent = null;
  isInitializing = false;
  isRunning = false;
  todoState = [];

  setStatus("Idle.", "idle");
  setSandboxState("idle", "Sandbox not initialized.");
  workspaceNode.textContent =
    "Workspace unavailable until the browser sandbox is initialized.";
  renderTodos([]);
  resetDownloadProgress();
  resetRunMetrics();
  resetDebugLog();

  if (announce) {
    appendMessage(
      "system",
      "Reset",
      "Local model and sandbox state were cleared. You can initialize again without reloading the page."
    );
  }

  syncUi();
}

function parseTodosFromText(text) {
  if (!text) {
    return null;
  }

  try {
    const parsed = JSON.parse(text);
    return normalizeTodoItems(parsed?.todos ?? parsed);
  } catch {
    return null;
  }
}

function handleToolCallChunks(namespace, toolCallChunks) {
  const sourceKey = getSourceKey(namespace);

  toolCallChunks.forEach((toolCall, index) => {
    appendDebugLine(
      `tool-call chunk from ${sourceKey}: ${toolCall.name ?? "pending"}`
    );
    const toolKey = `${sourceKey}:tool:${toolCall.id ?? toolCall.name ?? index}`;
    const title = `Tool Call: ${toolCall.name ?? "pending"}`;
    const card = getOrCreateStreamCard(toolKey, "tool", title);

    if (toolCall.name) {
      setActivityDetail(`Calling tool: ${toolCall.name}`);
    }

    if (toolCall.args) {
      card.content.textContent += toolCall.args;
      if (toolCall.name === "write_todos") {
        const todos = parseTodosFromText(card.content.textContent);
        if (todos) {
          updateTodos(todos);
        }
      }
    }
  });
}

function handleUpdateChunk(data) {
  appendDebugLine(
    `update event with nodes: ${Object.keys(data ?? {}).join(", ") || "(none)"}`
  );
  for (const [nodeName, update] of Object.entries(data)) {
    const updateTodosValue = normalizeTodoItems(update?.todos);
    if (updateTodosValue) {
      appendDebugLine(
        `todo state update from ${nodeName}: ${updateTodosValue.length} items`
      );
      updateTodos(updateTodosValue);
    }

    if (nodeName === "model_request") {
      runMetrics.llmCalls += 1;
      setRunMetric("llmCalls", String(runMetrics.llmCalls));
      setActivityDetail(`Model request #${runMetrics.llmCalls}`);
    }

    if (!update || !Array.isArray(update.messages)) {
      continue;
    }

    const lastAiMessage = update.messages.filter(AIMessage.isInstance).at(-1);
    if (!lastAiMessage?.tool_calls?.length) {
      continue;
    }

    for (const toolCall of lastAiMessage.tool_calls) {
      if (toolCall.name === "write_todos") {
        updateTodos(toolCall.args?.todos);
      }
    }
  }
}

function handleValueChunk(namespace, data) {
  const sourceKey = getSourceKey(namespace);
  const todos = normalizeTodoItems(data?.todos);

  appendDebugLine(
    `value event from ${sourceKey}: keys=${Object.keys(data ?? {}).join(", ") || "(none)"}`
  );

  if (todos) {
    appendDebugLine(
      `todo snapshot from ${sourceKey}: ${todos.length} items`
    );
    updateTodos(todos);
  }
}

function handleSingleMessageChunk(namespace, message) {
  const sourceKey = getSourceKey(namespace);
  const sourceLabel = getSourceLabel(sourceKey);
  appendDebugLine(
    `message event from ${sourceKey}: ${summarizeMessageForDebug(message)}`
  );

  const toolCallChunks = getToolCallChunks(message);
  if (
    (AIMessageChunk.isInstance(message) || toolCallChunks.length > 0) &&
    toolCallChunks.length
  ) {
    handleToolCallChunks(namespace, toolCallChunks);
  }

  if (isToolMessageLike(message)) {
    const body = getMessageText(message) || "(empty tool result)";
    appendDebugLine(
      `tool result from ${sourceKey}: ${message.name ?? "tool"} (${body.length} chars)`
    );
    appendMessage("tool", `Tool Result: ${message.name ?? "tool"}`, body);
    setActivityDetail(`Tool result: ${message.name ?? "tool"}`);
    if (message.name === "write_todos") {
      const todos = parseTodosFromText(body);
      if (todos) {
        updateTodos(todos);
      }
    }
    return;
  }

  const text = getMessageText(message);
  if (!text) {
    appendDebugLine(`message from ${sourceKey} had no visible text`);
    return;
  }

  appendDebugLine(
    `assistant text from ${sourceKey}: ${JSON.stringify(text.slice(0, 160))}`
  );

  const card = getOrCreateStreamCard(
    `${sourceKey}:assistant`,
    "assistant",
    sourceLabel
  );
  card.content.textContent += text;

  const now = performance.now();
  if (!runMetrics.firstTokenAt) {
    runMetrics.firstTokenAt = now;
  }
  runMetrics.lastTokenAt = now;
  runMetrics.outputText += text;

  const elapsedSec = Math.max(
    (now - (runMetrics.firstTokenAt || now)) / 1000,
    0.001
  );
  // Cheap proxy: ~4 chars/token. Replaced with the exact tokenizer count at finalize.
  const approxTokens = Math.round(runMetrics.outputText.length / 4);
  const approxRate = approxTokens / elapsedSec;
  setActivityDetail(
    `Streaming response — ~${approxTokens} tok @ ${formatTokenRate(approxRate)} tok/s`
  );
}

function handleMessageChunk(namespace, data) {
  if (!Array.isArray(data)) {
    return;
  }

  for (const message of data) {
    handleSingleMessageChunk(namespace, message);
  }
}

async function finalizeRunMetrics() {
  if (!model) {
    resetRunMetrics();
    return;
  }

  if (runMetrics.outputText) {
    runMetrics.outputTokens = await model.getNumTokens(runMetrics.outputText);
  } else {
    runMetrics.outputTokens = 0;
  }

  const startedAt = runMetrics.firstTokenAt || runMetrics.startedAt;
  const endedAt = runMetrics.lastTokenAt || performance.now();
  const elapsedSeconds =
    startedAt && endedAt > startedAt ? (endedAt - startedAt) / 1000 : 0;

  runMetrics.tokenRate =
    elapsedSeconds > 0 ? runMetrics.outputTokens / elapsedSeconds : 0;

  setRunMetric("outputTokens", String(runMetrics.outputTokens));
  setRunMetric("tokenRate", formatTokenRate(runMetrics.tokenRate));
  setRunMetric("llmCalls", String(runMetrics.llmCalls));
}

async function runDeepAgent(input) {
  resetStreamCards();
  runMetrics.startedAt = performance.now();
  setSandboxState("running", "Sandbox-backed deep agent is executing.");
  appendDebugLine(`run started: ${JSON.stringify(input)}`);
  setActivityDetail("Starting agent loop...");

  const stream = await deepAgent.stream(
    {
      messages: [new HumanMessage(input)],
    },
    {
      streamMode: ["updates", "messages", "values"],
      subgraphs: true,
    }
  );

  for await (const [namespace, mode, data] of stream) {
    appendDebugLine(
      `stream event mode=${mode} namespace=${Array.isArray(namespace) ? namespace.join(" > ") : String(namespace)}`
    );
    if (mode === "updates") {
      handleUpdateChunk(data);
      continue;
    }

    if (mode === "messages") {
      handleMessageChunk(namespace, data);
      continue;
    }

    if (mode === "values") {
      handleValueChunk(namespace, data);
    }
  }

  appendDebugLine("stream completed");
  if (!runMetrics.outputText.trim()) {
    appendDebugLine("run completed without visible assistant text");
  }

  await finalizeRunMetrics();
  await refreshWorkspacePanel();
  setSandboxState("ready", "Sandbox worker is running.");
}

async function buildDeepAgent(attemptId) {
  const modelAsset = await resolveModelAssetPath();
  const mediapipeRuntime = await resolveStaticAsset(
    WASM_ENTRYPOINT,
    "MediaPipe runtime asset"
  );
  const sandboxWorker = await resolveStaticAsset(
    SANDBOX_WORKER_URL,
    "wasmsh worker asset"
  );
  const sandboxLibrary = await resolveStaticAsset(
    SANDBOX_LIB_HEALTHCHECK,
    "wasmsh library asset"
  );

  validateModelAsset(modelAsset);
  await runRuntimeCompatibilityPreflight();
  throwIfAttemptStale(attemptId);

  const { modelAssetBuffer, progressPromise } = await prepareModelAssetBuffer(
    modelAsset,
    attemptId
  );
  throwIfAttemptStale(attemptId);

  model = new ChatMediaPipeGenAI({
    wasmRoot: WASM_ROOT,
    modelAssetPath: modelAsset.path,
    modelAssetBuffer,
    maxTokens: MODEL_MAX_TOKENS,
    temperature: 0.2,
    topK: 40,
    randomSeed: 101,
  });

  setStatus("Initializing MediaPipe fileset...", "working");
  await Promise.all([
    initializeModelWithGuards(attemptId, progressPromise),
    progressPromise,
  ]);
  throwIfAttemptStale(attemptId);

  sandbox = await buildSandbox();
  throwIfAttemptStale(attemptId);

  deepAgent = createDeepAgent({
    model,
    backend: sandbox,
    systemPrompt: [
      "You are a browser-local coding and analysis assistant running fully on-device.",
      "Work inside /workspace and prefer using the available filesystem or execute tools before guessing.",
      "Use write_todos for non-trivial tasks, especially if they involve multiple file or command steps.",
      "Use shell or Python when calculations or file transformations matter.",
      "Keep the final answer concise and mention any files you created or changed.",
    ].join(" "),
    name: "gemma-browser-lab",
  });

  await refreshWorkspacePanel();
  setSandboxState("ready", "Sandbox worker is running.");
}

initButton.addEventListener("click", async () => {
  if (isInitializing || model !== null) {
    return;
  }

  const attemptId = initAttemptId + 1;
  initAttemptId = attemptId;
  isInitializing = true;
  syncUi();

  try {
    await buildDeepAgent(attemptId);
    if (attemptId !== initAttemptId) {
      return;
    }

    appendMessage(
      "system",
      "Init",
      "Model initialized. The browser deep agent can now plan, inspect /workspace, edit files, and run shell or Python inside the wasm sandbox."
    );
    setStatus("Model and sandbox ready.", "ready");
  } catch (error) {
    if (attemptId !== initAttemptId && error?.name === "AbortError") {
      return;
    }

    const message = extractErrorMessage(
      error,
      "Unknown initialization error."
    );
    initAttemptId += 1;
    clearInitTimers();
    abortModelDownload();
    if (sandbox) {
      await sandbox.stop().catch(() => {});
      sandbox = null;
    }
    model = null;
    deepAgent = null;
    setSandboxState("error", message);
    setStatus(message, "error");
    appendMessage("error", "Initialization Error", message);
  } finally {
    if (attemptId === initAttemptId) {
      isInitializing = false;
    }
    syncUi();
  }
});

resetButton.addEventListener("click", async () => {
  if (isRunning) {
    return;
  }

  await resetDemoState({
    announce:
      model !== null ||
      deepAgent !== null ||
      sandbox !== null ||
      statusNode.dataset.mode === "error",
  });
});

sendButton.addEventListener("click", async () => {
  const input = promptInput.value.trim();
  if (!input || !deepAgent || isRunning) {
    return;
  }

  isRunning = true;
  resetRunMetrics();
  syncUi();
  appendMessage("user", "User", input);
  promptInput.value = "";
  setStatus("Running browser deep agent...", "working");

  try {
    await runDeepAgent(input);
    setStatus(
      `Model ready. Last run: ${formatTokenRate(runMetrics.tokenRate)} tok/s.`,
      "ready"
    );
  } catch (error) {
    const message = extractErrorMessage(error, "Unknown agent error.");
    appendDebugLine(`run failed: ${message}`);
    appendMessage("error", "Agent Error", message);
    setSandboxState("error", message);
    setStatus(message, "error");
  } finally {
    isRunning = false;
    syncUi();
  }
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendButton.click();
  }
});

renderTodos([]);
renderSandboxPanel();
workspaceNode.textContent =
  "Workspace unavailable until the browser sandbox is initialized.";
syncUi();
setStatus("Idle.", "idle");
resetDownloadProgress();
resetRunMetrics();
