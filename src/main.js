import { HumanMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import {
  AgentExecutor,
  createToolCallingAgent,
} from "@langchain/classic/agents";
import { ChatMediaPipeGenAI } from "./vendor/langchain-community/chat_models/mediapipe_genai.js";
import { z } from "zod";

const initButton = document.getElementById("init-button");
const resetButton = document.getElementById("reset-button");
const sendButton = document.getElementById("send-button");
const promptInput = document.getElementById("prompt-input");
const statusNode = document.getElementById("status");
const transcript = document.getElementById("transcript");

const diagnosticsNodes = {
  modelUrl: document.getElementById("diag-model-url"),
  wasmRoot: document.getElementById("diag-wasm-root"),
  webgpu: document.getElementById("diag-webgpu"),
  adapter: document.getElementById("diag-adapter"),
  stage: document.getElementById("diag-stage"),
};

const downloadProgressNode = document.getElementById("download-progress");
const downloadStatusNode = document.getElementById("download-status");

const imageUploadInput = document.getElementById("image-upload");
const imagePromptInput = document.getElementById("image-prompt");
const imagePreviewNode = document.getElementById("image-preview");
const imageStatusNode = document.getElementById("image-status");
const cameraPreviewNode = document.getElementById("camera-preview");
const startCameraButton = document.getElementById("start-camera-button");
const captureImageButton = document.getElementById("capture-image-button");
const stopCameraButton = document.getElementById("stop-camera-button");
const analyzeImageButton = document.getElementById("analyze-image-button");

const audioUploadInput = document.getElementById("audio-upload");
const audioPromptInput = document.getElementById("audio-prompt");
const audioPreviewNode = document.getElementById("audio-preview");
const audioStatusNode = document.getElementById("audio-status");
const startRecordingButton = document.getElementById("start-recording-button");
const stopRecordingButton = document.getElementById("stop-recording-button");
const analyzeAudioButton = document.getElementById("analyze-audio-button");

let agentExecutor = null;
let model = null;
let isInitializing = false;
let isRunning = false;
let isRecording = false;
let initAttemptId = 0;
let initLongLoadTimer = null;
let initTimeoutTimer = null;
let initDownloadController = null;
let cameraStream = null;
let microphoneStream = null;
let mediaRecorder = null;
let discardNextRecording = false;
let recordedChunks = [];
let selectedImage = null;
let selectedAudio = null;

const WASM_ROOT = "/vendor/mediapipe/tasks-genai/wasm";
const INIT_LONG_LOAD_MS = 20_000;
const INIT_TIMEOUT_MS = 180_000;
const MODEL_ASSET_PATH_CANDIDATES = [
  "/models/gemma/gemma-4-E2B-it-web.task",
  "/models/gemma-4-E2B-it-web.task",
];
const WASM_ENTRYPOINT = `${WASM_ROOT}/genai_wasm_internal.js`;
const diagnostics = {
  modelUrl: "unresolved",
  wasmRoot: WASM_ROOT,
  webgpu: "unchecked",
  adapter: "unchecked",
  stage: "idle",
};

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

function extractErrorMessage(error, fallback) {
  return error instanceof Error ? error.message : fallback;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed with ${response.status} ${response.statusText}`);
  }

  return response.json();
}

function describeWeatherCode(code) {
  const descriptions = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
  };

  return descriptions[code] ?? "unknown conditions";
}

function validateModelAsset(modelAsset) {
  const warnings = [];

  if (modelAsset.contentType.toLowerCase().includes("text/html")) {
    warnings.push(
      "The server is returning HTML instead of a binary model file."
    );
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

const weatherTool = tool(
  async ({ location }) => {
    const searchUrl = new URL("https://geocoding-api.open-meteo.com/v1/search");
    searchUrl.search = new URLSearchParams({
      name: location,
      count: "1",
      language: "en",
      format: "json",
    }).toString();

    const geocoding = await fetchJson(searchUrl);
    const match = geocoding.results?.[0];
    if (!match) {
      throw new Error(`No location match found for "${location}".`);
    }

    const forecastUrl = new URL("https://api.open-meteo.com/v1/forecast");
    forecastUrl.search = new URLSearchParams({
      latitude: String(match.latitude),
      longitude: String(match.longitude),
      current: [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "weather_code",
        "wind_speed_10m",
      ].join(","),
      timezone: "auto",
    }).toString();

    const forecast = await fetchJson(forecastUrl);
    const current = forecast.current;
    if (!current) {
      throw new Error(`No current weather data returned for "${location}".`);
    }

    return JSON.stringify(
      {
        source: "Open-Meteo",
        resolvedLocation: [match.name, match.admin1, match.country]
          .filter(Boolean)
          .join(", "),
        latitude: match.latitude,
        longitude: match.longitude,
        timezone: forecast.timezone,
        observationTime: current.time,
        temperatureC: current.temperature_2m,
        apparentTemperatureC: current.apparent_temperature,
        relativeHumidity: current.relative_humidity_2m,
        windSpeedKmh: current.wind_speed_10m,
        weatherCode: current.weather_code,
        weatherSummary: describeWeatherCode(current.weather_code),
      },
      null,
      2
    );
  },
  {
    name: "get_current_weather",
    description: "Get the current weather for a city using Open-Meteo.",
    schema: z.object({
      location: z.string().describe("City or place name."),
    }),
  }
);

const localTimeTool = tool(
  async ({ location }) => {
    const timeZoneMap = {
      berlin: "Europe/Berlin",
      paris: "Europe/Paris",
      london: "Europe/London",
      tokyo: "Asia/Tokyo",
      "new york": "America/New_York",
      "san francisco": "America/Los_Angeles",
    };

    const timeZone = timeZoneMap[location.trim().toLowerCase()] ?? "UTC";
    return new Intl.DateTimeFormat("en-US", {
      dateStyle: "full",
      timeStyle: "long",
      timeZone,
    }).format(new Date());
  },
  {
    name: "get_local_time",
    description: "Get the local time for a city or place.",
    schema: z.object({
      location: z.string().describe("City or place name."),
    }),
  }
);

const calculatorTool = tool(
  async ({ expression }) => {
    if (!/^[0-9+\-*/().\s]+$/.test(expression)) {
      throw new Error("Only basic arithmetic expressions are allowed.");
    }

    const value = Function(`"use strict"; return (${expression});`)();
    if (typeof value !== "number" || Number.isNaN(value)) {
      throw new Error("Expression did not evaluate to a valid number.");
    }

    return String(value);
  },
  {
    name: "calculate",
    description: "Evaluate a basic arithmetic expression.",
    schema: z.object({
      expression: z.string().describe("Arithmetic expression like 17 * 6"),
    }),
  }
);

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
}

function setStatus(text, mode = "idle") {
  statusNode.textContent = text;
  statusNode.dataset.mode = mode;
}

function setDiagnostic(key, value) {
  diagnostics[key] = value;
  diagnosticsNodes[key].textContent = value;
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

function revokeSelectionUrl(selection) {
  if (selection?.revoke && selection.url) {
    URL.revokeObjectURL(selection.url);
  }
}

function stopCameraStream() {
  if (!cameraStream) {
    cameraPreviewNode.hidden = true;
    cameraPreviewNode.srcObject = null;
    return;
  }

  for (const track of cameraStream.getTracks()) {
    track.stop();
  }
  cameraStream = null;
  cameraPreviewNode.srcObject = null;
  cameraPreviewNode.hidden = true;
}

function stopMicrophoneStream() {
  if (!microphoneStream) {
    return;
  }

  for (const track of microphoneStream.getTracks()) {
    track.stop();
  }
  microphoneStream = null;
}

function setImageSelection(selection) {
  revokeSelectionUrl(selectedImage);
  selectedImage = selection;

  if (!selection) {
    imagePreviewNode.hidden = true;
    imagePreviewNode.removeAttribute("src");
    imageStatusNode.textContent = "No image selected.";
    syncUi();
    return;
  }

  imagePreviewNode.src = selection.url;
  imagePreviewNode.hidden = false;
  imageStatusNode.textContent = `Ready: ${selection.label}`;
  syncUi();
}

function setAudioSelection(selection) {
  revokeSelectionUrl(selectedAudio);
  selectedAudio = selection;

  if (!selection) {
    audioPreviewNode.hidden = true;
    audioPreviewNode.pause();
    audioPreviewNode.removeAttribute("src");
    audioPreviewNode.load();
    audioStatusNode.textContent = "No audio selected.";
    syncUi();
    return;
  }

  audioPreviewNode.src = selection.url;
  audioPreviewNode.hidden = false;
  audioPreviewNode.load();
  audioStatusNode.textContent = `Ready: ${selection.label}`;
  syncUi();
}

function setImageStatus(text) {
  imageStatusNode.textContent = text;
}

function setAudioStatus(text) {
  audioStatusNode.textContent = text;
}

function extractResponseText(message) {
  if (typeof message.content === "string") {
    return message.content;
  }

  return message.content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }

      if (
        typeof part === "object" &&
        part !== null &&
        "type" in part &&
        part.type === "text" &&
        "text" in part
      ) {
        return part.text;
      }

      return "";
    })
    .filter(Boolean)
    .join("");
}

function resetDemoState(options = {}) {
  const { announce = false } = options;
  initAttemptId += 1;
  clearInitTimers();
  abortModelDownload();

  if (isRecording && mediaRecorder?.state === "recording") {
    discardNextRecording = true;
    mediaRecorder.stop();
  }
  stopMicrophoneStream();
  stopCameraStream();

  model = null;
  agentExecutor = null;
  isInitializing = false;
  isRunning = false;
  isRecording = false;
  mediaRecorder = null;
  recordedChunks = [];

  setStatus("Idle.", "idle");
  setDiagnostic("modelUrl", "unresolved");
  setDiagnostic("wasmRoot", WASM_ROOT);
  setDiagnostic("webgpu", "unchecked");
  setDiagnostic("adapter", "unchecked");
  setDiagnostic("stage", "idle");
  resetDownloadProgress();

  if (announce) {
    appendMessage(
      "system",
      "Reset",
      "Local model state was cleared. You can initialize again without reloading the page."
    );
  }

  syncUi();
}

function syncUi() {
  const hasRunnableModel = model !== null && agentExecutor !== null;
  const canInteractWithMedia = !isInitializing && !isRunning;

  initButton.disabled = isInitializing || model !== null;
  resetButton.disabled =
    isRunning ||
    (!isInitializing &&
      model === null &&
      agentExecutor === null &&
      !isRecording &&
      statusNode.dataset.mode !== "error" &&
      diagnostics.stage === "idle");

  sendButton.disabled = isInitializing || isRunning || agentExecutor === null;
  promptInput.disabled = isInitializing || isRunning || agentExecutor === null;

  imageUploadInput.disabled = !canInteractWithMedia;
  startCameraButton.disabled = !canInteractWithMedia || cameraStream !== null;
  captureImageButton.disabled = !canInteractWithMedia || cameraStream === null;
  stopCameraButton.disabled = !canInteractWithMedia || cameraStream === null;
  imagePromptInput.disabled = !canInteractWithMedia;
  analyzeImageButton.disabled =
    !hasRunnableModel || !canInteractWithMedia || !selectedImage;

  audioUploadInput.disabled = !canInteractWithMedia || isRecording;
  startRecordingButton.disabled =
    !canInteractWithMedia ||
    isRecording ||
    !navigator.mediaDevices?.getUserMedia ||
    typeof MediaRecorder === "undefined";
  stopRecordingButton.disabled = !isRecording;
  audioPromptInput.disabled = !canInteractWithMedia;
  analyzeAudioButton.disabled =
    !hasRunnableModel || !canInteractWithMedia || !selectedAudio;
}

async function resolveWasmAsset() {
  const response = await fetch(WASM_ENTRYPOINT, {
    method: "HEAD",
  });
  if (!response.ok) {
    throw new Error(
      `MediaPipe runtime asset check failed for ${WASM_ENTRYPOINT} with ${response.status} ${response.statusText}.`
    );
  }

  const contentType =
    response.headers.get("content-type") ?? "unknown content type";
  if (contentType.toLowerCase().includes("text/html")) {
    throw new Error(
      `MediaPipe runtime asset check failed for ${WASM_ENTRYPOINT}: the server returned HTML instead of JavaScript.`
    );
  }

  return {
    path: WASM_ENTRYPOINT,
    contentType,
  };
}

async function resolveModelAssetPath() {
  for (const candidate of MODEL_ASSET_PATH_CANDIDATES) {
    try {
      const response = await fetch(candidate, {
        method: "HEAD",
      });
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

async function warmModelAssetDownload(modelAsset, attemptId) {
  throwIfAttemptStale(attemptId);
  setDiagnostic("stage", "downloading-model");
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
    setDownloadProgress(
      total || 1,
      total || 1,
      `Model download completed (${formatBytes(total)}).`
    );
    initDownloadController = null;
    return;
  }

  const reader = response.body.getReader();
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
  } finally {
    initDownloadController = null;
  }

  setDownloadProgress(
    total || received || 1,
    total || received || 1,
    `Model download completed (${formatBytes(total || received)}).`
  );
}

async function runRuntimeCompatibilityPreflight() {
  setDiagnostic("stage", "runtime preflight");
  setDiagnostic("wasmRoot", WASM_ROOT);

  if (
    typeof navigator === "undefined" ||
    !("gpu" in navigator) ||
    !navigator.gpu
  ) {
    setDiagnostic("webgpu", "unavailable");
    setDiagnostic("adapter", "not requested");
    throw new Error(
      "WebGPU is not available in this browser. Use a recent Chromium-based browser with WebGPU enabled."
    );
  }

  setDiagnostic("webgpu", "available");
  setDiagnostic("adapter", "requesting");

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (error) {
    setDiagnostic("adapter", "request failed");
    throw new Error(
      `WebGPU adapter request failed: ${extractErrorMessage(
        error,
        "Unknown adapter request failure."
      )}`
    );
  }

  if (!adapter) {
    setDiagnostic("adapter", "unavailable");
    throw new Error(
      "WebGPU is exposed by the browser, but no GPU adapter could be acquired. This usually means GPU access is blocked, unsupported, or the device lacks enough resources."
    );
  }

  setDiagnostic("adapter", "acquired");
  return adapter;
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
    setDiagnostic("stage", "resolving-fileset");
    setStatus("Resolving MediaPipe fileset...", "working");
    return;
  }

  if (stage === "creating-inference") {
    setDiagnostic("stage", "creating-inference");
    setStatus("Creating Gemma inference instance...", "working");
    if (initLongLoadTimer === null) {
      initLongLoadTimer = window.setTimeout(() => {
        if (attemptId !== initAttemptId) {
          return;
        }
        setDiagnostic("stage", "creating-inference (long-running)");
        setStatus(
          "Still creating Gemma inference instance. Large local models can take significant time and may fail on unsupported GPUs or memory-constrained devices.",
          "working"
        );
      }, INIT_LONG_LOAD_MS);
    }
    return;
  }

  clearLongLoadTimer();
  setDiagnostic("stage", "ready");
  setStatus("Model ready.", "ready");
}

async function initializeModelWithGuards(attemptId) {
  const initPromise = model.initialize((progress) => {
    updateInitializationStage(attemptId, progress.stage);
  });

  const timeoutPromise = new Promise((_, reject) => {
    initTimeoutTimer = window.setTimeout(() => {
      reject(new Error(buildInitializationTimeoutMessage()));
    }, INIT_TIMEOUT_MS);
  });

  try {
    await Promise.race([initPromise, timeoutPromise]);
  } finally {
    clearInitTimers();
  }
}

async function buildAgent(attemptId) {
  const modelAsset = await resolveModelAssetPath();
  const wasmAsset = await resolveWasmAsset();
  validateModelAsset(modelAsset);
  await runRuntimeCompatibilityPreflight();
  throwIfAttemptStale(attemptId);
  setDiagnostic("modelUrl", modelAsset.path);

  appendMessage(
    "system",
    "Model Preflight",
    [
      `Resolved model URL: ${modelAsset.path}`,
      `Content-Type: ${modelAsset.contentType}`,
      `Content-Length: ${formatBytes(modelAsset.contentLength)}`,
      `Resolved WASM runtime: ${wasmAsset.path}`,
      `WASM Content-Type: ${wasmAsset.contentType}`,
      `WebGPU: ${diagnostics.webgpu}`,
      `Adapter: ${diagnostics.adapter}`,
      "Expected size is roughly 2.0 GB for the real Gemma 4 E2B web model.",
    ].join("\n")
  );

  await warmModelAssetDownload(modelAsset, attemptId);
  throwIfAttemptStale(attemptId);

  model = new ChatMediaPipeGenAI({
    wasmRoot: WASM_ROOT,
    modelAssetPath: modelAsset.path,
    maxTokens: 2048,
    temperature: 0.2,
    topK: 40,
    randomSeed: 101,
    maxNumImages: 1,
    supportAudio: true,
  });

  setDiagnostic("stage", "initializing");
  setStatus("Initializing MediaPipe fileset...", "working");
  await initializeModelWithGuards(attemptId);
  throwIfAttemptStale(attemptId);

  const tools = [weatherTool, localTimeTool, calculatorTool];
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      [
        "You are a concise browser agent running fully on-device.",
        "Use tools whenever they improve accuracy.",
        "When tools are used, synthesize their outputs into a short direct answer.",
      ].join(" "),
    ],
    ["human", "{input}"],
    ["placeholder", "{agent_scratchpad}"],
  ]);

  const agent = await createToolCallingAgent({
    llm: model,
    tools,
    prompt,
  });

  throwIfAttemptStale(attemptId);
  agentExecutor = new AgentExecutor({
    agent,
    tools,
    returnIntermediateSteps: true,
  });
}

async function startCameraPreview() {
  if (cameraStream) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access is unavailable in this browser.");
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
      },
    });
  } catch {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: true,
    });
  }

  cameraPreviewNode.srcObject = cameraStream;
  cameraPreviewNode.hidden = false;
  await cameraPreviewNode.play().catch(() => {});
  setImageStatus("Camera ready. Take a snapshot when the preview looks right.");
  syncUi();
}

function captureCameraSnapshot() {
  if (!cameraStream) {
    throw new Error("Start the camera before taking a snapshot.");
  }

  const width = cameraPreviewNode.videoWidth || 1280;
  const height = cameraPreviewNode.videoHeight || 720;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not capture a camera snapshot.");
  }

  context.drawImage(cameraPreviewNode, 0, 0, width, height);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Camera snapshot failed."));
          return;
        }

        const label = `Camera snapshot ${new Date().toLocaleTimeString()}`;
        resolve({
          url: URL.createObjectURL(blob),
          label,
          revoke: true,
        });
      },
      "image/jpeg",
      0.92
    );
  });
}

function pickRecordingMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return undefined;
  }

  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
  ];

  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

async function startMicrophoneRecording() {
  if (isRecording) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone access is unavailable in this browser.");
  }

  if (typeof MediaRecorder === "undefined") {
    throw new Error("MediaRecorder is unavailable in this browser.");
  }

  microphoneStream = await navigator.mediaDevices.getUserMedia({
    audio: true,
  });

  const mimeType = pickRecordingMimeType();
  mediaRecorder = mimeType
    ? new MediaRecorder(microphoneStream, { mimeType })
    : new MediaRecorder(microphoneStream);

  discardNextRecording = false;
  recordedChunks = [];

  mediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  });

  mediaRecorder.addEventListener(
    "stop",
    () => {
      isRecording = false;
      stopMicrophoneStream();

      if (discardNextRecording) {
        discardNextRecording = false;
        mediaRecorder = null;
        recordedChunks = [];
        syncUi();
        return;
      }

      if (recordedChunks.length === 0) {
        setAudioStatus("Recording stopped, but no audio data was captured.");
        mediaRecorder = null;
        syncUi();
        return;
      }

      const mime = mediaRecorder?.mimeType || mimeType || "audio/webm";
      const blob = new Blob(recordedChunks, { type: mime });
      setAudioSelection({
        blob,
        url: URL.createObjectURL(blob),
        label: `Microphone clip (${formatBytes(blob.size)})`,
        revoke: true,
      });
      recordedChunks = [];
      mediaRecorder = null;
      setAudioStatus("Microphone recording captured. Analyze it when ready.");
      syncUi();
    },
    { once: true }
  );

  mediaRecorder.start();
  isRecording = true;
  setAudioStatus("Recording from the microphone...");
  syncUi();
}

function stopMicrophoneRecording() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") {
    return;
  }

  mediaRecorder.stop();
}

function encodeAudioBufferToMonoWav(audioBuffer) {
  const sampleRate = audioBuffer.sampleRate;
  const frameCount = audioBuffer.length;
  const channelCount = audioBuffer.numberOfChannels;
  const mono = new Float32Array(frameCount);

  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const channelData = audioBuffer.getChannelData(channelIndex);
    for (let index = 0; index < frameCount; index += 1) {
      mono[index] += channelData[index] / channelCount;
    }
  }

  const buffer = new ArrayBuffer(44 + frameCount * 2);
  const view = new DataView(buffer);

  const writeAscii = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + frameCount * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, frameCount * 2, true);

  let offset = 44;
  for (let index = 0; index < frameCount; index += 1) {
    const sample = Math.max(-1, Math.min(1, mono[index]));
    view.setInt16(
      offset,
      sample < 0 ? sample * 0x8000 : sample * 0x7fff,
      true
    );
    offset += 2;
  }

  return new Uint8Array(buffer);
}

async function wavBytesToBase64(wavBytes) {
  const blob = new Blob([wavBytes], {
    type: "audio/wav",
  });

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not base64-encode audio."));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string" || !result.includes(",")) {
        reject(new Error("Could not base64-encode audio."));
        return;
      }

      resolve(result.split(",")[1]);
    };
    reader.readAsDataURL(blob);
  });
}

async function audioBlobToModelInput(blob) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) {
    throw new Error("The Web Audio API is unavailable in this browser.");
  }

  const audioContext = new AudioContextCtor();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const wavBytes = encodeAudioBufferToMonoWav(decoded);
    const data = await wavBytesToBase64(wavBytes);
    return {
      data,
      format: "wav",
    };
  } catch (error) {
    throw new Error(
      `Audio decoding failed: ${extractErrorMessage(
        error,
        "Unknown audio decode failure."
      )}`
    );
  } finally {
    await audioContext.close().catch(() => {});
  }
}

async function runImageAnalysis() {
  if (!model || !selectedImage || isRunning) {
    return;
  }

  const prompt = imagePromptInput.value.trim();
  if (!prompt) {
    setImageStatus("Enter an image prompt before analyzing.");
    return;
  }

  isRunning = true;
  syncUi();
  setStatus("Analyzing image...", "working");
  appendMessage("user", "Image Prompt", `${prompt}\n\nSource: ${selectedImage.label}`);

  try {
    const response = await model.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: selectedImage.url } },
        ],
      }),
    ]);

    appendMessage(
      "assistant",
      "Image Analysis",
      extractResponseText(response) || "(empty output)"
    );
    setStatus("Model ready.", "ready");
    setImageStatus(`Analyzed: ${selectedImage.label}`);
  } catch (error) {
    const message = extractErrorMessage(error, "Unknown image analysis error.");
    appendMessage("error", "Image Error", message);
    setStatus(message, "error");
    setImageStatus(message);
  } finally {
    isRunning = false;
    syncUi();
  }
}

async function runAudioAnalysis() {
  if (!model || !selectedAudio || isRunning) {
    return;
  }

  const prompt = audioPromptInput.value.trim();
  if (!prompt) {
    setAudioStatus("Enter an audio prompt before analyzing.");
    return;
  }

  isRunning = true;
  syncUi();
  setStatus("Converting audio for model input...", "working");
  appendMessage("user", "Audio Prompt", `${prompt}\n\nSource: ${selectedAudio.label}`);

  try {
    const inputAudio = await audioBlobToModelInput(selectedAudio.blob);
    setStatus("Analyzing audio...", "working");
    const response = await model.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: prompt },
          { type: "input_audio", input_audio: inputAudio },
        ],
      }),
    ]);

    appendMessage(
      "assistant",
      "Audio Analysis",
      extractResponseText(response) || "(empty output)"
    );
    setStatus("Model ready.", "ready");
    setAudioStatus(`Analyzed: ${selectedAudio.label}`);
  } catch (error) {
    const message = extractErrorMessage(error, "Unknown audio analysis error.");
    appendMessage("error", "Audio Error", message);
    setStatus(message, "error");
    setAudioStatus(message);
  } finally {
    isRunning = false;
    syncUi();
  }
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
    await buildAgent(attemptId);
    if (attemptId !== initAttemptId) {
      return;
    }
    appendMessage(
      "system",
      "Init",
      "Model initialized. The agent is ready for local tool-calling prompts and direct image/audio analysis."
    );
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
    setStatus(message, "error");
    setDiagnostic("stage", "failed");
    appendMessage("error", "Initialization Error", message);
    model = null;
    agentExecutor = null;
  } finally {
    if (attemptId === initAttemptId) {
      isInitializing = false;
    }
    syncUi();
  }
});

resetButton.addEventListener("click", () => {
  if (isRunning) {
    return;
  }

  resetDemoState({
    announce:
      model !== null ||
      agentExecutor !== null ||
      isRecording ||
      statusNode.dataset.mode === "error",
  });
});

sendButton.addEventListener("click", async () => {
  const input = promptInput.value.trim();
  if (!input || !agentExecutor || isRunning) {
    return;
  }

  isRunning = true;
  syncUi();
  appendMessage("user", "User", input);
  promptInput.value = "";
  setStatus("Running agent...", "working");

  try {
    const result = await agentExecutor.invoke({
      input,
    });

    for (const step of result.intermediateSteps ?? []) {
      appendMessage(
        "tool",
        `Tool: ${step.action.tool}`,
        JSON.stringify(
          {
            toolInput: step.action.toolInput,
            observation: step.observation,
          },
          null,
          2
        )
      );
    }

    appendMessage("assistant", "Assistant", result.output ?? "(empty output)");
    setStatus("Model ready.", "ready");
  } catch (error) {
    const message = extractErrorMessage(error, "Unknown agent error.");
    appendMessage("error", "Agent Error", message);
    setStatus(message, "error");
  } finally {
    isRunning = false;
    syncUi();
  }
});

promptInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendButton.click();
  }
});

imageUploadInput.addEventListener("change", () => {
  const file = imageUploadInput.files?.[0];
  if (!file) {
    setImageSelection(null);
    return;
  }

  setImageSelection({
    url: URL.createObjectURL(file),
    label: `${file.name} (${formatBytes(file.size)})`,
    revoke: true,
  });
});

startCameraButton.addEventListener("click", async () => {
  try {
    await startCameraPreview();
  } catch (error) {
    const message = extractErrorMessage(error, "Camera startup failed.");
    setImageStatus(message);
    appendMessage("error", "Camera Error", message);
  }
});

captureImageButton.addEventListener("click", async () => {
  try {
    const snapshot = await captureCameraSnapshot();
    setImageSelection(snapshot);
    setImageStatus(`Captured: ${snapshot.label}`);
  } catch (error) {
    const message = extractErrorMessage(error, "Camera snapshot failed.");
    setImageStatus(message);
    appendMessage("error", "Camera Error", message);
  }
});

stopCameraButton.addEventListener("click", () => {
  stopCameraStream();
  if (selectedImage) {
    setImageStatus(`Ready: ${selectedImage.label}`);
  } else {
    setImageStatus("Camera stopped. Upload an image or start the camera again.");
  }
  syncUi();
});

analyzeImageButton.addEventListener("click", async () => {
  await runImageAnalysis();
});

audioUploadInput.addEventListener("change", () => {
  const file = audioUploadInput.files?.[0];
  if (!file) {
    setAudioSelection(null);
    return;
  }

  setAudioSelection({
    blob: file,
    url: URL.createObjectURL(file),
    label: `${file.name} (${formatBytes(file.size)})`,
    revoke: true,
  });
});

startRecordingButton.addEventListener("click", async () => {
  try {
    await startMicrophoneRecording();
  } catch (error) {
    const message = extractErrorMessage(error, "Microphone recording failed.");
    setAudioStatus(message);
    appendMessage("error", "Microphone Error", message);
  }
});

stopRecordingButton.addEventListener("click", () => {
  stopMicrophoneRecording();
});

analyzeAudioButton.addEventListener("click", async () => {
  await runAudioAnalysis();
});

syncUi();
setStatus("Idle.", "idle");
setDiagnostic("wasmRoot", WASM_ROOT);
resetDownloadProgress();
