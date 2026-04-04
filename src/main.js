import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import {
  AgentExecutor,
  createToolCallingAgent,
} from "@langchain/classic/agents";
import { ChatMediaPipeGenAI } from "./vendor/langchain-community/chat_models/mediapipe_genai.js";
import { z } from "zod";

const initButton = document.getElementById("init-button");
const sendButton = document.getElementById("send-button");
const promptInput = document.getElementById("prompt-input");
const statusNode = document.getElementById("status");
const transcript = document.getElementById("transcript");

let agentExecutor = null;
let model = null;
let isInitializing = false;
let isRunning = false;

const MODEL_ASSET_PATH_CANDIDATES = [
  "/models/gemma/gemma-4-E2B-it-web.task",
  "/models/gemma-4-E2B-it-web.task",
];

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
        resolvedLocation: [
          match.name,
          match.admin1,
          match.country,
        ]
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

function syncUi() {
  initButton.disabled = isInitializing || model !== null;
  sendButton.disabled = isInitializing || isRunning || agentExecutor === null;
  promptInput.disabled = isInitializing || isRunning || agentExecutor === null;
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

async function buildAgent() {
  const modelAsset = await resolveModelAssetPath();
  validateModelAsset(modelAsset);
  appendMessage(
    "system",
    "Model Preflight",
    [
      `Resolved model URL: ${modelAsset.path}`,
      `Content-Type: ${modelAsset.contentType}`,
      `Content-Length: ${formatBytes(modelAsset.contentLength)}`,
      "Expected size is roughly 2.0 GB for the real Gemma 4 E2B web model.",
    ].join("\n")
  );

  model = new ChatMediaPipeGenAI({
    wasmRoot: "/vendor/mediapipe/tasks-genai/wasm",
    modelAssetPath: modelAsset.path,
    maxTokens: 2048,
    temperature: 0.2,
    topK: 40,
    randomSeed: 101,
  });

  setStatus("Initializing MediaPipe fileset...", "working");
  await model.initialize((progress) => {
    if (progress.stage === "resolving-fileset") {
      setStatus("Resolving MediaPipe fileset...", "working");
      return;
    }

    if (progress.stage === "creating-inference") {
      setStatus("Creating Gemma inference instance...", "working");
      return;
    }

    setStatus("Model ready.", "ready");
  });

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

  agentExecutor = new AgentExecutor({
    agent,
    tools,
    returnIntermediateSteps: true,
  });
}

initButton.addEventListener("click", async () => {
  if (isInitializing || model !== null) {
    return;
  }

  isInitializing = true;
  syncUi();

  try {
    await buildAgent();
    appendMessage(
      "system",
      "Init",
      "Model initialized. The agent is ready for local tool-calling prompts."
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown initialization error.";
    setStatus(message, "error");
    appendMessage("error", "Initialization Error", message);
    model = null;
    agentExecutor = null;
  } finally {
    isInitializing = false;
    syncUi();
  }
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
    const message =
      error instanceof Error ? error.message : "Unknown agent error.";
    appendMessage("error", "Agent Error", message);
    setStatus(message, "error");
  } finally {
    isRunning = false;
    syncUi();
  }
});

promptInput.addEventListener("keydown", async (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendButton.click();
  }
});

syncUi();
