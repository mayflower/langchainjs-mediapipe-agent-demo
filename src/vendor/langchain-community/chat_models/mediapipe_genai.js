import { extractVisibleText } from "./mediapipe_genai_internal/prompt_codec.js";
import { Gemma4Codec } from "./mediapipe_genai_internal/gemma4_codec.js";
import {
  makeInvalidToolCall,
  parseToolCall,
} from "@langchain/core/output_parsers/openai_tools";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk } from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";

function isFunctionToolDefinition(tool) {
  return (
    !!tool &&
    typeof tool === "object" &&
    "type" in tool &&
    tool.type === "function" &&
    "function" in tool &&
    !!tool.function &&
    typeof tool.function === "object" &&
    "name" in tool.function &&
    typeof tool.function.name === "string"
  );
}

function normalizeToolDefinitions(tools) {
  return (tools ?? []).map((tool) => {
    const openAITool = isFunctionToolDefinition(tool)
      ? tool
      : convertToOpenAITool(tool);

    return {
      type: "function",
      function: {
        name: openAITool.function.name,
        description: openAITool.function.description,
        parameters: openAITool.function.parameters,
      },
    };
  });
}

function normalizeToolChoice(toolChoice) {
  if (toolChoice === undefined) {
    return undefined;
  }

  if (toolChoice === "auto" || toolChoice === "none") {
    return toolChoice;
  }

  if (toolChoice === "any" || toolChoice === "required") {
    return "required";
  }

  if (typeof toolChoice === "string") {
    return {
      type: "function",
      function: { name: toolChoice },
    };
  }

  if (typeof toolChoice === "object" && toolChoice !== null && "type" in toolChoice) {
    if (toolChoice.type === "none") {
      return "none";
    }

    if (toolChoice.type === "auto") {
      return "auto";
    }

    if (
      toolChoice.type === "function" &&
      "function" in toolChoice &&
      toolChoice.function &&
      typeof toolChoice.function === "object" &&
      "name" in toolChoice.function &&
      typeof toolChoice.function.name === "string"
    ) {
      return {
        type: "function",
        function: { name: toolChoice.function.name },
      };
    }

    if (toolChoice.type === "tool" && "name" in toolChoice && typeof toolChoice.name === "string") {
      return {
        type: "function",
        function: { name: toolChoice.name },
      };
    }
  }

  throw new Error(
    `Unsupported tool_choice value for ChatMediaPipeGenAI: ${JSON.stringify(toolChoice)}`
  );
}

export class ChatMediaPipeGenAI extends BaseChatModel {
  static lc_name() {
    return "ChatMediaPipeGenAI";
  }

  llmInference;
  codec;
  wasmRoot;
  modelAssetPath;
  maxTokens;
  temperature;
  topK;
  randomSeed;

  get callKeys() {
    return ["tools", "tool_choice"];
  }

  constructor(fields) {
    super(fields);
    this.wasmRoot = fields.wasmRoot;
    this.modelAssetPath = fields.modelAssetPath;
    this.maxTokens = fields.maxTokens;
    this.temperature = fields.temperature;
    this.topK = fields.topK;
    this.randomSeed = fields.randomSeed;
    this.codec = new Gemma4Codec({ thoughtTagName: fields.thoughtTagName });
  }

  _llmType() {
    return "mediapipe-genai";
  }

  identifyingParams() {
    return {
      wasmRoot: this.wasmRoot,
      modelAssetPath: this.modelAssetPath,
      maxTokens: this.maxTokens,
      temperature: this.temperature,
      topK: this.topK,
      randomSeed: this.randomSeed,
    };
  }

  bindTools(tools, kwargs) {
    return this.withConfig({
      tools,
      ...kwargs,
    });
  }

  async initialize(progressCallback) {
    if (this.llmInference) {
      progressCallback?.({ stage: "ready" });
      return;
    }

    this.ensureSupportedEnvironment();
    progressCallback?.({ stage: "resolving-fileset" });

    const tasksModule = await this.importMediaPipeTasksGenAI();
    const fileset = await tasksModule.FilesetResolver.forGenAiTasks(this.wasmRoot);

    progressCallback?.({ stage: "creating-inference" });
    this.llmInference = await tasksModule.LlmInference.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: this.modelAssetPath },
      maxTokens: this.maxTokens,
      temperature: this.temperature,
      topK: this.topK,
      randomSeed: this.randomSeed,
    });

    progressCallback?.({ stage: "ready" });
  }

  ensureSupportedEnvironment() {
    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined" ||
      typeof document === "undefined"
    ) {
      throw new Error(
        "ChatMediaPipeGenAI is browser-only. Call initialize() from a WebGPU-capable browser environment."
      );
    }
  }

  async importMediaPipeTasksGenAI() {
    return await import("@mediapipe/tasks-genai");
  }

  getInference() {
    if (!this.llmInference) {
      throw new Error(
        "ChatMediaPipeGenAI must be initialized before invocation. Call initialize() first."
      );
    }

    return this.llmInference;
  }

  async _generate(messages, options) {
    this.validateCallOptions(options);

    const inference = this.getInference();
    const { prompt, tools } = this.renderPrompt(messages, options);
    const rawResponse = await inference.generateResponse(prompt);

    if (typeof rawResponse !== "string") {
      throw new Error(
        "MediaPipe LlmInference.generateResponse() did not return a string for non-streaming generation."
      );
    }

    const parsed = this.parseResponseArtifacts(rawResponse, tools);

    return {
      generations: [
        {
          text: typeof parsed.message.content === "string" ? parsed.message.content : "",
          message: parsed.message,
        },
      ],
      llmOutput: { rawResponse },
    };
  }

  async *_streamResponseChunks(messages, options, runManager) {
    this.validateCallOptions(options);

    const inference = this.getInference();
    const { prompt, tools } = this.renderPrompt(messages, options);

    let rawResponse = "";
    let emittedVisibleText = "";
    let done = false;
    let rejectedError;
    let wakeUp;
    const pendingText = [];

    const notify = () => {
      if (wakeUp) {
        const resolve = wakeUp;
        wakeUp = undefined;
        resolve();
      }
    };

    const streamPromise = inference
      .generateResponse(prompt, (partialResult, streamDone) => {
        rawResponse += partialResult;
        const visibleText = extractVisibleText(rawResponse);
        const delta = visibleText.slice(emittedVisibleText.length);

        if (delta) {
          emittedVisibleText = visibleText;
          pendingText.push(delta);
          notify();
        }

        if (streamDone) {
          done = true;
          notify();
        }
      })
      .then((maybeFullResponse) => {
        if (!rawResponse && typeof maybeFullResponse === "string") {
          rawResponse = maybeFullResponse;
        }

        done = true;
        notify();
      })
      .catch((error) => {
        rejectedError = error instanceof Error ? error : new Error(String(error));
        done = true;
        notify();
      });

    while (!done || pendingText.length > 0) {
      if (pendingText.length === 0) {
        await new Promise((resolve) => {
          wakeUp = resolve;
        });
      }

      while (pendingText.length > 0) {
        const text = pendingText.shift();
        if (!text) {
          continue;
        }

        yield new ChatGenerationChunk({
          text,
          message: new AIMessageChunk({ content: text }),
        });
        await runManager?.handleLLMNewToken(text);
      }
    }

    await streamPromise;

    if (rejectedError) {
      throw rejectedError;
    }

    const parsed = this.parseResponseArtifacts(rawResponse, tools);
    const finalVisibleDelta = parsed.visibleText.slice(emittedVisibleText.length);

    if (finalVisibleDelta) {
      yield new ChatGenerationChunk({
        text: finalVisibleDelta,
        message: new AIMessageChunk({ content: finalVisibleDelta }),
      });
      await runManager?.handleLLMNewToken(finalVisibleDelta);
    }

    if (parsed.toolCallChunks.length > 0 || (parsed.invalidToolCalls?.length ?? 0) > 0) {
      yield new ChatGenerationChunk({
        text: "",
        message: new AIMessageChunk({
          content: "",
          tool_call_chunks: parsed.toolCallChunks,
          invalid_tool_calls: parsed.invalidToolCalls,
          additional_kwargs: parsed.message.additional_kwargs,
        }),
      });
    }
  }

  parseResponseArtifacts(rawResponse, tools) {
    const parsedOutput = this.codec.parse(rawResponse);
    const toolNames = new Set(tools.map((tool) => tool.function.name));
    const parsedToolCalls = parsedOutput.toolCalls.map((toolCall, index) =>
      this.parseToolCallArtifact(toolCall, toolNames, index)
    );

    const toolCalls = parsedToolCalls.flatMap((toolCall) =>
      toolCall.parsed ? [toolCall.parsed] : []
    );
    const rawToolCalls = parsedToolCalls.flatMap((toolCall) =>
      toolCall.raw ? [toolCall.raw] : []
    );
    const toolCallChunks = parsedToolCalls.flatMap((toolCall) =>
      toolCall.chunk ? [toolCall.chunk] : []
    );
    const invalidToolCalls = parsedToolCalls.flatMap((toolCall) =>
      toolCall.invalid ? [toolCall.invalid] : []
    );

    return {
      visibleText: parsedOutput.visibleText,
      toolCallChunks,
      invalidToolCalls,
      message: new AIMessage({
        content: parsedOutput.content,
        tool_calls: toolCalls,
        invalid_tool_calls: invalidToolCalls,
        additional_kwargs: rawToolCalls.length > 0 ? { tool_calls: rawToolCalls } : {},
      }),
    };
  }

  renderPrompt(messages, options) {
    const tools = normalizeToolDefinitions(options?.tools);

    return {
      prompt: this.codec.render({
        messages,
        tools,
        toolChoice: normalizeToolChoice(options?.tool_choice),
      }),
      tools,
    };
  }

  validateCallOptions(options) {
    if (options?.stop?.length) {
      throw new Error("ChatMediaPipeGenAI does not support stop sequences.");
    }

    if (options?.signal) {
      throw new Error("ChatMediaPipeGenAI does not support AbortSignal cancellation.");
    }
  }

  parseToolCallArtifact(parsedToolCall, toolNames, index) {
    const toolCallId = parsedToolCall.id ?? `call_mediapipe_${index + 1}`;
    const name = parsedToolCall.name;
    const serializedArguments =
      typeof parsedToolCall.arguments === "string"
        ? parsedToolCall.arguments
        : JSON.stringify(parsedToolCall.arguments ?? {});
    const fallbackRawToolCall = {
      id: toolCallId,
      type: "function",
      function: {
        name: name ?? "",
        arguments: serializedArguments,
      },
    };

    if (parsedToolCall.error) {
      return { invalid: makeInvalidToolCall(fallbackRawToolCall, parsedToolCall.error) };
    }

    if (!name) {
      return {
        invalid: makeInvalidToolCall(
          fallbackRawToolCall,
          "Tool call is missing a tool name."
        ),
      };
    }

    if (!toolNames.has(name)) {
      return {
        invalid: makeInvalidToolCall(
          fallbackRawToolCall,
          `Tool "${name}" is not bound to this ChatMediaPipeGenAI instance.`
        ),
      };
    }

    try {
      return {
        parsed: parseToolCall(fallbackRawToolCall, { returnId: true }),
        raw: fallbackRawToolCall,
        chunk: {
          id: toolCallId,
          index,
          name,
          args: serializedArguments,
        },
      };
    } catch (error) {
      return {
        invalid: makeInvalidToolCall(
          fallbackRawToolCall,
          error instanceof Error
            ? error.message
            : "Failed to parse tool call arguments."
        ),
      };
    }
  }
}
