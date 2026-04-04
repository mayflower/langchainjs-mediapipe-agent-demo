import { ChatMessage, isAIMessage } from "@langchain/core/messages";

export const GEMMA_TURN_OPEN = "<|turn>";
export const GEMMA_TURN_CLOSE = "<turn|>";
export const GEMMA_TOOL_TAG_OPEN = "<|tool>";
export const GEMMA_TOOL_TAG_CLOSE = "<tool|>";
export const GEMMA_TOOL_CALL_TAG_OPEN = "<|tool_call>";
export const GEMMA_TOOL_CALL_TAG_CLOSE = "<tool_call|>";
export const GEMMA_TOOL_RESPONSE_TAG_OPEN = "<|tool_response>";
export const GEMMA_TOOL_RESPONSE_TAG_CLOSE = "<tool_response|>";
export const GEMMA_STRING_DELIMITER = `<|"|>`;

export function extractVisibleText(rawText) {
  let visibleText = removeTaggedBlocks(
    rawText,
    GEMMA_TOOL_CALL_TAG_OPEN,
    GEMMA_TOOL_CALL_TAG_CLOSE
  );
  visibleText = removeTaggedBlocks(
    visibleText,
    GEMMA_TOOL_RESPONSE_TAG_OPEN,
    GEMMA_TOOL_RESPONSE_TAG_CLOSE
  );

  if (hasTrailingToolBlock(rawText) && /(?:\r?\n)$/.test(visibleText)) {
    visibleText = visibleText.replace(/(?:\r?\n)$/, "");
  }

  return visibleText;
}

export function getMessageTextContent(message) {
  if (typeof message.content !== "string") {
    throw new Error("ChatMediaPipeGenAI v1 only supports string message content.");
  }

  return message.content;
}

export function stripHistoricalThoughts(content, thoughtTagName, preserveThoughts) {
  if (preserveThoughts || !thoughtTagName) {
    return content;
  }

  return removeTaggedBlocks(content, `<${thoughtTagName}>`, `</${thoughtTagName}>`).trim();
}

export function getNormalizedRole(message) {
  const messageType = message._getType();

  switch (messageType) {
    case "system":
    case "developer":
      return "system";
    case "human":
      return "user";
    case "ai":
      return "model";
    case "generic":
      if (!ChatMessage.isInstance(message)) {
        throw new Error("Invalid generic chat message.");
      }
      if (message.role === "assistant") {
        return "model";
      }
      if (message.role === "developer" || message.role === "system") {
        return "system";
      }
      return "user";
    default:
      throw new Error(
        `ChatMediaPipeGenAI does not support "${messageType}" messages in prompt rendering.`
      );
  }
}

export function getReplayToolCalls(message) {
  if (!isAIMessage(message)) {
    return [];
  }

  if (message.tool_calls?.length) {
    return message.tool_calls.map((toolCall) => ({
      id: toolCall.id,
      name: toolCall.name,
      arguments: toolCall.args,
    }));
  }

  const rawToolCalls = message.additional_kwargs?.tool_calls;
  if (!Array.isArray(rawToolCalls)) {
    return [];
  }

  return rawToolCalls.flatMap((rawToolCall) => {
    if (
      !rawToolCall ||
      typeof rawToolCall !== "object" ||
      !("function" in rawToolCall) ||
      !rawToolCall.function ||
      typeof rawToolCall.function !== "object" ||
      !("name" in rawToolCall.function) ||
      typeof rawToolCall.function.name !== "string"
    ) {
      return [];
    }

    let parsedArguments = {};
    if (
      "arguments" in rawToolCall.function &&
      typeof rawToolCall.function.arguments === "string"
    ) {
      try {
        parsedArguments = JSON.parse(rawToolCall.function.arguments);
      } catch {
        parsedArguments = rawToolCall.function.arguments;
      }
    }

    return [
      {
        id: "id" in rawToolCall && typeof rawToolCall.id === "string" ? rawToolCall.id : undefined,
        name: rawToolCall.function.name,
        arguments: parsedArguments,
      },
    ];
  });
}

export function serializeGemmaToolDeclaration(tool) {
  return `declaration:${tool.name}${serializeGemmaObject(
    omitUndefined({
      description: tool.description,
      parameters: tool.parameters,
    }),
    { schema: true }
  )}`;
}

export function serializeGemmaToolCall(toolCall) {
  const body =
    isPlainObject(toolCall.arguments) && !("id" in toolCall.arguments)
      ? omitUndefined({
          id: toolCall.id,
          ...toolCall.arguments,
        })
      : omitUndefined({
          id: toolCall.id,
          arguments: toolCall.arguments,
        });

  return `call:${toolCall.name}${serializeGemmaObject(body)}`;
}

export function serializeGemmaToolResponse(response) {
  const parsedContent = tryParseJson(response.content);
  const body =
    isPlainObject(parsedContent) && !("id" in parsedContent)
      ? omitUndefined({
          id: response.id,
          ...parsedContent,
        })
      : omitUndefined({
          id: response.id,
          content: parsedContent ?? response.content,
        });

  return `response:${response.name}${serializeGemmaObject(body)}`;
}

export function parseGemmaToolCall(text) {
  const trimmed = text.trim();

  if (trimmed.startsWith("call:")) {
    try {
      const { name, payload } = parseNamedGemmaBlock(trimmed, "call:");
      return {
        id: isPlainObject(payload) && typeof payload.id === "string" ? payload.id : undefined,
        name,
        arguments:
          isPlainObject(payload) && "arguments" in payload
            ? payload.arguments
            : isPlainObject(payload)
              ? omitUndefined({
                  ...payload,
                  id: undefined,
                })
              : payload,
      };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : "Invalid tool call block.",
      };
    }
  }

  try {
    const parsed = parseGemmaObject(trimmed);
    if (isPlainObject(parsed)) {
      return {
        id: typeof parsed.id === "string" ? parsed.id : undefined,
        name: typeof parsed.name === "string" ? parsed.name : undefined,
        arguments:
          "arguments" in parsed
            ? parsed.arguments
            : omitUndefined({
                ...parsed,
                id: undefined,
                name: undefined,
              }),
      };
    }

    return { arguments: parsed };
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : "Invalid tool call block.",
    };
  }
}

export function parseGemmaObject(text) {
  return JSON.parse(toJsonText(text));
}

function removeTaggedBlocks(text, openTag, closeTag) {
  let cursor = 0;
  let output = "";

  while (cursor < text.length) {
    const start = text.indexOf(openTag, cursor);
    if (start === -1) {
      output += text.slice(cursor);
      break;
    }

    output += text.slice(cursor, start);
    const end = text.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      break;
    }

    cursor = end + closeTag.length;
  }

  return output;
}

export function extractTaggedBlocks(text, openTag, closeTag) {
  const blocks = [];
  let cursor = 0;

  while (cursor < text.length) {
    const start = text.indexOf(openTag, cursor);
    if (start === -1) {
      break;
    }

    const end = text.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      break;
    }

    blocks.push(text.slice(start + openTag.length, end).trim());
    cursor = end + closeTag.length;
  }

  return blocks;
}

function parseNamedGemmaBlock(text, prefix) {
  const body = text.slice(prefix.length).trim();
  const objectStart = body.indexOf("{");
  if (objectStart === -1) {
    throw new Error("Gemma tool block is missing an object payload.");
  }

  const name = body.slice(0, objectStart).trim();
  if (!name) {
    throw new Error("Gemma tool block is missing a tool name.");
  }

  return {
    name,
    payload: parseGemmaObject(body.slice(objectStart)),
  };
}

function serializeGemmaObject(value, options = {}) {
  if (!isPlainObject(value)) {
    return serializeGemmaValue(value, options);
  }

  return `{${Object.entries(value)
    .filter(([, entryValue]) => entryValue !== undefined)
    .map(
      ([key, entryValue]) =>
        `${key}:${serializeGemmaValue(entryValue, {
          ...options,
          key,
        })}`
    )
    .join(",")}}`;
}

function serializeGemmaValue(value, options = {}) {
  if (value === null) {
    return "null";
  }

  if (Array.isArray(value)) {
    return `[${value.map((entry) => serializeGemmaValue(entry, options)).join(",")}]`;
  }

  if (isPlainObject(value)) {
    return serializeGemmaObject(value, options);
  }

  if (typeof value === "string") {
    return `${GEMMA_STRING_DELIMITER}${escapeGemmaString(
      options.schema && options.key === "type" ? normalizeSchemaType(value) : value
    )}${GEMMA_STRING_DELIMITER}`;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return JSON.stringify(value);
  }

  return `${GEMMA_STRING_DELIMITER}${escapeGemmaString(
    JSON.stringify(value)
  )}${GEMMA_STRING_DELIMITER}`;
}

function normalizeSchemaType(value) {
  switch (value.toLowerCase()) {
    case "string":
      return "STRING";
    case "number":
      return "NUMBER";
    case "integer":
      return "INTEGER";
    case "boolean":
      return "BOOLEAN";
    case "object":
      return "OBJECT";
    case "array":
      return "ARRAY";
    case "null":
      return "NULL";
    default:
      return value.toUpperCase();
  }
}

function escapeGemmaString(value) {
  return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function toJsonText(text) {
  return text
    .replaceAll(GEMMA_STRING_DELIMITER, '"')
    .replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)/g, '$1"$2"$3');
}

function tryParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function omitUndefined(value) {
  return Object.fromEntries(
    Object.entries(value).filter(([, entryValue]) => entryValue !== undefined)
  );
}

function isPlainObject(value) {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.getPrototypeOf(value) === Object.prototype
  );
}

function hasTrailingToolBlock(text) {
  const trimmed = text.trimEnd();
  const toolCallOpen = trimmed.lastIndexOf(GEMMA_TOOL_CALL_TAG_OPEN);
  const toolCallClose = trimmed.lastIndexOf(GEMMA_TOOL_CALL_TAG_CLOSE);
  const toolResponseOpen = trimmed.lastIndexOf(GEMMA_TOOL_RESPONSE_TAG_OPEN);
  const toolResponseClose = trimmed.lastIndexOf(GEMMA_TOOL_RESPONSE_TAG_CLOSE);

  return (
    (toolCallOpen !== -1 &&
      (toolCallOpen > toolCallClose || trimmed.endsWith("<tool_call|>"))) ||
    (toolResponseOpen !== -1 &&
      (toolResponseOpen > toolResponseClose ||
        trimmed.endsWith("<tool_response|>")))
  );
}
