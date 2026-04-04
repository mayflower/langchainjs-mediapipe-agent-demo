import {
  GEMMA_TOOL_CALL_TAG_CLOSE,
  GEMMA_TOOL_CALL_TAG_OPEN,
  GEMMA_TOOL_RESPONSE_TAG_CLOSE,
  GEMMA_TOOL_RESPONSE_TAG_OPEN,
  GEMMA_TOOL_TAG_CLOSE,
  GEMMA_TOOL_TAG_OPEN,
  GEMMA_TURN_CLOSE,
  GEMMA_TURN_OPEN,
  extractTaggedBlocks,
  extractVisibleText,
  getMessageTextContent,
  getNormalizedRole,
  getReplayToolCalls,
  parseGemmaToolCall,
  serializeGemmaToolCall,
  serializeGemmaToolDeclaration,
  serializeGemmaToolResponse,
  stripHistoricalThoughts,
} from "./prompt_codec.js";

function renderTurn(role, content) {
  return `${GEMMA_TURN_OPEN}${role}\n${content}\n${GEMMA_TURN_CLOSE}`;
}

function renderToolChoiceInstruction(toolChoice) {
  if (!toolChoice || toolChoice === "auto") {
    return [
      "Use declared tools when they improve accuracy.",
      `Emit ${GEMMA_TOOL_CALL_TAG_OPEN}call:name{...}${GEMMA_TOOL_CALL_TAG_CLOSE} for tool requests.`,
      `Replay tool results as ${GEMMA_TOOL_RESPONSE_TAG_OPEN}response:name{...}${GEMMA_TOOL_RESPONSE_TAG_CLOSE} before any final answer.`,
    ].join(" ");
  }

  if (toolChoice === "none") {
    return "Do not call tools. Respond with natural language only.";
  }

  if (toolChoice === "required") {
    return "You must emit at least one tool call before providing a final answer.";
  }

  return `You must call the tool named ${toolChoice.function.name} before providing a final answer.`;
}

export class Gemma4Codec {
  thoughtTagName;

  constructor(fields = {}) {
    this.thoughtTagName = fields.thoughtTagName;
  }

  render(input) {
    const sections = [];

    if (input.tools.length > 0 && input.toolChoice !== "none") {
      const content = [
        ...input.tools.map(
          (tool) =>
            `${GEMMA_TOOL_TAG_OPEN}${serializeGemmaToolDeclaration(tool.function)}${GEMMA_TOOL_TAG_CLOSE}`
        ),
        renderToolChoiceInstruction(input.toolChoice),
      ].join("\n");

      sections.push(renderTurn("system", content));
    }

    for (let index = 0; index < input.messages.length; index += 1) {
      const message = input.messages[index];

      if (message._getType() === "tool") {
        sections.push(this.renderOrphanToolMessage(message));
        continue;
      }

      if (message._getType() !== "ai") {
        sections.push(renderTurn(getNormalizedRole(message), getMessageTextContent(message)));
        continue;
      }

      const replayToolCalls = getReplayToolCalls(message);
      const trailingToolMessages = [];
      let trailingIndex = index + 1;

      while (
        trailingIndex < input.messages.length &&
        input.messages[trailingIndex]._getType() === "tool"
      ) {
        trailingToolMessages.push(input.messages[trailingIndex]);
        trailingIndex += 1;
      }

      sections.push(
        renderTurn(
          "model",
          this.renderAssistantTurn(message, replayToolCalls, trailingToolMessages)
        )
      );
      index = trailingIndex - 1;
    }

    sections.push(`${GEMMA_TURN_OPEN}model\n`);
    return sections.join("\n");
  }

  parse(rawText) {
    const visibleText = extractVisibleText(rawText);
    const toolCalls = extractTaggedBlocks(
      rawText,
      GEMMA_TOOL_CALL_TAG_OPEN,
      GEMMA_TOOL_CALL_TAG_CLOSE
    ).map((block) => parseGemmaToolCall(block));

    return {
      content: visibleText.trim(),
      visibleText,
      toolCalls,
    };
  }

  renderAssistantTurn(message, replayToolCalls, trailingToolMessages) {
    const preserveThoughts =
      replayToolCalls.length > 0 || trailingToolMessages.length > 0;

    return [
      stripHistoricalThoughts(
        getMessageTextContent(message),
        this.thoughtTagName,
        preserveThoughts
      ),
      ...replayToolCalls.map(
        (toolCall) =>
          `${GEMMA_TOOL_CALL_TAG_OPEN}${serializeGemmaToolCall(toolCall)}${GEMMA_TOOL_CALL_TAG_CLOSE}`
      ),
      ...trailingToolMessages.map((toolMessage) => {
        const toolCallId =
          "tool_call_id" in toolMessage && typeof toolMessage.tool_call_id === "string"
            ? toolMessage.tool_call_id
            : undefined;
        const matchedToolCall = replayToolCalls.find(
          (toolCall) => toolCall.id === toolCallId
        );
        return this.renderToolResponseBlock(toolMessage, matchedToolCall?.name);
      }),
    ]
      .filter(Boolean)
      .join("\n");
  }

  renderOrphanToolMessage(message) {
    return renderTurn("model", this.renderToolResponseBlock(message));
  }

  renderToolResponseBlock(message, fallbackName) {
    const content = getMessageTextContent(message);

    return `${GEMMA_TOOL_RESPONSE_TAG_OPEN}${serializeGemmaToolResponse({
      id:
        "tool_call_id" in message && typeof message.tool_call_id === "string"
          ? message.tool_call_id
          : undefined,
      name: typeof message.name === "string" ? message.name : fallbackName ?? "tool",
      content,
    })}${GEMMA_TOOL_RESPONSE_TAG_CLOSE}`;
  }
}
