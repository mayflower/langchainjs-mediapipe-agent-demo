import { AIMessage, HumanMessage, SystemMessage, ToolMessage, anthropicPromptCachingMiddleware, context, countTokensApproximately, createAgent, createMiddleware, humanInTheLoopMiddleware, todoListMiddleware, tool } from "langchain";
import { ChatAnthropic } from "@langchain/anthropic";
import { Command, REMOVE_ALL_MESSAGES, ReducedValue, StateSchema, getConfig, getCurrentTaskInput, getStore, isCommand } from "@langchain/langgraph";
import { z } from "zod/v4";
import micromatch from "micromatch";
import path, { basename } from "path";
import { HumanMessage as HumanMessage$1, RemoveMessage, getBufferString } from "@langchain/core/messages";
import { z as z$1 } from "zod";
import yaml from "yaml";
import { Client } from "@langchain/langgraph-sdk";
import { ContextOverflowError } from "@langchain/core/errors";
import { initChatModel } from "langchain/chat_models/universal";
//#region src/backends/utils.ts
/**
* Shared utility functions for memory backend implementations.
*
* This module contains both user-facing string formatters and structured
* helpers used by backends and the composite router. Structured helpers
* enable composition without fragile string parsing.
*/
const MAX_LINE_LENGTH = 5e3;
const TOOL_RESULT_TOKEN_LIMIT = 2e4;
const TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]";
const MIME_TYPES = {
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".gif": "image/gif",
	".webp": "image/webp",
	".svg": "image/svg+xml",
	".heic": "image/heic",
	".heif": "image/heif",
	".mp3": "audio/mpeg",
	".wav": "audio/wav",
	".aiff": "audio/aiff",
	".aac": "audio/aac",
	".ogg": "audio/ogg",
	".flac": "audio/flac",
	".mp4": "video/mp4",
	".webm": "video/webm",
	".mpeg": "video/mpeg",
	".mov": "video/quicktime",
	".avi": "video/x-msvideo",
	".flv": "video/x-flv",
	".mpg": "video/mpeg",
	".wmv": "video/x-ms-wmv",
	".3gpp": "video/3gpp",
	".pdf": "application/pdf",
	".ppt": "application/vnd.ms-powerpoint",
	".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
};
/**
* Sanitize tool_call_id to prevent path traversal and separator issues.
*
* Replaces dangerous characters (., /, \) with underscores.
*/
function sanitizeToolCallId(toolCallId) {
	return toolCallId.replace(/\./g, "_").replace(/\//g, "_").replace(/\\/g, "_");
}
/**
* Format file content with line numbers (cat -n style).
*
* Chunks lines longer than MAX_LINE_LENGTH with continuation markers (e.g., 5.1, 5.2).
*
* @param content - File content as string or list of lines
* @param startLine - Starting line number (default: 1)
* @returns Formatted content with line numbers and continuation markers
*/
function formatContentWithLineNumbers(content, startLine = 1) {
	let lines;
	if (typeof content === "string") {
		lines = content.split("\n");
		if (lines.length > 0 && lines[lines.length - 1] === "") lines = lines.slice(0, -1);
	} else lines = content;
	const resultLines = [];
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const lineNum = i + startLine;
		if (line.length <= 5e3) resultLines.push(`${lineNum.toString().padStart(6)}\t${line}`);
		else {
			const numChunks = Math.ceil(line.length / MAX_LINE_LENGTH);
			for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
				const start = chunkIdx * MAX_LINE_LENGTH;
				const end = Math.min(start + MAX_LINE_LENGTH, line.length);
				const chunk = line.substring(start, end);
				if (chunkIdx === 0) resultLines.push(`${lineNum.toString().padStart(6)}\t${chunk}`);
				else {
					const continuationMarker = `${lineNum}.${chunkIdx}`;
					resultLines.push(`${continuationMarker.padStart(6)}\t${chunk}`);
				}
			}
		}
	}
	return resultLines.join("\n");
}
/**
* Convert FileData to plain string content.
*
* @param fileData - FileData object with 'content' key
* @returns Content as string with lines joined by newlines
*/
function fileDataToString(fileData) {
	if (Array.isArray(fileData.content)) return fileData.content.join("\n");
	if (typeof fileData.content === "string") return fileData.content;
	throw new Error("Cannot convert binary FileData to string");
}
/**
* Type guard to check if FileData contains binary content (Uint8Array).
*
* @param data - FileData to check
* @returns True if the content is a Uint8Array (binary)
*/
function isFileDataBinary(data) {
	return ArrayBuffer.isView(data.content);
}
/**
* Create a FileData object.
*
* Defaults to v2 format (content as single string). Pass `fileFormat: "v1"` for
* backward compatibility with older readers during a rolling deployment.
* Binary content (Uint8Array) is only supported with v2.
*
* @param content - File content as a string or binary Uint8Array (v2 only)
* @param createdAt - Optional creation timestamp (ISO format), defaults to now
* @param fileFormat - Storage format: "v2" (default) or "v1" (legacy line array)
* @returns FileData in the requested format
*/
function createFileData(content, createdAt, fileFormat = "v2", mimeType) {
	const now = (/* @__PURE__ */ new Date()).toISOString();
	if (fileFormat === "v1" && ArrayBuffer.isView(content)) throw new Error("Binary data is not supported with v1 file formats. Please use v2 file format");
	if (fileFormat === "v2") {
		if (ArrayBuffer.isView(content)) return {
			content: new Uint8Array(content.buffer, content.byteOffset, content.byteLength),
			mimeType: mimeType ?? "application/octet-stream",
			created_at: createdAt || now,
			modified_at: now
		};
		return {
			content,
			mimeType: mimeType ?? "text/plain",
			created_at: createdAt || now,
			modified_at: now
		};
	}
	return {
		content: typeof content === "string" ? content.split("\n") : content,
		created_at: createdAt || now,
		modified_at: now
	};
}
/**
* Update FileData with new content, preserving creation timestamp.
*
* @param fileData - Existing FileData object
* @param content - New content as string
* @returns Updated FileData object
*/
function updateFileData(fileData, content) {
	const now = (/* @__PURE__ */ new Date()).toISOString();
	if (isFileDataV1(fileData)) return {
		content: typeof content === "string" ? content.split("\n") : content,
		created_at: fileData.created_at,
		modified_at: now
	};
	return {
		content,
		mimeType: fileData.mimeType,
		created_at: fileData.created_at,
		modified_at: now
	};
}
/**
* Perform string replacement with occurrence validation.
*
* @param content - Original content
* @param oldString - String to replace
* @param newString - Replacement string
* @param replaceAll - Whether to replace all occurrences
* @returns Tuple of [new_content, occurrences] on success, or error message string
*
* Special case: When both content and oldString are empty, this sets the initial
* content to newString. This allows editing empty files by treating empty oldString
* as "set initial content" rather than "replace nothing".
*/
function performStringReplacement(content, oldString, newString, replaceAll) {
	if (content === "" && oldString === "") return [newString, 0];
	if (oldString === "") return "Error: oldString cannot be empty when file has content";
	const occurrences = content.split(oldString).length - 1;
	if (occurrences === 0) return `Error: String not found in file: '${oldString}'`;
	if (occurrences > 1 && !replaceAll) return `Error: String '${oldString}' has multiple occurrences (appears ${occurrences} times) in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context.`;
	return [content.split(oldString).join(newString), occurrences];
}
/**
* Truncate list or string result if it exceeds token limit (rough estimate: 4 chars/token).
*/
function truncateIfTooLong(result) {
	if (Array.isArray(result)) {
		const totalChars = result.reduce((sum, item) => sum + item.length, 0);
		if (totalChars > 2e4 * 4) {
			const truncateAt = Math.floor(result.length * TOOL_RESULT_TOKEN_LIMIT * 4 / totalChars);
			return [...result.slice(0, truncateAt), TRUNCATION_GUIDANCE];
		}
		return result;
	}
	if (result.length > 2e4 * 4) return result.substring(0, TOOL_RESULT_TOKEN_LIMIT * 4) + "\n... [results truncated, try being more specific with your parameters]";
	return result;
}
/**
* Validate and normalize a directory path.
*
* Ensures paths are safe to use by preventing directory traversal attacks
* and enforcing consistent formatting. All paths are normalized to use
* forward slashes and start with a leading slash.
*
* This function is designed for virtual filesystem paths and rejects
* Windows absolute paths (e.g., C:/..., F:/...) to maintain consistency
* and prevent path format ambiguity.
*
* @param path - Path to validate
* @returns Normalized path starting with / and ending with /
* @throws Error if path is invalid
*
* @example
* ```typescript
* validatePath("foo/bar")  // Returns: "/foo/bar/"
* validatePath("/./foo//bar")  // Returns: "/foo/bar/"
* validatePath("../etc/passwd")  // Throws: Path traversal not allowed
* validatePath("C:\\Users\\file")  // Throws: Windows absolute paths not supported
* ```
*/
function validatePath(path) {
	const pathStr = path || "/";
	if (!pathStr || pathStr.trim() === "") throw new Error("Path cannot be empty");
	let normalized = pathStr.startsWith("/") ? pathStr : "/" + pathStr;
	if (!normalized.endsWith("/")) normalized += "/";
	return normalized;
}
/**
* Search files dict for paths matching glob pattern.
*
* @param files - Dictionary of file paths to FileData
* @param pattern - Glob pattern (e.g., `*.py`, `**\/*.ts`)
* @param path - Base path to search from
* @returns Newline-separated file paths, sorted by modification time (most recent first).
*          Returns "No files found" if no matches.
*
* @example
* ```typescript
* const files = {"/src/main.py": FileData(...), "/test.py": FileData(...)};
* globSearchFiles(files, "*.py", "/");
* // Returns: "/test.py\n/src/main.py" (sorted by modified_at)
* ```
*/
function globSearchFiles(files, pattern, path = "/") {
	let normalizedPath;
	try {
		normalizedPath = validatePath(path);
	} catch {
		return "No files found";
	}
	const filtered = Object.fromEntries(Object.entries(files).filter(([fp]) => fp.startsWith(normalizedPath)));
	const effectivePattern = pattern;
	const matches = [];
	for (const [filePath, fileData] of Object.entries(filtered)) {
		let relative = filePath.substring(normalizedPath.length);
		if (relative.startsWith("/")) relative = relative.substring(1);
		if (!relative) {
			const parts = filePath.split("/");
			relative = parts[parts.length - 1] || "";
		}
		if (micromatch.isMatch(relative, effectivePattern, {
			dot: true,
			nobrace: false
		})) matches.push([filePath, fileData.modified_at]);
	}
	matches.sort((a, b) => b[1].localeCompare(a[1]));
	if (matches.length === 0) return "No files found";
	return matches.map(([fp]) => fp).join("\n");
}
/**
* Return structured grep matches from an in-memory files mapping.
*
* Performs literal text search (not regex). Binary files are skipped.
* Returns an empty array when no matches are found or on invalid input.
*/
function grepMatchesFromFiles(files, pattern, path = null, glob = null) {
	let normalizedPath;
	try {
		normalizedPath = validatePath(path);
	} catch {
		return [];
	}
	let filtered = Object.fromEntries(Object.entries(files).filter(([fp]) => fp.startsWith(normalizedPath)));
	if (glob) filtered = Object.fromEntries(Object.entries(filtered).filter(([fp]) => micromatch.isMatch(basename(fp), glob, {
		dot: true,
		nobrace: false
	})));
	const matches = [];
	for (const [filePath, fileData] of Object.entries(filtered)) {
		if (!isTextMimeType(migrateToFileDataV2(fileData, filePath).mimeType)) continue;
		const lines = fileDataToString(fileData).split("\n");
		for (let i = 0; i < lines.length; i++) {
			const line = lines[i];
			const lineNum = i + 1;
			if (line.includes(pattern)) matches.push({
				path: filePath,
				line: lineNum,
				text: line
			});
		}
	}
	return matches;
}
/**
* Determine MIME type from a file path's extension.
*
* Returns "text/plain" for unknown extensions.
*
* @param filePath - File path to inspect
* @returns MIME type string (e.g., "image/png", "text/plain")
*/
function getMimeType(filePath) {
	return MIME_TYPES[path.extname(filePath).toLocaleLowerCase()] || "text/plain";
}
/**
* Check whether a MIME type represents text content.
*
* @param mimeType - MIME type string to check
* @returns True if the MIME type is text-based
*/
function isTextMimeType(mimeType) {
	return mimeType.startsWith("text/") || mimeType === "application/json" || mimeType === "application/javascript" || mimeType === "image/svg+xml";
}
/**
* Type guard to check if FileData is v1 format (content as line array).
*
* @param data - FileData to check
* @returns True if data is FileDataV1
*/
function isFileDataV1(data) {
	return Array.isArray(data.content);
}
/**
* Convert FileData to v2 format, joining v1 line arrays into a single string.
*
* If the data is already v2, returns it unchanged.
*
* @param data - FileData in either format
* @returns FileDataV2 with content as string (text) or Uint8Array (binary)
*/
function migrateToFileDataV2(data, filePath) {
	if (isFileDataV1(data)) return {
		content: data.content.join("\n"),
		mimeType: getMimeType(filePath),
		created_at: data.created_at,
		modified_at: data.modified_at
	};
	if (!("mimeType" in data) || !data.mimeType) return {
		...data,
		mimeType: getMimeType(filePath)
	};
	return data;
}
/**
* Adapt a v1 {@link BackendProtocol} to {@link BackendProtocolV2}.
*
* If the backend already implements v2, it is returned as-is.
* For v1 backends, wraps returns in Result types:
* - `read()` string returns wrapped in {@link ReadResult}
* - `readRaw()` FileData returns wrapped in {@link ReadRawResult}
* - `grep()` returns wrapped in {@link GrepResult}
* - `ls()` FileInfo[] returns wrapped in {@link LsResult}
* - `glob()` FileInfo[] returns wrapped in {@link GlobResult}
*
* Note: For sandbox instances, use {@link adaptSandboxProtocol} instead.
*
* @param backend - Backend instance (v1 or v2)
* @returns BackendProtocolV2-compatible backend
*/
function adaptBackendProtocol(backend) {
	return {
		async ls(path) {
			const result = await ("ls" in backend ? backend.ls(path) : backend.lsInfo(path));
			if (Array.isArray(result)) return { files: result };
			return result;
		},
		async readRaw(filePath) {
			const result = await backend.readRaw(filePath);
			if ("data" in result || "error" in result) return result;
			return { data: migrateToFileDataV2(result, filePath) };
		},
		async glob(pattern, path) {
			const result = await ("glob" in backend ? backend.glob(pattern, path) : backend.globInfo(pattern, path));
			if (Array.isArray(result)) return { files: result };
			return result;
		},
		write: (filePath, content) => backend.write(filePath, content),
		edit: (filePath, oldString, newString, replaceAll) => backend.edit(filePath, oldString, newString, replaceAll),
		uploadFiles: backend.uploadFiles ? (files) => backend.uploadFiles(files) : void 0,
		downloadFiles: backend.downloadFiles ? (paths) => backend.downloadFiles(paths) : void 0,
		async read(filePath, offset, limit) {
			const result = await backend.read(filePath, offset, limit);
			if (typeof result === "string") return { content: result };
			return result;
		},
		async grep(pattern, path, glob) {
			const result = await ("grep" in backend ? backend.grep(pattern, path, glob) : backend.grepRaw(pattern, path, glob));
			if (Array.isArray(result)) return { matches: result };
			if (typeof result === "string") return { error: result };
			return result;
		}
	};
}
/**
* Adapt a sandbox backend from v1 to v2 interface.
*
* This extends {@link adaptBackendProtocol} to also preserve sandbox-specific
* properties from {@link SandboxBackendProtocol}: `execute` and `id`.
*
* @param sandbox - Sandbox backend (v1 or v2)
* @returns SandboxBackendProtocolV2-compatible sandbox
*/
function adaptSandboxProtocol(sandbox) {
	const adapted = adaptBackendProtocol(sandbox);
	adapted.execute = (cmd) => sandbox.execute(cmd);
	Object.defineProperty(adapted, "id", {
		value: sandbox.id,
		enumerable: true,
		configurable: true
	});
	return adapted;
}
//#endregion
//#region src/backends/protocol.ts
/**
* Type guard to check if a backend supports execution.
*
* @param backend - Backend instance to check
* @returns True if the backend implements SandboxBackendProtocolV2
*/
function isSandboxBackend(backend) {
	return backend != null && typeof backend === "object" && typeof backend.execute === "function" && typeof backend.id === "string" && backend.id !== "";
}
/**
* Type guard to check if a backend is a sandbox protocol (v1 or v2).
*
* Checks for the presence of `execute` function and `id` string,
* which are the defining features of sandbox protocols.
*
* @param backend - Backend instance to check
* @returns True if the backend implements sandbox protocol (v1 or v2)
*/
function isSandboxProtocol(backend) {
	return backend != null && typeof backend === "object" && typeof backend.execute === "function" && typeof backend.id === "string" && backend.id !== "";
}
const SANDBOX_ERROR_SYMBOL = Symbol.for("sandbox.error");
/**
* Custom error class for sandbox operations.
*
* @param message - Human-readable error description
* @param code - Structured error code for programmatic handling
* @returns SandboxError with message and code
*
* @example
* ```typescript
* try {
*   await sandbox.execute("some command");
* } catch (error) {
*   if (error instanceof SandboxError) {
*     switch (error.code) {
*       case "NOT_INITIALIZED":
*         await sandbox.initialize();
*         break;
*       case "COMMAND_TIMEOUT":
*         console.error("Command took too long");
*         break;
*       default:
*         throw error;
*     }
*   }
* }
* ```
*/
var SandboxError = class SandboxError extends Error {
	/** Symbol for identifying sandbox error instances */
	[SANDBOX_ERROR_SYMBOL] = true;
	/** Error name for instanceof checks and logging */
	name = "SandboxError";
	/**
	* Creates a new SandboxError.
	*
	* @param message - Human-readable error description
	* @param code - Structured error code for programmatic handling
	*/
	constructor(message, code, cause) {
		super(message);
		this.code = code;
		this.cause = cause;
		Object.setPrototypeOf(this, SandboxError.prototype);
	}
	static isInstance(error) {
		return typeof error === "object" && error !== null && error[SANDBOX_ERROR_SYMBOL] === true;
	}
};
/**
* Resolve a backend instance or await a {@link BackendFactory}.
*
* Accepts {@link BackendRuntime} or {@link ToolRuntime} — store typing differs
* between LangGraph checkpoint stores and core `ToolRuntime`; factories receive
* a value that is structurally compatible at runtime.
*
* @internal
*/
async function resolveBackend(backend, runtime) {
	if (typeof backend === "function") {
		const resolved = await backend(runtime);
		return isSandboxProtocol(resolved) ? adaptSandboxProtocol(resolved) : adaptBackendProtocol(resolved);
	}
	return isSandboxProtocol(backend) ? adaptSandboxProtocol(backend) : adaptBackendProtocol(backend);
}
//#endregion
//#region src/backends/state.ts
const PREGEL_SEND_KEY = "__pregel_send";
/**
* Backend that stores files in agent state (ephemeral).
*
* Uses LangGraph's state management and checkpointing. Files persist within
* a conversation thread but not across threads. State is automatically
* checkpointed after each agent step.
*
* Special handling: Since LangGraph state must be updated via Command objects
* (not direct mutation), operations return filesUpdate in WriteResult/EditResult
* for the middleware to apply via Command.
*/
var StateBackend = class {
	runtime;
	fileFormat;
	constructor(runtimeOrOptions, options) {
		if (runtimeOrOptions != null && typeof runtimeOrOptions === "object" && "state" in runtimeOrOptions) {
			this.runtime = runtimeOrOptions;
			this.fileFormat = options?.fileFormat ?? "v2";
		} else {
			this.runtime = void 0;
			this.fileFormat = runtimeOrOptions?.fileFormat ?? "v2";
		}
	}
	/**
	* Whether this instance was constructed with the legacy factory pattern.
	*
	* When true, state is read from the injected `runtime` and `filesUpdate`
	* is returned to the caller. When false, state is read from LangGraph's
	* execution context and updates are sent via `__pregel_send`.
	*/
	get isLegacy() {
		return this.runtime !== void 0;
	}
	/**
	* Get files from current state.
	*
	* In legacy mode, reads from the injected {@link BackendRuntime}.
	* In zero-arg mode, reads from the LangGraph execution context via
	* {@link getCurrentTaskInput}.
	*/
	getFiles() {
		if (this.runtime) return this.runtime.state.files || {};
		return getCurrentTaskInput()?.files || {};
	}
	/**
	* Push a files state update through LangGraph's internal send channel.
	*
	* In zero-arg mode, sends the update via the `__pregel_send` function
	* from {@link getConfig}, mirroring Python's `CONFIG_KEY_SEND`.
	* In legacy mode, this is a no-op — the caller uses `filesUpdate`
	* from the return value instead.
	*
	* @param update - Map of file paths to their updated {@link FileData}
	*/
	sendFilesUpdate(update) {
		if (this.isLegacy) return;
		const send = getConfig().configurable?.[PREGEL_SEND_KEY];
		if (typeof send === "function") send([["files", update]]);
	}
	/**
	* List files and directories in the specified directory (non-recursive).
	*
	* @param path - Absolute path to directory
	* @returns LsResult with list of FileInfo objects on success or error on failure.
	*          Directories have a trailing / in their path and is_dir=true.
	*/
	ls(path) {
		const files = this.getFiles();
		const infos = [];
		const subdirs = /* @__PURE__ */ new Set();
		const normalizedPath = path.endsWith("/") ? path : path + "/";
		for (const [k, fd] of Object.entries(files)) {
			if (!k.startsWith(normalizedPath)) continue;
			const relative = k.substring(normalizedPath.length);
			if (relative.includes("/")) {
				const subdirName = relative.split("/")[0];
				subdirs.add(normalizedPath + subdirName + "/");
				continue;
			}
			const size = isFileDataV1(fd) ? fd.content.join("\n").length : isFileDataBinary(fd) ? fd.content.byteLength : fd.content.length;
			infos.push({
				path: k,
				is_dir: false,
				size,
				modified_at: fd.modified_at
			});
		}
		for (const subdir of Array.from(subdirs).sort()) infos.push({
			path: subdir,
			is_dir: true,
			size: 0,
			modified_at: ""
		});
		infos.sort((a, b) => a.path.localeCompare(b.path));
		return { files: infos };
	}
	/**
	* Read file content.
	*
	* Text files are paginated by line offset/limit.
	* Binary files return full Uint8Array content (offset/limit ignored).
	*
	* @param filePath - Absolute file path
	* @param offset - Line offset to start reading from (0-indexed)
	* @param limit - Maximum number of lines to read
	* @returns ReadResult with content on success or error on failure
	*/
	read(filePath, offset = 0, limit = 500) {
		const fileData = this.getFiles()[filePath];
		if (!fileData) return { error: `File '${filePath}' not found` };
		const fileDataV2 = migrateToFileDataV2(fileData, filePath);
		if (!isTextMimeType(fileDataV2.mimeType)) return {
			content: fileDataV2.content,
			mimeType: fileDataV2.mimeType
		};
		if (typeof fileDataV2.content !== "string") return { error: `File '${filePath}' has binary content but text MIME type` };
		return {
			content: fileDataV2.content.split("\n").slice(offset, offset + limit).join("\n"),
			mimeType: fileDataV2.mimeType
		};
	}
	/**
	* Read file content as raw FileData.
	*
	* @param filePath - Absolute file path
	* @returns ReadRawResult with raw file data on success or error on failure
	*/
	readRaw(filePath) {
		const fileData = this.getFiles()[filePath];
		if (!fileData) return { error: `File '${filePath}' not found` };
		return { data: fileData };
	}
	/**
	* Create a new file with content.
	* Returns WriteResult with filesUpdate to update LangGraph state.
	*/
	write(filePath, content) {
		if (filePath in this.getFiles()) return { error: `Cannot write to ${filePath} because it already exists. Read and then make an edit, or write to a new path.` };
		const mimeType = getMimeType(filePath);
		const newFileData = createFileData(content, void 0, this.fileFormat, mimeType);
		const update = { [filePath]: newFileData };
		if (!this.isLegacy) {
			this.sendFilesUpdate(update);
			return { path: filePath };
		}
		return {
			path: filePath,
			filesUpdate: { [filePath]: newFileData }
		};
	}
	/**
	* Edit a file by replacing string occurrences.
	* Returns EditResult with filesUpdate and occurrences.
	*/
	edit(filePath, oldString, newString, replaceAll = false) {
		const fileData = this.getFiles()[filePath];
		if (!fileData) return { error: `Error: File '${filePath}' not found` };
		const result = performStringReplacement(fileDataToString(fileData), oldString, newString, replaceAll);
		if (typeof result === "string") return { error: result };
		const [newContent, occurrences] = result;
		const newFileData = updateFileData(fileData, newContent);
		const update = { [filePath]: newFileData };
		if (!this.isLegacy) {
			this.sendFilesUpdate(update);
			return {
				path: filePath,
				occurrences
			};
		}
		return {
			path: filePath,
			filesUpdate: { [filePath]: newFileData },
			occurrences
		};
	}
	/**
	* Search file contents for a literal text pattern.
	* Binary files are skipped.
	*/
	grep(pattern, path = "/", glob = null) {
		return { matches: grepMatchesFromFiles(this.getFiles(), pattern, path, glob) };
	}
	/**
	* Structured glob matching returning FileInfo objects.
	*/
	glob(pattern, path = "/") {
		const files = this.getFiles();
		const result = globSearchFiles(files, pattern, path);
		if (result === "No files found") return { files: [] };
		const paths = result.split("\n");
		const infos = [];
		for (const p of paths) {
			const fd = files[p];
			const size = fd ? isFileDataV1(fd) ? fd.content.join("\n").length : isFileDataBinary(fd) ? fd.content.byteLength : fd.content.length : 0;
			infos.push({
				path: p,
				is_dir: false,
				size,
				modified_at: fd?.modified_at || ""
			});
		}
		return { files: infos };
	}
	/**
	* Upload multiple files.
	*
	* Note: Since LangGraph state must be updated via Command objects,
	* the caller must apply filesUpdate via Command after calling this method.
	*
	* @param files - List of [path, content] tuples to upload
	* @returns List of FileUploadResponse objects, one per input file
	*/
	uploadFiles(files) {
		const responses = [];
		const updates = {};
		for (const [path, content] of files) try {
			const mimeType = getMimeType(path);
			if (this.fileFormat === "v2" && !isTextMimeType(mimeType)) updates[path] = createFileData(content, void 0, "v2", mimeType);
			else updates[path] = createFileData(new TextDecoder().decode(content), void 0, this.fileFormat, mimeType);
			responses.push({
				path,
				error: null
			});
		} catch {
			responses.push({
				path,
				error: "invalid_path"
			});
		}
		if (!this.isLegacy) {
			if (Object.keys(updates).length > 0) this.sendFilesUpdate(updates);
			return responses;
		}
		const result = responses;
		result.filesUpdate = updates;
		return result;
	}
	/**
	* Download multiple files.
	*
	* @param paths - List of file paths to download
	* @returns List of FileDownloadResponse objects, one per input path
	*/
	downloadFiles(paths) {
		const files = this.getFiles();
		const responses = [];
		for (const path of paths) {
			const fileData = files[path];
			if (!fileData) {
				responses.push({
					path,
					content: null,
					error: "file_not_found"
				});
				continue;
			}
			const fileDataV2 = migrateToFileDataV2(fileData, path);
			if (typeof fileDataV2.content === "string") {
				const content = new TextEncoder().encode(fileDataV2.content);
				responses.push({
					path,
					content,
					error: null
				});
			} else responses.push({
				path,
				content: fileDataV2.content,
				error: null
			});
		}
		return responses;
	}
};
//#endregion
//#region src/middleware/fs.ts
/**
* Middleware for providing filesystem tools to an agent.
*
* Provides ls, read_file, write_file, edit_file, glob, and grep tools with support for:
* - Pluggable backends (StateBackend, StoreBackend, FilesystemBackend, CompositeBackend)
* - Tool result eviction for large outputs
*/
const INT_FORMATTER = new Intl.NumberFormat("en-US");
/**
* Tools that should be excluded from the large result eviction logic.
*
* This array contains tools that should NOT have their results evicted to the filesystem
* when they exceed token limits. Tools are excluded for different reasons:
*
* 1. Tools with built-in truncation (ls, glob, grep):
*    These tools truncate their own output when it becomes too large. When these tools
*    produce truncated output due to many matches, it typically indicates the query
*    needs refinement rather than full result preservation. In such cases, the truncated
*    matches are potentially more like noise and the LLM should be prompted to narrow
*    its search criteria instead.
*
* 2. Tools with problematic truncation behavior (read_file):
*    read_file is tricky to handle as the failure mode here is single long lines
*    (e.g., imagine a jsonl file with very long payloads on each line). If we try to
*    truncate the result of read_file, the agent may then attempt to re-read the
*    truncated file using read_file again, which won't help.
*
* 3. Tools that never exceed limits (edit_file, write_file):
*    These tools return minimal confirmation messages and are never expected to produce
*    output large enough to exceed token limits, so checking them would be unnecessary.
*/
/**
* All tool names registered by FilesystemMiddleware.
* This is the single source of truth — used by createDeepAgent to detect
* collisions with user-supplied tools at construction time.
*/
const FILESYSTEM_TOOL_NAMES = [
	"ls",
	"read_file",
	"write_file",
	"edit_file",
	"glob",
	"grep",
	"execute"
];
const TOOLS_EXCLUDED_FROM_EVICTION = [
	"ls",
	"glob",
	"grep",
	"read_file",
	"edit_file",
	"write_file"
];
/**
* Maximum size for binary (non-text) files read via read_file, in bytes.
* Base64-encoded content is ~33% larger, so 10MB raw ≈ 13.3MB in context.
* This keeps inline multimodal payloads within all major provider limits.
*/
const MAX_BINARY_READ_SIZE_BYTES = 10 * 1024 * 1024;
/**
* Template for truncation message in read_file.
* {file_path} will be filled in at runtime.
*/
const READ_FILE_TRUNCATION_MSG = `

[Output was truncated due to size limits. The file content is very large. Consider reformatting the file to make it easier to navigate. For example, if this is JSON, use execute(command='jq . {file_path}') to pretty-print it with line breaks. For other formats, you can use appropriate formatting tools to split long lines.]`;
/**
* Message template for evicted tool results.
*/
const TOO_LARGE_TOOL_MSG = context`
  Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}
  You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.
  You can do this by specifying an offset and limit in the read_file tool call.
  For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

  Here is a preview showing the head and tail of the result (lines of the form
  ... [N lines truncated] ...
  indicate omitted lines in the middle of the content):

  {content_sample}
`;
/**
* Message template for evicted HumanMessages.
*/
const TOO_LARGE_HUMAN_MSG = `Message content too large and was saved to the filesystem at: {file_path}

You can read the full content using the read_file tool with pagination (offset and limit parameters).

Here is a preview showing the head and tail of the content:

{content_sample}`;
/**
* Extract text content from a message.
*
* For string content, returns it directly. For array content (mixed block types
* like text + image), joins all text blocks. Returns empty string if no text found.
*/
function extractTextFromMessage(message) {
	if (typeof message.content === "string") return message.content;
	if (Array.isArray(message.content)) return message.content.filter((block) => block.type === "text" && typeof block.text === "string").map((block) => block.text).join("\n");
	return String(message.content);
}
/**
* Build replacement content for an evicted HumanMessage, preserving non-text blocks.
*
* For plain string content, returns the replacement text directly. For list content
* with mixed block types (e.g., text + image), replaces all text blocks with a single
* text block containing the replacement text while keeping non-text blocks intact.
*/
function buildEvictedHumanContent(message, replacementText) {
	if (typeof message.content === "string") return replacementText;
	if (Array.isArray(message.content)) {
		const mediaBlocks = message.content.filter((block) => typeof block === "object" && block !== null && block.type !== "text");
		if (mediaBlocks.length === 0) return replacementText;
		return [{
			type: "text",
			text: replacementText
		}, ...mediaBlocks];
	}
	return replacementText;
}
/**
* Build a truncated HumanMessage for the model request.
*
* Computes a preview from the full content still in state and returns a
* lightweight replacement the model will see. Pure string computation — no
* backend I/O.
*/
function buildTruncatedHumanMessage(message, filePath) {
	const contentSample = createContentPreview(extractTextFromMessage(message));
	return new HumanMessage({
		content: buildEvictedHumanContent(message, TOO_LARGE_HUMAN_MSG.replace("{file_path}", filePath).replace("{content_sample}", contentSample)),
		id: message.id,
		additional_kwargs: { ...message.additional_kwargs },
		response_metadata: { ...message.response_metadata }
	});
}
/**
* Create a preview of content showing head and tail with truncation marker.
*
* @param contentStr - The full content string to preview.
* @param headLines - Number of lines to show from the start (default: 5).
* @param tailLines - Number of lines to show from the end (default: 5).
* @returns Formatted preview string with line numbers.
*/
function createContentPreview(contentStr, headLines = 5, tailLines = 5) {
	const lines = contentStr.split("\n");
	if (lines.length <= headLines + tailLines) return formatContentWithLineNumbers(lines.map((line) => line.substring(0, 1e3)), 1);
	const head = lines.slice(0, headLines).map((line) => line.substring(0, 1e3));
	const tail = lines.slice(-tailLines).map((line) => line.substring(0, 1e3));
	const headSample = formatContentWithLineNumbers(head, 1);
	const truncationNotice = `\n... [${lines.length - headLines - tailLines} lines truncated] ...\n`;
	const tailSample = formatContentWithLineNumbers(tail, lines.length - tailLines + 1);
	return headSample + truncationNotice + tailSample;
}
/**
* Zod schema for legacy FileDataV1 (content as line array).
*/
const FileDataV1Schema = z.object({
	content: z.array(z.string()),
	created_at: z.string(),
	modified_at: z.string()
});
/**
* Zod schema for FileDataV2 (content as string for text or Uint8Array for binary).
*/
const FileDataV2Schema = z.object({
	content: z.union([z.string(), z.instanceof(Uint8Array)]),
	mimeType: z.string(),
	created_at: z.string(),
	modified_at: z.string()
});
/**
* Zod v3 schema for FileData (re-export from backends)
*/
const FileDataSchema = z.union([FileDataV1Schema, FileDataV2Schema]);
/**
* Reducer for files state that merges file updates with support for deletions.
* When a file value is null, the file is deleted from state.
* When a file value is non-null, it is added or updated in state.
*
* This reducer enables concurrent updates from parallel subagents by properly
* merging their file changes instead of requiring LastValue semantics.
*
* @param current - The current files record (from state)
* @param update - The new files record (from a subagent update), with null values for deletions
* @returns Merged files record with deletions applied
*/
function fileDataReducer(current, update) {
	if (update === void 0) return current || {};
	if (current === void 0) {
		const result = {};
		for (const [key, value] of Object.entries(update)) if (value !== null) result[key] = value;
		return result;
	}
	const result = { ...current };
	for (const [key, value] of Object.entries(update)) if (value === null) delete result[key];
	else result[key] = value;
	return result;
}
/**
* Shared filesystem state schema.
* Defined at module level to ensure the same object identity is used across all agents,
* preventing "Channel already exists with different type" errors when multiple agents
* use createFilesystemMiddleware.
*
* Uses ReducedValue for files to allow concurrent updates from parallel subagents.
*/
const FilesystemStateSchema = new StateSchema({ files: new ReducedValue(z.record(z.string(), FileDataSchema).default(() => ({})), {
	inputSchema: z.record(z.string(), FileDataSchema.nullable()).optional(),
	reducer: fileDataReducer
}) });
const FILESYSTEM_SYSTEM_PROMPT = context`
  ## Following Conventions

  - Read files before editing — understand existing content before making changes
  - Mimic existing style, naming conventions, and patterns

  ## Filesystem Tools \`ls\`, \`read_file\`, \`write_file\`, \`edit_file\`, \`glob\`, \`grep\`

  You have access to a filesystem which you can interact with using these tools.
  All file paths must start with a /.

  - ls: list files in a directory (requires absolute path)
  - read_file: read a file from the filesystem
  - write_file: write to a file in the filesystem
  - edit_file: edit a file in the filesystem
  - glob: find files matching a pattern (e.g., "**/*.py")
  - grep: search for text within files
`;
const LS_TOOL_DESCRIPTION = context`
  Lists all files in a directory.

  This is useful for exploring the filesystem and finding the right file to read or edit.
  You should almost ALWAYS use this tool before using the read_file or edit_file tools.
`;
const READ_FILE_TOOL_DESCRIPTION = context`
  Reads a file from the filesystem.

  Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

  Usage:
  - By default, it reads up to 100 lines starting from the beginning of the file
  - **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
    - First scan: read_file(path, limit=100) to see file structure
    - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
    - Only omit limit (read full file) when necessary for editing
  - Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
  - Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than ${INT_FORMATTER.format(MAX_LINE_LENGTH)} characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
  - You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
  - If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
  - You should ALWAYS make sure a file has been read before editing it.
`;
const WRITE_FILE_TOOL_DESCRIPTION = context`
  Writes to a new file in the filesystem.

  Usage:
  - The write_file tool will create a new file.
  - Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.
`;
const EDIT_FILE_TOOL_DESCRIPTION = context`
  Performs exact string replacements in files.

  Usage:
  - You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
  - When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
  - ALWAYS prefer editing existing files over creating new ones.
  - Only use emojis if the user explicitly requests it.
`;
const GLOB_TOOL_DESCRIPTION = context`
  Find files matching a glob pattern.

  Supports standard glob patterns: \`*\` (any characters), \`**\` (any directories), \`?\` (single character).
  Returns a list of absolute file paths that match the pattern.

  Examples:
  - \`**/*.py\` - Find all Python files
  - \`*.txt\` - Find all text files in root
  - \`/subdir/**/*.md\` - Find all markdown files under /subdir
`;
const GREP_TOOL_DESCRIPTION = context`
  Search for a text pattern across files.

  Searches for literal text (not regex) and returns matching files or content based on output_mode.
  Special characters like parentheses, brackets, pipes, etc. are treated as literal characters, not regex operators.

  Examples:
  - Search all files: \`grep(pattern="TODO")\`
  - Search Python files only: \`grep(pattern="import", glob="*.py")\`
  - Show matching lines: \`grep(pattern="error", output_mode="content")\`
  - Search for code with special chars: \`grep(pattern="def __init__(self):")\`
`;
const EXECUTE_TOOL_DESCRIPTION = context`
  Executes a shell command in an isolated sandbox environment.

  Usage:
  Executes a given command in the sandbox environment with proper handling and security measures.
  Before executing the command, please follow these steps:

  1. Directory Verification:
    - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
    - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory

  2. Command Execution:
    - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
    - Examples of proper quoting:
      - cd "/Users/name/My Documents" (correct)
      - cd /Users/name/My Documents (incorrect - will fail)
      - python "/path/with spaces/script.py" (correct)
      - python /path/with spaces/script.py (incorrect - will fail)
    - After ensuring proper quoting, execute the command
    - Capture the output of the command

  Usage notes:
    - Commands run in an isolated sandbox environment
    - Returns combined stdout/stderr output with exit code
    - If the output is very large, it may be truncated
    - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
    - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
      - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
      - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
    - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

  Examples:
    Good examples:
      - execute(command="pytest /foo/bar/tests")
      - execute(command="python /path/to/script.py")
      - execute(command="npm install && npm test")

    Bad examples (avoid these):
      - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
      - execute(command="cat file.txt")  # Use read_file tool instead
      - execute(command="find . -name '*.py'")  # Use glob tool instead
      - execute(command="grep -r 'pattern' .")  # Use grep tool instead

  Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
  If execution is not supported, the tool will return an error message.
`;
const EXECUTION_SYSTEM_PROMPT = context`
  ## Execute Tool \`execute\`

  You have access to an \`execute\` tool for running shell commands in a sandboxed environment.
  Use this tool to run commands, scripts, tests, builds, and other shell operations.

  - execute: run a shell command in the sandbox (returns output and exit code)
`;
/**
* Create ls tool using backend.
*/
function createLsTool(backend, options) {
	const { customDescription } = options;
	return tool(async (input, runtime) => {
		const resolvedBackend = await resolveBackend(backend, runtime);
		const path = input.path || "/";
		const lsResult = await resolvedBackend.ls(path);
		if (lsResult.error) return `Error listing files: ${lsResult.error}`;
		const infos = lsResult.files || [];
		if (infos.length === 0) return `No files found in ${path}`;
		const lines = [];
		for (const info of infos) if (info.is_dir) lines.push(`${info.path} (directory)`);
		else {
			const size = info.size ? ` (${info.size} bytes)` : "";
			lines.push(`${info.path}${size}`);
		}
		const result = truncateIfTooLong(lines);
		if (Array.isArray(result)) return result.join("\n");
		return result;
	}, {
		name: "ls",
		description: customDescription || LS_TOOL_DESCRIPTION,
		schema: z.object({ path: z.string().optional().default("/").describe("Directory path to list (default: /)") })
	});
}
/**
* Create read_file tool using backend.
*/
function createReadFileTool(backend, options) {
	const { customDescription, toolTokenLimitBeforeEvict } = options;
	return tool(async (input, runtime) => {
		const resolvedBackend = await resolveBackend(backend, runtime);
		const { file_path, offset = 0, limit = 100 } = input;
		const readResult = await resolvedBackend.read(file_path, offset, limit);
		if (readResult.error) return [{
			type: "text",
			text: `Error: ${readResult.error}`
		}];
		const mimeType = readResult.mimeType ?? getMimeType(file_path);
		if (!isTextMimeType(mimeType)) {
			const binaryContent = readResult.content;
			if (!binaryContent) return [{
				type: "text",
				text: `Error: expected binary content for '${file_path}'`
			}];
			let base64Data;
			if (typeof binaryContent === "string") base64Data = binaryContent;
			else if (ArrayBuffer.isView(binaryContent)) base64Data = Buffer.from(binaryContent).toString("base64");
			else {
				const values = Object.values(binaryContent);
				base64Data = Buffer.from(new Uint8Array(values)).toString("base64");
			}
			const sizeBytes = Math.ceil(base64Data.length * 3 / 4);
			if (sizeBytes > 10485760) return [{
				type: "text",
				text: `Error: file too large to read (${Math.round(sizeBytes / (1024 * 1024))}MB exceeds ${MAX_BINARY_READ_SIZE_BYTES / (1024 * 1024)}MB limit for binary files)`
			}];
			if (mimeType.startsWith("image/")) return [{
				type: "image",
				mimeType,
				data: base64Data
			}];
			if (mimeType.startsWith("audio/")) return [{
				type: "audio",
				mimeType,
				data: base64Data
			}];
			if (mimeType.startsWith("video/")) return [{
				type: "video",
				mimeType,
				data: base64Data
			}];
			return [{
				type: "file",
				mimeType,
				data: base64Data
			}];
		}
		let content = typeof readResult.content === "string" ? readResult.content : "";
		const lines = content.split("\n");
		if (lines.length > limit) content = lines.slice(0, limit).join("\n");
		let formatted = formatContentWithLineNumbers(content, offset + 1);
		if (toolTokenLimitBeforeEvict && formatted.length >= 4 * toolTokenLimitBeforeEvict) {
			const truncationMsg = READ_FILE_TRUNCATION_MSG.replace("{file_path}", file_path);
			const maxContentLength = 4 * toolTokenLimitBeforeEvict - truncationMsg.length;
			formatted = formatted.substring(0, maxContentLength) + truncationMsg;
		}
		return [{
			type: "text",
			text: formatted
		}];
	}, {
		name: "read_file",
		description: customDescription || READ_FILE_TOOL_DESCRIPTION,
		schema: z.object({
			file_path: z.string().describe("Absolute path to the file to read"),
			offset: z.coerce.number().optional().default(0).describe("Line offset to start reading from (0-indexed)"),
			limit: z.coerce.number().optional().default(100).describe("Maximum number of lines to read")
		})
	});
}
/**
* Create write_file tool using backend.
*/
function createWriteFileTool(backend, options) {
	const { customDescription } = options;
	return tool(async (input, runtime) => {
		const resolvedBackend = await resolveBackend(backend, runtime);
		const { file_path, content } = input;
		const result = await resolvedBackend.write(file_path, content);
		if (result.error) return result.error;
		const message = new ToolMessage({
			content: `Successfully wrote to '${file_path}'`,
			tool_call_id: runtime.toolCall?.id,
			name: "write_file",
			metadata: result.metadata
		});
		if (result.filesUpdate) return new Command({ update: {
			files: result.filesUpdate,
			messages: [message]
		} });
		return message;
	}, {
		name: "write_file",
		description: customDescription || WRITE_FILE_TOOL_DESCRIPTION,
		schema: z.object({
			file_path: z.string().describe("Absolute path to the file to write"),
			content: z.string().default("").describe("Content to write to the file")
		})
	});
}
/**
* Create edit_file tool using backend.
*/
function createEditFileTool(backend, options) {
	const { customDescription } = options;
	return tool(async (input, runtime) => {
		const resolvedBackend = await resolveBackend(backend, runtime);
		const { file_path, old_string, new_string, replace_all = false } = input;
		const result = await resolvedBackend.edit(file_path, old_string, new_string, replace_all);
		if (result.error) return result.error;
		const message = new ToolMessage({
			content: `Successfully replaced ${result.occurrences} occurrence(s) in '${file_path}'`,
			tool_call_id: runtime.toolCall?.id,
			name: "edit_file",
			metadata: result.metadata
		});
		if (result.filesUpdate) return new Command({ update: {
			files: result.filesUpdate,
			messages: [message]
		} });
		return message;
	}, {
		name: "edit_file",
		description: customDescription || EDIT_FILE_TOOL_DESCRIPTION,
		schema: z.object({
			file_path: z.string().describe("Absolute path to the file to edit"),
			old_string: z.string().describe("String to be replaced (must match exactly)"),
			new_string: z.string().describe("String to replace with"),
			replace_all: z.boolean().optional().default(false).describe("Whether to replace all occurrences")
		})
	});
}
/**
* Create glob tool using backend.
*/
function createGlobTool(backend, options) {
	const { customDescription } = options;
	return tool(async (input, runtime) => {
		const resolvedBackend = await resolveBackend(backend, runtime);
		const { pattern, path = "/" } = input;
		const globResult = await resolvedBackend.glob(pattern, path);
		if (globResult.error) return `Error finding files: ${globResult.error}`;
		const infos = globResult.files || [];
		if (infos.length === 0) return `No files found matching pattern '${pattern}'`;
		const result = truncateIfTooLong(infos.map((info) => info.path));
		if (Array.isArray(result)) return result.join("\n");
		return result;
	}, {
		name: "glob",
		description: customDescription || GLOB_TOOL_DESCRIPTION,
		schema: z.object({
			pattern: z.string().describe("Glob pattern (e.g., '*.py', '**/*.ts')"),
			path: z.string().optional().default("/").describe("Base path to search from (default: /)")
		})
	});
}
/**
* Create grep tool using backend.
*/
function createGrepTool(backend, options) {
	const { customDescription } = options;
	return tool(async (input, runtime) => {
		const resolvedBackend = await resolveBackend(backend, runtime);
		const { pattern, path = "/", glob = null } = input;
		const result = await resolvedBackend.grep(pattern, path, glob);
		if (result.error) return result.error;
		const matches = result.matches ?? [];
		if (matches.length === 0) return `No matches found for pattern '${pattern}'`;
		const lines = [];
		let currentFile = null;
		for (const match of matches) {
			if (match.path !== currentFile) {
				currentFile = match.path;
				lines.push(`\n${currentFile}:`);
			}
			lines.push(`  ${match.line}: ${match.text}`);
		}
		const truncated = truncateIfTooLong(lines);
		if (Array.isArray(truncated)) return truncated.join("\n");
		return truncated;
	}, {
		name: "grep",
		description: customDescription || GREP_TOOL_DESCRIPTION,
		schema: z.object({
			pattern: z.string().describe("Regex pattern to search for"),
			path: z.string().optional().default("/").describe("Base path to search from (default: /)"),
			glob: z.string().optional().nullable().describe("Optional glob pattern to filter files (e.g., '*.py')")
		})
	});
}
/**
* Create execute tool using backend.
*/
function createExecuteTool(backend, options) {
	const { customDescription } = options;
	return tool(async (input, runtime) => {
		const resolvedBackend = await resolveBackend(backend, runtime);
		if (!isSandboxBackend(resolvedBackend)) return "Error: Execution not available. This agent's backend does not support command execution (SandboxBackendProtocol). To use the execute tool, provide a backend that implements SandboxBackendProtocol.";
		const result = await resolvedBackend.execute(input.command);
		const parts = [result.output];
		if (result.exitCode !== null) {
			const status = result.exitCode === 0 ? "succeeded" : "failed";
			parts.push(`\n[Command ${status} with exit code ${result.exitCode}]`);
		}
		if (result.truncated) parts.push("\n[Output was truncated due to size limits]");
		return parts.join("");
	}, {
		name: "execute",
		description: customDescription || EXECUTE_TOOL_DESCRIPTION,
		schema: z.object({ command: z.string().describe("The shell command to execute") })
	});
}
/**
* Create filesystem middleware with all tools and features.
*/
function createFilesystemMiddleware(options = {}) {
	const { backend = (runtime) => new StateBackend(runtime), systemPrompt: customSystemPrompt = null, customToolDescriptions = null, toolTokenLimitBeforeEvict = 2e4, humanMessageTokenLimitBeforeEvict = 5e4 } = options;
	const baseSystemPrompt = customSystemPrompt || FILESYSTEM_SYSTEM_PROMPT;
	const allToolsByName = {
		ls: createLsTool(backend, { customDescription: customToolDescriptions?.ls }),
		read_file: createReadFileTool(backend, {
			customDescription: customToolDescriptions?.read_file,
			toolTokenLimitBeforeEvict
		}),
		write_file: createWriteFileTool(backend, { customDescription: customToolDescriptions?.write_file }),
		edit_file: createEditFileTool(backend, { customDescription: customToolDescriptions?.edit_file }),
		glob: createGlobTool(backend, { customDescription: customToolDescriptions?.glob }),
		grep: createGrepTool(backend, { customDescription: customToolDescriptions?.grep }),
		execute: createExecuteTool(backend, { customDescription: customToolDescriptions?.execute })
	};
	return createMiddleware({
		name: "FilesystemMiddleware",
		stateSchema: FilesystemStateSchema,
		tools: Object.values(allToolsByName),
		async beforeAgent(state) {
			if (!humanMessageTokenLimitBeforeEvict) return;
			const messages = state.messages;
			if (!messages || messages.length === 0) return;
			const last = messages[messages.length - 1];
			if (!HumanMessage.isInstance(last)) return;
			if (last.additional_kwargs?.lc_evicted_to) return;
			const contentStr = extractTextFromMessage(last);
			const threshold = 4 * humanMessageTokenLimitBeforeEvict;
			if (contentStr.length <= threshold) return;
			const resolvedBackend = await resolveBackend(backend, { state: state || {} });
			const filePath = `/conversation_history/${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;
			const writeResult = await resolvedBackend.write(filePath, contentStr);
			if (writeResult.error) return;
			const result = { messages: [new HumanMessage({
				content: last.content,
				id: last.id,
				additional_kwargs: {
					...last.additional_kwargs,
					lc_evicted_to: filePath
				},
				response_metadata: { ...last.response_metadata }
			})] };
			if (writeResult.filesUpdate) result.files = writeResult.filesUpdate;
			return result;
		},
		wrapModelCall: async (request, handler) => {
			const supportsExecution = isSandboxBackend(await resolveBackend(backend, {
				...request.runtime,
				state: request.state
			}));
			let tools = request.tools;
			if (!supportsExecution) tools = tools.filter((t) => t.name !== "execute");
			let filesystemPrompt = baseSystemPrompt;
			if (supportsExecution) filesystemPrompt = `${filesystemPrompt}\n\n${EXECUTION_SYSTEM_PROMPT}`;
			const newSystemMessage = request.systemMessage.concat(filesystemPrompt);
			let messages = request.messages;
			if (humanMessageTokenLimitBeforeEvict && messages) {
				if (messages.some((msg) => HumanMessage.isInstance(msg) && msg.additional_kwargs?.lc_evicted_to)) messages = messages.map((msg) => {
					if (HumanMessage.isInstance(msg) && msg.additional_kwargs?.lc_evicted_to) return buildTruncatedHumanMessage(msg, msg.additional_kwargs.lc_evicted_to);
					return msg;
				});
			}
			return handler({
				...request,
				tools,
				messages,
				systemMessage: newSystemMessage
			});
		},
		wrapToolCall: async (request, handler) => {
			if (!toolTokenLimitBeforeEvict) return handler(request);
			const toolName = request.toolCall?.name;
			if (toolName && TOOLS_EXCLUDED_FROM_EVICTION.includes(toolName)) return handler(request);
			const result = await handler(request);
			async function processToolMessage(msg, toolTokenLimitBeforeEvict) {
				if (typeof msg.content === "string" && msg.content.length > toolTokenLimitBeforeEvict * 4) {
					const resolvedBackend = await resolveBackend(backend, {
						...request.runtime,
						state: request.state
					});
					const evictPath = `/large_tool_results/${sanitizeToolCallId(request.toolCall?.id || msg.tool_call_id)}`;
					const writeResult = await resolvedBackend.write(evictPath, msg.content);
					if (writeResult.error) return {
						message: msg,
						filesUpdate: null
					};
					const contentSample = createContentPreview(msg.content);
					return {
						message: new ToolMessage({
							content: TOO_LARGE_TOOL_MSG.replace("{tool_call_id}", msg.tool_call_id).replace("{file_path}", evictPath).replace("{content_sample}", contentSample),
							tool_call_id: msg.tool_call_id,
							name: msg.name,
							id: msg.id,
							artifact: msg.artifact,
							status: msg.status,
							metadata: msg.metadata,
							additional_kwargs: msg.additional_kwargs,
							response_metadata: msg.response_metadata
						}),
						filesUpdate: writeResult.filesUpdate
					};
				}
				return {
					message: msg,
					filesUpdate: null
				};
			}
			if (ToolMessage.isInstance(result)) {
				const processed = await processToolMessage(result, toolTokenLimitBeforeEvict);
				if (processed.filesUpdate) return new Command({ update: {
					files: processed.filesUpdate,
					messages: [processed.message]
				} });
				return processed.message;
			}
			if (isCommand(result)) {
				const update = result.update;
				if (!update?.messages) return result;
				let hasLargeResults = false;
				const accumulatedFiles = update.files ? { ...update.files } : {};
				const processedMessages = [];
				for (const msg of update.messages) if (ToolMessage.isInstance(msg)) {
					const processed = await processToolMessage(msg, toolTokenLimitBeforeEvict);
					processedMessages.push(processed.message);
					if (processed.filesUpdate) {
						hasLargeResults = true;
						Object.assign(accumulatedFiles, processed.filesUpdate);
					}
				} else processedMessages.push(msg);
				if (hasLargeResults) return new Command({ update: {
					...update,
					messages: processedMessages,
					files: accumulatedFiles
				} });
			}
			return result;
		}
	});
}
//#endregion
//#region src/middleware/subagents.ts
/**
* Default system prompt for subagents.
* Provides a minimal base prompt that can be extended by specific subagent configurations.
*/
const DEFAULT_SUBAGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools.";
/**
* State keys that are excluded when passing state to subagents and when returning
* updates from subagents.
*
* When returning updates:
* 1. The messages key is handled explicitly to ensure only the final message is included
* 2. The todos and structuredResponse keys are excluded as they do not have a defined reducer
*    and no clear meaning for returning them from a subagent to the main agent.
* 3. The skillsMetadata and memoryContents keys are automatically excluded from subagent output
*    to prevent parent state from leaking to child agents. Each agent loads its own skills/memory
*    independently based on its middleware configuration.
*/
const EXCLUDED_STATE_KEYS = [
	"messages",
	"todos",
	"structuredResponse",
	"skillsMetadata",
	"memoryContents"
];
/**
* Default description for the general-purpose subagent.
* This description is shown to the model when selecting which subagent to use.
*/
const DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent.";
function getTaskToolDescription(subagentDescriptions) {
	return context`
    Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

    Available agent types and the tools they have access to:
    ${subagentDescriptions.join("\n")}

    When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

    ## Usage notes:
    1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
    2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
    3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
    4. The agent's outputs should generally be trusted
    5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
    6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
    7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

    ### Example usage of the general-purpose agent:

    <example_agent_descriptions>
    "general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
    </example_agent_descriptions>

    <example>
    User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
    Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
    Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
    <commentary>
    Research is a complex, multi-step task in it of itself.
    The research of each individual player is not dependent on the research of the other players.
    The assistant uses the task tool to break down the complex objective into three isolated tasks.
    Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
    This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
    </commentary>
    </example>

    <example>
    User: "Analyze a single large code repository for security vulnerabilities and generate a report."
    Assistant: *Launches a single \`task\` subagent for the repository analysis*
    Assistant: *Receives report and integrates results into final summary*
    <commentary>
    Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
    If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
    </commentary>
    </example>

    <example>
    User: "Schedule two meetings for me and prepare agendas for each."
    Assistant: *Calls the task tool in parallel to launch two \`task\` subagents (one per meeting) to prepare agendas*
    Assistant: *Returns final schedules and agendas*
    <commentary>
    Tasks are simple individually, but subagents help silo agenda preparation.
    Each subagent only needs to worry about the agenda for one meeting.
    </commentary>
    </example>

    <example>
    User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
    Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
    <commentary>
    The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
    It is better to just complete the task directly and NOT use the \`task\`tool.
    </commentary>
    </example>

    ### Example usage with custom agents:

    <example_agent_descriptions>
    "content-reviewer": use this agent after you are done creating significant content or documents
    "greeting-responder": use this agent when to respond to user greetings with a friendly joke
    "research-analyst": use this agent to conduct thorough research on complex topics
    </example_agent_description>

    <example>
    user: "Please write a function that checks if a number is prime"
    assistant: Sure let me write a function that checks if a number is prime
    assistant: First let me use the Write tool to write a function that checks if a number is prime
    assistant: I'm going to use the Write tool to write the following code:
    <code>
    function isPrime(n) {{
      if (n <= 1) return false
      for (let i = 2; i * i <= n; i++) {{
        if (n % i === 0) return false
      }}
      return true
    }}
    </code>
    <commentary>
    Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
    </commentary>
    assistant: Now let me use the content-reviewer agent to review the code
    assistant: Uses the Task tool to launch with the content-reviewer agent
    </example>

    <example>
    user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
    <commentary>
    This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
    </commentary>
    assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
    assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
    </example>

    <example>
    user: "Hello"
    <commentary>
    Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
    </commentary>
    assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
    </example>
  `;
}
/**
* System prompt section that explains how to use the task tool for spawning subagents.
*
* This prompt is automatically appended to the main agent's system prompt when
* using `createSubAgentMiddleware`. It provides guidance on:
* - When to use the task tool
* - Subagent lifecycle (spawn → run → return → reconcile)
* - When NOT to use the task tool
* - Best practices for parallel task execution
*
* You can provide a custom `systemPrompt` to `createSubAgentMiddleware` to override
* or extend this default.
*/
const TASK_SYSTEM_PROMPT = context`
  ## \`task\` (subagent spawner)

  You have access to a \`task\` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

  When to use the task tool:
  - When a task is complex and multi-step, and can be fully delegated in isolation
  - When a task is independent of other tasks and can run in parallel
  - When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
  - When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
  - When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

  Subagent lifecycle:
  1. **Spawn** → Provide clear role, instructions, and expected output
  2. **Run** → The subagent completes the task autonomously
  3. **Return** → The subagent provides a single structured result
  4. **Reconcile** → Incorporate or synthesize the result into the main thread

  When NOT to use the task tool:
  - If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
  - If the task is trivial (a few tool calls or simple lookup)
  - If delegating does not reduce token usage, complexity, or context switching
  - If splitting would add latency without benefit

  ## Important Task Tool Usage Notes to Remember
  - Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
  - Remember to use the \`task\` tool to silo independent tasks within a multi-part objective.
  - You should use the \`task\` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient.
`;
/**
* Base specification for the general-purpose subagent.
*
* This constant provides the default configuration for the general-purpose subagent
* that is automatically included when `generalPurposeAgent: true` (the default).
*
* The general-purpose subagent:
* - Has access to all tools from the main agent
* - Inherits skills from the main agent (when skills are configured)
* - Uses the same model as the main agent (by default)
* - Is ideal for delegating complex, multi-step tasks
*
* You can spread this constant and override specific properties when creating
* custom subagents that should behave similarly to the general-purpose agent:
*
* @example
* ```typescript
* import { GENERAL_PURPOSE_SUBAGENT, createDeepAgent } from "@anthropic/deepagents";
*
* // Use as-is (automatically included with generalPurposeAgent: true)
* const agent = createDeepAgent({ model: "claude-sonnet-4-5-20250929" });
*
* // Or create a custom variant with different tools
* const customGP: SubAgent = {
*   ...GENERAL_PURPOSE_SUBAGENT,
*   name: "research-gp",
*   tools: [webSearchTool, readFileTool],
* };
*
* const agent = createDeepAgent({
*   model: "claude-sonnet-4-5-20250929",
*   subagents: [customGP],
*   // Disable the default general-purpose agent since we're providing our own
*   // (handled automatically when using createSubAgentMiddleware directly)
* });
* ```
*/
const GENERAL_PURPOSE_SUBAGENT = {
	name: "general-purpose",
	description: DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
	systemPrompt: DEFAULT_SUBAGENT_PROMPT
};
/**
* Filter state to exclude certain keys when passing to subagents
*/
function filterStateForSubagent(state) {
	const filtered = {};
	for (const [key, value] of Object.entries(state)) if (!EXCLUDED_STATE_KEYS.includes(key)) filtered[key] = value;
	return filtered;
}
/**
* Invalid tool message block types
*/
const INVALID_TOOL_MESSAGE_BLOCK_TYPES = [
	"tool_use",
	"thinking",
	"redacted_thinking"
];
/**
* Create Command with filtered state update from subagent result
*/
function returnCommandWithStateUpdate(result, toolCallId) {
	const stateUpdate = filterStateForSubagent(result);
	let content;
	if (result.structuredResponse != null) content = JSON.stringify(result.structuredResponse);
	else {
		const messages = result.messages;
		content = (messages?.[messages.length - 1])?.content || "Task completed";
		if (Array.isArray(content)) {
			content = content.filter((block) => !INVALID_TOOL_MESSAGE_BLOCK_TYPES.includes(block.type));
			if (content.length === 0) content = "Task completed";
		}
	}
	return new Command({ update: {
		...stateUpdate,
		messages: [new ToolMessage({
			content,
			tool_call_id: toolCallId,
			name: "task"
		})]
	} });
}
/**
* Create subagent instances from specifications
*/
function getSubagents(options) {
	const { defaultModel, defaultTools, defaultMiddleware, generalPurposeMiddleware: gpMiddleware, defaultInterruptOn, subagents, generalPurposeAgent } = options;
	const defaultSubagentMiddleware = defaultMiddleware || [];
	const generalPurposeMiddlewareBase = gpMiddleware || defaultSubagentMiddleware;
	const agents = {};
	const subagentDescriptions = [];
	if (generalPurposeAgent) {
		const generalPurposeMiddleware = [...generalPurposeMiddlewareBase];
		if (defaultInterruptOn) generalPurposeMiddleware.push(humanInTheLoopMiddleware({ interruptOn: defaultInterruptOn }));
		agents["general-purpose"] = createAgent({
			model: defaultModel,
			systemPrompt: DEFAULT_SUBAGENT_PROMPT,
			tools: defaultTools,
			middleware: generalPurposeMiddleware,
			name: "general-purpose"
		});
		subagentDescriptions.push(`- general-purpose: ${DEFAULT_GENERAL_PURPOSE_DESCRIPTION}`);
	}
	for (const agentParams of subagents) {
		subagentDescriptions.push(`- ${agentParams.name}: ${agentParams.description}`);
		if ("runnable" in agentParams) agents[agentParams.name] = agentParams.runnable;
		else {
			const middleware = agentParams.middleware ? [...defaultSubagentMiddleware, ...agentParams.middleware] : [...defaultSubagentMiddleware];
			const interruptOn = agentParams.interruptOn || defaultInterruptOn;
			if (interruptOn) middleware.push(humanInTheLoopMiddleware({ interruptOn }));
			agents[agentParams.name] = createAgent({
				model: agentParams.model ?? defaultModel,
				systemPrompt: agentParams.systemPrompt,
				tools: agentParams.tools ?? defaultTools,
				middleware,
				name: agentParams.name,
				...agentParams.responseFormat != null && { responseFormat: agentParams.responseFormat }
			});
		}
	}
	return {
		agents,
		descriptions: subagentDescriptions
	};
}
/**
* Create the task tool for invoking subagents
*/
function createTaskTool(options) {
	const { defaultModel, defaultTools, defaultMiddleware, generalPurposeMiddleware, defaultInterruptOn, subagents, generalPurposeAgent, taskDescription } = options;
	const { agents: subagentGraphs, descriptions: subagentDescriptions } = getSubagents({
		defaultModel,
		defaultTools,
		defaultMiddleware,
		generalPurposeMiddleware,
		defaultInterruptOn,
		subagents,
		generalPurposeAgent
	});
	return tool(async (input, runtime) => {
		const { description, subagent_type } = input;
		if (!(subagent_type in subagentGraphs)) {
			const allowedTypes = Object.keys(subagentGraphs).map((k) => `\`${k}\``).join(", ");
			throw new Error(`Error: invoked agent of type ${subagent_type}, the only allowed types are ${allowedTypes}`);
		}
		const subagent = subagentGraphs[subagent_type];
		const currentState = runtime.state && typeof runtime.state === "object" ? runtime.state : getCurrentTaskInput();
		const subagentState = filterStateForSubagent(currentState);
		subagentState.messages = [new HumanMessage$1({ content: description })];
		const result = await subagent.invoke(subagentState, runtime.config);
		if (!runtime.toolCall?.id) {
			if (result.structuredResponse != null) return JSON.stringify(result.structuredResponse);
			const messages = result.messages;
			let content = (messages?.[messages.length - 1])?.content || "Task completed";
			if (Array.isArray(content)) {
				content = content.filter((block) => !INVALID_TOOL_MESSAGE_BLOCK_TYPES.includes(block.type));
				if (content.length === 0) return "Task completed";
				return content.map((block) => "text" in block ? block.text : JSON.stringify(block)).join("\n");
			}
			return content;
		}
		return returnCommandWithStateUpdate(result, runtime.toolCall.id);
	}, {
		name: "task",
		description: taskDescription ? taskDescription : getTaskToolDescription(subagentDescriptions),
		schema: z.object({
			description: z.string().describe("The task to execute with the selected agent"),
			subagent_type: z.string().describe(`Name of the agent to use. Available: ${Object.keys(subagentGraphs).join(", ")}`)
		})
	});
}
/**
* Create subagent middleware with task tool
*/
function createSubAgentMiddleware(options) {
	const { defaultModel, defaultTools = [], defaultMiddleware = null, generalPurposeMiddleware = null, defaultInterruptOn = null, subagents = [], systemPrompt = TASK_SYSTEM_PROMPT, generalPurposeAgent = true, taskDescription = null } = options;
	return createMiddleware({
		name: "subAgentMiddleware",
		tools: [createTaskTool({
			defaultModel,
			defaultTools,
			defaultMiddleware,
			generalPurposeMiddleware,
			defaultInterruptOn,
			subagents,
			generalPurposeAgent,
			taskDescription
		})],
		wrapModelCall: async (request, handler) => {
			if (systemPrompt !== null) return handler({
				...request,
				systemMessage: request.systemMessage.concat(new SystemMessage({ content: systemPrompt }))
			});
			return handler(request);
		}
	});
}
//#endregion
//#region src/middleware/patch_tool_calls.ts
/**
* Patch tool call / tool response parity in a messages array.
*
* Ensures strict 1:1 correspondence between AIMessage tool_calls and
* ToolMessage responses:
*
* 1. **Dangling tool_calls** — an AIMessage contains a tool_call with no
*    matching ToolMessage anywhere after it. A synthetic cancellation
*    ToolMessage is inserted immediately after the AIMessage.
*
* 2. **Orphaned ToolMessages** — a ToolMessage whose `tool_call_id` does not
*    match any tool_call in a preceding AIMessage. The ToolMessage is removed.
*
* Both directions are required for providers that enforce strict parity
* (e.g. Google Gemini returns 400 INVALID_ARGUMENT otherwise).
*
* @param messages - The messages array to patch
* @returns Object with patched messages and needsPatch flag
*/
function patchDanglingToolCalls(messages) {
	if (!messages || messages.length === 0) return {
		patchedMessages: [],
		needsPatch: false
	};
	const allToolCallIds = /* @__PURE__ */ new Set();
	for (const msg of messages) if (AIMessage.isInstance(msg) && msg.tool_calls != null) {
		for (const tc of msg.tool_calls) if (tc.id) allToolCallIds.add(tc.id);
	}
	const patchedMessages = [];
	let needsPatch = false;
	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i];
		if (ToolMessage.isInstance(msg)) {
			if (!allToolCallIds.has(msg.tool_call_id)) {
				needsPatch = true;
				continue;
			}
		}
		patchedMessages.push(msg);
		if (AIMessage.isInstance(msg) && msg.tool_calls != null) {
			for (const toolCall of msg.tool_calls) if (!messages.slice(i + 1).find((m) => ToolMessage.isInstance(m) && m.tool_call_id === toolCall.id)) {
				needsPatch = true;
				const toolMsg = `Tool call ${toolCall.name} with id ${toolCall.id} was cancelled - another message came in before it could be completed.`;
				patchedMessages.push(new ToolMessage({
					content: toolMsg,
					name: toolCall.name,
					tool_call_id: toolCall.id
				}));
			}
		}
	}
	return {
		patchedMessages,
		needsPatch
	};
}
/**
* Create middleware that enforces strict tool call / tool response parity in
* the messages history.
*
* Two kinds of violations are repaired:
* 1. **Dangling tool_calls** — an AIMessage contains tool_calls with no
*    matching ToolMessage responses. Synthetic cancellation ToolMessages are
*    injected so every tool_call has a response.
* 2. **Orphaned ToolMessages** — a ToolMessage exists whose `tool_call_id`
*    does not match any tool_call in a preceding AIMessage. These are removed.
*
* This is critical for providers like Google Gemini that reject requests with
* mismatched function call / function response counts (400 INVALID_ARGUMENT).
*
* This middleware patches in two places:
* 1. `beforeAgent`: Patches state at the start of the agent loop (handles most cases)
* 2. `wrapModelCall`: Patches the request right before model invocation (handles
*    edge cases like HITL rejection during graph resume where state updates from
*    beforeAgent may not be applied in time)
*
* @returns AgentMiddleware that enforces tool call / response parity
*
* @example
* ```typescript
* import { createAgent } from "langchain";
* import { createPatchToolCallsMiddleware } from "./middleware/patch_tool_calls";
*
* const agent = createAgent({
*   model: "claude-sonnet-4-5-20250929",
*   middleware: [createPatchToolCallsMiddleware()],
* });
* ```
*/
function createPatchToolCallsMiddleware() {
	return createMiddleware({
		name: "patchToolCallsMiddleware",
		beforeAgent: async (state) => {
			const messages = state.messages;
			if (!messages || messages.length === 0) return;
			const { patchedMessages, needsPatch } = patchDanglingToolCalls(messages);
			/**
			* Only trigger REMOVE_ALL_MESSAGES if patching is actually needed
			*/
			if (!needsPatch) return;
			return { messages: [new RemoveMessage({ id: REMOVE_ALL_MESSAGES }), ...patchedMessages] };
		},
		wrapModelCall: async (request, handler) => {
			const messages = request.messages;
			if (!messages || messages.length === 0) return handler(request);
			const { patchedMessages, needsPatch } = patchDanglingToolCalls(messages);
			if (!needsPatch) return handler(request);
			return handler({
				...request,
				messages: patchedMessages
			});
		}
	});
}
//#endregion
//#region src/values.ts
/**
* Shared state values for use in StateSchema definitions.
*
* This module provides pre-configured ReducedValue instances that can be
* reused across different state schemas, similar to LangGraph's messagesValue.
*/
/**
* Shared ReducedValue for file data state management.
*
* This provides a reusable pattern for managing file state with automatic
* merging of concurrent updates from parallel subagents. Files can be updated
* or deleted (using null values) and the reducer handles the merge logic.
*
* Similar to LangGraph's messagesValue, this encapsulates the common pattern
* of managing files in agent state so you don't have to manually configure
* the ReducedValue each time.
*
* @example
* ```typescript
* import { filesValue } from "@anthropic/deepagents";
* import { StateSchema } from "@langchain/langgraph";
*
* const MyStateSchema = new StateSchema({
*   files: filesValue,
*   // ... other state fields
* });
* ```
*/
const filesValue = new ReducedValue(z$1.record(z$1.string(), FileDataSchema).default(() => ({})), {
	inputSchema: z$1.record(z$1.string(), FileDataSchema.nullable()).optional(),
	reducer: fileDataReducer
});
//#endregion
//#region src/middleware/memory.ts
/**
* Middleware for loading agent memory/context from AGENTS.md files.
*
* This module implements support for the AGENTS.md specification (https://agents.md/),
* loading memory/context from configurable sources and injecting into the system prompt.
*
* ## Overview
*
* AGENTS.md files provide project-specific context and instructions to help AI agents
* work effectively. Unlike skills (which are on-demand workflows), memory is always
* loaded and provides persistent context.
*
* ## Usage
*
* ```typescript
* import { createMemoryMiddleware } from "@anthropic/deepagents";
* import { FilesystemBackend } from "@anthropic/deepagents";
*
* // Security: FilesystemBackend allows reading/writing from the entire filesystem.
* // Either ensure the agent is running within a sandbox OR add human-in-the-loop (HIL)
* // approval to file operations.
* const backend = new FilesystemBackend({ rootDir: "/" });
*
* const middleware = createMemoryMiddleware({
*   backend,
*   sources: [
*     "~/.deepagents/AGENTS.md",
*     "./.deepagents/AGENTS.md",
*   ],
* });
*
* const agent = createDeepAgent({ middleware: [middleware] });
* ```
*
* ## Memory Sources
*
* Sources are simply paths to AGENTS.md files that are loaded in order and combined.
* Multiple sources are concatenated in order, with all content included.
* Later sources appear after earlier ones in the combined prompt.
*
* ## File Format
*
* AGENTS.md files are standard Markdown with no required structure.
* Common sections include:
* - Project overview
* - Build/test commands
* - Code style guidelines
* - Architecture notes
*/
/**
* State schema for memory middleware.
*/
const MemoryStateSchema = new StateSchema({
	memoryContents: z$1.record(z$1.string(), z$1.string()).optional(),
	files: filesValue
});
/**
* Default system prompt template for memory.
* Ported from Python's comprehensive memory guidelines.
*/
const MEMORY_SYSTEM_PROMPT = context`
  <agent_memory>
  {memory_contents}
  </agent_memory>

  <memory_guidelines>
      The above <agent_memory> was loaded in from files in your filesystem. As you learn from your interactions with the user, you can save new knowledge by calling the \`edit_file\` tool.

      **Learning from feedback:**
      - One of your MAIN PRIORITIES is to learn from your interactions with the user. These learnings can be implicit or explicit. This means that in the future, you will remember this important information.
      - When you need to remember something, updating memory must be your FIRST, IMMEDIATE action - before responding to the user, before calling other tools, before doing anything else. Just update memory immediately.
      - When user says something is better/worse, capture WHY and encode it as a pattern.
      - Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions.
      - A great opportunity to update your memories is when the user interrupts a tool call and provides feedback. You should update your memories immediately before revising the tool call.
      - Look for the underlying principle behind corrections, not just the specific mistake.
      - The user might not explicitly ask you to remember something, but if they provide information that is useful for future use, you should update your memories immediately.

      **Asking for information:**
      - If you lack context to perform an action (e.g. send a Slack DM, requires a user ID/email) you should explicitly ask the user for this information.
      - It is preferred for you to ask for information, don't assume anything that you do not know!
      - When the user provides information that is useful for future use, you should update your memories immediately.

      **When to update memories:**
      - When the user explicitly asks you to remember something (e.g., "remember my email", "save this preference")
      - When the user describes your role or how you should behave (e.g., "you are a web researcher", "always do X")
      - When the user gives feedback on your work - capture what was wrong and how to improve
      - When the user provides information required for tool use (e.g., slack channel ID, email addresses)
      - When the user provides context useful for future tasks, such as how to use tools, or which actions to take in a particular situation
      - When you discover new patterns or preferences (coding styles, conventions, workflows)

      **When to NOT update memories:**
      - When the information is temporary or transient (e.g., "I'm running late", "I'm on my phone right now")
      - When the information is a one-time task request (e.g., "Find me a recipe", "What's 25 * 4?")
      - When the information is a simple question that doesn't reveal lasting preferences (e.g., "What day is it?", "Can you explain X?")
      - When the information is an acknowledgment or small talk (e.g., "Sounds good!", "Hello", "Thanks for that")
      - When the information is stale or irrelevant in future conversations
      - Never store API keys, access tokens, passwords, or any other credentials in any file, memory, or system prompt.
      - If the user asks where to put API keys or provides an API key, do NOT echo or save it.

      **Examples:**
      Example 1 (remembering user information):
      User: Can you connect to my google account?
      Agent: Sure, I'll connect to your google account, what's your google account email?
      User: john@example.com
      Agent: Let me save this to my memory.
      Tool Call: edit_file(...) -> remembers that the user's google account email is john@example.com

      Example 2 (remembering implicit user preferences):
      User: Can you write me an example for creating a deep agent in LangChain?
      Agent: Sure, I'll write you an example for creating a deep agent in LangChain <example code in Python>
      User: Can you do this in JavaScript
      Agent: Let me save this to my memory.
      Tool Call: edit_file(...) -> remembers that the user prefers to get LangChain code examples in JavaScript
      Agent: Sure, here is the JavaScript example<example code in JavaScript>

      Example 3 (do not remember transient information):
      User: I'm going to play basketball tonight so I will be offline for a few hours.
      Agent: Okay I'll add a block to your calendar.
      Tool Call: create_calendar_event(...) -> just calls a tool, does not commit anything to memory, as it is transient information
  </memory_guidelines>
`;
/**
* Format loaded memory contents for injection into prompt.
* Pairs memory locations with their contents for clarity.
*/
function formatMemoryContents(contents, sources) {
	if (Object.keys(contents).length === 0) return "(No memory loaded)";
	const sections = [];
	for (const path of sources) if (contents[path]) sections.push(`${path}\n${contents[path]}`);
	if (sections.length === 0) return "(No memory loaded)";
	return sections.join("\n\n");
}
/**
* Load memory content from a backend path.
*
* @param backend - Backend to load from.
* @param path - Path to the AGENTS.md file.
* @returns File content if found, null otherwise.
*/
async function loadMemoryFromBackend(backend, path) {
	const adaptedBackend = adaptBackendProtocol(backend);
	if (!adaptedBackend.downloadFiles) {
		const content = await adaptedBackend.read(path);
		if (content.error) return null;
		if (typeof content.content !== "string") return null;
		return content.content;
	}
	const results = await adaptedBackend.downloadFiles([path]);
	if (results.length !== 1) throw new Error(`Expected 1 response for path ${path}, got ${results.length}`);
	const response = results[0];
	if (response.error != null) {
		if (response.error === "file_not_found") return null;
		throw new Error(`Failed to download ${path}: ${response.error}`);
	}
	if (response.content != null) return new TextDecoder().decode(response.content);
	return null;
}
/**
* Create middleware for loading agent memory from AGENTS.md files.
*
* Loads memory content from configured sources and injects into the system prompt.
* Supports multiple sources that are combined together.
*
* @param options - Configuration options
* @returns AgentMiddleware for memory loading and injection
*
* @example
* ```typescript
* const middleware = createMemoryMiddleware({
*   backend: new FilesystemBackend({ rootDir: "/" }),
*   sources: [
*     "~/.deepagents/AGENTS.md",
*     "./.deepagents/AGENTS.md",
*   ],
* });
* ```
*/
function createMemoryMiddleware(options) {
	const { backend, sources, addCacheControl = false } = options;
	return createMiddleware({
		name: "MemoryMiddleware",
		stateSchema: MemoryStateSchema,
		async beforeAgent(state) {
			if ("memoryContents" in state && state.memoryContents != null) return;
			const resolvedBackend = await resolveBackend(backend, { state });
			const contents = {};
			for (const path of sources) try {
				const content = await loadMemoryFromBackend(resolvedBackend, path);
				if (content) contents[path] = content;
			} catch (error) {
				console.debug(`Failed to load memory from ${path}:`, error);
			}
			return { memoryContents: contents };
		},
		wrapModelCall(request, handler) {
			const formattedContents = formatMemoryContents(request.state?.memoryContents || {}, sources);
			const memorySection = MEMORY_SYSTEM_PROMPT.replace("{memory_contents}", formattedContents);
			const existingContent = request.systemMessage.content;
			const newSystemMessage = new SystemMessage({ content: [...typeof existingContent === "string" ? [{
				type: "text",
				text: existingContent
			}] : Array.isArray(existingContent) ? existingContent : [], {
				type: "text",
				text: memorySection,
				...addCacheControl && { cache_control: { type: "ephemeral" } }
			}] });
			return handler({
				...request,
				systemMessage: newSystemMessage
			});
		}
	});
}
//#endregion
//#region src/middleware/skills.ts
/**
* Backend-agnostic skills middleware for loading agent skills from any backend.
*
* This middleware implements Anthropic's agent skills pattern with progressive disclosure,
* loading skills from backend storage via configurable sources.
*
* ## Architecture
*
* Skills are loaded from one or more **sources** - paths in a backend where skills are
* organized. Sources are loaded in order, with later sources overriding earlier ones
* when skills have the same name (last one wins). This enables layering: base -> user
* -> project -> team skills.
*
* The middleware uses backend APIs exclusively (no direct filesystem access), making it
* portable across different storage backends (filesystem, state, remote storage, etc.).
*
* ## Usage
*
* ```typescript
* import { createSkillsMiddleware, FilesystemBackend } from "@anthropic/deepagents";
*
* const middleware = createSkillsMiddleware({
*   backend: new FilesystemBackend({ rootDir: "/" }),
*   sources: [
*     "/skills/user/",
*     "/skills/project/",
*   ],
* });
*
* const agent = createDeepAgent({ middleware: [middleware] });
* ```
*
* Or use the `skills` parameter on createDeepAgent:
*
* ```typescript
* const agent = createDeepAgent({
*   skills: ["/skills/user/", "/skills/project/"],
* });
* ```
*/
const MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024;
const MAX_SKILL_NAME_LENGTH = 64;
const MAX_SKILL_DESCRIPTION_LENGTH = 1024;
/**
* Zod schema for a single skill metadata entry.
*/
const SkillMetadataEntrySchema = z$1.object({
	name: z$1.string(),
	description: z$1.string(),
	path: z$1.string(),
	license: z$1.string().nullable().optional(),
	compatibility: z$1.string().nullable().optional(),
	metadata: z$1.record(z$1.string(), z$1.string()).optional(),
	allowedTools: z$1.array(z$1.string()).optional()
});
/**
* Reducer for skillsMetadata that merges arrays from parallel subagents.
* Skills are deduplicated by name, with later values overriding earlier ones.
*
* @param current - The current skillsMetadata array (from state)
* @param update - The new skillsMetadata array (from a subagent update)
* @returns Merged array with duplicates resolved by name (later values win)
*/
function skillsMetadataReducer(current, update) {
	if (!update || update.length === 0) return current || [];
	if (!current || current.length === 0) return update;
	const merged = /* @__PURE__ */ new Map();
	for (const skill of current) merged.set(skill.name, skill);
	for (const skill of update) merged.set(skill.name, skill);
	return Array.from(merged.values());
}
/**
* State schema for skills middleware.
* Uses ReducedValue for skillsMetadata to allow concurrent updates from parallel subagents.
*/
const SkillsStateSchema = new StateSchema({
	skillsMetadata: new ReducedValue(z$1.array(SkillMetadataEntrySchema).default(() => []), {
		inputSchema: z$1.array(SkillMetadataEntrySchema).optional(),
		reducer: skillsMetadataReducer
	}),
	files: filesValue
});
/**
* Skills System Documentation prompt template.
*/
const SKILLS_SYSTEM_PROMPT = `
## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you know they exist (name + description above), but you only read the full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches any skill's description
2. **Read the skill's full instructions**: The skill list above shows the exact path to use with read_file
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When you need specialized knowledge or structured workflows
- When a skill provides proven patterns for complex tasks

**Skills are Self-Documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it
- The skill list above shows the full path for each skill's SKILL.md file

**Executing Skill Scripts:**
Skills may contain scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills above → See "web-research" skill with its full path
2. Read the skill using the path shown in the list
3. Follow the skill's research workflow (search → organize → synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills are tools to make you more capable and consistent. When in doubt, check if a skill exists for the task!
`;
/**
* Validate skill name per Agent Skills specification.
*
* Constraints per Agent Skills specification:
*
* - 1-64 characters
* - Unicode lowercase alphanumeric and hyphens only (`a-z` and `-`).
* - Must not start or end with `-`
* - Must not contain consecutive `--`
* - Must match the parent directory name containing the `SKILL.md` file
*
* Unicode lowercase alphanumeric means any lowercase or decimal digit, which
* covers accented Latin characters (e.g., `'café'`, `'über-tool'`) and other
* scripts.
*
* @param name - The skill name from YAML frontmatter
* @param directoryName - The parent directory name
* @returns `{ valid, error }` tuple. Error is empty string if valid.
*/
function validateSkillName(name, directoryName) {
	if (!name) return {
		valid: false,
		error: "name is required"
	};
	if (name.length > 64) return {
		valid: false,
		error: "name exceeds 64 characters"
	};
	if (name.startsWith("-") || name.endsWith("-") || name.includes("--")) return {
		valid: false,
		error: "name must be lowercase alphanumeric with single hyphens only"
	};
	for (const c of name) {
		if (c === "-") continue;
		if (/\p{Ll}/u.test(c) || /\p{Nd}/u.test(c)) continue;
		return {
			valid: false,
			error: "name must be lowercase alphanumeric with single hyphens only"
		};
	}
	if (name !== directoryName) return {
		valid: false,
		error: `name '${name}' must match directory name '${directoryName}'`
	};
	return {
		valid: true,
		error: ""
	};
}
/**
* Validate and normalize the metadata field from YAML frontmatter.
*
* YAML parsing can return any type for the `metadata` key. This ensures the
* value in {@link SkillMetadata} is always a `Record<string, string>` by
* coercing via `String()` and rejecting non-object inputs.
*
* @param raw - Raw value from `frontmatterData.metadata`.
* @param skillPath - Path to the `SKILL.md` file (for warning messages).
* @returns A validated `Record<string, string>`.
*/
function validateMetadata(raw, skillPath) {
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		if (raw) console.warn(`Ignoring non-object metadata in ${skillPath} (got ${typeof raw})`);
		return {};
	}
	const result = {};
	for (const [k, v] of Object.entries(raw)) result[String(k)] = String(v);
	return result;
}
/**
* Build a parenthetical annotation string from optional skill fields.
*
* Combines license and compatibility into a comma-separated string for
* display in the system prompt skill listing.
*
* @param skill - Skill metadata to extract annotations from.
* @returns Annotation string like `'License: MIT, Compatibility: Python 3.10+'`,
*   or empty string if neither field is set.
*/
function formatSkillAnnotations(skill) {
	const parts = [];
	if (skill.license) parts.push(`License: ${skill.license}`);
	if (skill.compatibility) parts.push(`Compatibility: ${skill.compatibility}`);
	return parts.join(", ");
}
/**
* Parse YAML frontmatter from `SKILL.md` content.
*
* Extracts metadata per Agent Skills specification from YAML frontmatter
* delimited by `---` markers at the start of the content.
*
* @param content - Content of the `SKILL.md` file
* @param skillPath - Path to the `SKILL.md` file (for error messages and metadata)
* @param directoryName - Name of the parent directory containing the skill
* @returns `SkillMetadata` if parsing succeeds, `null` if parsing fails or
*   validation errors occur
*/
function parseSkillMetadataFromContent(content, skillPath, directoryName) {
	if (content.length > 10485760) {
		console.warn(`Skipping ${skillPath}: content too large (${content.length} bytes)`);
		return null;
	}
	const match = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
	if (!match) {
		console.warn(`Skipping ${skillPath}: no valid YAML frontmatter found`);
		return null;
	}
	const frontmatterStr = match[1];
	let frontmatterData;
	try {
		frontmatterData = yaml.parse(frontmatterStr);
	} catch (e) {
		console.warn(`Invalid YAML in ${skillPath}:`, e);
		return null;
	}
	if (!frontmatterData || typeof frontmatterData !== "object") {
		console.warn(`Skipping ${skillPath}: frontmatter is not a mapping`);
		return null;
	}
	const name = String(frontmatterData.name ?? "").trim();
	const description = String(frontmatterData.description ?? "").trim();
	if (!name || !description) {
		console.warn(`Skipping ${skillPath}: missing required 'name' or 'description'`);
		return null;
	}
	const validation = validateSkillName(name, directoryName);
	if (!validation.valid) console.warn(`Skill '${name}' in ${skillPath} does not follow Agent Skills specification: ${validation.error}. Consider renaming for spec compliance.`);
	let descriptionStr = description;
	if (descriptionStr.length > 1024) {
		console.warn(`Description exceeds ${MAX_SKILL_DESCRIPTION_LENGTH} characters in ${skillPath}, truncating`);
		descriptionStr = descriptionStr.slice(0, MAX_SKILL_DESCRIPTION_LENGTH);
	}
	const rawTools = frontmatterData["allowed-tools"];
	let allowedTools;
	if (rawTools) if (Array.isArray(rawTools)) allowedTools = rawTools.map((t) => String(t).trim()).filter(Boolean);
	else allowedTools = String(rawTools).split(/\s+/).filter(Boolean);
	else allowedTools = [];
	let compatibilityStr = String(frontmatterData.compatibility ?? "").trim() || null;
	if (compatibilityStr && compatibilityStr.length > 500) {
		console.warn(`Compatibility exceeds 500 characters in ${skillPath}, truncating`);
		compatibilityStr = compatibilityStr.slice(0, 500);
	}
	return {
		name,
		description: descriptionStr,
		path: skillPath,
		metadata: validateMetadata(frontmatterData.metadata ?? {}, skillPath),
		license: String(frontmatterData.license ?? "").trim() || null,
		compatibility: compatibilityStr,
		allowedTools
	};
}
/**
* List all skills from a backend source.
*/
async function listSkillsFromBackend(backend, sourcePath) {
	const adaptedBackend = adaptBackendProtocol(backend);
	const skills = [];
	const pathSep = sourcePath.includes("\\") ? "\\" : "/";
	const normalizedPath = sourcePath.endsWith("/") || sourcePath.endsWith("\\") ? sourcePath : `${sourcePath}${pathSep}`;
	let fileInfos;
	try {
		const lsResult = await adaptedBackend.ls(normalizedPath);
		if (lsResult.error || !lsResult.files) return [];
		fileInfos = lsResult.files;
	} catch {
		return [];
	}
	const entries = fileInfos.map((info) => ({
		name: info.path.replace(/[/\\]$/, "").split(/[/\\]/).pop() || "",
		type: info.is_dir ? "directory" : "file"
	}));
	for (const entry of entries) {
		if (entry.type !== "directory") continue;
		const skillMdPath = `${normalizedPath}${entry.name}${pathSep}SKILL.md`;
		let content;
		if (adaptedBackend.downloadFiles) {
			const results = await adaptedBackend.downloadFiles([skillMdPath]);
			if (results.length !== 1) continue;
			const response = results[0];
			if (response.error != null || response.content == null) continue;
			content = new TextDecoder().decode(response.content);
		} else {
			const readResult = await adaptedBackend.read(skillMdPath);
			if (readResult.error) continue;
			if (typeof readResult.content !== "string") continue;
			content = readResult.content;
		}
		const metadata = parseSkillMetadataFromContent(content, skillMdPath, entry.name);
		if (metadata) skills.push(metadata);
	}
	return skills;
}
/**
* Format skills locations for display in system prompt.
* Shows priority indicator for the last source (highest priority).
*/
function formatSkillsLocations(sources) {
	if (sources.length === 0) return "**Skills Sources:** None configured";
	const lines = [];
	for (let i = 0; i < sources.length; i++) {
		const sourcePath = sources[i];
		const name = sourcePath.replace(/[/\\]$/, "").split(/[/\\]/).filter(Boolean).pop()?.replace(/^./, (c) => c.toUpperCase()) || "Skills";
		const suffix = i === sources.length - 1 ? " (higher priority)" : "";
		lines.push(`**${name} Skills**: \`${sourcePath}\`${suffix}`);
	}
	return lines.join("\n");
}
/**
* Format skills metadata for display in system prompt.
* Shows allowed tools for each skill if specified.
*/
function formatSkillsList(skills, sources) {
	if (skills.length === 0) return `(No skills available yet. You can create skills in ${sources.map((s) => `\`${s}\``).join(" or ")})`;
	const lines = [];
	for (const skill of skills) {
		const annotations = formatSkillAnnotations(skill);
		let descLine = `- **${skill.name}**: ${skill.description}`;
		if (annotations) descLine += ` (${annotations})`;
		lines.push(descLine);
		if (skill.allowedTools && skill.allowedTools.length > 0) lines.push(`  → Allowed tools: ${skill.allowedTools.join(", ")}`);
		lines.push(`  → Read \`${skill.path}\` for full instructions`);
	}
	return lines.join("\n");
}
/**
* Create backend-agnostic middleware for loading and exposing agent skills.
*
* This middleware loads skills from configurable backend sources and injects
* skill metadata into the system prompt. It implements the progressive disclosure
* pattern: skill names and descriptions are shown in the prompt, but the agent
* reads full SKILL.md content only when needed.
*
* @param options - Configuration options
* @returns AgentMiddleware for skills loading and injection
*
* @example
* ```typescript
* const middleware = createSkillsMiddleware({
*   backend: new FilesystemBackend({ rootDir: "/" }),
*   sources: ["/skills/user/", "/skills/project/"],
* });
* ```
*/
function createSkillsMiddleware(options) {
	const { backend, sources } = options;
	let loadedSkills = [];
	return createMiddleware({
		name: "SkillsMiddleware",
		stateSchema: SkillsStateSchema,
		async beforeAgent(state) {
			if (loadedSkills.length > 0) return;
			if ("skillsMetadata" in state && Array.isArray(state.skillsMetadata) && state.skillsMetadata.length > 0) {
				loadedSkills = state.skillsMetadata;
				return;
			}
			const resolvedBackend = await resolveBackend(backend, { state });
			const allSkills = /* @__PURE__ */ new Map();
			for (const sourcePath of sources) try {
				const skills = await listSkillsFromBackend(resolvedBackend, sourcePath);
				for (const skill of skills) allSkills.set(skill.name, skill);
			} catch (error) {
				console.debug(`[BackendSkillsMiddleware] Failed to load skills from ${sourcePath}:`, error);
			}
			loadedSkills = Array.from(allSkills.values());
			return { skillsMetadata: loadedSkills };
		},
		wrapModelCall(request, handler) {
			const skillsMetadata = loadedSkills.length > 0 ? loadedSkills : request.state?.skillsMetadata || [];
			const skillsLocations = formatSkillsLocations(sources);
			const skillsList = formatSkillsList(skillsMetadata, sources);
			const skillsSection = SKILLS_SYSTEM_PROMPT.replace("{skills_locations}", skillsLocations).replace("{skills_list}", skillsList);
			const newSystemMessage = request.systemMessage.concat(skillsSection);
			return handler({
				...request,
				systemMessage: newSystemMessage
			});
		}
	});
}
//#endregion
//#region src/middleware/summarization.ts
/**
* Summarization middleware with backend support for conversation history offloading.
*
* This module extends the base LangChain summarization middleware with additional
* backend-based features for persisting conversation history before summarization.
*
* ## Usage
*
* ```typescript
* import { createSummarizationMiddleware } from "@anthropic/deepagents";
* import { FilesystemBackend } from "@anthropic/deepagents";
*
* const backend = new FilesystemBackend({ rootDir: "/data" });
*
* const middleware = createSummarizationMiddleware({
*   model: "gpt-4o-mini",
*   backend,
*   trigger: { type: "fraction", value: 0.85 },
*   keep: { type: "fraction", value: 0.10 },
* });
*
* const agent = createDeepAgent({ middleware: [middleware] });
* ```
*
* ## Storage
*
* Offloaded messages are stored as markdown at `/conversation_history/{thread_id}.md`.
*
* Each summarization event appends a new section to this file, creating a running log
* of all evicted messages.
*
* ## Relationship to LangChain Summarization Middleware
*
* The base `summarizationMiddleware` from `langchain` provides core summarization
* functionality. This middleware adds:
* - Backend-based conversation history offloading
* - Tool argument truncation for old messages
*
* For simple use cases without backend offloading, use `summarizationMiddleware`
* from `langchain` directly.
*/
const DEFAULT_MESSAGES_TO_KEEP = 20;
const DEFAULT_TRIM_TOKEN_LIMIT = 4e3;
const FALLBACK_TRIGGER = {
	type: "tokens",
	value: 17e4
};
const FALLBACK_KEEP = {
	type: "messages",
	value: 6
};
const FALLBACK_TRUNCATE_ARGS = {
	trigger: {
		type: "messages",
		value: 20
	},
	keep: {
		type: "messages",
		value: 20
	}
};
const PROFILE_TRIGGER = {
	type: "fraction",
	value: .85
};
const PROFILE_KEEP = {
	type: "fraction",
	value: .1
};
const PROFILE_TRUNCATE_ARGS = {
	trigger: {
		type: "fraction",
		value: .85
	},
	keep: {
		type: "fraction",
		value: .1
	}
};
/**
* Compute summarization defaults based on model profile.
* Mirrors Python's `_compute_summarization_defaults`.
*
* If the model has a profile with `maxInputTokens`, uses fraction-based
* settings. Otherwise, uses fixed token/message counts.
*
* @param resolvedModel - The resolved chat model instance.
*/
function computeSummarizationDefaults(resolvedModel) {
	if (resolvedModel.profile && typeof resolvedModel.profile === "object" && "maxInputTokens" in resolvedModel.profile && typeof resolvedModel.profile.maxInputTokens === "number") return {
		trigger: PROFILE_TRIGGER,
		keep: PROFILE_KEEP,
		truncateArgsSettings: PROFILE_TRUNCATE_ARGS
	};
	return {
		trigger: FALLBACK_TRIGGER,
		keep: FALLBACK_KEEP,
		truncateArgsSettings: FALLBACK_TRUNCATE_ARGS
	};
}
const DEFAULT_SUMMARY_PROMPT = `You are a conversation summarizer. Your task is to create a concise summary of the conversation that captures:
1. The main topics discussed
2. Key decisions or conclusions reached
3. Any important context that would be needed for continuing the conversation

Keep the summary focused and informative. Do not include unnecessary details.

Conversation to summarize:
{conversation}

Summary:`;
/**
* Zod schema for a summarization event that tracks what was summarized and
* where the cutoff is.
*
* Instead of rewriting LangGraph state with `RemoveMessage(REMOVE_ALL_MESSAGES)`,
* the middleware stores this event and uses it to reconstruct the effective message
* list on subsequent calls.
*/
const SummarizationEventSchema = z$1.object({
	cutoffIndex: z$1.number(),
	summaryMessage: z$1.instanceof(HumanMessage),
	filePath: z$1.string().nullable()
});
/**
* State schema for summarization middleware.
*/
const SummarizationStateSchema = z$1.object({
	_summarizationSessionId: z$1.string().optional(),
	_summarizationEvent: SummarizationEventSchema.optional()
});
/**
* Check if a message is a previous summarization message.
* Summary messages are HumanMessage objects with lc_source='summarization' in additional_kwargs.
*/
function isSummaryMessage(msg) {
	if (!HumanMessage.isInstance(msg)) return false;
	return msg.additional_kwargs?.lc_source === "summarization";
}
/**
* Create summarization middleware with backend support for conversation history offloading.
*
* This middleware:
* 1. Monitors conversation length against configured thresholds
* 2. When triggered, offloads old messages to backend storage
* 3. Generates a summary of offloaded messages
* 4. Replaces old messages with the summary, preserving recent context
*
* @param options - Configuration options
* @returns AgentMiddleware for summarization and history offloading
*/
function createSummarizationMiddleware(options) {
	const { model, backend, summaryPrompt = DEFAULT_SUMMARY_PROMPT, trimTokensToSummarize = DEFAULT_TRIM_TOKEN_LIMIT, historyPathPrefix = "/conversation_history" } = options;
	let trigger = options.trigger;
	let keep = options.keep ?? {
		type: "messages",
		value: DEFAULT_MESSAGES_TO_KEEP
	};
	let truncateArgsSettings = options.truncateArgsSettings;
	let defaultsComputed = trigger != null;
	let truncateTrigger = truncateArgsSettings?.trigger;
	let truncateKeep = truncateArgsSettings?.keep ?? {
		type: "messages",
		value: 20
	};
	let maxArgLength = truncateArgsSettings?.maxLength ?? 2e3;
	let truncationText = truncateArgsSettings?.truncationText ?? "...(argument truncated)";
	/**
	* Lazily compute defaults from model profile when trigger was not provided.
	* Called once when the model is first resolved.
	*/
	function applyModelDefaults(resolvedModel) {
		if (defaultsComputed) return;
		defaultsComputed = true;
		const defaults = computeSummarizationDefaults(resolvedModel);
		trigger = defaults.trigger;
		keep = options.keep ?? defaults.keep;
		if (!options.truncateArgsSettings) {
			truncateArgsSettings = defaults.truncateArgsSettings;
			truncateTrigger = defaults.truncateArgsSettings.trigger;
			truncateKeep = defaults.truncateArgsSettings.keep ?? {
				type: "messages",
				value: 20
			};
			maxArgLength = defaults.truncateArgsSettings.maxLength ?? 2e3;
			truncationText = defaults.truncateArgsSettings.truncationText ?? "...(argument truncated)";
		}
	}
	let sessionId = null;
	let tokenEstimationMultiplier = 1;
	/**
	* Get or create session ID for history file naming.
	*/
	function getSessionId(state) {
		if (state._summarizationSessionId) return state._summarizationSessionId;
		if (!sessionId) sessionId = `session_${crypto.randomUUID().substring(0, 8)}`;
		return sessionId;
	}
	/**
	* Get the history file path.
	*/
	function getHistoryPath(state) {
		return `${historyPathPrefix}/${getSessionId(state)}.md`;
	}
	/**
	* Cached resolved model to avoid repeated initChatModel calls
	*/
	let cachedModel = void 0;
	/**
	* Resolve the chat model.
	* Uses initChatModel to support any model provider from a string name.
	* The resolved model is cached for subsequent calls.
	*/
	async function getChatModel() {
		if (cachedModel) return cachedModel;
		if (typeof model === "string") cachedModel = await initChatModel(model);
		else cachedModel = model;
		return cachedModel;
	}
	/**
	* Get the max input tokens from the model's profile.
	* Similar to Python's _get_profile_limits.
	*
	* When the profile is unavailable, returns undefined. In that case the
	* middleware uses fixed token/message-count fallback defaults for
	* trigger/keep, and relies on the ContextOverflowError catch as a
	* safety net if the prompt still exceeds the model's actual limit.
	*/
	function getMaxInputTokens(resolvedModel) {
		const profile = resolvedModel.profile;
		if (profile && typeof profile === "object" && "maxInputTokens" in profile && typeof profile.maxInputTokens === "number") return profile.maxInputTokens;
	}
	/**
	* Check if summarization should be triggered.
	*/
	function shouldSummarize(messages, totalTokens, maxInputTokens) {
		if (!trigger) return false;
		const adjustedTokens = totalTokens * tokenEstimationMultiplier;
		const triggers = Array.isArray(trigger) ? trigger : [trigger];
		for (const t of triggers) {
			if (t.type === "messages" && messages.length >= t.value) return true;
			if (t.type === "tokens" && adjustedTokens >= t.value) return true;
			if (t.type === "fraction" && maxInputTokens) {
				if (adjustedTokens >= Math.floor(maxInputTokens * t.value)) return true;
			}
		}
		return false;
	}
	/**
	* Find a safe cutoff point that doesn't split AI/Tool message pairs.
	*
	* If the message at `cutoffIndex` is a ToolMessage, this adjusts the boundary
	* so that related AI and Tool messages stay together. Two strategies are used:
	*
	* 1. **Move backward** to include the AIMessage that produced the tool calls,
	*    keeping the pair in the preserved set. Preferred when it doesn't move
	*    the cutoff too far back.
	*
	* 2. **Advance forward** past all consecutive ToolMessages, putting the entire
	*    pair into the summarized set. Used when moving backward would preserve
	*    too many messages (e.g., a single AIMessage made 20+ tool calls).
	*/
	function findSafeCutoffPoint(messages, cutoffIndex) {
		if (cutoffIndex >= messages.length || !ToolMessage.isInstance(messages[cutoffIndex])) return cutoffIndex;
		let forwardIdx = cutoffIndex;
		while (forwardIdx < messages.length && ToolMessage.isInstance(messages[forwardIdx])) forwardIdx++;
		const toolCallIds = /* @__PURE__ */ new Set();
		for (let i = cutoffIndex; i < forwardIdx; i++) {
			const toolMsg = messages[i];
			if (toolMsg.tool_call_id) toolCallIds.add(toolMsg.tool_call_id);
		}
		let backwardIdx = null;
		for (let i = cutoffIndex - 1; i >= 0; i--) {
			const msg = messages[i];
			if (AIMessage.isInstance(msg) && msg.tool_calls) {
				const aiToolCallIds = new Set(msg.tool_calls.map((tc) => tc.id).filter((id) => id != null));
				for (const id of toolCallIds) if (aiToolCallIds.has(id)) {
					backwardIdx = i;
					break;
				}
				if (backwardIdx !== null) break;
			}
		}
		if (backwardIdx === null) return forwardIdx;
		if (cutoffIndex - backwardIdx > cutoffIndex / 2 && cutoffIndex > 2) return forwardIdx;
		return backwardIdx;
	}
	/**
	* Determine cutoff index for messages to summarize.
	* Messages at index < cutoff will be summarized.
	* Messages at index >= cutoff will be preserved.
	*
	* Uses findSafeCutoffPoint to ensure tool call/result pairs stay together.
	*/
	function determineCutoffIndex(messages, maxInputTokens) {
		let rawCutoff;
		if (keep.type === "messages") {
			if (messages.length <= keep.value) return 0;
			rawCutoff = messages.length - keep.value;
		} else if (keep.type === "tokens" || keep.type === "fraction") {
			const targetTokenCount = keep.type === "fraction" && maxInputTokens ? Math.floor(maxInputTokens * keep.value) : keep.value;
			let tokensKept = 0;
			rawCutoff = 0;
			for (let i = messages.length - 1; i >= 0; i--) {
				const msgTokens = countTokensApproximately([messages[i]]);
				if (tokensKept + msgTokens > targetTokenCount) {
					rawCutoff = i + 1;
					break;
				}
				tokensKept += msgTokens;
			}
		} else return 0;
		return findSafeCutoffPoint(messages, rawCutoff);
	}
	/**
	* Check if argument truncation should be triggered.
	*/
	function shouldTruncateArgs(messages, totalTokens, maxInputTokens) {
		if (!truncateTrigger) return false;
		const adjustedTokens = totalTokens * tokenEstimationMultiplier;
		if (truncateTrigger.type === "messages") return messages.length >= truncateTrigger.value;
		if (truncateTrigger.type === "tokens") return adjustedTokens >= truncateTrigger.value;
		if (truncateTrigger.type === "fraction" && maxInputTokens) return adjustedTokens >= Math.floor(maxInputTokens * truncateTrigger.value);
		return false;
	}
	/**
	* Determine cutoff index for argument truncation.
	* Uses findSafeCutoffPoint to ensure tool call/result pairs stay together.
	*/
	function determineTruncateCutoffIndex(messages, maxInputTokens) {
		let rawCutoff;
		if (truncateKeep.type === "messages") {
			if (messages.length <= truncateKeep.value) return messages.length;
			rawCutoff = messages.length - truncateKeep.value;
		} else if (truncateKeep.type === "tokens" || truncateKeep.type === "fraction") {
			const targetTokenCount = truncateKeep.type === "fraction" && maxInputTokens ? Math.floor(maxInputTokens * truncateKeep.value) : truncateKeep.value;
			let tokensKept = 0;
			rawCutoff = 0;
			for (let i = messages.length - 1; i >= 0; i--) {
				const msgTokens = countTokensApproximately([messages[i]]);
				if (tokensKept + msgTokens > targetTokenCount) {
					rawCutoff = i + 1;
					break;
				}
				tokensKept += msgTokens;
			}
		} else return messages.length;
		return findSafeCutoffPoint(messages, rawCutoff);
	}
	/**
	* Count tokens including system message and tools, matching Python's approach.
	* This gives a more accurate picture of what actually gets sent to the model.
	*/
	function countTotalTokens(messages, systemMessage, tools) {
		return countTokensApproximately(systemMessage && SystemMessage.isInstance(systemMessage) ? [systemMessage, ...messages] : [...messages], tools && Array.isArray(tools) && tools.length > 0 ? tools : null);
	}
	/**
	* Truncate ToolMessage content so that the total payload fits within the
	* model's context window. Each ToolMessage gets an equal share of the
	* remaining token budget after accounting for non-tool messages, system
	* message, and tool schemas.
	*
	* This is critical for conversations where a single AIMessage triggers
	* many tool calls whose results collectively exceed the context window.
	* Without this, findSafeCutoffPoint cannot split the AI/Tool group and
	* summarization would discard everything, causing the model to re-call
	* the same tools in an infinite loop.
	*/
	function compactToolResults(messages, maxInputTokens, systemMessage, tools) {
		const toolMessageIndices = [];
		for (let i = 0; i < messages.length; i++) if (ToolMessage.isInstance(messages[i])) toolMessageIndices.push(i);
		if (toolMessageIndices.length === 0) return {
			messages,
			modified: false
		};
		const overheadTokens = countTotalTokens(messages.filter((m) => !ToolMessage.isInstance(m)), systemMessage, tools);
		const adjustedMax = maxInputTokens / tokenEstimationMultiplier;
		const budgetForTools = Math.max(adjustedMax * .7 - overheadTokens, 1e3);
		const perToolBudgetChars = Math.floor(budgetForTools / toolMessageIndices.length) * 4;
		let modified = false;
		const result = [...messages];
		for (const idx of toolMessageIndices) {
			const msg = messages[idx];
			const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
			if (content.length > perToolBudgetChars) {
				result[idx] = new ToolMessage({
					content: content.substring(0, perToolBudgetChars) + "\n...(result truncated)",
					tool_call_id: msg.tool_call_id,
					name: msg.name
				});
				modified = true;
			}
		}
		return {
			messages: result,
			modified
		};
	}
	/**
	* Truncate large tool arguments in old messages.
	*/
	function truncateArgs(messages, maxInputTokens, systemMessage, tools) {
		if (!shouldTruncateArgs(messages, countTotalTokens(messages, systemMessage, tools), maxInputTokens)) return {
			messages,
			modified: false
		};
		const cutoffIndex = determineTruncateCutoffIndex(messages, maxInputTokens);
		if (cutoffIndex >= messages.length) return {
			messages,
			modified: false
		};
		const truncatedMessages = [];
		let modified = false;
		for (let i = 0; i < messages.length; i++) {
			const msg = messages[i];
			if (i < cutoffIndex && AIMessage.isInstance(msg) && msg.tool_calls) {
				const truncatedToolCalls = msg.tool_calls.map((toolCall) => {
					const args = toolCall.args || {};
					const truncatedArgs = {};
					let toolModified = false;
					for (const [key, value] of Object.entries(args)) if (typeof value === "string" && value.length > maxArgLength && (toolCall.name === "write_file" || toolCall.name === "edit_file")) {
						truncatedArgs[key] = value.substring(0, 20) + truncationText;
						toolModified = true;
					} else truncatedArgs[key] = value;
					if (toolModified) {
						modified = true;
						return {
							...toolCall,
							args: truncatedArgs
						};
					}
					return toolCall;
				});
				if (modified) {
					const truncatedMsg = new AIMessage({
						content: msg.content,
						tool_calls: truncatedToolCalls,
						additional_kwargs: msg.additional_kwargs
					});
					truncatedMessages.push(truncatedMsg);
				} else truncatedMessages.push(msg);
			} else truncatedMessages.push(msg);
		}
		return {
			messages: truncatedMessages,
			modified
		};
	}
	/**
	* Filter out previous summary messages.
	*/
	function filterSummaryMessages(messages) {
		return messages.filter((msg) => !isSummaryMessage(msg));
	}
	/**
	* Offload messages to backend by appending to the history file.
	*
	* Uses uploadFiles() directly with raw byte concatenation instead of
	* edit() to avoid downloading the file twice and performing a full
	* string search-and-replace. This keeps peak memory at ~2x file size
	* (existing bytes + combined bytes) instead of ~6x with the old
	* download → edit(oldContent, newContent) approach.
	*/
	async function offloadToBackend(resolvedBackend, messages, state) {
		const filePath = getHistoryPath(state);
		const filteredMessages = filterSummaryMessages(messages);
		const newSection = `## Summarized at ${(/* @__PURE__ */ new Date()).toISOString()}\n\n${getBufferString(filteredMessages)}\n\n`;
		const sectionBytes = new TextEncoder().encode(newSection);
		try {
			let existingBytes = null;
			if (resolvedBackend.downloadFiles) try {
				const responses = await resolvedBackend.downloadFiles([filePath]);
				if (responses.length > 0 && responses[0].content && !responses[0].error) existingBytes = responses[0].content;
			} catch {}
			let result;
			if (existingBytes && resolvedBackend.uploadFiles) {
				const combined = new Uint8Array(existingBytes.byteLength + sectionBytes.byteLength);
				combined.set(existingBytes, 0);
				combined.set(sectionBytes, existingBytes.byteLength);
				const uploadResults = await resolvedBackend.uploadFiles([[filePath, combined]]);
				result = uploadResults[0].error ? { error: uploadResults[0].error } : { path: filePath };
			} else if (!existingBytes) result = await resolvedBackend.write(filePath, newSection);
			else {
				const existingContent = new TextDecoder().decode(existingBytes);
				result = await resolvedBackend.edit(filePath, existingContent, existingContent + newSection);
			}
			if (result.error) {
				console.warn(`Failed to offload conversation history to ${filePath}: ${result.error}`);
				return null;
			}
			return filePath;
		} catch (e) {
			console.warn(`Exception offloading conversation history to ${filePath}:`, e);
			return null;
		}
	}
	/**
	* Create summary of messages.
	*/
	async function createSummary(messages, chatModel) {
		let messagesToSummarize = messages;
		if (countTokensApproximately(messages) > trimTokensToSummarize) {
			let kept = 0;
			const trimmedMessages = [];
			for (let i = messages.length - 1; i >= 0; i--) {
				const msgTokens = countTokensApproximately([messages[i]]);
				if (kept + msgTokens > trimTokensToSummarize) break;
				trimmedMessages.unshift(messages[i]);
				kept += msgTokens;
			}
			messagesToSummarize = trimmedMessages;
		}
		const conversation = getBufferString(messagesToSummarize);
		const prompt = summaryPrompt.replace("{conversation}", conversation);
		const response = await chatModel.invoke([new HumanMessage({ content: prompt })]);
		return typeof response.content === "string" ? response.content : JSON.stringify(response.content);
	}
	/**
	* Build the summary message with file path reference.
	*/
	function buildSummaryMessage(summary, filePath) {
		let content;
		if (filePath) content = context`
        You are in the middle of a conversation that has been summarized.

        The full conversation history has been saved to ${filePath} should you need to refer back to it for details.

        A condensed summary follows:

        <summary>
        ${summary}
        </summary>
      `;
		else content = `Here is a summary of the conversation to date:\n\n${summary}`;
		return new HumanMessage({
			content,
			additional_kwargs: { lc_source: "summarization" }
		});
	}
	/**
	* Reconstruct the effective message list based on any previous summarization event.
	*
	* After summarization, instead of using all messages from state, we use the summary
	* message plus messages after the cutoff index. This avoids full state rewrites.
	*/
	function getEffectiveMessages(messages, state) {
		const event = state._summarizationEvent;
		if (!event) return messages;
		const result = [event.summaryMessage];
		result.push(...messages.slice(event.cutoffIndex));
		return result;
	}
	/**
	* Summarize a set of messages using the given model and build the
	* summary message + backend offload. Returns the summary message,
	* the file path, and the state cutoff index.
	*/
	async function summarizeMessages(messagesToSummarize, resolvedModel, state, previousCutoffIndex, cutoffIndex) {
		const filePath = await offloadToBackend(await resolveBackend(backend, { state }), messagesToSummarize, state);
		if (filePath === null) console.warn(`[SummarizationMiddleware] Backend offload failed during summarization. Proceeding with summary generation.`);
		return {
			summaryMessage: buildSummaryMessage(await createSummary(messagesToSummarize, resolvedModel), filePath),
			filePath,
			stateCutoffIndex: previousCutoffIndex != null ? previousCutoffIndex + cutoffIndex - 1 : cutoffIndex
		};
	}
	/**
	* Check if an error (possibly wrapped in MiddlewareError layers) is a
	* ContextOverflowError by walking the `cause` chain.
	*/
	function isContextOverflow(err) {
		let cause = err;
		for (;;) {
			if (!cause) break;
			if (ContextOverflowError.isInstance(cause)) return true;
			cause = typeof cause === "object" && "cause" in cause ? cause.cause : void 0;
		}
		return false;
	}
	async function performSummarization(request, handler, truncatedMessages, resolvedModel, maxInputTokens) {
		const cutoffIndex = determineCutoffIndex(truncatedMessages, maxInputTokens);
		if (cutoffIndex <= 0) return handler({
			...request,
			messages: truncatedMessages
		});
		const messagesToSummarize = truncatedMessages.slice(0, cutoffIndex);
		const preservedMessages = truncatedMessages.slice(cutoffIndex);
		if (preservedMessages.length === 0 && maxInputTokens) {
			const compact = compactToolResults(truncatedMessages, maxInputTokens, request.systemMessage, request.tools);
			if (compact.modified) try {
				return await handler({
					...request,
					messages: compact.messages
				});
			} catch (err) {
				if (!isContextOverflow(err)) throw err;
			}
		}
		const previousEvent = request.state._summarizationEvent;
		const previousCutoffIndex = previousEvent != null ? previousEvent.cutoffIndex : void 0;
		const { summaryMessage, filePath, stateCutoffIndex } = await summarizeMessages(messagesToSummarize, resolvedModel, request.state, previousCutoffIndex, cutoffIndex);
		let modifiedMessages = [summaryMessage, ...preservedMessages];
		const modifiedTokens = countTotalTokens(modifiedMessages, request.systemMessage, request.tools);
		let finalStateCutoffIndex = stateCutoffIndex;
		let finalSummaryMessage = summaryMessage;
		let finalFilePath = filePath;
		try {
			await handler({
				...request,
				messages: modifiedMessages
			});
		} catch (err) {
			if (!isContextOverflow(err)) throw err;
			if (maxInputTokens && modifiedTokens > 0) {
				const observedRatio = maxInputTokens / modifiedTokens;
				if (observedRatio > tokenEstimationMultiplier) tokenEstimationMultiplier = observedRatio * 1.1;
			}
			const reSumResult = await summarizeMessages([...messagesToSummarize, ...preservedMessages], resolvedModel, request.state, previousCutoffIndex, truncatedMessages.length);
			finalSummaryMessage = reSumResult.summaryMessage;
			finalFilePath = reSumResult.filePath;
			finalStateCutoffIndex = reSumResult.stateCutoffIndex;
			modifiedMessages = [reSumResult.summaryMessage];
			await handler({
				...request,
				messages: modifiedMessages
			});
		}
		return new Command({ update: {
			_summarizationEvent: {
				cutoffIndex: finalStateCutoffIndex,
				summaryMessage: finalSummaryMessage,
				filePath: finalFilePath
			},
			_summarizationSessionId: getSessionId(request.state)
		} });
	}
	return createMiddleware({
		name: "SummarizationMiddleware",
		stateSchema: SummarizationStateSchema,
		async wrapModelCall(request, handler) {
			const effectiveMessages = getEffectiveMessages(request.messages ?? [], request.state);
			if (effectiveMessages.length === 0) return handler(request);
			/**
			* Resolve the chat model and get max input tokens from its profile.
			*/
			const resolvedModel = await getChatModel();
			const maxInputTokens = getMaxInputTokens(resolvedModel);
			applyModelDefaults(resolvedModel);
			/**
			* Step 1: Truncate args if configured
			*/
			const { messages: truncatedMessages } = truncateArgs(effectiveMessages, maxInputTokens, request.systemMessage, request.tools);
			/**
			* Step 2: Check if summarization should happen.
			* Count tokens including system message and tools to match what's
			* actually sent to the model (matching Python implementation).
			*/
			const totalTokens = countTotalTokens(truncatedMessages, request.systemMessage, request.tools);
			/**
			* If no summarization needed, try passing through.
			* If the handler throws a ContextOverflowError, fall back to
			* emergency summarization (matching Python's behavior).
			*/
			if (!shouldSummarize(truncatedMessages, totalTokens, maxInputTokens)) try {
				return await handler({
					...request,
					messages: truncatedMessages
				});
			} catch (err) {
				if (!isContextOverflow(err)) throw err;
				if (maxInputTokens && totalTokens > 0) {
					const observedRatio = maxInputTokens / totalTokens;
					if (observedRatio > tokenEstimationMultiplier) tokenEstimationMultiplier = observedRatio * 1.1;
				}
			}
			/**
			* Step 3: Perform summarization
			*/
			return performSummarization(request, handler, truncatedMessages, resolvedModel, maxInputTokens);
		}
	});
}
//#endregion
//#region src/middleware/async_subagents.ts
function toolCallIdFromRuntime(runtime) {
	return runtime.toolCall?.id ?? runtime.toolCallId ?? "";
}
/**
* Zod schema for {@link AsyncTask}.
*
* Used by the {@link ReducedValue} in the state schema so that LangGraph
* can validate and serialize task records stored in `asyncTasks`.
*/
const AsyncTaskSchema = z.object({
	taskId: z.string(),
	agentName: z.string(),
	threadId: z.string(),
	runId: z.string(),
	status: z.string(),
	createdAt: z.string(),
	description: z.string().optional(),
	updatedAt: z.string().optional(),
	checkedAt: z.string().optional()
});
/**
* State schema for the async subagent middleware.
*
* Declares `asyncTasks` as a reduced state channel so that individual
* tool updates (launch, check, update, cancel, list) merge into the existing
* tasks dict rather than replacing it wholesale.
*/
const AsyncTaskStateSchema = new StateSchema({ asyncTasks: new ReducedValue(z.record(z.string(), AsyncTaskSchema).default(() => ({})), {
	inputSchema: z.record(z.string(), AsyncTaskSchema).optional(),
	reducer: asyncTasksReducer
}) });
/**
* Reducer for the `asyncTasks` state channel.
*
* Merges task updates into the existing tasks dict using shallow spread.
* This allows individual tools to update a single task without overwriting
* the full map — only the keys present in `update` are replaced.
*
* @param existing - The current tasks dict from state (may be undefined on first write).
* @param update - New or updated task entries to merge in.
* @returns Merged tasks dict.
*/
function asyncTasksReducer(existing, update) {
	return {
		...existing || {},
		...update || {}
	};
}
/**
* Description template for the `start_async_task` tool.
*
* The `{available_agents}` placeholder is replaced at middleware creation
* time with a formatted list of configured async subagent names and descriptions.
*/
const ASYNC_TASK_TOOL_DESCRIPTION = `Launch an async subagent on a remote server. The subagent runs in the background and returns a task ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background task and returns immediately with a task ID. Report the task ID to the user and stop — do NOT immediately check status.
2. Use \`check_async_task\` only when the user asks for a status update or result.
3. Use \`update_async_task\` to send new instructions to a running task.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs on a remote server, so it has its own tools and capabilities.`;
/**
* Default system prompt appended to the main agent's system message when
* async subagent middleware is active.
*
* Provides the agent with instructions on how to use the five async subagent
* tools (launch, check, update, cancel, list) including workflow ordering,
* critical rules about polling behavior, and guidance on when to use async
* subagents vs. synchronous delegation.
*/
const ASYNC_TASK_SYSTEM_PROMPT = `## Async subagents (remote servers)

You have access to async subagent tools that launch background tasks on remote servers.

### Tools:
- \`start_async_task\`: Start a new background task. Returns a task ID immediately.
- \`check_async_task\`: Check the status of a running task. Returns status and result if complete.
- \`update_async_task\`: Send an update or new instructions to a running task.
- \`cancel_async_task\`: Cancel a running task that is no longer needed.
- \`list_async_tasks\`: List all tracked tasks with live statuses. Use this to check all tasks at once.

### Workflow:
1. **Launch** — Use \`start_async_task\` to start a task. Report the task ID to the user and stop.
   Do NOT immediately check the status — the task runs in the background while you and the user continue other work.
2. **Check (on request)** — Only use \`check_async_task\` when the user explicitly asks for a status update or
   result. If the status is "running", report that and stop — do not poll in a loop.
3. **Update** (optional) — Use \`update_async_task\` to send new instructions to a running task. This interrupts
   the current run and starts a fresh one on the same thread. The task_id stays the same.
4. **Cancel** (optional) — Use \`cancel_async_task\` to stop a task that is no longer needed.
5. **Collect** — When \`check_async_task\` returns status "success", the result is included in the response.
6. **List** — Use \`list_async_tasks\` to see live statuses for all tasks at once, or to recall task IDs after context compaction.

### Critical rules:
- After launching, ALWAYS return control to the user immediately. Never auto-check after launching.
- Never poll \`check_async_task\` in a loop. Check once per user request, then stop.
- If a check returns "running", tell the user and wait for them to ask again.
- Task statuses in conversation history are ALWAYS stale — a task that was "running" may now be done.
  NEVER report a status from a previous tool result. ALWAYS call a tool to get the current status:
  use \`list_async_tasks\` when the user asks about multiple tasks or "all tasks",
  use \`check_async_task\` when the user asks about a specific task.
- Always show the full task_id — never truncate or abbreviate it.

### When to use async subagents:
- Long-running tasks that would block the main agent
- Tasks that benefit from running on specialized remote deployments
- When you want to run multiple tasks concurrently and collect results later`;
/**
* Task statuses that will never change.
*
* When listing tasks, live-status fetches are skipped for tasks whose
* cached status is in this set, since they are guaranteed to be final.
*/
/**
* Names of the tools added by the async subagent middleware.
*
* Exported so `agent.ts` can include them in `BUILTIN_TOOL_NAMES` and
* surface a `ConfigurationError` if a user-provided tool collides.
*/
const ASYNC_TASK_TOOL_NAMES = [
	"start_async_task",
	"check_async_task",
	"update_async_task",
	"cancel_async_task",
	"list_async_tasks"
];
const TERMINAL_STATUSES = new Set([
	"cancelled",
	"success",
	"error",
	"timeout",
	"interrupted"
]);
/**
* Look up a tracked task from state by its `taskId`.
*
* @param taskId - The task ID to look up (will be trimmed).
* @param state - The current agent state containing `asyncTasks`.
* @returns The tracked task on success, or an error string.
*/
function resolveTrackedTask(taskId, state) {
	const tracked = (state.asyncTasks ?? {})[taskId.trim()];
	if (!tracked) return `No tracked task found for taskId: '${taskId}'`;
	return tracked;
}
/**
* Build a check result from a run's current status and thread state values.
*
* For successful runs, extracts the last message's content from the remote
* thread's state values. For errored runs, includes a generic error message.
*
* @param run - The run object from the SDK.
* @param threadId - The thread ID for the run.
* @param threadValues - The `values` from `ThreadState` (the remote subagent's state).
*/
function buildCheckResult(run, threadId, threadValues) {
	const checkResult = {
		status: run.status,
		threadId
	};
	if (run.status === "success") {
		const messages = (Array.isArray(threadValues) ? {} : threadValues)?.messages ?? [];
		if (messages.length > 0) {
			const last = messages[messages.length - 1];
			const rawContent = typeof last === "object" && last !== null && "content" in last ? last.content : last;
			checkResult.result = typeof rawContent === "string" ? rawContent : JSON.stringify(rawContent);
		} else checkResult.result = "Completed with no output messages.";
	} else if (run.status === "error") checkResult.error = "The async subagent encountered an error.";
	return checkResult;
}
/**
* Filter tasks by cached status from agent state.
*
* Filtering uses the cached status, not live server status. Live statuses
* are fetched after filtering by the calling tool.
*
* @param tasks - All tracked tasks from state.
* @param statusFilter - If nullish or `'all'`, return all tasks.
*   Otherwise return only tasks whose cached status matches.
*/
function filterTasks(tasks, statusFilter) {
	if (!statusFilter || statusFilter === "all") return Object.values(tasks);
	return Object.values(tasks).filter((task) => task.status === statusFilter);
}
/**
* Fetch the current run status from the server.
*
* Returns the cached status immediately for terminal tasks (avoiding
* unnecessary API calls). Falls back to the cached status on SDK errors.
*/
async function fetchLiveTaskStatus(clients, task) {
	if (TERMINAL_STATUSES.has(task.status)) return task.status;
	try {
		return (await clients.getClient(task.agentName).runs.get(task.threadId, task.runId)).status;
	} catch {
		return task.status;
	}
}
/**
* Format a single task as a display string for list output.
*/
function formatTaskEntry(task, status) {
	return `- taskId: ${task.taskId} agent: ${task.agentName} status: ${status}`;
}
/**
* Lazily-created, cached LangGraph SDK clients keyed by (url, headers).
*
* Agents that share the same URL and headers will reuse a single `Client`
* instance, avoiding unnecessary connections.
*/
var ClientCache = class {
	agents;
	clients = /* @__PURE__ */ new Map();
	constructor(agents) {
		this.agents = agents;
	}
	/**
	* Build headers for a remote Agent Protocol server.
	*
	* Adds `x-auth-scheme: langsmith` by default unless already provided.
	* For self-hosted servers that don't require this header, it is typically
	* ignored. Override via the `headers` field on the AsyncSubAgent config.
	*/
	resolveHeaders(spec) {
		const headers = { ...spec.headers || {} };
		if (!("x-auth-scheme" in headers)) headers["x-auth-scheme"] = "langsmith";
		return headers;
	}
	/**
	* Build a stable cache key from a spec's url and resolved headers.
	*/
	cacheKey(spec) {
		const headers = this.resolveHeaders(spec);
		const headerStr = Object.entries(headers).sort().flat().join(":");
		return `${spec.url ?? ""}|${headerStr}`;
	}
	/**
	* Get or create a `Client` for the named agent.
	*/
	getClient(name) {
		const spec = this.agents[name];
		const key = this.cacheKey(spec);
		const existing = this.clients.get(key);
		if (existing) return existing;
		const headers = this.resolveHeaders(spec);
		const client = new Client({
			apiUrl: spec.url,
			defaultHeaders: headers
		});
		this.clients.set(key, client);
		return client;
	}
};
/**
* Extract the callback thread ID from the tool runtime.
*
* The thread ID is included in the subagent's input state so the subagent
* can notify the parent when it completes (via
* `CompletionCallbackMiddleware`).
*
* @returns Object with `callbackThreadId` if available. Empty object otherwise.
*/
function extractCallbackContext(runtime) {
	const threadId = (runtime.config?.configurable)?.thread_id;
	if (typeof threadId === "string" && threadId) return { callbackThreadId: threadId };
	return {};
}
/**
* Build the `start_async_task` tool.
*
* Creates a thread on the remote server, starts a run, and returns a
* `Command` that persists the new task in state.
*/
function buildStartTool(agentMap, clients, toolDescription) {
	return tool(async (input, runtime) => {
		if (!(input.agentName in agentMap)) {
			const allowed = Object.keys(agentMap).map((k) => `\`${k}\``).join(", ");
			return `Unknown async subagent type \`${input.agentName}\`. Available types: ${allowed}`;
		}
		const spec = agentMap[input.agentName];
		const callbackContext = extractCallbackContext(runtime);
		try {
			const client = clients.getClient(input.agentName);
			const thread = await client.threads.create();
			const run = await client.runs.create(thread.thread_id, spec.graphId, { input: {
				messages: [{
					role: "user",
					content: input.description
				}],
				...callbackContext
			} });
			const taskId = thread.thread_id;
			const task = {
				taskId,
				agentName: input.agentName,
				threadId: taskId,
				runId: run.run_id,
				status: "running",
				createdAt: (/* @__PURE__ */ new Date()).toISOString(),
				description: input.description
			};
			return new Command({ update: {
				messages: [new ToolMessage({
					content: `Launched async subagent. taskId: ${taskId}`,
					tool_call_id: toolCallIdFromRuntime(runtime)
				})],
				asyncTasks: { [taskId]: task }
			} });
		} catch (e) {
			return `Failed to launch async subagent '${input.agentName}': ${e}`;
		}
	}, {
		name: "start_async_task",
		description: toolDescription,
		schema: z.object({
			description: z.string().describe("A detailed description of the task for the async subagent to perform."),
			agentName: z.string().describe("The type of async subagent to use. Must be one of the available types listed in the tool description.")
		})
	});
}
/**
* Build the `check_async_task` tool.
*
* Fetches the current run status from the remote server and, if the run
* succeeded, retrieves the thread state to extract the result.
*/
function buildCheckTool(clients) {
	return tool(async (input, runtime) => {
		const task = resolveTrackedTask(input.taskId, runtime.state);
		if (typeof task === "string") return task;
		const client = clients.getClient(task.agentName);
		let run;
		try {
			run = await client.runs.get(task.threadId, task.runId);
		} catch (e) {
			return `Failed to get run status: ${e}`;
		}
		let threadValues = {};
		if (run.status === "success") try {
			threadValues = (await client.threads.getState(task.threadId)).values || {};
		} catch {}
		const result = buildCheckResult(run, task.threadId, threadValues);
		const updatedTask = {
			taskId: task.taskId,
			agentName: task.agentName,
			threadId: task.threadId,
			runId: task.runId,
			status: result.status,
			createdAt: task.createdAt,
			updatedAt: result.status !== task.status ? (/* @__PURE__ */ new Date()).toISOString() : task.updatedAt,
			checkedAt: (/* @__PURE__ */ new Date()).toISOString()
		};
		return new Command({ update: {
			messages: [new ToolMessage({
				content: JSON.stringify(result),
				tool_call_id: toolCallIdFromRuntime(runtime)
			})],
			asyncTasks: { [task.taskId]: updatedTask }
		} });
	}, {
		name: "check_async_task",
		description: "Check the status of an async subagent task. Returns the current status and, if complete, the result.",
		schema: z.object({ taskId: z.string().describe("The exact taskId string returned by start_async_task. Pass it verbatim.") })
	});
}
/**
* Build the `update_async_task` tool.
*
* Sends a follow-up message to a running async subagent by creating a new
* run on the same thread with `multitaskStrategy: "interrupt"`. The subagent
* sees the full conversation history plus the new message. The `taskId`
* remains the same; only the internal `runId` is updated.
*/
function buildUpdateTool(agentMap, clients) {
	return tool(async (input, runtime) => {
		const tracked = resolveTrackedTask(input.taskId, runtime.state);
		if (typeof tracked === "string") return tracked;
		const spec = agentMap[tracked.agentName];
		try {
			const run = await clients.getClient(tracked.agentName).runs.create(tracked.threadId, spec.graphId, {
				input: { messages: [{
					role: "user",
					content: input.message
				}] },
				multitaskStrategy: "interrupt"
			});
			const task = {
				taskId: tracked.taskId,
				agentName: tracked.agentName,
				threadId: tracked.threadId,
				runId: run.run_id,
				status: "running",
				createdAt: tracked.createdAt,
				description: input.message,
				updatedAt: (/* @__PURE__ */ new Date()).toISOString(),
				checkedAt: tracked.checkedAt
			};
			return new Command({ update: {
				messages: [new ToolMessage({
					content: `Updated async subagent. taskId: ${tracked.taskId}`,
					tool_call_id: toolCallIdFromRuntime(runtime)
				})],
				asyncTasks: { [tracked.taskId]: task }
			} });
		} catch (e) {
			return `Failed to update async subagent: ${e}`;
		}
	}, {
		name: "update_async_task",
		description: "send updated instructions to an async subagent. Interrupts the current run and starts a new one on the same thread so the subagent sees the full conversation history plus your new message. The taskId remains the same.",
		schema: z.object({
			taskId: z.string().describe("The exact taskId string returned by start_async_task. Pass it verbatim."),
			message: z.string().describe("Follow-up instructions or context to send to the subagent")
		})
	});
}
/**
* Build the `cancel_async_task` tool.
*
* Cancels the current run on the remote server and updates the task's
* cached status to `"cancelled"`.
*/
function buildCancelTool(clients) {
	return tool(async (input, runtime) => {
		const tracked = resolveTrackedTask(input.taskId, runtime.state);
		if (typeof tracked === "string") return tracked;
		const client = clients.getClient(tracked.agentName);
		try {
			await client.runs.cancel(tracked.threadId, tracked.runId);
		} catch (e) {
			return `Failed to cancel run: ${e}`;
		}
		const updated = {
			taskId: tracked.taskId,
			agentName: tracked.agentName,
			threadId: tracked.threadId,
			runId: tracked.runId,
			status: "cancelled",
			createdAt: tracked.createdAt,
			updatedAt: (/* @__PURE__ */ new Date()).toISOString(),
			checkedAt: tracked.checkedAt
		};
		return new Command({ update: {
			messages: [new ToolMessage({
				content: `Cancelled async subagent task: ${tracked.taskId}`,
				tool_call_id: toolCallIdFromRuntime(runtime)
			})],
			asyncTasks: { [tracked.taskId]: updated }
		} });
	}, {
		name: "cancel_async_task",
		description: "Cancel a running async subagent task. Use this to stop a task that is no longer needed.",
		schema: z.object({ taskId: z.string().describe("The exact taskId string returned by start_async_task. Pass it verbatim.") })
	});
}
/**
* Build the `list_async_tasks` tool.
*
* Lists all tracked tasks with their live statuses fetched in parallel.
* Supports optional filtering by cached status.
*/
function buildListTool(clients) {
	return tool(async (input, runtime) => {
		const filtered = filterTasks(runtime.state.asyncTasks ?? {}, input.statusFilter ?? void 0);
		if (filtered.length === 0) return "No async subagent tasks tracked";
		const statuses = await Promise.all(filtered.map((task) => fetchLiveTaskStatus(clients, task)));
		const updatedTasks = {};
		const entries = [];
		for (let idx = 0; idx < filtered.length; idx++) {
			const task = filtered[idx];
			const status = statuses[idx];
			const taskEntry = formatTaskEntry(task, status);
			entries.push(taskEntry);
			updatedTasks[task.taskId] = {
				taskId: task.taskId,
				agentName: task.agentName,
				threadId: task.threadId,
				runId: task.runId,
				status,
				createdAt: task.createdAt,
				updatedAt: status !== task.status ? (/* @__PURE__ */ new Date()).toISOString() : task.updatedAt,
				checkedAt: task.checkedAt
			};
		}
		return new Command({ update: {
			messages: [new ToolMessage({
				content: `${entries.length} tracked task(s):\n${entries.join("\n")}`,
				tool_call_id: toolCallIdFromRuntime(runtime)
			})],
			asyncTasks: updatedTasks
		} });
	}, {
		name: "list_async_tasks",
		description: "List tracked async subagent tasks with their current live statuses. Be default shows all tasks. Use `statusFilter` to narrow by status (e.g., 'running', 'success', 'error', 'cancelled'). Use `check_async_task` to get the full result of a specific completed task.",
		schema: z.object({ statusFilter: z.string().nullish().describe("Filter tasks by status. One of: 'running', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.") })
	});
}
/**
* Create middleware that adds async subagent tools to an agent.
*
* Provides five tools for launching, checking, updating, cancelling, and
* listing background tasks on remote Agent Protocol servers. Task state is
* persisted in the `asyncTasks` state channel so it survives
* context compaction.
*
* Works with any Agent Protocol-compliant server — LangGraph Platform (managed)
* or self-hosted (e.g. a Hono/Express server implementing the Agent Protocol spec).
*
* @throws {Error} If no async subagents are provided or names are duplicated.
*
* @example
* ```ts
* const middleware = createAsyncSubAgentMiddleware({
*   asyncSubAgents: [{
*     name: "researcher",
*     description: "Research agent for deep analysis",
*     url: "https://my-agent-protocol-server.example.com",
*     graphId: "research_agent",
*   }],
* });
* ```
*/
/**
* Type guard to distinguish async SubAgents from sync SubAgents/CompiledSubAgents.
*
* Uses the presence of the `graphId` field as the runtime discriminant —
* `AsyncSubAgent` requires it, while `SubAgent` and `CompiledSubAgent` do not have it.
*/
function isAsyncSubAgent(subAgent) {
	return "graphId" in subAgent;
}
function createAsyncSubAgentMiddleware(options) {
	const { asyncSubAgents, systemPrompt = ASYNC_TASK_SYSTEM_PROMPT } = options;
	if (!asyncSubAgents || asyncSubAgents.length === 0) throw new Error("At least one async subagent must be specified");
	const names = asyncSubAgents.map((a) => a.name);
	const duplicates = names.filter((n, i) => names.indexOf(n) !== i);
	if (duplicates.length > 0) throw new Error(`Duplicate async subagent names: ${[...new Set(duplicates)].join(", ")}`);
	const agentMap = Object.fromEntries(asyncSubAgents.map((a) => [a.name, a]));
	const clients = new ClientCache(agentMap);
	const agentsDescription = asyncSubAgents.map((a) => `- ${a.name}: ${a.description}`).join("\n");
	const tools = [
		buildStartTool(agentMap, clients, ASYNC_TASK_TOOL_DESCRIPTION.replace("{available_agents}", agentsDescription)),
		buildCheckTool(clients),
		buildUpdateTool(agentMap, clients),
		buildCancelTool(clients),
		buildListTool(clients)
	];
	const fullSystemPrompt = systemPrompt ? `${systemPrompt}\n\nAvailable async subagent types:\n${agentsDescription}` : null;
	return createMiddleware({
		name: "asyncSubAgentMiddleware",
		stateSchema: AsyncTaskStateSchema,
		tools,
		wrapModelCall: async (request, handler) => {
			if (fullSystemPrompt !== null) return handler({
				...request,
				systemMessage: request.systemMessage.concat(new SystemMessage({ content: fullSystemPrompt }))
			});
			return handler(request);
		}
	});
}
//#endregion
//#region src/backends/store.ts
/**
* StoreBackend: Adapter for LangGraph's BaseStore (persistent, cross-thread).
*/
const NAMESPACE_COMPONENT_RE = /^[A-Za-z0-9\-_.@+:~]+$/;
/**
* Validate a namespace array.
*
* Each component must be a non-empty string containing only safe characters:
* alphanumeric (a-z, A-Z, 0-9), hyphen (-), underscore (_), dot (.),
* at sign (@), plus (+), colon (:), and tilde (~).
*
* Characters like *, ?, [, ], {, } etc. are rejected to prevent
* wildcard or glob injection in store lookups.
*/
function validateNamespace(namespace) {
	if (namespace.length === 0) throw new Error("Namespace array must not be empty.");
	for (let i = 0; i < namespace.length; i++) {
		const component = namespace[i];
		if (typeof component !== "string") throw new TypeError(`Namespace component at index ${i} must be a string, got ${typeof component}.`);
		if (!component) throw new Error(`Namespace component at index ${i} must not be empty.`);
		if (!NAMESPACE_COMPONENT_RE.test(component)) throw new Error(`Namespace component at index ${i} contains disallowed characters: "${component}". Only alphanumeric characters, hyphens, underscores, dots, @, +, colons, and tildes are allowed.`);
	}
	return namespace;
}
/**
* Backend that stores files in LangGraph's BaseStore (persistent).
*
* Uses LangGraph's Store for persistent, cross-conversation storage.
* Files are organized via namespaces and persist across all threads.
*
* The namespace can be customized via a factory function for flexible
* isolation patterns (user-scoped, org-scoped, etc.), or falls back
* to legacy assistant_id-based isolation.
*/
var StoreBackend = class {
	stateAndStore;
	_namespace;
	fileFormat;
	constructor(stateAndStoreOrOptions, options) {
		let opts;
		if (stateAndStoreOrOptions != null && typeof stateAndStoreOrOptions === "object" && "state" in stateAndStoreOrOptions) {
			this.stateAndStore = stateAndStoreOrOptions;
			opts = options;
		} else {
			this.stateAndStore = void 0;
			opts = stateAndStoreOrOptions;
		}
		if (opts?.namespace) this._namespace = validateNamespace(opts.namespace);
		this.fileFormat = opts?.fileFormat ?? "v2";
	}
	/**
	* Get the BaseStore instance for persistent storage operations.
	*
	* In legacy mode, reads from the injected {@link StateAndStore}.
	* In zero-arg mode, retrieves the store from the LangGraph execution
	* context via {@link getLangGraphStore}.
	*
	* @returns BaseStore instance
	* @throws Error if no store is available in either mode
	*/
	getStore() {
		if (this.stateAndStore) {
			const store = this.stateAndStore.store;
			if (!store) throw new Error("Store is required but not available in runtime");
			return store;
		}
		const store = getStore();
		if (!store) throw new Error("Store is required but not available in LangGraph execution context. Ensure the graph was configured with a store.");
		return store;
	}
	/**
	* Get the namespace for store operations.
	*
	* Resolution order:
	* 1. Explicit namespace from constructor options (both modes)
	* 2. Legacy mode: `[assistantId, "filesystem"]` fallback from {@link StateAndStore}
	* 3. Zero-arg mode without namespace: `["filesystem"]` with a deprecation warning
	*    nudging callers to pass an explicit namespace
	* 4. Legacy mode without assistantId: `["filesystem"]`
	*/
	getNamespace() {
		if (this._namespace) return this._namespace;
		if (this.stateAndStore) {
			const assistantId = this.stateAndStore.assistantId;
			if (assistantId) return [assistantId, "filesystem"];
		}
		return ["filesystem"];
	}
	/**
	* Convert a store Item to FileData format.
	*
	* @param storeItem - The store Item containing file data
	* @returns FileData object
	* @throws Error if required fields are missing or have incorrect types
	*/
	convertStoreItemToFileData(storeItem) {
		const value = storeItem.value;
		if (!(value.content !== void 0 && (Array.isArray(value.content) || typeof value.content === "string" || ArrayBuffer.isView(value.content))) || typeof value.created_at !== "string" || typeof value.modified_at !== "string") throw new Error(`Store item does not contain valid FileData fields. Got keys: ${Object.keys(value).join(", ")}`);
		return {
			content: value.content,
			...value.mimeType ? { mimeType: value.mimeType } : {},
			created_at: value.created_at,
			modified_at: value.modified_at
		};
	}
	/**
	* Convert FileData to a value suitable for store.put().
	*
	* @param fileData - The FileData to convert
	* @returns Object with content, mimeType, created_at, and modified_at fields
	*/
	convertFileDataToStoreValue(fileData) {
		return {
			content: fileData.content,
			..."mimeType" in fileData ? { mimeType: fileData.mimeType } : {},
			created_at: fileData.created_at,
			modified_at: fileData.modified_at
		};
	}
	/**
	* Search store with automatic pagination to retrieve all results.
	*
	* @param store - The store to search
	* @param namespace - Hierarchical path prefix to search within
	* @param options - Optional query, filter, and page_size
	* @returns List of all items matching the search criteria
	*/
	async searchStorePaginated(store, namespace, options = {}) {
		const { query, filter, pageSize = 100 } = options;
		const allItems = [];
		let offset = 0;
		while (true) {
			const pageItems = await store.search(namespace, {
				query,
				filter,
				limit: pageSize,
				offset
			});
			if (!pageItems || pageItems.length === 0) break;
			allItems.push(...pageItems);
			if (pageItems.length < pageSize) break;
			offset += pageSize;
		}
		return allItems;
	}
	/**
	* List files and directories in the specified directory (non-recursive).
	*
	* @param path - Absolute path to directory
	* @returns LsResult with list of FileInfo objects on success or error on failure.
	*          Directories have a trailing / in their path and is_dir=true.
	*/
	async ls(path) {
		const store = this.getStore();
		const namespace = this.getNamespace();
		const items = await this.searchStorePaginated(store, namespace);
		const infos = [];
		const subdirs = /* @__PURE__ */ new Set();
		const normalizedPath = path.endsWith("/") ? path : path + "/";
		for (const item of items) {
			const itemKey = String(item.key);
			if (!itemKey.startsWith(normalizedPath)) continue;
			const relative = itemKey.substring(normalizedPath.length);
			if (relative.includes("/")) {
				const subdirName = relative.split("/")[0];
				subdirs.add(normalizedPath + subdirName + "/");
				continue;
			}
			try {
				const fd = this.convertStoreItemToFileData(item);
				const size = isFileDataV1(fd) ? fd.content.join("\n").length : isFileDataBinary(fd) ? fd.content.byteLength : fd.content.length;
				infos.push({
					path: itemKey,
					is_dir: false,
					size,
					modified_at: fd.modified_at
				});
			} catch {
				continue;
			}
		}
		for (const subdir of Array.from(subdirs).sort()) infos.push({
			path: subdir,
			is_dir: true,
			size: 0,
			modified_at: ""
		});
		infos.sort((a, b) => a.path.localeCompare(b.path));
		return { files: infos };
	}
	/**
	* Read file content.
	*
	* Text files are paginated by line offset/limit.
	* Binary files return full Uint8Array content (offset/limit ignored).
	*
	* @param filePath - Absolute file path
	* @param offset - Line offset to start reading from (0-indexed)
	* @param limit - Maximum number of lines to read
	* @returns ReadResult with content on success or error on failure
	*/
	async read(filePath, offset = 0, limit = 500) {
		try {
			const readRawResult = await this.readRaw(filePath);
			if (readRawResult.error || !readRawResult.data) return { error: readRawResult.error || "File data not found" };
			const fileDataV2 = migrateToFileDataV2(readRawResult.data, filePath);
			if (!isTextMimeType(fileDataV2.mimeType)) return {
				content: fileDataV2.content,
				mimeType: fileDataV2.mimeType
			};
			if (typeof fileDataV2.content !== "string") return { error: `File '${filePath}' has binary content but text MIME type` };
			return {
				content: fileDataV2.content.split("\n").slice(offset, offset + limit).join("\n"),
				mimeType: fileDataV2.mimeType
			};
		} catch (e) {
			return { error: e.message };
		}
	}
	/**
	* Read file content as raw FileData.
	*
	* @param filePath - Absolute file path
	* @returns ReadRawResult with raw file data on success or error on failure
	*/
	async readRaw(filePath) {
		const store = this.getStore();
		const namespace = this.getNamespace();
		const item = await store.get(namespace, filePath);
		if (!item) return { error: `File '${filePath}' not found` };
		return { data: this.convertStoreItemToFileData(item) };
	}
	/**
	* Create a new file with content.
	* Returns WriteResult. External storage sets filesUpdate=null.
	*/
	async write(filePath, content) {
		const store = this.getStore();
		const namespace = this.getNamespace();
		if (await store.get(namespace, filePath)) return { error: `Cannot write to ${filePath} because it already exists. Read and then make an edit, or write to a new path.` };
		const mimeType = getMimeType(filePath);
		const fileData = createFileData(content, void 0, this.fileFormat, mimeType);
		const storeValue = this.convertFileDataToStoreValue(fileData);
		await store.put(namespace, filePath, storeValue);
		return {
			path: filePath,
			filesUpdate: null
		};
	}
	/**
	* Edit a file by replacing string occurrences.
	* Returns EditResult. External storage sets filesUpdate=null.
	*/
	async edit(filePath, oldString, newString, replaceAll = false) {
		const store = this.getStore();
		const namespace = this.getNamespace();
		const item = await store.get(namespace, filePath);
		if (!item) return { error: `Error: File '${filePath}' not found` };
		try {
			const fileData = this.convertStoreItemToFileData(item);
			const result = performStringReplacement(fileDataToString(fileData), oldString, newString, replaceAll);
			if (typeof result === "string") return { error: result };
			const [newContent, occurrences] = result;
			const newFileData = updateFileData(fileData, newContent);
			const storeValue = this.convertFileDataToStoreValue(newFileData);
			await store.put(namespace, filePath, storeValue);
			return {
				path: filePath,
				filesUpdate: null,
				occurrences
			};
		} catch (e) {
			return { error: `Error: ${e.message}` };
		}
	}
	/**
	* Search file contents for a literal text pattern.
	* Binary files are skipped.
	*/
	async grep(pattern, path = "/", glob = null) {
		const store = this.getStore();
		const namespace = this.getNamespace();
		const items = await this.searchStorePaginated(store, namespace);
		const files = {};
		for (const item of items) try {
			files[item.key] = this.convertStoreItemToFileData(item);
		} catch {
			continue;
		}
		return { matches: grepMatchesFromFiles(files, pattern, path, glob) };
	}
	/**
	* Structured glob matching returning FileInfo objects.
	*/
	async glob(pattern, path = "/") {
		const store = this.getStore();
		const namespace = this.getNamespace();
		const items = await this.searchStorePaginated(store, namespace);
		const files = {};
		for (const item of items) try {
			files[item.key] = this.convertStoreItemToFileData(item);
		} catch {
			continue;
		}
		const result = globSearchFiles(files, pattern, path);
		if (result === "No files found") return { files: [] };
		const paths = result.split("\n");
		const infos = [];
		for (const p of paths) {
			const fd = files[p];
			const size = fd ? isFileDataV1(fd) ? fd.content.join("\n").length : isFileDataBinary(fd) ? fd.content.byteLength : fd.content.length : 0;
			infos.push({
				path: p,
				is_dir: false,
				size,
				modified_at: fd?.modified_at || ""
			});
		}
		return { files: infos };
	}
	/**
	* Upload multiple files.
	*
	* @param files - List of [path, content] tuples to upload
	* @returns List of FileUploadResponse objects, one per input file
	*/
	async uploadFiles(files) {
		const store = this.getStore();
		const namespace = this.getNamespace();
		const responses = [];
		for (const [path, content] of files) try {
			const mimeType = getMimeType(path);
			const isBinary = this.fileFormat === "v2" && !isTextMimeType(mimeType);
			let fileData;
			if (isBinary) fileData = createFileData(content, void 0, "v2", mimeType);
			else fileData = createFileData(new TextDecoder().decode(content), void 0, this.fileFormat, mimeType);
			const storeValue = this.convertFileDataToStoreValue(fileData);
			await store.put(namespace, path, storeValue);
			responses.push({
				path,
				error: null
			});
		} catch {
			responses.push({
				path,
				error: "invalid_path"
			});
		}
		return responses;
	}
	/**
	* Download multiple files.
	*
	* @param paths - List of file paths to download
	* @returns List of FileDownloadResponse objects, one per input path
	*/
	async downloadFiles(paths) {
		const store = this.getStore();
		const namespace = this.getNamespace();
		const responses = [];
		for (const path of paths) try {
			const item = await store.get(namespace, path);
			if (!item) {
				responses.push({
					path,
					content: null,
					error: "file_not_found"
				});
				continue;
			}
			const fileDataV2 = migrateToFileDataV2(this.convertStoreItemToFileData(item), path);
			if (typeof fileDataV2.content === "string") {
				const content = new TextEncoder().encode(fileDataV2.content);
				responses.push({
					path,
					content,
					error: null
				});
			} else responses.push({
				path,
				content: fileDataV2.content,
				error: null
			});
		} catch {
			responses.push({
				path,
				content: null,
				error: "file_not_found"
			});
		}
		return responses;
	}
};
//#endregion
//#region src/backends/composite.ts
/**
* Backend that routes file operations to different backends based on path prefix.
*
* This enables hybrid storage strategies like:
* - `/memories/` → StoreBackend (persistent, cross-thread)
* - Everything else → StateBackend (ephemeral, per-thread)
*
* The CompositeBackend handles path prefix stripping/re-adding transparently.
*/
var CompositeBackend = class {
	default;
	routes;
	sortedRoutes;
	constructor(defaultBackend, routes) {
		this.default = isSandboxProtocol(defaultBackend) ? adaptSandboxProtocol(defaultBackend) : adaptBackendProtocol(defaultBackend);
		this.routes = Object.fromEntries(Object.entries(routes).map(([k, v]) => [k, isSandboxProtocol(v) ? adaptSandboxProtocol(v) : adaptBackendProtocol(v)]));
		this.sortedRoutes = Object.entries(this.routes).sort((a, b) => b[0].length - a[0].length);
	}
	/** Delegates to default backend's id if it is a sandbox, otherwise empty string. */
	get id() {
		return isSandboxBackend(this.default) ? this.default.id : "";
	}
	/**
	* Determine which backend handles this key and strip prefix.
	*
	* @param key - Original file path
	* @returns Tuple of [backend, stripped_key] where stripped_key has the route
	*          prefix removed (but keeps leading slash).
	*/
	getBackendAndKey(key) {
		for (const [prefix, backend] of this.sortedRoutes) if (key.startsWith(prefix)) {
			const suffix = key.substring(prefix.length);
			return [backend, suffix ? "/" + suffix : "/"];
		}
		return [this.default, key];
	}
	/**
	* List files and directories in the specified directory (non-recursive).
	*
	* @param path - Absolute path to directory
	* @returns LsResult with list of FileInfo objects (with route prefixes added) on success or error on failure.
	*          Directories have a trailing / in their path and is_dir=true.
	*/
	async ls(path) {
		for (const [routePrefix, backend] of this.sortedRoutes) if (path.startsWith(routePrefix.replace(/\/$/, ""))) {
			const suffix = path.substring(routePrefix.length);
			const searchPath = suffix ? "/" + suffix : "/";
			const result = await backend.ls(searchPath);
			if (result.error) return result;
			const prefixed = [];
			for (const fi of result.files || []) prefixed.push({
				...fi,
				path: routePrefix.slice(0, -1) + fi.path
			});
			return { files: prefixed };
		}
		if (path === "/") {
			const results = [];
			const defaultResult = await this.default.ls(path);
			if (defaultResult.error) return defaultResult;
			results.push(...defaultResult.files || []);
			for (const [routePrefix] of this.sortedRoutes) results.push({
				path: routePrefix,
				is_dir: true,
				size: 0,
				modified_at: ""
			});
			results.sort((a, b) => a.path.localeCompare(b.path));
			return { files: results };
		}
		return await this.default.ls(path);
	}
	/**
	* Read file content, routing to appropriate backend.
	*
	* @param filePath - Absolute file path
	* @param offset - Line offset to start reading from (0-indexed)
	* @param limit - Maximum number of lines to read
	* @returns Formatted file content with line numbers, or error message
	*/
	async read(filePath, offset = 0, limit = 500) {
		const [backend, strippedKey] = this.getBackendAndKey(filePath);
		return await backend.read(strippedKey, offset, limit);
	}
	/**
	* Read file content as raw FileData.
	*
	* @param filePath - Absolute file path
	* @returns ReadRawResult with raw file data on success or error on failure
	*/
	async readRaw(filePath) {
		const [backend, strippedKey] = this.getBackendAndKey(filePath);
		return await backend.readRaw(strippedKey);
	}
	/**
	* Structured search results or error string for invalid input.
	*/
	async grep(pattern, path = "/", glob = null) {
		for (const [routePrefix, backend] of this.sortedRoutes) if (path.startsWith(routePrefix.replace(/\/$/, ""))) {
			const searchPath = path.substring(routePrefix.length - 1);
			const raw = await backend.grep(pattern, searchPath || "/", glob);
			if (raw.error) return raw;
			return { matches: (raw.matches || []).map((m) => ({
				...m,
				path: routePrefix.slice(0, -1) + m.path
			})) };
		}
		const allMatches = [];
		const rawDefault = await this.default.grep(pattern, path, glob);
		if (rawDefault.error) return rawDefault;
		allMatches.push(...rawDefault.matches || []);
		for (const [routePrefix, backend] of Object.entries(this.routes)) {
			const raw = await backend.grep(pattern, "/", glob);
			if (raw.error) return raw;
			const matches = (raw.matches || []).map((m) => ({
				...m,
				path: routePrefix.slice(0, -1) + m.path
			}));
			allMatches.push(...matches);
		}
		return { matches: allMatches };
	}
	/**
	* Structured glob matching returning FileInfo objects.
	*/
	async glob(pattern, path = "/") {
		const results = [];
		for (const [routePrefix, backend] of this.sortedRoutes) if (path.startsWith(routePrefix.replace(/\/$/, ""))) {
			const searchPath = path.substring(routePrefix.length - 1);
			const result = await backend.glob(pattern, searchPath || "/");
			if (result.error) return result;
			return { files: (result.files || []).map((fi) => ({
				...fi,
				path: routePrefix.slice(0, -1) + fi.path
			})) };
		}
		const defaultResult = await this.default.glob(pattern, path);
		if (defaultResult.error) return defaultResult;
		results.push(...defaultResult.files || []);
		for (const [routePrefix, backend] of Object.entries(this.routes)) {
			const result = await backend.glob(pattern, "/");
			if (result.error) continue;
			const files = (result.files || []).map((fi) => ({
				...fi,
				path: routePrefix.slice(0, -1) + fi.path
			}));
			results.push(...files);
		}
		results.sort((a, b) => a.path.localeCompare(b.path));
		return { files: results };
	}
	/**
	* Create a new file, routing to appropriate backend.
	*
	* @param filePath - Absolute file path
	* @param content - File content as string
	* @returns WriteResult with path or error
	*/
	async write(filePath, content) {
		const [backend, strippedKey] = this.getBackendAndKey(filePath);
		return await backend.write(strippedKey, content);
	}
	/**
	* Edit a file, routing to appropriate backend.
	*
	* @param filePath - Absolute file path
	* @param oldString - String to find and replace
	* @param newString - Replacement string
	* @param replaceAll - If true, replace all occurrences
	* @returns EditResult with path, occurrences, or error
	*/
	async edit(filePath, oldString, newString, replaceAll = false) {
		const [backend, strippedKey] = this.getBackendAndKey(filePath);
		return await backend.edit(strippedKey, oldString, newString, replaceAll);
	}
	/**
	* Execute a command via the default backend.
	* Execution is not path-specific, so it always delegates to the default backend.
	*
	* @param command - Full shell command string to execute
	* @returns ExecuteResponse with combined output, exit code, and truncation flag
	* @throws Error if the default backend doesn't support command execution
	*/
	execute(command) {
		if (!isSandboxBackend(this.default)) throw new Error("Default backend doesn't support command execution (SandboxBackendProtocol). To enable execution, provide a default backend that implements SandboxBackendProtocol.");
		return Promise.resolve(this.default.execute(command));
	}
	/**
	* Upload multiple files, batching by backend for efficiency.
	*
	* @param files - List of [path, content] tuples to upload
	* @returns List of FileUploadResponse objects, one per input file
	*/
	async uploadFiles(files) {
		const results = Array.from({ length: files.length }, () => null);
		const batchesByBackend = /* @__PURE__ */ new Map();
		for (let idx = 0; idx < files.length; idx++) {
			const [path, content] = files[idx];
			const [backend, strippedPath] = this.getBackendAndKey(path);
			if (!batchesByBackend.has(backend)) batchesByBackend.set(backend, []);
			batchesByBackend.get(backend).push({
				idx,
				path: strippedPath,
				content
			});
		}
		for (const [backend, batch] of batchesByBackend) {
			if (!backend.uploadFiles) throw new Error("Backend does not support uploadFiles");
			const batchFiles = batch.map((b) => [b.path, b.content]);
			const batchResponses = await backend.uploadFiles(batchFiles);
			for (let i = 0; i < batch.length; i++) {
				const originalIdx = batch[i].idx;
				results[originalIdx] = {
					path: files[originalIdx][0],
					error: batchResponses[i]?.error ?? null
				};
			}
		}
		return results;
	}
	/**
	* Download multiple files, batching by backend for efficiency.
	*
	* @param paths - List of file paths to download
	* @returns List of FileDownloadResponse objects, one per input path
	*/
	async downloadFiles(paths) {
		const results = Array.from({ length: paths.length }, () => null);
		const batchesByBackend = /* @__PURE__ */ new Map();
		for (let idx = 0; idx < paths.length; idx++) {
			const path = paths[idx];
			const [backend, strippedPath] = this.getBackendAndKey(path);
			if (!batchesByBackend.has(backend)) batchesByBackend.set(backend, []);
			batchesByBackend.get(backend).push({
				idx,
				path: strippedPath
			});
		}
		for (const [backend, batch] of batchesByBackend) {
			if (!backend.downloadFiles) throw new Error("Backend does not support downloadFiles");
			const batchPaths = batch.map((b) => b.path);
			const batchResponses = await backend.downloadFiles(batchPaths);
			for (let i = 0; i < batch.length; i++) {
				const originalIdx = batch[i].idx;
				results[originalIdx] = {
					path: paths[originalIdx],
					content: batchResponses[i]?.content ?? null,
					error: batchResponses[i]?.error ?? null
				};
			}
		}
		return results;
	}
};
//#endregion
//#region src/backends/sandbox.ts
/**
* Shell-quote a string using single quotes (POSIX).
* Escapes embedded single quotes with the '\'' technique.
*/
function shellQuote(s) {
	return "'" + s.replace(/'/g, "'\\''") + "'";
}
/**
* Convert a glob pattern to a path-aware RegExp.
*
* Inspired by the just-bash project's glob utilities:
* - `*`  matches any characters except `/`
* - `**` matches any characters including `/` (recursive)
* - `?`  matches a single character except `/`
* - `[...]` character classes
*/
function globToPathRegex(pattern) {
	let regex = "^";
	let i = 0;
	while (i < pattern.length) {
		const c = pattern[i];
		if (c === "*") if (i + 1 < pattern.length && pattern[i + 1] === "*") {
			i += 2;
			if (i < pattern.length && pattern[i] === "/") {
				regex += "(.*/)?";
				i++;
			} else regex += ".*";
		} else {
			regex += "[^/]*";
			i++;
		}
		else if (c === "?") {
			regex += "[^/]";
			i++;
		} else if (c === "[") {
			let j = i + 1;
			while (j < pattern.length && pattern[j] !== "]") j++;
			regex += pattern.slice(i, j + 1);
			i = j + 1;
		} else if (c === "." || c === "+" || c === "^" || c === "$" || c === "{" || c === "}" || c === "(" || c === ")" || c === "|" || c === "\\") {
			regex += `\\${c}`;
			i++;
		} else {
			regex += c;
			i++;
		}
	}
	regex += "$";
	return new RegExp(regex);
}
/**
* Parse a single line of stat/find output in the format: size\tmtime\ttype\tpath
*
* The first three tab-delimited fields are always fixed (number, number, string),
* so we safely take everything after the third tab as the file path — even if the
* path itself contains tabs.
*
* The type field varies by platform / tool:
* - GNU find -printf %y: single letter "d", "f", "l"
* - BSD stat -f %Sp: permission strings like "drwxr-xr-x", "-rw-r--r--"
*
* The mtime field may be a float (GNU find %T@ → "1234567890.0000000000")
* or an integer (BSD stat %m → "1234567890"); parseInt handles both.
*/
function parseStatLine(line) {
	const firstTab = line.indexOf("	");
	if (firstTab === -1) return null;
	const secondTab = line.indexOf("	", firstTab + 1);
	if (secondTab === -1) return null;
	const thirdTab = line.indexOf("	", secondTab + 1);
	if (thirdTab === -1) return null;
	const size = parseInt(line.slice(0, firstTab), 10);
	const mtime = parseInt(line.slice(firstTab + 1, secondTab), 10);
	const fileType = line.slice(secondTab + 1, thirdTab);
	const fullPath = line.slice(thirdTab + 1);
	if (isNaN(size) || isNaN(mtime)) return null;
	return {
		size,
		mtime,
		isDir: fileType === "d" || fileType === "directory" || fileType.startsWith("d"),
		fullPath
	};
}
/**
* BusyBox/Alpine fallback script for stat -c.
*
* Determines file type with POSIX test builtins, then uses stat -c
* (supported by both GNU coreutils and BusyBox) for size and mtime.
* printf handles tab-delimited output formatting.
*/
const STAT_C_SCRIPT = "for f; do if [ -d \"$f\" ]; then t=d; elif [ -L \"$f\" ]; then t=l; else t=f; fi; sz=$(stat -c %s \"$f\" 2>/dev/null) || continue; mt=$(stat -c %Y \"$f\" 2>/dev/null) || continue; printf \"%s\\t%s\\t%s\\t%s\\n\" \"$sz\" \"$mt\" \"$t\" \"$f\"; done";
/**
* Shell command for listing directory contents with metadata.
*
* Detects the environment at runtime with three-way probing:
* 1. GNU find (full Linux): uses built-in `-printf` (most efficient)
* 2. BusyBox / Alpine: uses `find -exec sh -c` with `stat -c` fallback
* 3. BSD / macOS: uses `find -exec stat -f`
*
* Output format per line: size\tmtime\ttype\tpath
*/
function buildLsCommand(dirPath) {
	const quotedPath = shellQuote(dirPath);
	const findBase = `find ${quotedPath} -maxdepth 1 -not -path ${quotedPath}`;
	return `if find /dev/null -maxdepth 0 -printf '' 2>/dev/null; then ${findBase} -printf '%s\\t%T@\\t%y\\t%p\\n' 2>/dev/null; elif stat -c %s /dev/null >/dev/null 2>&1; then ${findBase} -exec sh -c '${STAT_C_SCRIPT}' _ {} +; else ${findBase} -exec stat -f '%z\t%m\t%Sp\t%N' {} + 2>/dev/null; fi || true`;
}
/**
* Shell command for listing files recursively with metadata.
* Same three-way detection as buildLsCommand (GNU -printf / stat -c / BSD stat -f).
*
* Output format per line: size\tmtime\ttype\tpath
*/
function buildFindCommand(searchPath) {
	const quotedPath = shellQuote(searchPath);
	const findBase = `find ${quotedPath} -not -path ${quotedPath}`;
	return `if find /dev/null -maxdepth 0 -printf '' 2>/dev/null; then ${findBase} -printf '%s\\t%T@\\t%y\\t%p\\n' 2>/dev/null; elif stat -c %s /dev/null >/dev/null 2>&1; then ${findBase} -exec sh -c '${STAT_C_SCRIPT}' _ {} +; else ${findBase} -exec stat -f '%z\t%m\t%Sp\t%N' {} + 2>/dev/null; fi || true`;
}
/**
* Pure POSIX shell command for reading files with line numbers.
* Uses awk for line numbering with offset/limit — works on any Linux including Alpine.
*/
function buildReadCommand(filePath, offset, limit) {
	const quotedPath = shellQuote(filePath);
	const safeOffset = Number.isFinite(offset) && offset > 0 ? Math.floor(offset) : 0;
	const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.min(Math.floor(limit), 999999999) : 999999999;
	const start = safeOffset + 1;
	const end = safeOffset + safeLimit;
	return [
		`if [ ! -f ${quotedPath} ]; then echo "Error: File not found"; exit 1; fi`,
		`if [ ! -s ${quotedPath} ]; then echo "System reminder: File exists but has empty contents"; exit 0; fi`,
		`awk 'NR >= ${start} && NR <= ${end} { printf "%6d\\t%s\\n", NR, $0 }' ${quotedPath}`
	].join("; ");
}
/**
* Build a grep command for literal (fixed-string) search.
* Uses grep -rHnF for recursive, with-filename, with-line-number, fixed-string search.
*
* When a glob pattern is provided, uses `find -name GLOB -exec grep` instead of
* `grep --include=GLOB` for universal compatibility (BusyBox grep lacks --include).
*
* @param pattern - Literal string to search for (NOT regex).
* @param searchPath - Base path to search in.
* @param globPattern - Optional glob pattern to filter files.
*/
function buildGrepCommand(pattern, searchPath, globPattern) {
	const patternEscaped = shellQuote(pattern);
	const searchPathQuoted = shellQuote(searchPath);
	if (globPattern) return `find ${searchPathQuoted} -type f -name ${shellQuote(globPattern)} -exec grep -HnF -e ${patternEscaped} {} + 2>/dev/null || true`;
	return `grep -rHnF -e ${patternEscaped} ${searchPathQuoted} 2>/dev/null || true`;
}
/**
* Base sandbox implementation with execute() as the only abstract method.
*
* This class provides default implementations for all SandboxBackendProtocol
* methods using shell commands executed via execute(). Concrete implementations
* only need to implement execute(), uploadFiles(), and downloadFiles().
*
* All shell commands use pure POSIX utilities (awk, grep, find, stat) that are
* available on any Linux including Alpine/busybox. No Python, Node.js, or
* other runtime is required on the sandbox host.
*/
var BaseSandbox = class {
	/**
	* List files and directories in the specified directory (non-recursive).
	*
	* Uses pure POSIX shell (find + stat) via execute() — works on any Linux
	* including Alpine. No Python or Node.js needed.
	*
	* @param path - Absolute path to directory
	* @returns LsResult with list of FileInfo objects on success or error on failure.
	*/
	async ls(path) {
		const command = buildLsCommand(path);
		const result = await this.execute(command);
		const infos = [];
		const lines = result.output.trim().split("\n").filter(Boolean);
		for (const line of lines) {
			const parsed = parseStatLine(line);
			if (!parsed) continue;
			infos.push({
				path: parsed.isDir ? parsed.fullPath + "/" : parsed.fullPath,
				is_dir: parsed.isDir,
				size: parsed.size,
				modified_at: (/* @__PURE__ */ new Date(parsed.mtime * 1e3)).toISOString()
			});
		}
		return { files: infos };
	}
	/**
	* Read file content with line numbers.
	*
	* Uses pure POSIX shell (awk) via execute() — only the requested slice
	* is returned over the wire, making this efficient for large files.
	* Works on any Linux including Alpine (no Python or Node.js needed).
	*
	* @param filePath - Absolute file path
	* @param offset - Line offset to start reading from (0-indexed)
	* @param limit - Maximum number of lines to read
	* @returns Formatted file content with line numbers, or error message
	*/
	async read(filePath, offset = 0, limit = 500) {
		const mimeType = getMimeType(filePath);
		if (!isTextMimeType(mimeType)) {
			const results = await this.downloadFiles([filePath]);
			if (results[0].error || !results[0].content) return { error: `File '${filePath}' not found` };
			return {
				content: results[0].content,
				mimeType
			};
		}
		if (limit === 0) return {
			content: "",
			mimeType
		};
		const command = buildReadCommand(filePath, offset, limit);
		const result = await this.execute(command);
		if (result.exitCode !== 0) return { error: `File '${filePath}' not found` };
		return {
			content: result.output,
			mimeType
		};
	}
	/**
	* Read file content as raw FileData.
	*
	* Uses downloadFiles() directly — no runtime needed on the sandbox host.
	*
	* @param filePath - Absolute file path
	* @returns ReadRawResult with raw file data on success or error on failure
	*/
	async readRaw(filePath) {
		const results = await this.downloadFiles([filePath]);
		if (results[0].error || !results[0].content) return { error: `File '${filePath}' not found` };
		const now = (/* @__PURE__ */ new Date()).toISOString();
		const mimeType = getMimeType(filePath);
		if (!isTextMimeType(mimeType)) return { data: {
			content: results[0].content,
			mimeType,
			created_at: now,
			modified_at: now
		} };
		return { data: {
			content: new TextDecoder().decode(results[0].content),
			mimeType,
			created_at: now,
			modified_at: now
		} };
	}
	/**
	* Search for a literal text pattern in files using grep.
	*
	* @param pattern - Literal string to search for (NOT regex).
	* @param path - Directory or file path to search in.
	* @param glob - Optional glob pattern to filter which files to search.
	* @returns List of GrepMatch dicts containing path, line number, and matched text.
	*/
	async grep(pattern, path = "/", glob = null) {
		const command = buildGrepCommand(pattern, path, glob);
		const output = (await this.execute(command)).output.trim();
		if (!output) return { matches: [] };
		const matches = [];
		for (const line of output.split("\n")) {
			const parts = line.split(":");
			if (parts.length >= 3) {
				const filePath = parts[0];
				if (!isTextMimeType(getMimeType(filePath))) continue;
				const lineNum = parseInt(parts[1], 10);
				if (!isNaN(lineNum)) matches.push({
					path: filePath,
					line: lineNum,
					text: parts.slice(2).join(":")
				});
			}
		}
		return { matches };
	}
	/**
	* Structured glob matching returning FileInfo objects.
	*
	* Uses pure POSIX shell (find + stat) via execute() to list all files,
	* then applies glob-to-regex matching in TypeScript. No Python or Node.js
	* needed on the sandbox host.
	*
	* Glob patterns are matched against paths relative to the search base:
	* - `*`  matches any characters except `/`
	* - `**` matches any characters including `/` (recursive)
	* - `?`  matches a single character except `/`
	* - `[...]` character classes
	*/
	async glob(pattern, path = "/") {
		const command = buildFindCommand(path);
		const result = await this.execute(command);
		const regex = globToPathRegex(pattern);
		const infos = [];
		const lines = result.output.trim().split("\n").filter(Boolean);
		const basePath = path.endsWith("/") ? path.slice(0, -1) : path;
		for (const line of lines) {
			const parsed = parseStatLine(line);
			if (!parsed) continue;
			const relPath = parsed.fullPath.startsWith(basePath + "/") ? parsed.fullPath.slice(basePath.length + 1) : parsed.fullPath;
			if (regex.test(relPath)) infos.push({
				path: relPath,
				is_dir: parsed.isDir,
				size: parsed.size,
				modified_at: (/* @__PURE__ */ new Date(parsed.mtime * 1e3)).toISOString()
			});
		}
		return { files: infos };
	}
	/**
	* Create a new file with content.
	*
	* Uses downloadFiles() to check existence and uploadFiles() to write.
	* No runtime needed on the sandbox host.
	*/
	async write(filePath, content) {
		const mimeType = getMimeType(filePath);
		let fileContent;
		if (isTextMimeType(mimeType)) fileContent = new TextEncoder().encode(content);
		else fileContent = Buffer.from(content, "base64");
		const results = await this.uploadFiles([[filePath, fileContent]]);
		if (results[0].error) return { error: `Failed to write to ${filePath}: ${results[0].error}` };
		return {
			path: filePath,
			filesUpdate: null
		};
	}
	/**
	* Edit a file by replacing string occurrences.
	*
	* Uses downloadFiles() to read, performs string replacement in TypeScript,
	* then uploadFiles() to write back. No runtime needed on the sandbox host.
	*
	* Memory-conscious: releases intermediate references early so the GC can
	* reclaim buffers before the next large allocation is made.
	*/
	async edit(filePath, oldString, newString, replaceAll = false) {
		const results = await this.downloadFiles([filePath]);
		if (results[0].error || !results[0].content) return { error: `Error: File '${filePath}' not found` };
		const text = new TextDecoder().decode(results[0].content);
		results[0].content = null;
		/**
		* are we editing an empty file?
		*/
		if (oldString.length === 0) {
			/**
			* if the file is not empty, we cannot edit it with an empty oldString
			*/
			if (text.length !== 0) return { error: "oldString must not be empty unless the file is empty" };
			/**
			* if the newString is empty, we can just return the file as is
			*/
			if (newString.length === 0) return {
				path: filePath,
				filesUpdate: null,
				occurrences: 0
			};
			/**
			* if the newString is not empty, we can edit the file
			*/
			const encoded = new TextEncoder().encode(newString);
			const uploadResults = await this.uploadFiles([[filePath, encoded]]);
			/**
			* if the upload fails, we return an error
			*/
			if (uploadResults[0].error) return { error: `Failed to write edited file '${filePath}': ${uploadResults[0].error}` };
			return {
				path: filePath,
				filesUpdate: null,
				occurrences: 1
			};
		}
		const firstIdx = text.indexOf(oldString);
		if (firstIdx === -1) return { error: `String not found in file '${filePath}'` };
		if (oldString === newString) return {
			path: filePath,
			filesUpdate: null,
			occurrences: 1
		};
		let newText;
		let count;
		if (replaceAll) {
			newText = text.replaceAll(oldString, newString);
			/**
			* Derive count from the length delta to avoid a separate O(n) counting pass
			*/
			const lenDiff = oldString.length - newString.length;
			if (lenDiff !== 0) count = (text.length - newText.length) / lenDiff;
			else {
				/**
				* Lengths are equal — count via indexOf (we already found the first)
				*/
				count = 1;
				let pos = firstIdx + oldString.length;
				while (pos <= text.length) {
					const idx = text.indexOf(oldString, pos);
					if (idx === -1) break;
					count++;
					pos = idx + oldString.length;
				}
			}
		} else {
			if (text.indexOf(oldString, firstIdx + oldString.length) !== -1) return { error: `Multiple occurrences found in '${filePath}'. Use replaceAll=true to replace all.` };
			count = 1;
			/**
			* Build result from the known index — avoids a redundant search by .replace()
			*/
			newText = text.slice(0, firstIdx) + newString + text.slice(firstIdx + oldString.length);
		}
		const encoded = new TextEncoder().encode(newText);
		const uploadResults = await this.uploadFiles([[filePath, encoded]]);
		if (uploadResults[0].error) return { error: `Failed to write edited file '${filePath}': ${uploadResults[0].error}` };
		return {
			path: filePath,
			filesUpdate: null,
			occurrences: count
		};
	}
};
//#endregion
//#region src/errors.ts
const CONFIGURATION_ERROR_SYMBOL = Symbol.for("deepagents.configuration_error");
/**
* Thrown when `createDeepAgent` receives invalid configuration.
*
* Follows the same pattern as {@link SandboxError}: a human-readable
* `message`, a structured `code` for programmatic handling, and a
* static `isInstance` guard that works across realms.
*
* @example
* ```typescript
* try {
*   createDeepAgent({ tools: [myTool] });
* } catch (error) {
*   if (ConfigurationError.isInstance(error)) {
*     switch (error.code) {
*       case "TOOL_NAME_COLLISION":
*         console.error("Rename your tool:", error.message);
*         break;
*     }
*   }
* }
* ```
*/
var ConfigurationError = class ConfigurationError extends Error {
	[CONFIGURATION_ERROR_SYMBOL] = true;
	name = "ConfigurationError";
	constructor(message, code, cause) {
		super(message);
		this.code = code;
		this.cause = cause;
		Object.setPrototypeOf(this, ConfigurationError.prototype);
	}
	static isInstance(error) {
		return typeof error === "object" && error !== null && error[CONFIGURATION_ERROR_SYMBOL] === true;
	}
};
//#endregion
//#region src/middleware/cache.ts
/**
* Creates a middleware that places a cache breakpoint at the end of the static
* system prompt content.
*
* This middleware tags the last block of the system message with
* `cache_control: { type: "ephemeral" }` at the time it runs, capturing all
* static content injected by preceding middleware (e.g. todo list instructions,
* filesystem tools, subagent instructions) in a single cache breakpoint.
*
* This should run after all static system prompt middleware and before any
* dynamic middleware (e.g. memory) so the breakpoint sits at the boundary
* between stable and changing content.
*
* When used alongside memory middleware (which adds its own breakpoint on the
* memory block), the result is two separate cache breakpoints:
* - One covering all static content
* - One covering the memory block
*
* This is a no-op when the system message has no content blocks.
*/
function createCacheBreakpointMiddleware() {
	return createMiddleware({
		name: "CacheBreakpointMiddleware",
		wrapModelCall(request, handler) {
			const existingContent = request.systemMessage.content;
			const existingBlocks = typeof existingContent === "string" ? [{
				type: "text",
				text: existingContent
			}] : Array.isArray(existingContent) ? [...existingContent] : [];
			if (existingBlocks.length === 0) return handler(request);
			existingBlocks[existingBlocks.length - 1] = {
				...existingBlocks[existingBlocks.length - 1],
				cache_control: { type: "ephemeral" }
			};
			return handler({
				...request,
				systemMessage: new SystemMessage({ content: existingBlocks })
			});
		}
	});
}
//#endregion
//#region src/agent.ts
const BASE_AGENT_PROMPT = context`
  You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

  ## Core Behavior

  - Be concise and direct. Don't over-explain unless asked.
  - NEVER add unnecessary preamble (\"Sure!\", \"Great question!\", \"I'll now...\").
  - Don't say \"I'll now do X\" — just do it.
  - If the request is ambiguous, ask questions before acting.
  - If asked how to approach something, explain first, then act.

  ## Professional Objectivity

  - Prioritize accuracy over validating the user's beliefs
  - Disagree respectfully when the user is incorrect
  - Avoid unnecessary superlatives, praise, or emotional validation

  ## Doing Tasks

  When the user asks you to do something:

  1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
  2. **Act** — implement the solution. Work quickly but accurately.
  3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

  Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.

  **When things go wrong:**
  - If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
  - If you're blocked, tell the user what's wrong and ask for guidance.

  ## Progress Updates

  For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next.
`;
const BUILTIN_TOOL_NAMES = new Set([
	...FILESYSTEM_TOOL_NAMES,
	...ASYNC_TASK_TOOL_NAMES,
	"task",
	"write_todos"
]);
/**
* Detect whether a model is an Anthropic model.
* Used to gate Anthropic-specific prompt caching optimizations (cache_control breakpoints).
*/
function isAnthropicModel(model) {
	if (typeof model === "string") {
		if (model.includes(":")) return model.split(":")[0] === "anthropic";
		return model.startsWith("claude");
	}
	if (model.getName() === "ConfigurableModel") return model._defaultConfig?.modelProvider === "anthropic";
	return model.getName() === "ChatAnthropic";
}
/**
* Create a Deep Agent.
*
* This is the main entry point for building a production-style agent with
* deepagents. It gives you a strong default runtime (filesystem, tasks,
* subagents, summarization) and lets you opt into skills, memory,
* human-in-the-loop interrupts, async subagents, and custom middleware.
*
* The runtime is intentionally opinionated: defaults work out of the box, and
* when you customize behavior, the middleware ordering stays deterministic.
*
* @param params Configuration parameters for the agent
* @returns Deep Agent instance with inferred state/response types
*
* @example
* ```typescript
* // Middleware with custom state
* const ResearchMiddleware = createMiddleware({
*   name: "ResearchMiddleware",
*   stateSchema: z.object({ research: z.string().default("") }),
* });
*
* const agent = createDeepAgent({
*   middleware: [ResearchMiddleware],
* });
*
* const result = await agent.invoke({ messages: [...] });
* // result.research is properly typed as string
* ```
*/
function createDeepAgent(params = {}) {
	const { model = new ChatAnthropic("claude-sonnet-4-6"), tools = [], systemPrompt, middleware: customMiddleware = [], subagents = [], responseFormat, contextSchema, checkpointer, store, backend = (config) => new StateBackend(config), filesystemOptions, interruptOn, name, memory, skills } = params;
	const collidingTools = tools.map((t) => t.name).filter((n) => typeof n === "string" && BUILTIN_TOOL_NAMES.has(n));
	if (collidingTools.length > 0) throw new ConfigurationError(`Tool name(s) [${collidingTools.join(", ")}] conflict with built-in tools. Rename your custom tools to avoid this.`, "TOOL_NAME_COLLISION");
	const anthropicModel = isAnthropicModel(model);
	const cacheMiddleware = anthropicModel ? [anthropicPromptCachingMiddleware({
		unsupportedModelBehavior: "ignore",
		minMessagesToCache: 1
	}), createCacheBreakpointMiddleware()] : [];
	/**
	* Process subagents to add SkillsMiddleware for those with their own skills.
	*
	* Custom subagents do NOT inherit skills from the main agent by default.
	* Only the general-purpose subagent inherits the main agent's skills.
	* If a custom subagent needs skills, it must specify its own `skills` array.
	*/
	const normalizeSubagentSpec = (input) => {
		const subagentMiddleware = [
			todoListMiddleware(),
			createFilesystemMiddleware({
				backend,
				...filesystemOptions
			}),
			createSummarizationMiddleware({
				backend,
				model
			}),
			createPatchToolCallsMiddleware(),
			...input.skills != null && input.skills.length > 0 ? [createSkillsMiddleware({
				backend,
				sources: input.skills
			})] : [],
			...input.middleware ?? [],
			...cacheMiddleware
		];
		return {
			...input,
			tools: input.tools ?? [],
			middleware: subagentMiddleware
		};
	};
	const allSubagents = subagents;
	const asyncSubAgents = allSubagents.filter((item) => isAsyncSubAgent(item));
	const inlineSubagents = allSubagents.filter((item) => !isAsyncSubAgent(item)).map((item) => "runnable" in item ? item : normalizeSubagentSpec(item));
	if (!inlineSubagents.some((item) => item.name === GENERAL_PURPOSE_SUBAGENT["name"])) {
		const generalPurposeSpec = normalizeSubagentSpec({
			...GENERAL_PURPOSE_SUBAGENT,
			model,
			skills,
			tools
		});
		inlineSubagents.unshift(generalPurposeSpec);
	}
	const skillsMiddleware = skills != null && skills.length > 0 ? [createSkillsMiddleware({
		backend,
		sources: skills
	})] : [];
	const [todoMiddleware, fsMiddleware, subagentMiddleware, summarizationMiddleware, patchToolCallsMiddleware] = [
		todoListMiddleware(),
		createFilesystemMiddleware({
			backend,
			...filesystemOptions
		}),
		createSubAgentMiddleware({
			defaultModel: model,
			defaultTools: tools,
			defaultInterruptOn: interruptOn,
			subagents: inlineSubagents,
			generalPurposeAgent: false
		}),
		createSummarizationMiddleware({
			model,
			backend
		}),
		createPatchToolCallsMiddleware()
	];
	const middleware = [
		todoMiddleware,
		...skillsMiddleware,
		fsMiddleware,
		subagentMiddleware,
		summarizationMiddleware,
		patchToolCallsMiddleware,
		...asyncSubAgents.length > 0 ? [createAsyncSubAgentMiddleware({ asyncSubAgents })] : [],
		...customMiddleware,
		...cacheMiddleware,
		...memory && memory.length > 0 ? [createMemoryMiddleware({
			backend,
			sources: memory,
			addCacheControl: anthropicModel
		})] : [],
		...interruptOn ? [humanInTheLoopMiddleware({ interruptOn })] : []
	];
	/**
	* Return as DeepAgent with proper DeepAgentTypeConfig
	* - Response: InferStructuredResponse<TResponse> (unwraps ToolStrategy<T>/ProviderStrategy<T> → T)
	* - State: undefined (state comes from middleware)
	* - Context: ContextSchema
	* - Middleware: AllMiddleware (built-in + custom + subagent middleware for state inference)
	* - Tools: TTools
	* - Subagents: TSubagents (for type-safe streaming)
	*/
	return createAgent({
		model,
		systemPrompt: typeof systemPrompt === "string" ? new SystemMessage({ contentBlocks: [{
			type: "text",
			text: systemPrompt
		}, {
			type: "text",
			text: BASE_AGENT_PROMPT
		}] }) : SystemMessage.isInstance(systemPrompt) ? new SystemMessage({ contentBlocks: [...systemPrompt.contentBlocks, {
			type: "text",
			text: BASE_AGENT_PROMPT
		}] }) : new SystemMessage({ contentBlocks: [{
			type: "text",
			text: BASE_AGENT_PROMPT
		}] }),
		tools,
		middleware,
		...responseFormat !== null && { responseFormat },
		contextSchema,
		checkpointer,
		store,
		name
	}).withConfig({
		recursionLimit: 1e4,
		metadata: {
			ls_integration: "deepagents",
			lc_agent_name: name
		}
	});
}
//#endregion
export { BaseSandbox, CompositeBackend, ConfigurationError, DEFAULT_GENERAL_PURPOSE_DESCRIPTION, DEFAULT_SUBAGENT_PROMPT, GENERAL_PURPOSE_SUBAGENT, MAX_SKILL_DESCRIPTION_LENGTH, MAX_SKILL_FILE_SIZE, MAX_SKILL_NAME_LENGTH, SandboxError, StateBackend, StoreBackend, TASK_SYSTEM_PROMPT, computeSummarizationDefaults, createDeepAgent, createFilesystemMiddleware, createMemoryMiddleware, createPatchToolCallsMiddleware, createSkillsMiddleware, createSubAgentMiddleware, createSummarizationMiddleware, filesValue, isSandboxBackend, resolveBackend };

//# sourceMappingURL=index.browser.js.map