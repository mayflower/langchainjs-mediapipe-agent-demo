import { BaseSandbox } from "deepagents";
import { createBrowserWorkerSession } from "@mayflowergmbh/wasmsh-pyodide/browser";

function shellQuote(value) {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

function getDiagnosticError(events) {
  if (!events) {
    return undefined;
  }

  for (const event of events) {
    if (
      event &&
      typeof event === "object" &&
      "Diagnostic" in event &&
      Array.isArray(event.Diagnostic)
    ) {
      const [, message] = event.Diagnostic;
      return message;
    }
  }

  return undefined;
}

function mapDownloadError(message) {
  const normalized = message?.toLowerCase() ?? "";
  if (normalized.includes("not found")) {
    return "file_not_found";
  }
  if (normalized.includes("directory")) {
    return "is_directory";
  }
  if (normalized.includes("permission")) {
    return "permission_denied";
  }
  return "invalid_path";
}

export class WasmshSandbox extends BaseSandbox {
  #session = null;

  #workerUrl;

  #worker;

  #assetBaseUrl;

  #stepBudget;

  #workingDirectory;

  #initialFiles;

  #id;

  constructor(options) {
    super();
    this.#workerUrl = options.workerUrl;
    this.#worker = options.worker ?? null;
    this.#assetBaseUrl = options.assetBaseUrl;
    this.#stepBudget = options.stepBudget ?? 0;
    this.#workingDirectory = options.workingDirectory ?? "/workspace";
    this.#initialFiles = options.initialFiles ?? {};
    this.#id = `wasmsh-browser-${Date.now()}`;
  }

  get id() {
    return this.#id;
  }

  get isRunning() {
    return this.#session !== null;
  }

  static async createBrowserWorker(options) {
    if (!options.worker && !options.workerUrl) {
      throw new Error(
        "WasmshSandbox.createBrowserWorker requires either worker or workerUrl."
      );
    }

    const sandbox = new WasmshSandbox({
      workerUrl: options.workerUrl,
      worker: options.worker,
      assetBaseUrl: options.assetBaseUrl,
      stepBudget: options.stepBudget,
      workingDirectory: options.workingDirectory,
      initialFiles: options.initialFiles,
    });
    await sandbox.initialize();
    return sandbox;
  }

  async initialize() {
    if (this.#session) {
      throw new Error("WasmshSandbox is already initialized");
    }

    this.#session = await createBrowserWorkerSession({
      worker:
        this.#worker ??
        new Worker(this.#workerUrl, {
          name: "wasmsh-pyodide",
        }),
      assetBaseUrl: this.#assetBaseUrl,
      stepBudget: this.#stepBudget,
      initialFiles: Object.entries(this.#initialFiles).map(([path, content]) => ({
        path,
        content:
          typeof content === "string"
            ? new TextEncoder().encode(content)
            : content,
      })),
    });
  }

  async stop() {
    await this.close();
  }

  async close() {
    if (!this.#session) {
      return;
    }

    try {
      await this.#session.close();
    } finally {
      this.#session = null;
    }
  }

  async execute(command) {
    if (!this.#session) {
      throw new Error("WasmshSandbox is not initialized");
    }

    const fullCommand = `cd ${shellQuote(this.#workingDirectory)} && ${command}`;
    const result = await this.#session.run(fullCommand);

    return {
      output: result.output ?? `${result.stdout ?? ""}${result.stderr ?? ""}`,
      exitCode: result.exitCode ?? null,
      truncated: false,
    };
  }

  async uploadFiles(files) {
    if (!this.#session) {
      throw new Error("WasmshSandbox is not initialized");
    }

    const responses = [];
    for (const [path, content] of files) {
      if (!path.startsWith("/")) {
        responses.push({ path, error: "invalid_path" });
        continue;
      }

      try {
        const result = await this.#session.writeFile(path, content);
        const diagnostic = getDiagnosticError(result?.events);
        responses.push({
          path,
          error: diagnostic ? mapDownloadError(diagnostic) : null,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        responses.push({ path, error: mapDownloadError(message) });
      }
    }

    return responses;
  }

  async downloadFiles(paths) {
    if (!this.#session) {
      throw new Error("WasmshSandbox is not initialized");
    }

    const responses = [];
    for (const path of paths) {
      if (!path.startsWith("/")) {
        responses.push({ path, content: null, error: "invalid_path" });
        continue;
      }

      try {
        const result = await this.#session.readFile(path);
        const diagnostic = getDiagnosticError(result?.events);
        responses.push(
          diagnostic
            ? { path, content: null, error: mapDownloadError(diagnostic) }
            : {
                path,
                content: result.content,
                error: null,
              }
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        responses.push({
          path,
          content: null,
          error: mapDownloadError(message),
        });
      }
    }

    return responses;
  }
}
