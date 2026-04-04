import { RunTree } from "./run_trees.js";

export class Client {
  constructor(fields = {}) {
    Object.assign(this, fields);
  }

  async createRun() {}

  async updateRun() {}

  async readRun() {
    return undefined;
  }

  async listRuns() {
    return [];
  }
}

export function getDefaultProjectName() {
  return "default";
}

export function traceable(configOrFn, maybeFn) {
  const config =
    typeof configOrFn === "function" ? { name: configOrFn.name || "<lambda>" } : configOrFn ?? { name: "<lambda>" };
  const fn = typeof configOrFn === "function" ? configOrFn : maybeFn;

  if (typeof fn !== "function") {
    return (wrapped) => traceable(config, wrapped);
  }

  fn["langsmith:traceable"] = config;
  return fn;
}

export { RunTree };
