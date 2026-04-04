function makeRunId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }

  return `run-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function isRunTree(value) {
  return value instanceof RunTree;
}

export function convertToDottedOrderFormat(startTime, runId, executionOrder = 1) {
  const microsecondPrecisionDatestring = new Date(startTime).toISOString();
  const dottedOrder = `${microsecondPrecisionDatestring}.${runId}.${executionOrder}`;

  return { dottedOrder, microsecondPrecisionDatestring };
}

export class RunTree {
  constructor(fields = {}) {
    Object.assign(this, fields);
    this.id ??= makeRunId();
    this.name ??= "<run>";
    this.child_runs ??= [];
    this.extra ??= {};
    this.events ??= [];
    this.outputs ??= {};
    this.inputs ??= {};
  }

  createChild(fields = {}) {
    const child = new RunTree({
      ...fields,
      parent_run: this,
      client: fields.client ?? this.client,
      project_name: fields.project_name ?? this.project_name,
    });
    this.child_runs.push(child);
    return child;
  }

  async postRun() {}

  async patchRun() {}
}
