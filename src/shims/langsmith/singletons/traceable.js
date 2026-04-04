export function getCurrentRunTree() {
  return undefined;
}

export function isTraceableFunction(value) {
  return typeof value === "function" && "langsmith:traceable" in value;
}
