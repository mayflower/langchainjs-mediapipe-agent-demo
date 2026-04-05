import { access, cp, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const demoRoot = path.resolve(__dirname, "..");
const sourceRoot = "/Users/johann/src/ml/deepagentsjs";

const copies = [
  {
    source: path.join(sourceRoot, "libs", "deepagents", "dist", "index.browser.js"),
    target: path.join(demoRoot, "src", "vendor", "deepagents", "index.browser.js"),
  },
];

for (const item of copies) {
  await access(item.source);
  await mkdir(path.dirname(item.target), { recursive: true });
  await cp(item.source, item.target, { force: true });
  console.log(`Synced ${item.source} -> ${item.target}`);
}

const deepagentsTarget = path.join(
  demoRoot,
  "src",
  "vendor",
  "deepagents",
  "index.browser.js"
);
const deepagentsSource = await readFile(deepagentsTarget, "utf8");
const patchedDeepagentsSource = deepagentsSource
  .replace(
    "return tool(async (input, config) => {",
    "return tool(async (input, runtime) => {"
  )
  .replace(
    "const subagentState = filterStateForSubagent(getCurrentTaskInput());",
    'const currentState = runtime.state && typeof runtime.state === "object" ? runtime.state : getCurrentTaskInput();\n\t\tconst subagentState = filterStateForSubagent(currentState);'
  )
  .replace(
    "const result = await subagent.invoke(subagentState, config);",
    "const result = await subagent.invoke(subagentState, runtime.config);"
  )
  .replace(
    "if (!config.toolCall?.id) {",
    "if (!runtime.toolCall?.id) {"
  )
  .replace(
    "return returnCommandWithStateUpdate(result, config.toolCall.id);",
    "return returnCommandWithStateUpdate(result, runtime.toolCall.id);"
  );

if (patchedDeepagentsSource === deepagentsSource) {
  throw new Error(
    "Could not apply browser subagent patch to vendored deepagents bundle."
  );
}

await writeFile(deepagentsTarget, patchedDeepagentsSource);
console.log(
  "Patched vendored deepagents bundle to use browser runtime.state for subagents."
);

console.log(
  [
    "Vendored browser deep-agent sources were refreshed from:",
    `- ${sourceRoot} (expected branch: feat/wasmsh-sandbox)`,
    "- src/vendor/wasmsh/index.js remains the demo adapter layer, but now delegates browser session creation to @mayflowergmbh/wasmsh-pyodide/browser.",
  ].join("\n")
);
