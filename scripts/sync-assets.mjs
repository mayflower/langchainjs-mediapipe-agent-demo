import { cp, mkdir, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const demoRoot = path.resolve(__dirname, "..");
const publicDir = path.join(demoRoot, "public");
const mediaPipeWasmTargetDir = path.join(
  publicDir,
  "vendor",
  "mediapipe",
  "tasks-genai",
  "wasm"
);
const wasmshRootDir = path.join(publicDir, "vendor", "wasmsh");
const wasmshAssetsTargetDir = path.join(wasmshRootDir, "assets");
const wasmshLibTargetDir = path.join(wasmshRootDir, "lib");
const wasmshWorkerTarget = path.join(wasmshRootDir, "browser-worker.js");
const require = createRequire(import.meta.url);

async function resolvePackageDir(resolvedEntry) {
  const searchRoots = [demoRoot];

  for (const searchRoot of searchRoots) {
    try {
      const packageEntryPath = require.resolve(resolvedEntry, {
        paths: [searchRoot],
      });
      const packageDir = path.dirname(packageEntryPath);
      await access(packageDir);
      return packageDir;
    } catch {}
  }

  throw new Error(
    `Could not resolve ${resolvedEntry} from this demo installation. Run pnpm install first.`
  );
}

const mediaPipePackageDir = await resolvePackageDir("@mediapipe/tasks-genai");
const wasmshPackageDir = await resolvePackageDir(
  "@mayflowergmbh/wasmsh-pyodide"
);
const wasmSourceDir = path.join(mediaPipePackageDir, "wasm");
const wasmshAssetsSourceDir = path.join(wasmshPackageDir, "assets");
const wasmshLibSourceDir = path.join(wasmshPackageDir, "lib");
const wasmshWorkerSource = path.join(wasmshPackageDir, "browser-worker.js");

await mkdir(publicDir, { recursive: true });
await cp(wasmSourceDir, mediaPipeWasmTargetDir, {
  recursive: true,
  force: true,
});
await mkdir(wasmshRootDir, { recursive: true });
await cp(wasmshAssetsSourceDir, wasmshAssetsTargetDir, {
  recursive: true,
  force: true,
});
await cp(wasmshLibSourceDir, wasmshLibTargetDir, {
  recursive: true,
  force: true,
});
await cp(wasmshWorkerSource, wasmshWorkerTarget, { force: true });

console.log(`Copied MediaPipe WASM assets to ${mediaPipeWasmTargetDir}`);
console.log(`Copied wasmsh assets to ${wasmshRootDir}`);
