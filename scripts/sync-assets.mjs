import { cp, mkdir, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const demoRoot = path.resolve(__dirname, "..");
const publicDir = path.join(demoRoot, "public");
const wasmTargetDir = path.join(
  publicDir,
  "vendor",
  "mediapipe",
  "tasks-genai",
  "wasm"
);
const require = createRequire(import.meta.url);

async function resolveMediaPipePackageDir() {
  const searchRoots = [demoRoot];

  for (const searchRoot of searchRoots) {
    try {
      const packageEntryPath = require.resolve(
        "@mediapipe/tasks-genai",
        {
          paths: [searchRoot],
        }
      );
      const packageDir = path.dirname(packageEntryPath);
      await access(packageDir);
      return packageDir;
    } catch {}
  }

  throw new Error(
    "Could not resolve @mediapipe/tasks-genai from this demo installation. Run pnpm install first."
  );
}

const mediaPipePackageDir = await resolveMediaPipePackageDir();
const wasmSourceDir = path.join(mediaPipePackageDir, "wasm");

await mkdir(publicDir, { recursive: true });
await cp(wasmSourceDir, wasmTargetDir, {
  recursive: true,
  force: true,
});

console.log(`Copied MediaPipe WASM assets to ${wasmTargetDir}`);
