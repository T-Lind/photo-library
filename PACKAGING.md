# Packaging photolib as a desktop app

The goal is a thing someone can double-click. This describes how that is
built and why it's put together the way it is.

## What ships

A single native application containing:

| Piece | What it is |
|---|---|
| `photolib-server` | The Python backend, frozen with PyInstaller |
| `models/siglip2-base` | The image/text model, exported to ONNX |
| `web/` | The UI, statically exported from Next.js |
| `photolib.exe` | A small Tauri shell that starts the server and opens a window |

There is no Python to install, no Node.js at runtime, and no PyTorch.

## Why ONNX instead of PyTorch

PyTorch is about 2.5 GB installed and is the single most difficult
dependency to freeze — it loads native libraries through paths PyInstaller's
static analysis cannot follow, and the resulting binaries are enormous.

The model is exported once to ONNX and run with onnxruntime, which is about
50 MB and was already a dependency for face recognition. The installer drops
from roughly 3 GB to roughly 500 MB, cold start gets faster, and the whole
class of PyTorch freezing problems disappears.

The cost is that preprocessing has to be reimplemented outside 🤗
`transformers`. That is the one genuinely risky part of this approach, so it
is guarded three ways:

1. The exporter reads preprocessing parameters **off the real processor** and
   writes them to `preprocess.json`. Nothing is hardcoded or guessed —
   including whether the processor canonicalises text, which is determined by
   probing it rather than keying off the model name.
2. The exporter records **golden vectors**: reference tokenisations and
   embeddings from the real PyTorch model.
3. `photolib verify-model` replays those through the NumPy runtime and fails
   if anything drifts. CI runs it against the frozen binary, so a
   preprocessing regression breaks the build rather than quietly degrading
   search months later.

## Building

Everything below runs in CI (`.github/workflows/desktop.yml`). Run it from
the Actions tab, or push a `v*` tag to attach installers to a draft release.
To do it by hand:

```bash
# 1. Export the model (needs PyTorch; only needed once per model)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.49" onnx onnxruntime tokenizers sentencepiece
python tools/export_onnx.py --model google/siglip2-base-patch16-224 \
                            --out models/siglip2-base

# 2. Build the UI (in the frontend repo)
npm ci && npm run build:export
cp -r out ../photo-library/web

# 3. Freeze the server
pip install -r requirements-min.txt onnxruntime tokenizers insightface pyinstaller
pyinstaller packaging/photolib.spec --noconfirm --clean

# 4. Build the desktop app
mkdir -p desktop/src-tauri/binaries
cp -r dist/photolib-server/* desktop/src-tauri/binaries/
mv desktop/src-tauri/binaries/photolib-server.exe \
   desktop/src-tauri/binaries/photolib-server-x86_64-pc-windows-msvc.exe
cd desktop && npm install && npx tauri build
```

Installers land in `desktop/src-tauri/target/release/bundle/`.

Windows needs the MSVC build tools and WebView2 (present on Windows 10 21H2
and later). macOS needs Xcode command line tools. Linux needs
`libwebkit2gtk` and `libgtk-3` development packages.

## How it starts

1. The Tauri shell spawns `photolib-server --no-browser` as a sidecar.
2. The server picks a **free port** — hardcoding 8000 fails on any machine
   where something already holds it, in a way a non-technical user has no
   hope of diagnosing — and prints `PHOTOLIB_READY {"url": ...}` on stdout.
3. The shell reads that line and only then creates the window pointing at it,
   so nobody ever sees a connection-refused page mid-startup.
4. If the server dies before becoming ready, the shell exits rather than
   leaving an invisible process running.

## Where data goes

A frozen binary's working directory is wherever the shortcut happened to
point, and it may be installed somewhere read-only, so nothing is written
next to the executable:

| Platform | Location |
|---|---|
| Windows | `%LOCALAPPDATA%\photolib` |
| macOS | `~/Library/Application Support/photolib` |
| Linux | `~/.local/share/photolib` |

That holds the library, thumbnails, logs, and downloaded face models.
Deleting that folder resets the app completely. Any `PHOTO_*` environment
variable still overrides its default.

## Network behaviour

The packaged app makes exactly one kind of network request, and only when
the user triggers it: downloading the face recognition weights on first use.

- The image/text model is **bundled** — SigLIP 2 is Apache-2.0, so it can be
  redistributed. Search works offline from the very first launch.
- Face weights are **downloaded once** from the InsightFace project's own
  release page. They are published for non-commercial research use, so
  redistributing them inside an installer is a question best avoided;
  fetching them on the user's machine is not. `GET /api/v1/admin/models`
  reports the exact URL, size, and licence before anything is fetched.
- `PHOTO_OFFLINE=1` refuses even that, for an air-gapped install. Copy the
  model directory across by hand instead.
- The launcher sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` so the ML
  libraries cannot contact a model hub behind the application's back.
- Next.js telemetry is disabled in every build script.
- The UI uses a system font stack rather than a web font, so it never
  contacts a font CDN and the build works offline.

The only outbound request the UI can make is if someone clicks a photo's
coordinates, which opens OpenStreetMap in their browser. No map tiles are
ever loaded in-app.

## Sizes

Approximate, for the base model on Windows:

| | |
|---|---|
| onnxruntime + Python runtime | ~120 MB |
| SigLIP 2 base, ONNX fp32 | ~750 MB |
| SigLIP 2 base, ONNX int8 (`--quantize`) | ~200 MB |
| Web UI | ~1.5 MB |
| Face weights (downloaded on first use) | ~290 MB |

`--quantize` produces markedly smaller installers. It does cost some
retrieval quality, so check it against your own photos before shipping it —
`photolib verify-model` confirms the plumbing, not that ranking is still
good enough for you.
