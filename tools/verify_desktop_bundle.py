"""Verify the frozen backend from the layout shipped in each installer.

Runs after Tauri bundling in CI. Uses only the standard library so the host
Python environment cannot accidentally satisfy a missing frozen dependency.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
BUNDLES = ROOT / "desktop/src-tauri/target/release/bundle"


def verify(sidecar: Path, work: Path) -> None:
    binary = sidecar / ("photolib-server.exe" if sys.platform == "win32" else "photolib-server")
    if not binary.is_file() or not (sidecar / "_internal").is_dir():
        raise RuntimeError(f"Incomplete installed sidecar: {sidecar}")
    # Do not let the build-time library search path mask a packaging error.
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_LIBRARY_PATH_ORIG", None)
    subprocess.run([str(binary), "--verify-model", "--data-dir", str(work / "verify")],
                   env=env, check=True, timeout=180)
    log_path = work / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(binary), "--no-browser", "--data-dir", str(work / "data")],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env,
        )
        try:
            deadline = time.monotonic() + 90
            url = None
            while time.monotonic() < deadline:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("PHOTOLIB_READY "):
                        url = json.loads(line.removeprefix("PHOTOLIB_READY "))["url"]
                        break
                if url:
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Installed sidecar exited early:\n{text}")
                time.sleep(0.25)
            if not url:
                raise RuntimeError(f"Installed sidecar never became ready:\n{text}")
            for path in ("/", "/api/v1/health", "/api/v1/admin/models"):
                with urlopen(url.rstrip("/") + path, timeout=20) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Installed sidecar returned {response.status} for {path}")
                    if path == "/" and b"folderPath" not in response.read():
                        raise RuntimeError("Installed UI is missing the source-folder controls")
            process.stdin.write(b"PHOTOLIB_SHUTDOWN\n")
            process.stdin.flush()
            if process.wait(timeout=20) != 0:
                raise RuntimeError("Installed sidecar did not shut down cleanly")
            print(f"Verified installed model, UI, API and shutdown: {sidecar}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdin.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="photolib-bundle-") as temporary:
        work = Path(temporary)
        if sys.platform == "win32":
            verify(ROOT / "installer-smoke/sidecar", work)
        elif sys.platform == "darwin":
            mount = work / "dmg"
            mount.mkdir()
            dmg = next((BUNDLES / "dmg").glob("*.dmg"))
            subprocess.run(["hdiutil", "attach", str(dmg), "-nobrowse", "-readonly",
                            "-mountpoint", str(mount)], check=True)
            try:
                app = next(mount.glob("*.app"))
                verify(app / "Contents/Resources/sidecar", work)
            finally:
                subprocess.run(["hdiutil", "detach", str(mount)], check=True)
        else:
            deb = next((BUNDLES / "deb").glob("*.deb"))
            extracted = work / "deb"
            subprocess.run(["dpkg-deb", "--extract", str(deb), str(extracted)], check=True)
            deb_work = work / "deb-test"
            deb_work.mkdir()
            verify(extracted / "usr/lib/photolib/sidecar", deb_work)
            appimage = next((BUNDLES / "appimage").glob("*.AppImage"))
            image_work = work / "appimage-test"
            image_work.mkdir()
            subprocess.run([str(appimage), "--appimage-extract"], cwd=image_work,
                           check=True, stdout=subprocess.DEVNULL)
            verify(image_work / "squashfs-root/usr/lib/photolib/sidecar", image_work)


if __name__ == "__main__":
    main()
