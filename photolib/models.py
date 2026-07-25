"""Model asset management: what's present, what's missing, how to fetch it.

The packaged application ships the image/text model (SigLIP 2 is Apache-2.0,
so it can be redistributed) but downloads the face recognition weights on
first launch. That is partly a size decision and partly a licensing one:
InsightFace's ``buffalo_l`` is published for non-commercial research use, so
redistributing it inside an installer is a question best avoided. Fetching it
on the user's own machine, once, with clear attribution, is not.

Everything here is explicit and inspectable — a URL, a size, a destination —
so "is this thing phoning home?" has an auditable answer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]

# Published by the InsightFace project. Pinned to a release tag rather than a
# moving target so an installer built today fetches the same weights next year.
INSIGHTFACE_BASE = "https://github.com/deepinsight/insightface/releases/download/v0.7"


class OfflineError(RuntimeError):
    """Raised when a download is needed but the build is pinned offline."""


class DownloadError(RuntimeError):
    pass


@dataclass
class ModelSpec:
    name: str
    kind: str                     # "face" | "embed"
    url: str
    # Files that must exist under the destination for the model to be usable.
    members: List[str] = field(default_factory=list)
    approx_bytes: int = 0
    sha256: Optional[str] = None
    licence: str = ""

    def destination(self, root: Path) -> Path:
        return root / self.name

    def installed(self, root: Path) -> bool:
        dest = self.destination(root)
        if not dest.is_dir():
            return False
        if not self.members:
            return any(dest.iterdir())
        return all((dest / m).exists() for m in self.members)


FACE_MODELS: Dict[str, ModelSpec] = {
    "buffalo_l": ModelSpec(
        name="buffalo_l",
        kind="face",
        url=f"{INSIGHTFACE_BASE}/buffalo_l.zip",
        members=["det_10g.onnx", "w600k_r50.onnx"],
        approx_bytes=288_000_000,
        licence="InsightFace model licence — non-commercial research use",
    ),
    "buffalo_s": ModelSpec(
        name="buffalo_s",
        kind="face",
        url=f"{INSIGHTFACE_BASE}/buffalo_s.zip",
        members=["det_500m.onnx", "w600k_mbf.onnx"],
        approx_bytes=16_000_000,
        licence="InsightFace model licence — non-commercial research use",
    ),
}


def offline() -> bool:
    """Is this build pinned offline? Set by the packaged app after first run."""
    return os.environ.get("PHOTO_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def enforce_offline_env() -> None:
    """Stop the ML libraries from reaching the network behind our back.

    ``transformers`` will happily contact the model hub to check for updated
    files even when a local copy exists. For an application whose entire
    premise is that nothing leaves the machine, that has to be impossible
    rather than merely unlikely.
    """
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(key, "1")


def face_model_root(settings=None) -> Path:
    """Directory holding face model folders.

    InsightFace resolves a model as ``<home>/models/<name>``, so the folder
    that actually contains ``buffalo_l/`` is one level below the configured
    home. Matching its convention exactly means ``FaceAnalysis(root=...)``
    finds what this module downloads, with no path juggling in between.
    """
    from .config import get_settings

    settings = settings or get_settings()
    return Path(settings.face_model_root).expanduser() / "models"


def status(settings=None) -> dict:
    """What is installed, what is missing, and how big the gap is."""
    from .config import get_settings

    settings = settings or get_settings()
    root = face_model_root(settings)

    entries = []
    if settings.face_backend == "insightface":
        spec = FACE_MODELS.get(settings.face_model)
        if spec is not None:
            entries.append({
                "name": spec.name,
                "kind": spec.kind,
                "installed": spec.installed(root),
                "path": str(spec.destination(root)),
                "approx_bytes": spec.approx_bytes,
                "url": spec.url,
                "licence": spec.licence,
            })

    if settings.embed_backend == "onnx":
        model_dir = Path(settings.onnx_model_dir).expanduser()
        entries.append({
            "name": Path(settings.onnx_model_dir).name,
            "kind": "embed",
            "installed": (model_dir / "preprocess.json").exists(),
            "path": str(model_dir),
            "approx_bytes": 0,
            "url": "",
            "licence": "Apache-2.0 (SigLIP 2) — bundled with the application",
        })

    missing = [e for e in entries if not e["installed"]]
    return {
        "models": entries,
        "ready": not missing,
        "missing": [e["name"] for e in missing],
        "download_bytes": sum(e["approx_bytes"] for e in missing),
        "offline": offline(),
    }


def ensure_face_model(settings=None, progress: Optional[ProgressFn] = None) -> Path:
    """Make sure the configured face model is on disk; download if not."""
    from .config import get_settings

    settings = settings or get_settings()
    root = face_model_root(settings)
    spec = FACE_MODELS.get(settings.face_model)
    if spec is None:
        raise DownloadError(
            f"Unknown face model {settings.face_model!r}. "
            f"Known: {', '.join(sorted(FACE_MODELS))}")

    dest = spec.destination(root)
    if spec.installed(root):
        return dest
    if offline():
        raise OfflineError(
            f"{spec.name} is not installed at {dest} and this build is pinned "
            "offline. Unset PHOTO_OFFLINE, or copy the model directory across "
            "from a machine that has it.")

    logger.info("Downloading %s (~%d MB) from %s",
                spec.name, spec.approx_bytes // 1_000_000, spec.url)
    _download_and_extract(spec, dest, progress)

    if not spec.installed(root):
        raise DownloadError(
            f"{spec.name} downloaded but is missing expected files "
            f"({', '.join(spec.members)}) in {dest}")
    return dest


def _download_and_extract(spec: ModelSpec, dest: Path,
                          progress: Optional[ProgressFn]) -> None:
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="photolib-model-", dir=dest.parent))
    archive = tmp_dir / f"{spec.name}.zip"

    try:
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(spec.url, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or spec.approx_bytes)
                done = 0
                with open(archive, "wb") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(spec.name, done, total)
        except urllib.error.URLError as exc:
            raise DownloadError(
                f"Could not download {spec.name} from {spec.url}: {exc.reason}. "
                "Check the machine's internet connection and try again."
            ) from exc

        if spec.sha256 and digest.hexdigest() != spec.sha256:
            raise DownloadError(
                f"{spec.name} failed its checksum — the download was corrupted "
                "or the file at that URL changed.")

        staging = tmp_dir / "extracted"
        with zipfile.ZipFile(archive) as zf:
            _safe_extract(zf, staging)

        # Some archives contain a top-level folder, some don't.
        contents = [p for p in staging.iterdir()]
        source = contents[0] if len(contents) == 1 and contents[0].is_dir() else staging

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(source), str(dest))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _safe_extract(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract without letting an archive write outside the target.

    A downloaded archive is untrusted input even when the URL is trusted;
    Python's ``extractall`` happily honours ``../`` members.
    """
    target.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()
    for member in zf.infolist():
        if member.is_dir():
            continue
        out_path = (target / member.filename).resolve()
        if not out_path.is_relative_to(resolved_target):
            raise DownloadError(
                f"Refusing to extract {member.filename!r}: it escapes the "
                "destination directory.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
