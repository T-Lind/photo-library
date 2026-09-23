"""photolib — a local, private, open-source photo library.

Semantic (natural-language) search, face search and recognition, and EXIF
organisation over a LanceDB store. Nothing leaves the machine.
"""

from __future__ import annotations

import os as _os


def _preload_msvc_runtime() -> None:
    """Load the system C++ runtime before pyarrow gets a chance to.

    PyArrow's Windows wheels bundle an old ``msvcp140.dll`` (14.28, VS 2019).
    Importing pyarrow first loads that copy process-wide, and PyTorch 2.x —
    built against a newer MSVC runtime — then fails with
    ``WinError 1114`` while initialising ``c10.dll``. Preloading the system
    ``msvcp140.dll`` (14.4x+) makes the loader reuse it, so both libraries
    work regardless of import order. A no-op off Windows.
    """
    if _os.name != "nt":
        return
    import ctypes

    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        try:
            ctypes.CDLL(name)
        except OSError:
            pass


_preload_msvc_runtime()

__version__ = "2.0.3"
