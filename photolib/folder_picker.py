"""Native folder selection for the desktop UI.

The web UI is served from the local API process, so a normal browser file
input cannot reveal the absolute path that the indexer needs. The packaged
app therefore asks the operating system to show its native folder picker.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class FolderChoice:
    path: str = ""
    supported: bool = True
    cancelled: bool = False
    detail: str = ""


def _run(command: Sequence[str], timeout: int = 300) -> FolderChoice:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=flags,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return FolderChoice(supported=False, detail=str(exc))

    selected = result.stdout.strip().lstrip("\ufeff")
    if result.returncode == 0 and selected:
        path = Path(selected).expanduser()
        if path.is_dir():
            return FolderChoice(path=str(path.resolve()))
        return FolderChoice(
            supported=False,
            detail="The folder picker returned a path that is not a directory.",
        )
    if result.returncode in (0, 1):
        return FolderChoice(cancelled=True)
    return FolderChoice(
        supported=False,
        detail=result.stderr.strip() or f"Folder picker exited with {result.returncode}.",
    )


def choose_photo_folder() -> FolderChoice:
    """Show a native folder chooser and return an absolute directory path."""
    if sys.platform == "win32":
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if not powershell:
            return FolderChoice(
                supported=False,
                detail="Windows PowerShell is unavailable; enter the path manually.",
            )
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$d.Description = 'Choose the folder containing your photos'; "
            "$d.ShowNewFolderButton = $false; "
            "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "[Console]::Write($d.SelectedPath) }"
        )
        return _run([powershell, "-NoProfile", "-STA", "-Command", script])

    if sys.platform == "darwin":
        osascript = shutil.which("osascript")
        if not osascript:
            return FolderChoice(supported=False, detail="osascript is unavailable.")
        script = (
            'POSIX path of (choose folder with prompt '
            '"Choose the folder containing your photos")'
        )
        return _run([osascript, "-e", script])

    zenity = shutil.which("zenity")
    if zenity:
        return _run([
            zenity,
            "--file-selection",
            "--directory",
            "--title=Choose the folder containing your photos",
        ])
    kdialog = shutil.which("kdialog")
    if kdialog:
        return _run([
            kdialog,
            "--getexistingdirectory",
            ".",
            "--title",
            "Choose the folder containing your photos",
        ])
    return FolderChoice(
        supported=False,
        detail="No supported folder picker was found; enter the path manually.",
    )
