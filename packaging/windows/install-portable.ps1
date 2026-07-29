param(
    [switch]$NonInteractive,
    [switch]$NoLaunch,
    [switch]$SkipProcessCheck
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName PresentationFramework

function Show-PhotolibMessage {
    param(
        [string]$Message,
        [string]$Title = "photolib setup",
        [System.Windows.MessageBoxImage]$Icon = [System.Windows.MessageBoxImage]::Information
    )

    if ($NonInteractive) {
        Write-Output "$Title`: $Message"
        return
    }

    [System.Windows.MessageBox]::Show(
        $Message,
        $Title,
        [System.Windows.MessageBoxButton]::OK,
        $Icon
    ) | Out-Null
}

$mutex = [System.Threading.Mutex]::new($false, "Local\photolib-installer")
$ownsMutex = $false

try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        Show-PhotolibMessage `
            "Another photolib setup is already running. Let it finish, then open photolib from the Start menu." `
            -Icon Warning
        exit 2
    }

    $running = if ($SkipProcessCheck) { @() } else {
        @(Get-Process -Name "photolib", "photolib-server" -ErrorAction SilentlyContinue)
    }
    if ($running.Count -gt 0) {
        Show-PhotolibMessage `
            "photolib is currently open. Close it and wait a few seconds, then run this setup again. Your library and photo data will not be removed." `
            -Icon Warning
        exit 3
    }

    if (-not $env:LOCALAPPDATA) {
        throw "Windows did not provide a LOCALAPPDATA folder."
    }

    $programsDir = Join-Path $env:LOCALAPPDATA "Programs"
    $installDir = Join-Path $programsDir "photolib"
    $expectedParent = [System.IO.Path]::GetFullPath($programsDir).TrimEnd("\") + "\"
    $resolvedInstall = [System.IO.Path]::GetFullPath($installDir)

    if (-not $resolvedInstall.StartsWith(
        $expectedParent,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to install outside the current user's Programs folder."
    }

    New-Item -ItemType Directory -Force -Path $installDir | Out-Null

    & robocopy.exe $PSScriptRoot $installDir /MIR /R:2 /W:1 /XF "install-portable.ps1" | Out-Null
    $copyCode = $LASTEXITCODE
    if ($copyCode -ge 8) {
        throw "Windows could not copy the application files (robocopy exit code $copyCode)."
    }

    $appExe = Join-Path $installDir "photolib.exe"
    if (-not (Test-Path -LiteralPath $appExe)) {
        throw "The application executable was not installed."
    }

    $startMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
    New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null
    $shortcutPath = Join-Path $startMenuDir "photolib.lnk"

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $appExe
    $shortcut.WorkingDirectory = $installDir
    $shortcut.IconLocation = "$appExe,0"
    $shortcut.Description = "A local, private photo library"
    $shortcut.Save()

    if (-not $NoLaunch) {
        Start-Process -FilePath $appExe -WorkingDirectory $installDir
    }
}
catch {
    Show-PhotolibMessage `
        ("Setup could not complete:`n`n" + $_.Exception.Message) `
        -Icon Error
    exit 1
}
finally {
    if ($ownsMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
