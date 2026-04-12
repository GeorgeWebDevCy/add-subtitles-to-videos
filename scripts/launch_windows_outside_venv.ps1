Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Get-ExternalPythonRoot {
    $uvPythonDirectory = Join-Path $env:APPDATA "uv\python"
    if (Test-Path $uvPythonDirectory) {
        $candidate = Get-ChildItem -Path $uvPythonDirectory -Directory -Filter "cpython-3.12*-windows-x86_64-none" |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($null -ne $candidate) {
            return $candidate.FullName
        }
    }

    return $null
}

function Ensure-ExternalPythonRoot {
    $pythonRoot = Get-ExternalPythonRoot
    if ($null -ne $pythonRoot) {
        return $pythonRoot
    }

    Write-Host "Installing a user-level CPython 3.12 runtime with uv..."
    uv python install 3.12

    $pythonRoot = Get-ExternalPythonRoot
    if ($null -eq $pythonRoot) {
        throw "Couldn't locate the uv-managed CPython 3.12 runtime after installation."
    }

    return $pythonRoot
}

function Test-ExternalAppInstall {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe
    )

    $checkScript = "import importlib.util, sys; required = ['torch', 'whisper', 'faster_whisper', 'PySide6', 'requests', 'add_subtitles_to_videos']; missing = [name for name in required if importlib.util.find_spec(name) is None]; print('missing:' + ','.join(missing) if missing else sys.executable); sys.exit(1 if missing else 0)"

    & $PythonExe -c $checkScript *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-ExternalAppInstall {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe
    )

    if (Test-ExternalAppInstall -PythonExe $PythonExe) {
        return
    }

    Write-Host "Installing Subtitle Foundry into the external Python runtime..."
    & $PythonExe -m pip install --break-system-packages -e $repoRoot

    if (-not (Test-ExternalAppInstall -PythonExe $PythonExe)) {
        throw "Subtitle Foundry dependencies are still missing from the external Python runtime."
    }
}

$pythonRoot = Ensure-ExternalPythonRoot
$pythonExe = Join-Path $pythonRoot "python.exe"
$pythonwExe = Join-Path $pythonRoot "pythonw.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Couldn't find python.exe under the external runtime: $pythonRoot"
}

if (-not (Test-Path $pythonwExe)) {
    throw "Couldn't find pythonw.exe under the external runtime: $pythonRoot"
}

Ensure-ExternalAppInstall -PythonExe $pythonExe

if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
}

Start-Process -FilePath $pythonwExe -ArgumentList "-m", "add_subtitles_to_videos" -WorkingDirectory $repoRoot
