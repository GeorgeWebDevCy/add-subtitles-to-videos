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
$launcherExe = Join-Path $pythonRoot "Scripts\add-subtitles-to-videos.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Couldn't find python.exe under the external runtime: $pythonRoot"
}

Ensure-ExternalAppInstall -PythonExe $pythonExe

if (-not (Test-Path $launcherExe)) {
    throw "Couldn't find the installed Subtitle Foundry launcher at $launcherExe"
}

Start-Process -FilePath $launcherExe -WorkingDirectory $repoRoot
