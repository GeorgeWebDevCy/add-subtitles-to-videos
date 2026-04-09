Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& "$PSScriptRoot\build_windows_bundle.ps1"

$iscc = (Get-Command ISCC -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)

if (-not $iscc) {
    $candidatePaths = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Antigravity\resources\app\node_modules\innosetup\bin\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )

    $iscc = $candidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $iscc) {
    throw "Could not find ISCC.exe. Install Inno Setup 6 first."
}

& $iscc "packaging\windows\subtitle_foundry.iss"
