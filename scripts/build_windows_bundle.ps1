Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Get-Process SubtitleFoundry -ErrorAction SilentlyContinue | Stop-Process -Force

uv run python scripts/generate_branding_assets.py

if (Test-Path ".\build\pyinstaller") {
    Remove-Item -Recurse -Force ".\build\pyinstaller"
}

if (Test-Path ".\dist\SubtitleFoundry") {
    Remove-Item -Recurse -Force ".\dist\SubtitleFoundry"
}

uv run python -m PyInstaller `
    --clean `
    --noconfirm `
    --distpath ".\dist" `
    --workpath ".\build\pyinstaller" `
    "packaging\windows\subtitle_foundry.spec"
