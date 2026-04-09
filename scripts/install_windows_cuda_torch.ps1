Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Project virtual environment not found. Running 'uv sync' first..."
    uv sync
}

Write-Host "Syncing the project environment with the CUDA-enabled PyTorch wheel..."
uv sync --reinstall-package torch

Write-Host "Verifying CUDA visibility..."
uv run python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"
