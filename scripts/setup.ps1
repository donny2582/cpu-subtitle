# 建立 .venv 並安裝套件。用法：powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-Python python]
param([string]$Python = "python", [switch]$Dev)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
}
& .venv\Scripts\python -m pip install --upgrade pip
& .venv\Scripts\python -m pip install -r requirements.txt
if ($Dev) { & .venv\Scripts\python -m pip install pytest }
if (-not (Test-Path "config.toml")) { Copy-Item config.example.toml config.toml }
Write-Host "完成。執行：powershell -ExecutionPolicy Bypass -File scripts\run.ps1"
