# 啟動即時字幕。預設開控制視窗；其餘參數原樣傳給程式，例如：
#   scripts\run.ps1 --cli                              # 不開視窗，在終端機執行
#   scripts\run.ps1 --cli --source loopback --mode print
#   scripts\run.ps1 --list-devices
#   scripts\run.ps1 --benchmark
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"
& .venv\Scripts\python -m voicetype @args
exit $LASTEXITCODE
