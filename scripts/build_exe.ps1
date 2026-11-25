# 打包成不需要 Python 的 Windows 應用程式
# 用法：powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
# 產出：release\LiveSubtitle\（整個資料夾可帶到別台電腦）與 release\LiveSubtitle.zip
param([switch]$NoZip)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

$py = ".venv\Scripts\python.exe"
& $py -m pip install --quiet pyinstaller
if ($LASTEXITCODE -ne 0) { throw "安裝 PyInstaller 失敗" }

Write-Host "== 1/4 打包程式"
& $py -m PyInstaller packaging\LiveSubtitle.spec --noconfirm --distpath build\dist --workpath build\work
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失敗" }

Write-Host "== 2/4 準備發布資料夾"
$out = "release\LiveSubtitle"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item -Recurse "build\dist\LiveSubtitle\*" $out
Copy-Item config.example.toml $out
Copy-Item "packaging\使用說明.txt" $out

# 模型不打包：使用者第一次按「開始字幕」時才自動下載到程式旁邊的 models\，之後重複使用
Write-Host "== 3/4 （模型不打包，第一次使用時自動下載）"

if (-not $NoZip) {
    Write-Host "== 4/4 壓縮成 zip"
    $zip = "release\LiveSubtitle.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    & $py -c "import shutil; shutil.make_archive(r'release\LiveSubtitle', 'zip', r'release', 'LiveSubtitle')"
    if ($LASTEXITCODE -ne 0) { throw "壓縮失敗" }
}
Write-Host "完成：$out"
