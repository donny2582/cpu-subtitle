# PyInstaller 設定：打包成「資料夾版」應用程式（dist\LiveSubtitle\LiveSubtitle.exe）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
#
# 為什麼用資料夾版而不是單一 exe：單一 exe 每次啟動都要先把幾百 MB 解壓到暫存資料夾，
# 啟動慢、也容易被防毒軟體誤判。資料夾版整個壓縮後就能帶到別台電腦。

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = SPECPATH + "\\.."

datas = []
datas += collect_data_files("opencc")          # 簡繁轉換的字典與設定檔
datas += collect_data_files("faster_whisper")  # 模型設定用到的 assets

binaries = []
binaries += collect_dynamic_libs("ctranslate2")  # 推論引擎 DLL（含 Intel OpenMP）

a = Analysis(
    [ROOT + "\\packaging\\launcher.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "voicetype.gui", "voicetype.overlay", "voicetype.audio", "voicetype.asr",
        "voicetype.translate", "voicetype.output", "voicetype.hotkey",
        "pyaudiowpatch", "webrtcvad", "soxr",
    ],
    excludes=[
        "onnxruntime",   # 只有 faster-whisper 內建 VAD 會用到，我們用自己的 VAD
        "torch", "tensorflow", "matplotlib", "pytest", "IPython",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiveSubtitle",
    console=False,          # 不顯示黑色主控台；錯誤訊息寫在 logs\voicetype.log
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="LiveSubtitle",
    upx=False,
)
