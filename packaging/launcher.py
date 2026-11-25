"""PyInstaller 的進入點（打包後的 LiveSubtitle.exe 從這裡開始）。"""

import sys

from voicetype.app import main

if __name__ == "__main__":
    sys.exit(main())
