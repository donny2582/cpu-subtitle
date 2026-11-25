"""逐字稿：每句一行（原文），有翻譯時下一行是中文。每次開始字幕一個檔案。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class Output:
    def __init__(self, transcript_dir: Path | None) -> None:
        self._file = None
        self.transcript_path: Path | None = None
        if transcript_dir:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            self.transcript_path = transcript_dir / f"{datetime.now():%Y%m%d_%H%M%S}.txt"
            self._file = self.transcript_path.open("a", encoding="utf-8")

    def emit(self, text: str, language: str, translation: str = "") -> None:
        if not self._file:
            return
        line = f"[{datetime.now():%H:%M:%S}] [{language}] {text}\n"
        if translation:
            line += f"           [中] {translation}\n"
        self._file.write(line)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
