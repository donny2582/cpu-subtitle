"""設定檔讀取（config.toml）。沒寫到的項目使用這裡的預設值。"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    source: str = "loopback"       # mic = 麥克風；loopback = 系統聲音（會議、影片）；both = 兩者混音
    mic_device_index: int = -1     # -1 = 系統預設麥克風；用 --list-devices 查詢
    loopback_device_index: int = -1  # -1 = 系統預設喇叭／耳機


@dataclass
class VadConfig:
    engine: str = "webrtc"         # webrtc / energy
    aggressiveness: int = 2        # 0~3，環境越吵調越高
    energy_threshold_dbfs: float = -45.0
    frame_ms: int = 30             # webrtcvad 只接受 10/20/30
    end_silence_ms: int = 500      # 停頓多久算「一句講完」
    min_speech_ms: int = 250       # 人聲短於此（咳嗽、敲鍵盤）直接丟掉
    pre_roll_ms: int = 300         # 句首往前多保留的音訊，避免吃掉第一個字
    # 講者一直講不停頓時主動斷句，不要一直累積成一大句：
    soft_cut_s: float = 4.0        # 一句超過這麼長之後，短短換氣（soft_end_silence_ms）就切
    soft_end_silence_ms: int = 200
    max_utterance_s: float = 8.0   # 還是沒停頓就在最近 1.5 秒內最安靜的地方硬切（最多 28）
    idle_flush_s: float = 0.6      # loopback 斷流時，多久沒資料就當作句子結束


@dataclass
class AsrConfig:
    model: str = "small"           # small / medium / large-v3-turbo / large-v3 ...
    compute_type: str = "int8"     # CPU 建議 int8（量化，速度約 float32 的 2~4 倍）
    cpu_threads: int = 0           # 0 = 自動（4；大小核筆電用 --benchmark 找最佳值）
    beam_size: int = 1             # 1 = 最快；5 = 略準但長句慢 30~50%
    dynamic_window: bool = True    # 只編碼實際長度（快 3~6 倍）；false = Whisper 原本的固定 30 秒
    window_padding_s: float = 2.0  # 動態視窗在句尾多補幾秒靜音
    min_window_s: float = 5.0      # 動態視窗最短幾秒（太短的話短句容易重複輸出）
    # 講者可能會講的語言：zh / en / ja / ko。只勾一種 = 固定該語言；勾多種 = 每句自動判斷。
    # 中文（含中英夾雜）直接顯示；英／日／韓翻成中文並附原文。
    languages: list[str] = field(default_factory=lambda: ["zh"])
    mixed_zh_prob: float = 0.15    # 判成英文但中文機率 ≥ 這個值 → 當作中英夾雜，用中文辨識（直接顯示）
    language: str = "zh"           # （由 languages 推算，不用手動設定）
    allowed_languages: list[str] = field(default_factory=lambda: ["zh"])  # （同上）
    sticky_language_prob: float = 0.5  # 短句信心低於此值時沿用上一句語言
    initial_prompt: str = ""       # 給模型的提示詞（專有名詞、標點風格）
    zh_convert: str = "s2twp"      # 中文結果轉台灣繁體（OpenCC 設定名）；空字串 = 不轉
    model_dir: str = "models"
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4


@dataclass
class TranslateConfig:
    model: str = "JustFrederik/nllb-200-distilled-600M-ct2-int8"  # 或 OpenNMT/nllb-200-distilled-1.3B-ct2-int8（較準較慢）
    target: str = "zh-tw"          # zh-tw = 台灣繁體用語；zh-hant = 繁體；zh-cn = 簡體
    beam_size: int = 2
    cpu_threads: int = 0           # 0 = 和語音辨識相同


@dataclass
class OutputConfig:
    transcript_dir: str = "transcripts"  # 空字串 = 不存逐字稿
    save_audio: bool = False       # 同時把擷取到的聲音存成 WAV（16 kHz 單聲道，約每小時 115 MB）


@dataclass
class OverlayConfig:
    enabled: bool = True           # 執行時顯示大字幕視窗（只有控制視窗模式有）
    font_family: str = "Microsoft JhengHei UI"
    font_size: int = 32
    bold: bool = True
    text_color: str = "#FFFFFF"    # 預設灰底白字；白底黑框可改 text "#000000"、bg "#FFFFFF"、border "#000000"
    bg_color: str = "#3C3C3C"
    border_color: str = "#3C3C3C"
    border_width: int = 0          # 外框粗細（像素），0 = 無框
    opacity: float = 0.70          # 整個字幕視窗的不透明度 0.3~1.0（越小越透明）
    corner_radius: int = 18        # 圓角半徑（像素），0 = 直角
    max_lines: int = 2             # 同時顯示幾句
    width_ratio: float = 0.8       # 字幕寬度佔螢幕寬度比例
    bottom_margin: int = 80        # 距離螢幕底部（像素）
    clear_after_s: float = 10.0    # 幾秒沒有新句子就清空；0 = 不清空
    show_language: bool = False    # 每句前面加 [英]/[日]/[韓]/[中]
    show_source: bool = True       # 翻譯時在中文下面用小字顯示原文
    source_font_ratio: float = 0.55  # 原文字級 = 中文字級 × 這個比例
    click_through: bool = False    # 滑鼠點擊穿透字幕（不能拖曳）


@dataclass
class HotkeyConfig:
    toggle: str = "ctrl+alt+f9"  # 暫停／繼續字幕
    quit: str = "ctrl+alt+q"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)


def install_dir() -> Path:
    """程式所在資料夾（打包成 exe 時為 exe 所在資料夾；附帶的 config.example.toml 在這裡）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


_DATA_ROOT: Path | None = None
# 模型快取裡的檔案路徑本身就長約 150 字元（models--作者--模型名\blobs\64 位雜湊.incomplete），
# Windows 預設路徑上限 260 字元，所以程式資料夾路徑超過這個長度就改用路徑短的 %LOCALAPPDATA%
MAX_DATA_DIR_LEN = 90


def _usable_data_dir(root: Path) -> bool:
    """資料夾可寫入、而且路徑夠短（不會讓模型檔超過 Windows 260 字元上限）。"""
    if len(str(root)) > MAX_DATA_DIR_LEN:
        return False
    try:
        probe = root / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def app_root() -> Path:
    """資料資料夾：設定、模型、逐字稿、紀錄都放這裡。

    預設就是程式所在資料夾（整個資料夾帶著走就好）；但如果程式放在 C:\\Program Files
    這類沒有寫入權限的位置，改用 %LOCALAPPDATA%\\LiveSubtitle。
    """
    global _DATA_ROOT
    if _DATA_ROOT is None:
        root = install_dir()
        if not _usable_data_dir(root):
            root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LiveSubtitle"
            root.mkdir(parents=True, exist_ok=True)
        _DATA_ROOT = root
    return _DATA_ROOT


CONFIG_WARNINGS: list[str] = []  # 載入設定時發現的問題（控制視窗啟動時會顯示給使用者）


def _coerce(current: Any, value: Any) -> Any:
    """把設定檔的值轉成和預設值相同的型別；轉不了就丟出 TypeError。"""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(current, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif isinstance(current, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(current, str):
        if isinstance(value, str):
            return value
    elif isinstance(current, list):
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return list(value)
    else:
        return value
    raise TypeError(type(current).__name__)


_TYPE_NAMES = {"bool": "true 或 false", "int": "整數", "float": "數字", "str": "文字（要加引號）", "list": "清單"}


def _apply(obj: Any, data: dict[str, Any], path: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            log.warning("設定檔有未知的項目，已忽略：%s%s", path, key)
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                CONFIG_WARNINGS.append(f"{path}{key} 應該是一個區段，已改用預設值")
                continue
            _apply(current, value, f"{path}{key}.")
            continue
        try:
            setattr(obj, key, _coerce(current, value))
        except TypeError as exc:
            msg = f"{path}{key} = {value!r} 格式不對（應該是{_TYPE_NAMES.get(str(exc), str(exc))}），已改用預設值 {current!r}"
            log.warning(msg)
            CONFIG_WARNINGS.append(msg)


def load_config(path: Path | None = None) -> Config:
    """讀取 config.toml；不存在時從 config.example.toml 複製一份。"""
    root = app_root()
    path = path or root / "config.toml"
    if not path.exists():
        example = install_dir() / "config.example.toml"
        if example.exists():
            shutil.copyfile(example, path)
    cfg = Config()
    CONFIG_WARNINGS.clear()
    if path.exists():
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            # 設定檔寫壞了：不要讓程式打不開，這次先用預設值（檔案保持原樣，讓使用者自己修或刪掉）
            msg = f"設定檔 {path.name} 格式有誤，這次先使用預設設定。\n錯誤位置：{exc}\n可以修正它，或刪掉它讓程式重新建立。"
            log.warning(msg)
            CONFIG_WARNINGS.append(msg)
            data = {}
        _apply(cfg, data, "")
    try:
        validate(cfg)
    except ValueError as exc:
        CONFIG_WARNINGS.append(f"{exc}，這次先使用預設設定")
        cfg = Config()
        validate(cfg)
    return cfg


def foreign_languages(cfg: Config) -> list[str]:
    """講者語言中需要翻譯成中文的（中文以外的）。"""
    return [code for code in cfg.asr.languages if code != "zh"]


def apply_language_policy(cfg: Config) -> None:
    """依「講者可能講的語言」決定辨識方式：
    只勾一種 → 固定用那個語言辨識（最快、最準）；勾多種 → 每句只在勾選的語言之間自動判斷。
    都沒勾 → 當作中文。
    """
    langs: list[str] = []
    for code in cfg.asr.languages:
        code = code.strip().lower()
        if code and code not in langs:
            langs.append(code)
    langs = langs or ["zh"]
    cfg.asr.languages = langs
    cfg.asr.allowed_languages = list(langs)
    cfg.asr.language = langs[0] if len(langs) == 1 else "auto"


def validate(cfg: Config) -> None:
    if cfg.audio.source not in {"mic", "loopback", "both"}:
        raise ValueError("audio.source 只能是 mic、loopback 或 both")
    if cfg.vad.frame_ms not in {10, 20, 30}:
        raise ValueError("vad.frame_ms 只能是 10、20、30")
    cfg.vad.max_utterance_s = min(float(cfg.vad.max_utterance_s), 28.0)
    apply_language_policy(cfg)
    cfg.asr.allowed_languages = [s.strip().lower() for s in cfg.asr.allowed_languages if s.strip()]


# ---------------------------------------------------------------------- 存檔
def _toml_value(v: Any) -> str:
    import json

    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"不支援的設定值型別：{type(v)}")


def _comment_start(text: str) -> int:
    """回傳字串外第一個 # 的位置；沒有註解時回傳字串長度。"""
    quote = ""
    escaped = False
    for i, ch in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return i
    return len(text)


def save_config(cfg: Config, path: Path | None = None) -> Path:
    """把設定寫回 config.toml：只替換數值，保留原本的註解與排列；檔案沒有的項目加在該區段最後。"""
    import re

    path = path or app_root() / "config.toml"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    wanted: dict[str, dict[str, Any]] = {
        f.name: {g.name: getattr(getattr(cfg, f.name), g.name) for g in fields(getattr(cfg, f.name))}
        for f in fields(cfg)
    }
    # 由 [translate] languages 推算出來的值，不寫進檔案
    done: set[tuple[str, str]] = {("asr", "language"), ("asr", "allowed_languages")}  # 由 asr.languages 推算
    section = ""
    section_end: dict[str, int] = {}
    key_re = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*)$")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]").strip()
            section_end[section] = i + 1
            continue
        m = key_re.match(line)
        if m and section in wanted and m.group(2) in wanted[section]:
            key = m.group(2)
            rest = m.group(4)
            cut = _comment_start(rest)
            value_part, comment = rest[:cut], rest[cut:]
            new_val = _toml_value(wanted[section][key])
            if comment:
                width = max(len(value_part.rstrip()), len(new_val))
                lines[i] = f"{m.group(1)}{key}{m.group(3)}{new_val.ljust(width)} {comment.lstrip()}"
            else:
                lines[i] = f"{m.group(1)}{key}{m.group(3)}{new_val}"
            done.add((section, key))
        if section:
            section_end[section] = i + 1
    inserts: dict[int, list[str]] = {}
    tail: list[str] = []
    for sec, values in wanted.items():
        missing = [f"{k} = {_toml_value(v)}" for k, v in values.items() if (sec, k) not in done]
        if not missing:
            continue
        if sec in section_end:
            inserts.setdefault(section_end[sec], []).extend(missing)
        else:
            tail += ["", f"[{sec}]", *missing]
    for idx in sorted(inserts, reverse=True):
        lines[idx:idx] = inserts[idx]
    path.write_text("\n".join(lines + tail) + "\n", encoding="utf-8")
    return path
