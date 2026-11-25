"""即時字幕（本機 CPU）：講完一句、停頓約 0.5 秒，整句字幕跳出；英／日／韓翻成中文。

流程：系統聲音／麥克風 → VAD 切句 → faster-whisper（CPU INT8）→ NLLB 翻譯 → 字幕視窗 + 逐字稿
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, app_root, apply_language_policy, foreign_languages, load_config
from .segmenter import SAMPLE_RATE, Segmenter, Utterance, make_vad

log = logging.getLogger("voicetype")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="voicetype", description="本機 CPU 即時字幕（VAD + faster-whisper + NLLB 翻譯）")
    p.add_argument("--config", type=Path, help="設定檔路徑（預設 config.toml）")
    p.add_argument("--model", help="覆寫 asr.model，例如 small、large-v3-turbo")
    p.add_argument("--threads", type=int, help="覆寫 asr.cpu_threads")
    p.add_argument("--languages", help="講者可能講的語言，逗號分隔：zh,en,ja,ko（預設 zh）")
    p.add_argument("--source", choices=["mic", "loopback", "both"], help="覆寫 audio.source（mic / loopback 系統聲音 / both 兩者）")
    p.add_argument("--silence-ms", type=int, help="覆寫 vad.end_silence_ms（停頓多久算一句）")
    p.add_argument("--cli", action="store_true", help="不開視窗，在終端機執行")
    p.add_argument("--list-devices", action="store_true", help="列出音訊裝置後結束")
    p.add_argument("--benchmark", action="store_true", help="測試不同 CPU 執行緒數的速度，找出最佳 cpu_threads")
    p.add_argument("--file", type=Path, help="測試用：辨識音訊檔（走同樣的 VAD 切句流程），只印出結果")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _apply_overrides(cfg: Config, a: argparse.Namespace) -> None:
    if a.model:
        cfg.asr.model = a.model
    if a.threads:
        cfg.asr.cpu_threads = a.threads
    if a.languages:
        cfg.asr.languages = a.languages.split(",")
    if a.source:
        cfg.audio.source = a.source
    if a.silence_ms:
        cfg.vad.end_silence_ms = a.silence_ms


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else app_root() / p


def _report(r: Any, utt: Utterance | None = None) -> None:
    rtf = r.elapsed_s / max(r.audio_s, 1e-6)
    print(f"[{r.language} {r.language_prob:.2f}] {r.text}    "
          f"（音訊 {r.audio_s:.1f}s，辨識 {r.elapsed_s:.2f}s，RTF {rtf:.2f}）", flush=True)


# ---------------------------------------------------------------------- 檔案測試
def run_file(cfg: Config, path: Path) -> int:
    from faster_whisper import decode_audio

    from .asr import WhisperCpu

    asr = WhisperCpu(cfg.asr, _resolve(cfg.asr.model_dir))
    print(f"載入模型 {cfg.asr.model}（CPU {cfg.asr.compute_type}）…", flush=True)
    with _DownloadWatcher(_resolve(cfg.asr.model_dir), "語音辨識模型", _console_events, cfg.asr.model):
        asr.load()
    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    seg = Segmenter(cfg.vad, make_vad(cfg.vad))
    utts: list[Utterance] = []
    step = SAMPLE_RATE // 10
    for i in range(0, audio.size, step):
        utts += seg.push(audio[i:i + step])
    utts += seg.push(np.zeros(SAMPLE_RATE, dtype=np.float32)) + seg.flush()
    print(f"VAD 切出 {len(utts)} 句", flush=True)
    translator = None
    if foreign_languages(cfg):
        from .asr import default_threads
        from .translate import NllbTranslator

        translator = NllbTranslator(cfg.translate, _resolve(cfg.asr.model_dir),
                                    cfg.translate.cpu_threads or cfg.asr.cpu_threads or default_threads())
        with _DownloadWatcher(_resolve(cfg.asr.model_dir), "翻譯模型", _console_events, cfg.translate.model):
            translator.load()
    for u in utts:
        r = asr.transcribe(u)
        if not r:
            continue
        _report(r, u)
        if translator and r.language in foreign_languages(cfg):
            t0 = time.perf_counter()
            r.translation = translator.translate(r.text, r.language)
            r.translate_s = time.perf_counter() - t0
            _console_events("translated", r)
    return 0


def run_benchmark(cfg: Config) -> int:
    """量「一句 3 秒的話」編碼一次要多久（編碼是 CPU 上最花時間的部分），找出最快的執行緒數。"""
    import os

    from faster_whisper import WhisperModel
    from faster_whisper.audio import pad_or_trim

    n = os.cpu_count() or 4
    candidates = sorted({t for t in (1, 2, 3, 4, 6, 8, n // 2, n) if 1 <= t <= n})
    audio = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 3) * 0.05).astype(np.float32)
    print(f"模型 {cfg.asr.model}（{cfg.asr.compute_type}），邏輯處理器 {n} 個；每個設定量 3 次取最快", flush=True)
    results: dict[int, float] = {}
    for t in candidates:
        m = WhisperModel(cfg.asr.model, device="cpu", compute_type=cfg.asr.compute_type, cpu_threads=t,
                         download_root=str(_resolve(cfg.asr.model_dir)))
        frames = 3000
        if cfg.asr.dynamic_window:  # 和實際辨識一樣的視窗大小（3 秒的句子）
            frames = min(3000, max(int(cfg.asr.min_window_s * 100), 300 + int(cfg.asr.window_padding_s * 100)))
        feats = pad_or_trim(m.feature_extractor(audio), frames)
        m.encode(feats)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            m.encode(feats)
            times.append(time.perf_counter() - t0)
        results[t] = min(times)
        print(f"  cpu_threads = {t:>2}：每句至少 {results[t]:.2f} 秒", flush=True)
        del m
    best = min(results, key=results.__getitem__)
    print(f"建議：config.toml 的 [asr] cpu_threads = {best}", flush=True)
    return 0


# ---------------------------------------------------------------------- 即時模式
EventHandler = Callable[[str, Any], None]


def _console_events(kind: str, payload: Any) -> None:
    """命令列模式：把事件印在終端機。"""
    if kind == "result":
        _report(payload, None)
    elif kind == "translated":
        print(f"    → {payload.translation}（翻譯 {payload.translate_s:.2f}s）", flush=True)
    elif kind in {"status", "error"}:
        print(payload, flush=True)
    elif kind == "progress" and payload:
        print("  " + payload, flush=True)


# 常用模型的大約大小（MB），用來顯示下載百分比；不在表裡的只顯示 MB
MODEL_SIZES_MB = {
    "small": 464, "base": 140, "tiny": 75, "medium": 1470, "large-v3-turbo": 1550, "large-v3": 2950,
    "JustFrederik/nllb-200-distilled-600M-ct2-int8": 613, "OpenNMT/nllb-200-distilled-1.3B-ct2-int8": 1380,
}


class _DownloadWatcher:
    """模型第一次下載時，每秒量一次 models 資料夾長大了多少，用 progress 事件回報（不然看起來像當掉）。"""

    def __init__(self, folder: Path, label: str, emit: EventHandler, model: str = "") -> None:
        self._folder, self._label, self._emit = folder, label, emit
        self._total = MODEL_SIZES_MB.get(model, 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _size(self) -> int:
        total = 0
        for p in self._folder.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                pass
        return total

    def _run(self) -> None:
        start = self._size()
        last_t, last_b = time.monotonic(), start
        reported = False
        while not self._stop.wait(1.0):
            now_b, now_t = self._size(), time.monotonic()
            got = (now_b - start) / 1e6
            if got < 1 and not reported:
                continue  # 本機已經有模型，只是在載入
            speed = (now_b - last_b) / 1e6 / max(now_t - last_t, 1e-6)
            last_t, last_b = now_t, now_b
            if self._total:
                pct = min(99, int(got / self._total * 100))
                text = f"下載{self._label}中… {pct}%（{got:.0f} / {self._total} MB，{speed:.1f} MB/s）"
            else:
                text = f"下載{self._label}中… 已下載 {got:.0f} MB（{speed:.1f} MB/s）"
            self._emit("progress", text)
            reported = True
        if reported:
            self._emit("progress", "")

    def __enter__(self) -> _DownloadWatcher:
        self._folder.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="download-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2)


class App:
    """字幕主流程。可由命令列（run 阻塞執行）或圖形介面（在背景執行緒 run）控制。

    事件（on_event(kind, payload)）：
      status 文字狀態｜result Transcription（原文，先出）｜translated Transcription（加上中文翻譯）｜paused bool｜level 音量 dBFS｜pending 待辨識句數
      error 錯誤訊息｜stopped None
    """

    def __init__(self, cfg: Config, on_event: EventHandler | None = None, asr: Any = None,
                 translator: Any = None) -> None:
        from .asr import WhisperCpu, default_threads
        from .translate import NllbTranslator

        self.cfg = cfg
        self._emit = on_event or _console_events
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2000)
        self.utt_q: queue.Queue[Utterance | None] = queue.Queue(maxsize=20)
        self.mt_q: queue.Queue[Any] = queue.Queue()
        self.asr = asr or WhisperCpu(cfg.asr, _resolve(cfg.asr.model_dir))
        apply_language_policy(cfg)
        self.translator = None
        if foreign_languages(cfg):
            threads = cfg.translate.cpu_threads or cfg.asr.cpu_threads or default_threads()
            self.translator = translator or NllbTranslator(cfg.translate, _resolve(cfg.asr.model_dir), threads)
        self.output: Any = None
        self.paused = threading.Event()
        self.stopping = threading.Event()

    def toggle(self) -> None:
        if self.paused.is_set():
            self.paused.clear()
            self._emit("status", "▶ 繼續")
        else:
            self.paused.set()
            self._emit("status", "⏸ 暫停（再按一次繼續）")
        self._emit("paused", self.paused.is_set())

    def quit(self) -> None:
        self.stopping.set()

    def _asr_worker(self) -> None:
        while True:
            utt = self.utt_q.get()
            if utt is None:
                return
            if self.stopping.is_set():
                continue  # 已按停止：佇列裡剩下的句子不用再辨識，停止比較快
            try:
                r = self.asr.transcribe(utt)
            except Exception as exc:  # noqa: BLE001
                log.exception("辨識失敗")
                self._emit("error", f"辨識失敗：{exc}")
                continue
            finally:
                self._emit("pending", self.utt_q.qsize())
            if not r or self.stopping.is_set():
                continue
            self._emit("result", r)  # 原文先顯示，翻譯好再補上中文
            # 每一句（不論要不要翻譯）都交給同一個執行緒：翻譯不會擋住下一句的辨識，
            # 逐字稿也只有這一個執行緒在寫，順序和講話順序一致、不會兩個執行緒同時寫檔
            self.mt_q.put(r)

    def _mt_worker(self) -> None:
        while True:
            r = self.mt_q.get()
            if r is None:
                return
            if self.translator and r.language in foreign_languages(self.cfg) and not self.stopping.is_set():
                t0 = time.perf_counter()
                try:
                    r.translation = self.translator.translate(r.text, r.language)
                    r.translate_s = time.perf_counter() - t0
                except Exception as exc:  # noqa: BLE001
                    log.exception("翻譯失敗")
                    self._emit("error", f"翻譯失敗：{exc}")
                if r.translation:
                    self._emit("translated", r)
            self._output(r)  # 停止時也把已辨識的句子寫進逐字稿，不會漏

    def _output(self, r: Any) -> None:
        try:
            self.output.emit(r.text, r.language, r.translation)
        except Exception as exc:  # noqa: BLE001
            log.exception("輸出失敗")
            self._emit("error", f"輸出失敗：{exc}")

    def run(self) -> int:
        try:
            return self._run()
        except Exception as exc:  # noqa: BLE001
            log.exception("執行失敗")
            self._emit("error", f"執行失敗：{exc}")
            return 1
        finally:
            self._emit("stopped", None)

    def _run(self) -> int:
        from .audio import AudioCapture
        from .hotkey import HotkeyListener
        from .output import Output

        c = self.cfg
        models_dir = _resolve(c.asr.model_dir)
        if not self.asr.loaded:
            self._emit("status", f"載入語音辨識模型 {c.asr.model}（第一次使用會自動下載，之後就不用）…")
            with _DownloadWatcher(models_dir, "語音辨識模型", self._emit, c.asr.model):
                self.asr.load()
        if self.translator and not self.translator.loaded:
            self._emit("status", "載入翻譯模型（第一次使用會自動下載約 600 MB，之後就不用）…")
            with _DownloadWatcher(models_dir, "翻譯模型", self._emit, c.translate.model):
                self.translator.load()
        if self.stopping.is_set():
            return 0  # 下載／載入模型期間按了停止：模型已存好，這次就不開始收音
        capture = AudioCapture(c.audio, self.audio_q, warn=lambda m: self._emit("error", m),
                               notify=lambda m: self._emit("status", m))
        capture.start()  # 先確定收得到聲音，失敗就不會留下空的逐字稿檔案
        tdir = _resolve(c.output.transcript_dir) if c.output.transcript_dir else None
        try:
            self.output = Output(tdir)
        except Exception:
            capture.stop()
            raise
        hotkeys = HotkeyListener({c.hotkey.toggle: self.toggle, c.hotkey.quit: self.quit})
        hotkeys.start()
        worker = threading.Thread(target=self._asr_worker, name="asr", daemon=True)
        worker.start()
        mt_worker = threading.Thread(target=self._mt_worker, name="translate", daemon=True)
        mt_worker.start()

        lang = c.asr.language if c.asr.language != "auto" else f"auto（限 {'/'.join(c.asr.allowed_languages)}）"
        lines = [f"🎙 開始收音：{capture.device_name}",
                 f"   模型 {c.asr.model}｜語言 {lang}｜停頓 {c.vad.end_silence_ms} ms 出字"
                 f"｜翻譯 {'開' if self.translator else '關'}",
                 f"   快捷鍵：{c.hotkey.toggle} 暫停/繼續、{c.hotkey.quit} 停止"]
        missing = [k for k in (c.hotkey.toggle, c.hotkey.quit) if k not in hotkeys.registered]
        if missing:
            lines.append(f"   ⚠ 這些快捷鍵被其他程式占用，請換一組：{', '.join(missing)}")
        if self.output.transcript_path:
            lines.append(f"   逐字稿：{self.output.transcript_path}")
        recorder, wav_path = self._open_recorder()
        if wav_path is not None:
            lines.append(f"   錄音：{wav_path}")
        if c.audio.source == "both":
            lines.append("   提醒：兩者混音時請戴耳機，否則喇叭的聲音會被麥克風再收一次")
        self._emit("status", "\n".join(lines))

        seg = Segmenter(c.vad, make_vad(c.vad))
        last_audio = last_level = time.monotonic()
        peak = 0.0
        try:
            while not self.stopping.is_set():
                try:
                    chunk = self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    chunk = None
                now = time.monotonic()
                if chunk is not None and chunk.size:
                    peak = max(peak, float(np.sqrt(np.mean(np.square(chunk)))))
                if now - last_level >= 0.1:  # 音量表每 0.1 秒更新一次
                    self._emit("level", 20 * np.log10(peak + 1e-9))
                    peak, last_level = 0.0, now
                if self.paused.is_set():
                    seg.reset()  # 暫停：丟掉講到一半的句子（已講完的句子照常輸入）
                    continue
                if chunk is None:
                    # loopback 在沒聲音時不送資料：斷流太久就把進行中的句子結束掉
                    if seg.in_speech and now - last_audio >= c.vad.idle_flush_s:
                        self._submit(seg.flush())
                    continue
                last_audio = now
                if recorder is not None:
                    recorder.writeframes((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
                self._submit(seg.push(chunk))
        except KeyboardInterrupt:
            pass
        finally:
            self._emit("status", "停止中…")
            capture.stop()
            hotkeys.stop()
            self.utt_q.put(None)
            worker.join(5)
            self.mt_q.put(None)
            mt_worker.join(5)
            self.output.close()
            if recorder is not None:
                recorder.close()
        self._emit("status", "■ 已停止")
        return 0

    def _open_recorder(self) -> tuple[Any, Path | None]:
        """output.save_audio = true 時，把執行期間的聲音存成 WAV（暫停期間不錄）。"""
        import wave
        from datetime import datetime

        if not self.cfg.output.save_audio:
            return None, None
        d = _resolve(self.cfg.output.transcript_dir or "transcripts")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{datetime.now():%Y%m%d_%H%M%S}.wav"
        w = wave.open(str(path), "wb")  # noqa: SIM115
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        return w, path

    def _submit(self, utts: list[Utterance]) -> None:
        for u in utts:
            try:
                self.utt_q.put_nowait(u)
            except queue.Full:
                self._emit("error", "辨識跟不上，丟棄一句（可改用較小模型或 beam_size = 1）")
        if utts:
            self._emit("pending", self.utt_q.qsize())


def main(argv: list[str] | None = None) -> int:
    # 模型下載改用一般 HTTP（邊下載邊寫入 models，才看得到「已下載 xxx MB」的進度）；
    # 預設的 hf_xet 會在最後才一次寫入檔案，下載過程中看起來像當掉
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if sys.stdout is None or sys.stderr is None:
        # pythonw（沒有主控台）啟動：輸出改寫到 logs/voicetype.log，避免下載模型的進度條找不到輸出而出錯
        log_dir = app_root() / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "voicetype.log"
        try:
            if log_file.exists() and log_file.stat().st_size > 5_000_000:  # 超過 5 MB 換新檔，只留前一份
                log_file.replace(log_dir / "voicetype.old.log")
        except OSError:
            pass
        sink = open(log_file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = sys.stdout or sink
        sys.stderr = sys.stderr or sink
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if args.list_devices:
        from .audio import list_devices

        list_devices()
        return 0
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    apply_language_policy(cfg)
    # 命令列模式的錯誤在這裡接住：打包成視窗程式（沒有主控台）時，沒接住的例外會跳出
    # PyInstaller 的錯誤對話框等人按確定，程式就一直停在那裡
    try:
        if args.benchmark:
            return run_benchmark(cfg)
        if args.file:
            return run_file(cfg, args.file)
    except Exception as exc:  # noqa: BLE001
        log.debug("詳細錯誤", exc_info=True)
        print(f"錯誤：{exc}", file=sys.stderr, flush=True)
        return 1
    if args.cli:
        return App(cfg).run()
    from .gui import run_gui

    return run_gui(cfg, args.config)


if __name__ == "__main__":
    sys.exit(main())
