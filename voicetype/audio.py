"""音訊擷取（PyAudioWPatch）：麥克風、系統播放聲音（WASAPI loopback）或兩者混音，轉成 16 kHz 單聲道 float32。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from .config import AudioConfig
from .segmenter import SAMPLE_RATE

log = logging.getLogger(__name__)

WATCH_INTERVAL_S = 1.0  # 每隔多久檢查一次裝置狀態
MIC_STALL_S = 3.0       # 麥克風超過幾秒沒送資料就視為斷線
RETRY_S = 2.0           # 找不到裝置時多久重試一次


def to_mono_float(data: bytes, channels: int) -> np.ndarray:
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm[: pcm.size - pcm.size % channels].reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32, copy=False)


class Resampler:
    """串流式重新取樣（soxr）；沒有 soxr 時用線性內插。"""

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE) -> None:
        self._in, self._out = in_rate, out_rate
        self._stream: Any = None
        if in_rate != out_rate:
            try:
                import soxr

                self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32", quality="HQ")
            except ImportError:
                log.warning("找不到 soxr，改用線性內插")

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._in == self._out or not x.size:
            return x
        if self._stream is not None:
            return np.asarray(self._stream.resample_chunk(x), dtype=np.float32)
        n = int(round(x.size * self._out / self._in))
        return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def enumerate_devices() -> list[tuple[int, str, str]]:
    """回傳 WASAPI 輸入裝置清單：(編號, "mic" 或 "loopback", 名稱)。"""
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        out = []
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d["hostApi"] != wasapi or d["maxInputChannels"] <= 0:
                continue
            out.append((i, "loopback" if d.get("isLoopbackDevice") else "mic", str(d["name"])))
        return out
    finally:
        pa.terminate()


def list_devices() -> None:
    print("WASAPI 裝置（device_index 填左邊的編號）：")
    for i, kind, name in enumerate_devices():
        print(f"  [{i:>3}] {kind:<8} {name}")


class Mixer:
    """把系統聲音混進麥克風（source = "both"）。

    麥克風是主時鐘：每收到一塊麥克風音訊，就取出等長的系統聲音相加。
    WASAPI loopback 在電腦沒播放聲音時完全不送資料，所以不能等它，不足的部分當作靜音；
    系統聲音最多緩衝 max_lag_s，超過就丟掉最舊的，避免延遲越積越多。
    """

    def __init__(self, max_lag_s: float = 0.5) -> None:
        self._max = int(SAMPLE_RATE * max_lag_s)
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def push_secondary(self, x: np.ndarray) -> None:
        if not x.size:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf, x])[-self._max:]

    def mix_primary(self, x: np.ndarray) -> np.ndarray:
        with self._lock:
            n = min(x.size, self._buf.size)
            other, self._buf = self._buf[:n], self._buf[n:]
        out = x.astype(np.float32, copy=True)
        out[:n] += other
        np.clip(out, -1.0, 1.0, out=out)
        return out


class AudioCapture:
    """背景擷取音訊（麥克風、系統聲音或兩者混音），16 kHz 單聲道 float32 放入佇列。"""

    def __init__(self, cfg: AudioConfig, out: queue.Queue[np.ndarray],
                 warn: Any = None, notify: Any = None) -> None:
        self._cfg = cfg
        self._q = out
        self._warn = warn or (lambda msg: log.warning(msg))      # 問題提示
        self._notify = notify or (lambda msg: log.info(msg))     # 一般通知（例如已重新連接）
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._last_data: dict[str, float] = {}  # 每種來源最後一次收到資料的時間
        self._streams: list[Any] = []
        self.device_name = ""
        self.reconnects = 0
        self.error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self.error:
            raise self.error

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3)

    def _put(self, chunk: np.ndarray) -> None:
        if chunk.size:
            try:
                self._q.put_nowait(chunk)
            except queue.Full:
                pass  # 辨識跟不上時丟掉，避免延遲越積越多

    def _device(self, pa: Any, kind: str, index: int) -> dict[str, Any]:
        """依編號找裝置；編號失效（換過耳機、插拔 USB 麥克風後編號會變）或種類不對時改用系統預設。"""
        if index >= 0:
            try:
                dev = dict(pa.get_device_info_by_index(index))
                is_loop = bool(dev.get("isLoopbackDevice"))
                if dev.get("maxInputChannels", 0) > 0 and is_loop == (kind == "loopback"):
                    return dev
                reason = "不是" + ("系統聲音" if kind == "loopback" else "麥克風") + "裝置"
            except OSError:
                reason = "已不存在"
            name = "系統聲音" if kind == "loopback" else "麥克風"
            self._warn(f"設定的{name}裝置（編號 {index}）{reason}，改用系統預設；可在設定裡按「重新整理」重新選擇")
        if kind == "loopback":
            return dict(pa.get_default_wasapi_loopback())
        return dict(pa.get_default_wasapi_device(d_in=True))

    def _open(self, pa: Any, pyaudio: Any, dev: dict[str, Any], sink: Any, kind: str) -> Any:
        channels = max(1, min(2, int(dev["maxInputChannels"])))
        rate = int(dev["defaultSampleRate"])
        resampler = Resampler(rate)

        def callback(data: bytes, _frames: int, _t: Any, _s: Any) -> tuple[None, int]:
            self._last_data[kind] = time.monotonic()
            sink(resampler.process(to_mono_float(data, channels)))
            return None, pyaudio.paContinue

        self._last_data[kind] = time.monotonic()
        stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                         input_device_index=int(dev["index"]), frames_per_buffer=int(rate * 0.03),
                         stream_callback=callback)
        stream.start_stream()
        return stream

    # ------------------------------------------------------------------ 連線／自動重新連線
    def _connect(self, pyaudio: Any) -> tuple[Any, list[Any]]:
        """開一個新的 PortAudio（重新讀取裝置清單）並開啟需要的串流。"""
        pa = pyaudio.PyAudio()
        streams: list[Any] = []
        src = self._cfg.source
        try:
            if src == "both":
                mixer = Mixer()
                loop = self._device(pa, "loopback", self._cfg.loopback_device_index)
                mic = self._device(pa, "mic", self._cfg.mic_device_index)
                streams.append(self._open(pa, pyaudio, loop, mixer.push_secondary, "loopback"))
                streams.append(self._open(pa, pyaudio, mic, lambda x: self._put(mixer.mix_primary(x)), "mic"))
                self.device_name = f"{mic['name']} ＋ {loop['name']}"
            else:
                index = self._cfg.loopback_device_index if src == "loopback" else self._cfg.mic_device_index
                dev = self._device(pa, src, index)
                streams.append(self._open(pa, pyaudio, dev, self._put, src))
                self.device_name = str(dev["name"])
        except Exception:
            self._close(pa, streams)
            raise
        self._streams = streams  # 目前的串流（測試與除錯用）
        return pa, streams

    @staticmethod
    def _close(pa: Any, streams: list[Any]) -> None:
        for s in streams:
            try:
                s.stop_stream()
            except Exception:  # noqa: BLE001 - 裝置已經拔掉時關閉也可能出錯，忽略
                pass
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            pa.terminate()
        except Exception:  # noqa: BLE001
            pass

    def _watched_defaults(self) -> dict[str, str | None]:
        """使用「系統預設」的來源，記下目前 Windows 預設裝置的 ID（之後比對有沒有換掉）。"""
        out: dict[str, str | None] = {}
        try:
            from .devices_win import E_CAPTURE, E_RENDER, default_device_id

            src = self._cfg.source
            if src in {"loopback", "both"} and self._cfg.loopback_device_index < 0:
                out["loopback"] = default_device_id(E_RENDER)
            if src in {"mic", "both"} and self._cfg.mic_device_index < 0:
                out["mic"] = default_device_id(E_CAPTURE)
        except Exception as exc:  # noqa: BLE001 - 查不到就不偵測「換預設裝置」，其他功能照常
            log.debug("無法讀取 Windows 預設裝置：%s", exc)
        return out

    def _problem(self, streams: list[Any], defaults: dict[str, str | None]) -> str | None:
        """檢查目前的連線；有問題時回傳原因。"""
        for s in streams:
            try:
                if not s.is_active():
                    return "音訊裝置中斷"
            except Exception:  # noqa: BLE001
                return "音訊裝置中斷"
        # 麥克風就算沒人講話也會一直送資料（靜音），超過 MIC_STALL_S 秒沒資料就是斷了；
        # 系統聲音在沒播放時本來就可能不送資料，所以不用這個方法判斷
        if "mic" in self._last_data and time.monotonic() - self._last_data["mic"] > MIC_STALL_S:
            return "麥克風沒有聲音資料"
        if defaults and self._watched_defaults() != defaults:
            return "Windows 預設裝置已變更"
        return None

    def _run(self) -> None:
        import pyaudiowpatch as pyaudio

        try:
            pa, streams = self._connect(pyaudio)
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self._ready.set()
            return
        self._ready.set()
        defaults = self._watched_defaults()
        try:
            while not self._stop.wait(WATCH_INTERVAL_S):
                reason = self._problem(streams, defaults)
                if not reason:
                    continue
                log.warning("%s，重新連接音訊裝置", reason)
                self._close(pa, streams)
                pa, streams = None, []
                warned = False
                while not self._stop.is_set():
                    try:
                        self._last_data.clear()
                        pa, streams = self._connect(pyaudio)
                        break
                    except Exception as exc:  # noqa: BLE001
                        if not warned:
                            self._warn(f"{reason}，目前找不到可用的音訊裝置（{exc}），每 {RETRY_S:.0f} 秒重試一次…")
                            warned = True
                        self._stop.wait(RETRY_S)
                if streams:
                    defaults = self._watched_defaults()
                    self.reconnects += 1
                    self._notify(f"🔄 {reason}，已自動重新連接：{self.device_name}")
        finally:
            if pa is not None:
                self._close(pa, streams)
