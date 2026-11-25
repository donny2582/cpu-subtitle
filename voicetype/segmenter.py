"""VAD 切句：偵測到你停頓 end_silence_ms（預設 0.5 秒）就把整句音訊交給辨識。

    等待說話 ──偵測到人聲──► 句子進行中 ──停頓 ≥ end_silence_ms──► 整句輸出
        ▲                       │ 超過 max_utterance_s 也強制輸出         │
        └───────────────────────┴───────────────────────────────────────┘

音訊以 30 ms 的「音框」為單位，每個音框問 VAD 一次「是不是人聲」。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import VadConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_TAIL_KEEP_MS = 200  # 句尾保留一點靜音，避免切到尾音


@dataclass
class Utterance:
    seq: int
    audio: np.ndarray   # 16 kHz 單聲道 float32
    started_at: float   # 相對開始時間（秒）
    ended_at: float
    forced_cut: bool = False


class VoiceDetector(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...


class EnergyVad:
    """以音量（RMS）判斷人聲；webrtcvad 不可用時的備援。"""

    def __init__(self, threshold_dbfs: float = -45.0) -> None:
        self._threshold = 10 ** (threshold_dbfs / 20)

    def is_speech(self, frame: np.ndarray) -> bool:
        if not frame.size:
            return False
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) >= self._threshold


class WebRtcVad:
    def __init__(self, aggressiveness: int) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray) -> bool:
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return bool(self._vad.is_speech(pcm, SAMPLE_RATE))


def make_vad(cfg: VadConfig) -> VoiceDetector:
    if cfg.engine == "webrtc":
        try:
            return WebRtcVad(cfg.aggressiveness)
        except ImportError:
            log.warning("找不到 webrtcvad，改用音量式 VAD")
    return EnergyVad(cfg.energy_threshold_dbfs)


class Segmenter:
    """把連續音訊切成一句一句。只能由單一執行緒呼叫。"""

    def __init__(self, cfg: VadConfig, vad: VoiceDetector) -> None:
        self._cfg = cfg
        self._vad = vad
        self._frame_len = SAMPLE_RATE * cfg.frame_ms // 1000
        self._frame_s = cfg.frame_ms / 1000
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, cfg.pre_roll_ms // cfg.frame_ms))
        self._pending = np.zeros(0, dtype=np.float32)
        self._clock = 0.0  # 已處理的音訊長度（秒）
        self._seq = 0
        self._reset_utterance()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, samples: np.ndarray) -> list[Utterance]:
        """餵入 16 kHz 單聲道 float32，回傳這次切出的完整句子。"""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not samples.size:
            return []
        buf = np.concatenate((self._pending, samples)) if self._pending.size else samples
        n = buf.size // self._frame_len
        out: list[Utterance] = []
        for i in range(n):
            out.extend(self._process(buf[i * self._frame_len:(i + 1) * self._frame_len]))
            self._clock += self._frame_s
        self._pending = buf[n * self._frame_len:].copy()
        return out

    def flush(self) -> list[Utterance]:
        """強制結束目前這句（暫停、斷流、結束程式時）。"""
        return self._finalize(forced=False) if self._in_speech else []

    def reset(self) -> None:
        self._reset_utterance()
        self._pre_roll.clear()
        self._pending = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    def _process(self, frame: np.ndarray) -> list[Utterance]:
        speech = self._vad.is_speech(frame)
        if not self._in_speech:
            if speech:
                self._in_speech = True
                self._frames = [*self._pre_roll, frame]
                self._flags = [False] * len(self._pre_roll) + [True]
                self._start = self._clock - len(self._pre_roll) * self._frame_s
                self._pre_roll.clear()
                self._speech_frames = 1
                self._silence_run = 0
            else:
                self._pre_roll.append(frame)
            return []

        self._frames.append(frame)
        self._flags.append(speech)
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
        duration = len(self._frames) * self._frame_s
        end_silence = self._cfg.end_silence_ms
        if duration >= self._cfg.soft_cut_s:
            # 已經講很久了：短短換氣就切，不必等完整的 0.5 秒停頓
            end_silence = min(end_silence, self._cfg.soft_end_silence_ms)
        if self._silence_run * self._cfg.frame_ms >= end_silence:
            return self._finalize(forced=False)
        if duration >= self._cfg.max_utterance_s:
            return self._split()
        return []

    def _split(self) -> list[Utterance]:
        """一直沒有停頓：在最近 1.5 秒（且不早於這句的一半）內音量最小的地方切開，
        前半段先送辨識，後半段留下來繼續累積，避免把一個字切成兩半。"""
        frames = self._frames
        n = len(frames)
        lo = max(n // 2, n - int(1.5 / self._frame_s))
        energy = [float(np.mean(np.square(f, dtype=np.float64))) for f in frames[lo:]]
        cut = lo + int(np.argmin(energy)) + 1
        head, tail = frames[:cut], frames[cut:]
        tail_flags = self._flags[cut:]
        start = self._start
        self._seq += 1
        audio = np.concatenate(head)
        utt = Utterance(self._seq, audio, start, start + audio.size / SAMPLE_RATE, forced_cut=True)
        self._frames = list(tail)
        self._flags = list(tail_flags)
        self._start = start + len(head) * self._frame_s
        self._speech_frames = sum(tail_flags)
        self._silence_run = 0
        for flag in reversed(tail_flags):
            if flag:
                break
            self._silence_run += 1
        return [utt]

    def _finalize(self, forced: bool) -> list[Utterance]:
        frames = self._frames
        trim = max(0, self._silence_run - _TAIL_KEEP_MS // self._cfg.frame_ms)
        if trim:
            frames = frames[:-trim]
        speech_ms = self._speech_frames * self._cfg.frame_ms
        start = self._start
        self._reset_utterance()
        if speech_ms < self._cfg.min_speech_ms or not frames:
            return []
        self._seq += 1
        audio = np.concatenate(frames)
        return [Utterance(self._seq, audio, start, start + audio.size / SAMPLE_RATE, forced)]

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._flags: list[bool] = []  # 每個音框是不是人聲（斷句時用）
        self._speech_frames = 0
        self._silence_run = 0
        self._start = 0.0
