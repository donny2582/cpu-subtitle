"""本機 CPU 語音辨識：faster-whisper（CTranslate2）+ INT8 量化。

為什麼不直接呼叫 WhisperModel.transcribe()？
    Whisper 的編碼器（encoder）每次都處理補滿到 30 秒的音訊，這是 CPU 上最花時間的部分。
    transcribe(language=None) 會「先為了偵測語言跑一次編碼器、再為了辨識跑一次」，CPU 時間加倍。
    這裡改成：編碼一次 → 用同一份編碼結果偵測語言 → 同一份結果直接解碼成文字，
    和 faster-whisper 的 multilingual=True 做法相同（以編碼結果逐段偵測），但每一句都偵測、
    而且只在 allowed_languages 裡選，所以「這句英文、下一句日文」的會議可以直接處理。

動態視窗（dynamic_window，預設開啟）：
    Whisper 原本不管句子多短都把音訊補滿 30 秒再編碼。CTranslate2 其實接受較短的輸入，
    所以只編碼「實際長度 + window_padding_s」（至少 min_window_s），4 秒的句子編碼量約剩 1/5，
    i7-1255U 筆電上從每句 5 秒降到 1 秒以內。等同 whisper.cpp 的 -ac（audio_ctx）。
    補白太少時短句容易重複（「好的好的好的」），所以保留至少 5 秒的視窗。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AsrConfig
from .filters import collapse_repeats, is_hallucination
from .segmenter import SAMPLE_RATE, Utterance

log = logging.getLogger(__name__)

MAX_AUDIO_S = 29.5  # Whisper 單次最多 30 秒


@dataclass
class Transcription:
    seq: int
    text: str
    language: str
    language_prob: float
    audio_s: float
    elapsed_s: float
    translation: str = ""
    translate_s: float = 0.0


_REPEATED_RE = re.compile(r"(.{4,}?)\1+")


def looks_repeated(text: str) -> bool:
    """同一段文字（4 個字以上）連續出現、而且占了大半句 → 模型卡在重複輸出。"""
    t = re.sub(r"[\s\W_]+", "", text)
    if len(t) < 8:
        return False
    m = _REPEATED_RE.search(t)
    return bool(m and len(m.group(0)) >= 0.6 * len(t))


def default_threads() -> int:
    """Intel 12 代以後的筆電是「大小核」，用太多執行緒會被小核（E-core）拖慢，所以保守一點。"""
    n = os.cpu_count() or 4
    return 4 if n >= 8 else max(1, n // 2)


class _ChineseConverter:
    """Whisper 的中文常輸出簡體與半形標點，轉成台灣繁體用語與全形標點。"""

    _PUNCT = str.maketrans({",": "，", "?": "？", "!": "！", ":": "：", ";": "；"})

    def __init__(self, profile: str) -> None:
        self._cc: Any = None
        if not profile:
            return
        try:
            from opencc import OpenCC

            self._cc = OpenCC(profile)
        except ImportError:
            log.warning("找不到 opencc，中文不做繁體轉換")

    def __call__(self, text: str) -> str:
        if self._cc:
            text = self._cc.convert(text)
        return re.sub(r"(?<=[㐀-鿿])\s*[,?!:;]", lambda m: m.group(0).strip().translate(self._PUNCT),
                      text)


class WhisperCpu:
    def __init__(self, cfg: AsrConfig, model_root: Path) -> None:
        self._cfg = cfg
        self._root = model_root
        self._model: Any = None
        self._last_lang: str | None = None
        self._zh = _ChineseConverter(cfg.zh_convert)
        self.threads = cfg.cpu_threads or default_threads()

    def can_reuse(self, cfg: AsrConfig) -> bool:
        """模型、精度、執行緒數相同時，可以沿用已載入的模型（不用重新載入）。"""
        wanted = (cfg.model, cfg.compute_type, cfg.cpu_threads or default_threads())
        same = wanted == (self._cfg.model, self._cfg.compute_type, self.threads)
        return same and self._model is not None

    def update_config(self, cfg: AsrConfig) -> None:
        """套用不需要重新載入模型的設定（語言、beam、提示詞、繁體轉換…）。"""
        if cfg.zh_convert != self._cfg.zh_convert:
            self._zh = _ChineseConverter(cfg.zh_convert)
        self._cfg = cfg
        self._last_lang = None  # 新的一次開始：不沿用上一次的語言

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def fixed_language(self) -> str | None:
        lang = self._cfg.language.strip().lower()
        return None if lang in {"", "auto"} else lang

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._root.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        kwargs = dict(device="cpu", compute_type=self._cfg.compute_type, cpu_threads=self.threads,
                      num_workers=1, download_root=str(self._root))
        try:
            # 先只用本機已下載的模型（沒網路也能啟動，也不用每次連線檢查）
            self._model = WhisperModel(self._cfg.model, local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - 本機沒有 → 才上網下載
            try:
                self._model = WhisperModel(self._cfg.model, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"找不到語音辨識模型 {self._cfg.model}，也無法從網路下載。"
                    "第一次使用需要連上網路下載模型（small 約 470 MB），下載一次之後就不需要網路。"
                    f"（詳細原因：{type(exc).__name__}）"
                ) from exc
        # 暖機：第一次推論特別慢，先跑一次讓第一句真正的話不用等
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 1e-3).astype(np.float32)
        self._decode(noise)
        log.info("模型就緒：%s（cpu / %s / %d 執行緒，%.1f 秒）",
                 self._cfg.model, self._cfg.compute_type, self.threads, time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def _choose_language(self, probs: dict[str, float]) -> tuple[str | None, float]:
        """只在允許語言中選；信心太低（短句）時沿用上一句語言。"""
        allowed = [code for code in self._cfg.allowed_languages if code in probs] or list(probs)
        lang = max(allowed, key=lambda code: probs[code])
        prob = probs[lang]
        if prob < 0.02:  # 允許的語言都幾乎不可能 → 多半是雜音或別的語言
            return None, prob
        # 中英夾雜：模型常判成英文，但中文機率明顯不是 0 → 用中文辨識，英文單字會原樣保留
        if lang == "en" and "zh" in allowed and probs.get("zh", 0.0) >= self._cfg.mixed_zh_prob:
            lang = "zh"
            prob = probs["zh"]
        total = sum(probs[c] for c in allowed) or 1.0
        rel = prob / total  # 在允許語言之間的相對信心
        # 上一句的語言也必須是目前勾選的語言（換了勾選後，舊的「上一句語言」不能再沿用）
        if (rel < self._cfg.sticky_language_prob and self._last_lang in allowed
                and self._last_lang != lang):
            return self._last_lang, probs[self._last_lang] / total
        return lang, rel

    def _decode(self, audio: np.ndarray) -> tuple[str, str, float] | None:
        """編碼一次 → 偵測語言 → 解碼。回傳 (文字, 語言, 語言信心)；不可信時回傳 None。"""
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_compression_ratio, get_suppressed_tokens

        c, m = self._cfg, self._model
        audio = audio[: int(MAX_AUDIO_S * SAMPLE_RATE)]
        mel = m.feature_extractor(audio)  # 每 10 ms 一格（frame），30 秒 = 3000 格
        frames = 3000
        if c.dynamic_window:
            frames = min(3000, max(int(c.min_window_s * 100), mel.shape[-1] + int(c.window_padding_s * 100)))
        encoded = m.encode(pad_or_trim(mel, frames))
        # 輸出長度上限：每秒最多約 25 個 token（中日文一秒講不到 10 個字）。
        # 模型偶爾會把同一句一直重複輸出，有上限就能很快發現、不會浪費時間解碼一大串
        duration = audio.size / SAMPLE_RATE
        max_new = int(40 + 25 * duration)

        fixed = self.fixed_language
        if fixed:
            lang, prob = fixed, 1.0
        else:
            ranked = m.model.detect_language(encoded)[0]
            lang, prob = self._choose_language({tok[2:-2]: p for tok, p in ranked})
            if lang is None:
                return None

        tokenizer = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language=lang)
        previous = tokenizer.encode(" " + c.initial_prompt.strip()) if c.initial_prompt.strip() else []
        prompt = m.get_prompt(tokenizer, previous, without_timestamps=True)
        suppress = get_suppressed_tokens(tokenizer, [-1])

        # 依序嘗試：① 動態視窗（快）→ ② 30 秒完整視窗（動態視窗在很短的句子偶爾會一直重複同一句，
        # 完整視窗就正常）→ ③ 完整視窗＋隨機取樣。大部分句子第一次就成功，只有少數會多花約 1 秒
        attempts = [(encoded, 0.0)]
        if frames < 3000:
            attempts.append((None, 0.0))  # None = 需要時才用 30 秒視窗重新編碼
        attempts.append((None, 0.4))
        full_encoded = encoded if frames >= 3000 else None
        text, avg_logprob, no_speech, ratio = "", -9.9, 1.0, 99.0
        max_length = min(m.max_length, len(prompt) + max_new)
        for enc, temperature in attempts:
            if enc is None:
                if full_encoded is None:
                    full_encoded = m.encode(pad_or_trim(mel, 3000))
                enc = full_encoded
            sampling = ({"beam_size": c.beam_size, "patience": 1.0} if temperature == 0
                        else {"beam_size": 1, "sampling_topk": 0, "sampling_temperature": temperature})
            r = m.model.generate(enc, [prompt], max_length=max_length, return_scores=True,
                                 return_no_speech_prob=True, suppress_blank=True, suppress_tokens=suppress,
                                 length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
                                 **sampling)[0]
            tokens = r.sequences_ids[0]
            text = tokenizer.decode(tokens).strip()
            avg_logprob = r.scores[0] * len(tokens) / (len(tokens) + 1)
            no_speech = r.no_speech_prob
            ratio = get_compression_ratio(text) if text else 0.0
            if (ratio <= c.compression_ratio_threshold and avg_logprob >= c.log_prob_threshold
                    and not looks_repeated(text)):
                break
        if no_speech > c.no_speech_threshold and avg_logprob < c.log_prob_threshold:
            return None  # 模型認為這段不是人聲
        if ratio > c.compression_ratio_threshold:
            return None  # 重複字串（典型幻覺）
        return text, lang, prob

    def transcribe(self, utt: Utterance) -> Transcription | None:
        if self._model is None:
            raise RuntimeError("模型尚未載入")
        t0 = time.perf_counter()
        out = self._decode(utt.audio)
        if out is None:
            return None
        text, lang, prob = out
        if lang == "en" and re.search(r"[\u3400-\u9fff]", text):
            lang = "zh"  # 判成英文但文字裡有中文 → 中英夾雜，直接顯示不翻譯
        if lang == "zh":
            text = self._zh(text)
        text = collapse_repeats(text)
        if is_hallucination(text):
            return None
        self._last_lang = lang
        return Transcription(utt.seq, text, lang, prob, utt.audio.size / SAMPLE_RATE,
                             time.perf_counter() - t0)
