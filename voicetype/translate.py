"""本機 CPU 翻譯：英／日／韓 → 繁體中文（NLLB-200，CTranslate2 INT8）。

* NLLB 是 Meta 的多語翻譯模型，一個模型涵蓋 200 種語言，所以英日韓都能直接翻成中文。
* 用 CTranslate2 跑 INT8 量化版（和 faster-whisper 同一套推論引擎，不需要另外裝 PyTorch）。
* 先翻成簡體再用 OpenCC s2twp 轉成台灣繁體用語：NLLB 的簡體訓練資料多很多，品質明顯比直接輸出繁體好。
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from .config import TranslateConfig

log = logging.getLogger(__name__)

NLLB_CODES = {
    "en": "eng_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "vi": "vie_Latn", "th": "tha_Thai",
}
TARGETS = {"zh-tw": ("zho_Hans", "s2twp"), "zh-hant": ("zho_Hant", ""), "zh-cn": ("zho_Hans", "")}


class NllbTranslator:
    def __init__(self, cfg: TranslateConfig, model_root: Path, threads: int) -> None:
        self._cfg = cfg
        self._root = model_root
        self._threads = threads
        self._model: Any = None
        self._tok: Any = None
        self._cc: Any = None
        self.model_name = cfg.model

    def update_config(self, cfg: TranslateConfig) -> None:
        """換模型以外的設定（beam、目標文字）不用重新載入。"""
        if cfg.target != self._cfg.target:
            self._cc = None
            _code, profile = TARGETS.get(cfg.target, TARGETS["zh-tw"])
            if profile:
                from opencc import OpenCC

                self._cc = OpenCC(profile)
        self._cfg = cfg

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        import ctranslate2
        from tokenizers import Tokenizer

        t0 = time.perf_counter()
        path = Path(self._cfg.model)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            opts = dict(cache_dir=str(self._root), allow_patterns=["*.json", "*.bin", "*.txt"])
            try:
                # 先只用本機已下載的模型（沒網路也能啟動）；本機沒有才上網下載
                path = Path(snapshot_download(self._cfg.model, local_files_only=True, **opts))
            except Exception:  # noqa: BLE001
                try:
                    path = Path(snapshot_download(self._cfg.model, **opts))
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"找不到翻譯模型 {self._cfg.model}，也無法從網路下載。"
                        "第一次使用英／日／韓翻譯需要連上網路下載模型（約 600 MB），下載一次之後就不需要網路。"
                        f"（詳細原因：{type(exc).__name__}）"
                    ) from exc
        self._model = ctranslate2.Translator(str(path), device="cpu", compute_type="int8",
                                             inter_threads=1, intra_threads=self._threads)
        self._tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        _code, profile = TARGETS.get(self._cfg.target, TARGETS["zh-tw"])
        if profile:
            try:
                from opencc import OpenCC

                self._cc = OpenCC(profile)
            except ImportError:
                log.warning("找不到 opencc，翻譯結果不做繁體轉換")
        self.translate("Hello.", "en")  # 暖機
        log.info("翻譯模型就緒：%s（%.1f 秒）", self._cfg.model, time.perf_counter() - t0)

    def translate(self, text: str, language: str) -> str:
        """把一句話翻成中文；不支援的語言回傳空字串。"""
        src = NLLB_CODES.get(language)
        if not src or not text.strip():
            return ""
        tgt, _profile = TARGETS.get(self._cfg.target, TARGETS["zh-tw"])
        # NLLB 是用「單句」訓練的：一段裡有好幾句時常會漏翻，所以先拆句、一次批次翻譯再接回來
        sentences = split_sentences(text)
        sources = [[src, *self._tok.encode(s, add_special_tokens=False).tokens[:400], "</s>"] for s in sentences]
        results = self._model.translate_batch(
            sources, target_prefix=[[tgt]] * len(sources), beam_size=self._cfg.beam_size,
            max_decoding_length=256, repetition_penalty=1.1,
        )
        parts = []
        for result in results:
            out = [t for t in result.hypotheses[0] if t != tgt]
            ids = [i for i in (self._tok.token_to_id(t) for t in out) if i is not None]
            zh = self._tok.decode(ids, skip_special_tokens=True).strip()
            parts.append(fullwidth_punct(self._cc.convert(zh) if self._cc else zh))
        return "".join(parts)


def split_sentences(text: str) -> list[str]:
    """依句尾標點拆句（英文 . ? !、日韓 。？！），太短的片段併到前一句。"""
    pieces = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", text.strip()) if p.strip()]
    merged: list[str] = []
    for p in pieces:
        if merged and len(p) < 4:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged or [text.strip()]


_CJK = r"[　-ヿ㐀-鿿＀-￯]"


def fullwidth_punct(text: str) -> str:
    """NLLB 的中文常帶半形標點（「你好, 我們開始.」），改成全形；數字裡的 1,000、3.5、3:00 不動。"""
    text = re.sub(r"(?<!\d),|,(?!\d)", "，", text)
    text = re.sub(rf"(?<={_CJK})\s*\.(?!\d)|\.$", "。", text)
    text = re.sub(r"\?", "？", text)
    text = re.sub(r"!", "！", text)
    text = re.sub(r"(?<!\d):|:(?!\d)", "：", text)
    text = re.sub(r";", "；", text)
    return re.sub(r"\s*([，。？！：；])\s*", r"\1", text).strip()
