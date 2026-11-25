"""Whisper 幻覺過濾與文字整理。

Whisper 在靜音或雜音段常憑空產生 "Thank you for watching"、「ご視聴ありがとうございました」
這類 YouTube 片尾字幕，出現在字幕和逐字稿裡會很困擾，所以先擋掉。
"""

from __future__ import annotations

import re
import unicodedata

_EXACT = {
    "thank you for watching", "thanks for watching", "please subscribe",
    "please like and subscribe", "subtitles by",
    "ご視聴ありがとうございました", "ご視聴いただきありがとうございました",
    "チャンネル登録お願いします", "チャンネル登録よろしくお願いします",
    "시청해주셔서 감사합니다", "시청해 주셔서 감사합니다", "구독과 좋아요 부탁드립니다",
    "謝謝觀看", "谢谢观看", "感謝收看", "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目",
}
_CONTAINS = ("amaraorg", "字幕由", "字幕製作", "字幕制作", "明镜与点点", "明鏡與點點", "mbc 뉴스")
_REPEAT_RE = re.compile(r"(.{1,12}?)\1{4,}")
NO_SPACE_LANGS = {"ja", "zh", "yue"}


def _key(text: str) -> str:
    out = []
    for ch in text.lower():
        cat = unicodedata.category(ch)
        if cat[0] in {"P", "S", "Z"} or ch.isspace():
            continue
        out.append(ch)
    return "".join(out)


_EXACT_KEYS = {_key(p) for p in _EXACT}
_CONTAINS_KEYS = tuple(_key(p) for p in _CONTAINS)


def collapse_repeats(text: str) -> str:
    """Whisper 有時會把同一句整句再輸出一次（「A。A。」），只留一次。"""
    t = text.strip()
    # 整段剛好是同一段文字重複兩次以上
    for n in (2, 3, 4):
        if len(t) % n == 0:
            unit = t[: len(t) // n]
            if unit * n == t and len(_key(unit)) >= 2:
                return unit.strip()
    # 依句尾標點拆開，連續相同的句子只留一句
    parts = re.split(r"(?<=[.!?。！？])\s*", t)
    out: list[str] = []
    for p in parts:
        if p and (not out or _key(p) != _key(out[-1])):
            out.append(p)
    sep = "" if re.search(r"[぀-ヿ㐀-鿿]", t) else " "
    return sep.join(s.strip() for s in out).strip()


def is_hallucination(text: str) -> bool:
    key = _key(text)
    if not key or key in _EXACT_KEYS:
        return True
    if any(k in key for k in _CONTAINS_KEYS):
        return True
    m = _REPEAT_RE.search(key)
    return bool(m and len(m.group(0)) >= 0.6 * len(key) and len(key) >= 8)


def join_segments(texts: list[str], language: str) -> str:
    parts = [t.strip() for t in texts if t and t.strip()]
    return ("" if language in NO_SPACE_LANGS else " ").join(parts).strip()
