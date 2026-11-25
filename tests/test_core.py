import numpy as np
import pytest

from voicetype.config import VadConfig
from voicetype.filters import is_hallucination, join_segments
from voicetype.hotkey import parse_hotkey
from voicetype.segmenter import SAMPLE_RATE, EnergyVad, Segmenter


def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _feed(seg: Segmenter, audio: np.ndarray):
    out = []
    for i in range(0, audio.size, 480):
        out += seg.push(audio[i:i + 480])
    return out


def test_sentence_emitted_after_pause():
    seg = Segmenter(VadConfig(end_silence_ms=500), EnergyVad())
    assert _feed(seg, _tone(1.0)) == []
    assert _feed(seg, _silence(0.3)) == []          # 停頓還不夠久
    utts = _feed(seg, _silence(0.3))                # 累積超過 0.5 秒 → 整句跳出
    assert len(utts) == 1
    assert 1.0 <= utts[0].audio.size / SAMPLE_RATE <= 1.6


def test_two_sentences_and_short_noise_dropped():
    seg = Segmenter(VadConfig(end_silence_ms=500, min_speech_ms=250), EnergyVad())
    audio = np.concatenate([_tone(0.1), _silence(0.8), _tone(1.0), _silence(0.8), _tone(0.8), _silence(0.8)])
    utts = _feed(seg, audio)
    assert [u.seq for u in utts] == [1, 2]


def test_nonstop_speech_is_split_without_losing_audio():
    seg = Segmenter(VadConfig(max_utterance_s=2.0, soft_cut_s=1.0), EnergyVad())
    utts = _feed(seg, _tone(7.0)) + seg.flush()
    assert len(utts) >= 3
    assert all(u.audio.size / SAMPLE_RATE <= 2.0 + 1e-6 for u in utts)
    assert all(u.forced_cut for u in utts[:-1])
    total = sum(u.audio.size for u in utts) / SAMPLE_RATE
    assert abs(total - 7.0) < 0.1  # 硬切後剩下的部分接到下一句，不會掉字


def test_split_prefers_quiet_point():
    # 3.5 秒處有一小段變小聲（不到靜音門檻的停頓）：應該從那裡切，而不是硬切在 4 秒
    audio = np.concatenate([_tone(3.5), _tone(0.12, amp=0.003), _tone(0.9)])
    seg = Segmenter(VadConfig(max_utterance_s=4.0, soft_cut_s=10.0, end_silence_ms=500), EnergyVad())
    utts = _feed(seg, audio)
    assert len(utts) == 1 and utts[0].forced_cut
    assert 3.4 <= utts[0].audio.size / SAMPLE_RATE <= 3.65


def test_long_speech_cut_at_short_breath():
    # 講超過 soft_cut_s 之後，0.24 秒的換氣就足以斷句（平常要 0.5 秒）
    audio = np.concatenate([_tone(1.5), _silence(0.24), _tone(3.0), _silence(0.24), _tone(1.0)])
    seg = Segmenter(VadConfig(end_silence_ms=500, soft_cut_s=4.0, soft_end_silence_ms=200,
                              max_utterance_s=20.0), EnergyVad())
    utts = _feed(seg, audio)
    assert len(utts) == 1 and not utts[0].forced_cut
    assert 4.5 <= utts[0].audio.size / SAMPLE_RATE <= 5.2


def test_hallucination_filter():
    assert is_hallucination("Thank you for watching!")
    assert is_hallucination("ご視聴ありがとうございました。")
    assert is_hallucination("   ")
    assert not is_hallucination("Thank you, see you tomorrow.")
    assert not is_hallucination("안녕하세요, 회의를 시작하겠습니다.")


def test_join_segments():
    assert join_segments(["こんにちは", "世界"], "ja") == "こんにちは世界"
    assert join_segments(["Hello", "world"], "en") == "Hello world"


def test_parse_hotkey():
    assert parse_hotkey("ctrl+alt+space") == (0x2 | 0x1, 0x20)
    assert parse_hotkey("Ctrl+Shift+F9") == (0x2 | 0x4, 0x78)
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+alt")


def test_save_config_keeps_comments(tmp_path):
    import shutil
    import tomllib
    from pathlib import Path

    from voicetype.config import Config, _apply, save_config

    p = tmp_path / "config.toml"
    shutil.copy(Path(__file__).parent.parent / "config.example.toml", p)
    cfg = Config()
    _apply(cfg, tomllib.loads(p.read_text(encoding="utf-8")), "")
    cfg.asr.model = "medium"
    cfg.vad.end_silence_ms = 650
    cfg.asr.initial_prompt = 'A # "quoted"'
    save_config(cfg, p)
    text = p.read_text(encoding="utf-8")
    again = Config()
    _apply(again, tomllib.loads(text), "")
    assert again == cfg
    assert "停頓多久算一句講完" in text  # 註解還在


def test_mixer_adds_secondary_and_pads_missing():
    from voicetype.audio import Mixer

    m = Mixer(max_lag_s=0.5)
    m.push_secondary(np.full(100, 0.25, dtype=np.float32))
    out = m.mix_primary(np.full(160, 0.5, dtype=np.float32))
    assert np.allclose(out[:100], 0.75) and np.allclose(out[100:], 0.5)  # 系統聲音不足的部分當靜音
    m.push_secondary(np.full(20000, 0.9, dtype=np.float32))  # 超過 0.5 秒緩衝只留最新的
    out = m.mix_primary(np.full(16000, 0.5, dtype=np.float32))
    assert np.allclose(out[:8000], 1.0) and np.allclose(out[8000:], 0.5)  # 相加後截在 1.0


def test_fullwidth_punct():
    from voicetype.translate import fullwidth_punct

    assert fullwidth_punct("您好,大家早上,讓我們開始每週專案會議.") == "您好，大家早上，讓我們開始每週專案會議。"
    assert fullwidth_punct("在今天結束之前, 我希望您確認嗎?") == "在今天結束之前，我希望您確認嗎？"
    assert fullwidth_punct("價格是 1,000 元,約 3.5 小時,3:00 開始.") == "價格是 1,000 元，約 3.5 小時，3:00 開始。"


def test_transcript_writes_original_and_translation(tmp_path):
    from voicetype.output import Output

    out = Output(tmp_path)
    out.emit("Hello.", "en", "你好。")
    out.emit("我們開始吧", "zh")
    out.close()
    text = out.transcript_path.read_text(encoding="utf-8")
    assert "[en] Hello." in text and "[中] 你好。" in text and "[zh] 我們開始吧" in text


def test_speaker_language_policy():
    from voicetype.config import Config, apply_language_policy, foreign_languages

    cfg = Config()
    apply_language_policy(cfg)
    assert (cfg.asr.language, cfg.asr.allowed_languages, foreign_languages(cfg)) == ("zh", ["zh"], [])
    cfg.asr.languages = ["ja"]
    apply_language_policy(cfg)
    assert (cfg.asr.language, foreign_languages(cfg)) == ("ja", ["ja"])  # 只勾一種 = 固定該語言
    cfg.asr.languages = ["zh", "EN", "ko", "en"]
    apply_language_policy(cfg)
    assert cfg.asr.language == "auto" and cfg.asr.allowed_languages == ["zh", "en", "ko"]
    assert foreign_languages(cfg) == ["en", "ko"]
    cfg.asr.languages = []
    apply_language_policy(cfg)
    assert cfg.asr.languages == ["zh"]  # 都沒勾 = 中文


def test_collapse_repeats():
    from voicetype.filters import collapse_repeats

    s = "請把 Meeting Notes 放到 shared folder 裡面。"
    assert collapse_repeats(s + s) == s
    assert collapse_repeats("Good morning. Good morning. Let's start.") == "Good morning. Let's start."
    assert collapse_repeats("好的，我們開始。") == "好的，我們開始。"
    assert collapse_repeats("哈哈") == "哈哈"


def test_split_sentences():
    from voicetype.translate import split_sentences

    assert split_sentences("Add your comments by Friday. Good morning everyone.") == [
        "Add your comments by Friday.", "Good morning everyone."]
    assert split_sentences("今日は晴れです。明日は雨です。") == ["今日は晴れです。", "明日は雨です。"]
    assert split_sentences("It costs 3.5 dollars. OK.") == ["It costs 3.5 dollars. OK."]


def test_data_dir_rules(tmp_path):
    from voicetype.config import _usable_data_dir

    assert _usable_data_dir(tmp_path) or len(str(tmp_path)) > 90  # 一般可寫入的短路徑可以用
    deep = tmp_path / ("x" * 100)
    deep.mkdir()
    assert not _usable_data_dir(deep)  # 路徑太長 → 改用 %LOCALAPPDATA%


def test_sticky_language_never_returns_unchecked_language():
    from pathlib import Path

    from voicetype.asr import WhisperCpu
    from voicetype.config import AsrConfig

    w = WhisperCpu(AsrConfig(), Path("."))
    w._last_lang = "ja"  # 上一次用過日文
    w._cfg.allowed_languages = ["zh", "en"]  # 現在只勾中英
    assert w._choose_language({"zh": 0.30, "en": 0.35, "ja": 0.30})[0] in {"zh", "en"}


def test_broken_or_mistyped_config_falls_back_to_defaults(tmp_path):
    from voicetype.config import CONFIG_WARNINGS, Config, load_config

    p = tmp_path / "config.toml"
    p.write_text("[overlay]\nopacity = \n", encoding="utf-8")  # TOML 語法錯誤
    cfg = load_config(p)
    assert cfg.overlay.opacity == Config().overlay.opacity and CONFIG_WARNINGS
    p.write_text('[overlay]\nfont_size = "大"\nopacity = 1\n[vad]\nend_silence_ms = 450.0\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.overlay.font_size == Config().overlay.font_size  # 型別錯誤 → 預設值
    assert cfg.overlay.opacity == 1.0 and isinstance(cfg.overlay.opacity, float)  # 整數自動轉成小數
    assert cfg.vad.end_silence_ms == 450 and isinstance(cfg.vad.end_silence_ms, int)
    assert len(CONFIG_WARNINGS) == 1


def test_transcript_order_follows_speech_order(tmp_path):
    """中文句不用翻譯、英文句要翻譯：逐字稿仍要依講話順序寫入（只有一個執行緒寫檔）。"""
    import time
    import types

    from voicetype.app import App
    from voicetype.config import Config
    from voicetype.output import Output

    cfg = Config()
    cfg.asr.languages = ["zh", "en"]

    class SlowTranslator:
        loaded = True

        def translate(self, text, lang):
            time.sleep(0.3)  # 翻譯比較慢
            return "（中文）"

    app = App(cfg, on_event=lambda k, p: None, asr=types.SimpleNamespace(loaded=True), translator=SlowTranslator())
    app.output = Output(tmp_path)
    import threading

    t = threading.Thread(target=app._mt_worker)
    t.start()
    R = lambda seq, text, lang: types.SimpleNamespace(seq=seq, text=text, language=lang, translation="", translate_s=0.0)
    app.mt_q.put(R(1, "First sentence.", "en"))
    app.mt_q.put(R(2, "第二句中文", "zh"))
    app.mt_q.put(None)
    t.join(5)
    app.output.close()
    lines = app.output.transcript_path.read_text(encoding="utf-8").splitlines()
    assert "First sentence." in lines[0] and "（中文）" in lines[1] and "第二句中文" in lines[2]


def test_looks_repeated():
    from voicetype.asr import looks_repeated

    s = "絵を描くのにはもう飽きてしまった"
    assert looks_repeated(s * 3 + "絵を描く")
    assert looks_repeated("Good morning everyone. Good morning everyone.")
    assert not looks_repeated(s)
    assert not looks_repeated("好的，我們開始吧。")
    assert not looks_repeated("哈哈哈")


def test_audio_watchdog_detects_problems(monkeypatch):
    """自動重新連線的判斷（不需要真的音效卡）：裝置中斷、麥克風沒資料、預設裝置變更。"""
    import queue
    import time
    import types

    import voicetype.audio as audio
    from voicetype.audio import AudioCapture
    from voicetype.config import AudioConfig

    cap = AudioCapture(AudioConfig(source="mic"), queue.Queue())
    ok = types.SimpleNamespace(is_active=lambda: True)
    dead = types.SimpleNamespace(is_active=lambda: False)
    cap._last_data["mic"] = time.monotonic()
    assert cap._problem([ok], {}) is None
    assert cap._problem([dead], {}) == "音訊裝置中斷"
    cap._last_data["mic"] = time.monotonic() - audio.MIC_STALL_S - 1
    assert cap._problem([ok], {}) == "麥克風沒有聲音資料"
    cap._last_data["mic"] = time.monotonic()
    monkeypatch.setattr(cap, "_watched_defaults", lambda: {"mic": "新裝置"})
    assert cap._problem([ok], {"mic": "舊裝置"}) == "Windows 預設裝置已變更"
    # 系統聲音沒播放時本來就不送資料：不能當成斷線
    cap2 = AudioCapture(AudioConfig(source="loopback"), queue.Queue())
    cap2._last_data["loopback"] = time.monotonic() - 60
    assert cap2._problem([ok], {}) is None
