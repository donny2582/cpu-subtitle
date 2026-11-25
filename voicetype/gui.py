"""控制視窗（tkinter，Python 內建）：開始／停止／暫停、即時音量、辨識結果、辨識翻譯設定與大字幕外觀。

背景執行緒（App）透過 on_event 丟事件到佇列，介面每 50 ms 取出更新，避免跨執行緒操作 tk 元件。
"""

from __future__ import annotations

import copy
import logging
import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import colorchooser, messagebox, ttk
from tkinter import font as tkfont
from typing import Any

from .config import Config, app_root, apply_language_policy, save_config
from .overlay import PRESETS, SubtitleOverlay

log = logging.getLogger(__name__)
MAX_LOG_LINES = 2000  # 辨識結果區最多保留幾行（開好幾個小時也不會越來越慢）

MODELS = ["small", "base", "medium", "large-v3-turbo", "large-v3"]
SPEAKER_LANGS = [("zh", "中文"), ("en", "英文"), ("ja", "日文"), ("ko", "韓文")]
SOURCES = [("mic", "麥克風"), ("loopback", "系統聲音（會議／影片）"), ("both", "兩者混音")]
MT_MODELS = [("JustFrederik/nllb-200-distilled-600M-ct2-int8", "快（NLLB 600M，約 0.8 秒／句）"),
             ("OpenNMT/nllb-200-distilled-1.3B-ct2-int8", "準（NLLB 1.3B，約 1.4 秒／句）")]
LANG_TAG = {"en": "英", "ja": "日", "ko": "韓", "zh": "中"}
UI_FONT = "Microsoft JhengHei UI"


class ControlWindow:
    def __init__(self, cfg: Config, config_path: Path | None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.app: Any = None
        self.thread: threading.Thread | None = None
        self.asr: Any = None  # 停止後保留已載入的模型，下次開始不用重新載入
        self.translator: Any = None
        self.devices: list[tuple[int, str, str]] = []
        self.running = False
        self._loading_widgets = False

        self.root = tk.Tk()
        self.root.title("即時字幕（本機 CPU）")
        self.root.minsize(600, 640)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        self.root.option_add("*Font", (UI_FONT, 10))
        style.configure("Big.TButton", font=(UI_FONT, 12, "bold"), padding=8)
        style.configure("Status.TLabel", font=(UI_FONT, 11, "bold"))
        style.configure("Hint.TLabel", foreground="#777")

        self._build()
        self.overlay = SubtitleOverlay(self.root, self.cfg.overlay, on_change=self._overlay_changed_outside)
        self._load_into_widgets()
        self._refresh_devices()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_job = self.root.after(50, self._poll)

    # ================================================================== 版面
    def _build(self) -> None:
        r = self.root
        top = ttk.Frame(r, padding=(10, 10, 10, 4))
        top.pack(fill="x")
        self.btn_start = ttk.Button(top, text="▶ 開始字幕", style="Big.TButton", command=self._toggle_run)
        self.btn_start.pack(side="left")
        self.btn_pause = ttk.Button(top, text="⏸ 暫停", style="Big.TButton", command=self._pause, state="disabled")
        self.btn_pause.pack(side="left", padx=8)
        stat = ttk.Frame(top)
        stat.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.var_status = tk.StringVar(value="● 未啟動")
        self.lbl_status = ttk.Label(stat, textvariable=self.var_status, style="Status.TLabel", foreground="#666")
        self.lbl_status.pack(anchor="w")
        meter = ttk.Frame(stat)
        meter.pack(fill="x", anchor="w")
        ttk.Label(meter, text="音量").pack(side="left")
        self.level = ttk.Progressbar(meter, maximum=60, length=160)
        self.level.pack(side="left", padx=4)
        self.var_pending = tk.StringVar(value="")
        ttk.Label(meter, textvariable=self.var_pending, foreground="#a60").pack(side="left", padx=4)
        ttk.Label(r, text="講完一句、停頓約 0.5 秒後整句出字幕；中文直接顯示，勾選的外語會翻成中文。",
                  foreground="#555", padding=(10, 0)).pack(fill="x")

        nb = ttk.Notebook(r)
        nb.pack(fill="x", padx=10, pady=6)
        tab1 = ttk.Frame(nb, padding=8)
        tab2 = ttk.Frame(nb, padding=8)
        nb.add(tab1, text="  辨識與翻譯  ")
        nb.add(tab2, text="  字幕外觀  ")
        self._build_dictation_tab(tab1)
        self._build_overlay_tab(tab2)

        brow = ttk.Frame(r, padding=(10, 0))
        brow.pack(fill="x")
        self.btn_save = ttk.Button(brow, text="儲存設定", command=self._save)
        self.btn_save.pack(side="left")
        ttk.Button(brow, text="開啟逐字稿資料夾", command=self._open_transcripts).pack(side="left", padx=6)
        self.btn_bench = ttk.Button(brow, text="測試 CPU 速度", command=self._benchmark)
        self.btn_bench.pack(side="left")

        logf = ttk.LabelFrame(r, text="辨識結果", padding=6)
        logf.pack(fill="both", expand=True, padx=10, pady=(6, 10))
        self.txt = tk.Text(logf, height=8, wrap="word", state="disabled", relief="flat",
                           font=(UI_FONT, 11), background="#fafafa")
        sb = ttk.Scrollbar(logf, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(fill="both", expand=True)
        self.txt.tag_configure("meta", foreground="#888", font=(UI_FONT, 9))
        self.txt.tag_configure("sys", foreground="#666")
        self.txt.tag_configure("err", foreground="#c00")
        self.txt.tag_configure("zh", foreground="#05a", font=(UI_FONT, 11, "bold"))
        ttk.Button(logf, text="清除", command=self._clear_log).pack(anchor="e", pady=(4, 0))

        self.dictation_widgets: list[Any] = [
            self.cmb_model, *self.chk_lang, *self.rad_source, self.cmb_mic, self.cmb_loop,
            self.btn_refresh, self.chk_save_audio, self.scl_silence, self.spn_threads,
            self.spn_beam, self.chk_zh, self.ent_prompt, self.btn_bench, self.cmb_mt_model,
        ]

    def _row(self, parent: ttk.Frame, row: int, label: str) -> ttk.Frame:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=4)
        f = ttk.Frame(parent)
        f.grid(row=row, column=1, sticky="we", pady=4)
        return f

    def _build_dictation_tab(self, t: ttk.Frame) -> None:
        t.columnconfigure(1, weight=1)

        f = self._row(t, 0, "模型")
        self.var_model = tk.StringVar()
        self.cmb_model = ttk.Combobox(f, textvariable=self.var_model, values=MODELS, width=18)
        self.cmb_model.pack(side="left")
        ttk.Label(f, text="small 最快；large-v3-turbo 較準但 CPU 慢很多", style="Hint.TLabel").pack(side="left", padx=6)

        f = self._row(t, 1, "講者語言")
        self.var_lang: dict[str, tk.BooleanVar] = {}
        self.chk_lang: list[ttk.Checkbutton] = []
        for code, name in SPEAKER_LANGS:
            v = tk.BooleanVar()
            self.var_lang[code] = v
            c = ttk.Checkbutton(f, text=name, variable=v, command=self._update_lang_hint)
            c.pack(side="left", padx=(0, 6))
            self.chk_lang.append(c)
        ttk.Label(f, text="（勾選講者可能會講的語言）", style="Hint.TLabel").pack(side="left")
        self.var_lang_hint = tk.StringVar()
        ttk.Label(t, textvariable=self.var_lang_hint, foreground="#05a").grid(row=6, column=1, sticky="w")

        f = self._row(t, 2, "收音來源")
        self.var_source = tk.StringVar()
        self.rad_source = [ttk.Radiobutton(f, text=name, value=code, variable=self.var_source,
                                           command=self._update_device_state) for code, name in SOURCES]
        for w in self.rad_source:
            w.pack(side="left", padx=(0, 10))

        f = self._row(t, 3, "麥克風")
        self.var_mic = tk.StringVar()
        self.cmb_mic = ttk.Combobox(f, textvariable=self.var_mic, state="readonly", width=46)
        self.cmb_mic.pack(side="left", fill="x", expand=True)
        self.btn_refresh = ttk.Button(f, text="重新整理", command=self._refresh_devices)
        self.btn_refresh.pack(side="left", padx=4)

        f = self._row(t, 4, "系統聲音")
        self.var_loop = tk.StringVar()
        self.cmb_loop = ttk.Combobox(f, textvariable=self.var_loop, state="readonly", width=46)
        self.cmb_loop.pack(side="left", fill="x", expand=True)
        self.lbl_both = ttk.Label(t, text="兩者混音時請戴耳機，否則喇叭的聲音會被麥克風再收一次。",
                                  foreground="#a60")
        self.lbl_both.grid(row=5, column=1, sticky="w")

        f = self._row(t, 7, "紀錄")
        self.var_save_audio = tk.BooleanVar()
        self.chk_save_audio = ttk.Checkbutton(f, text="同時存錄音檔（WAV，約每小時 115 MB）",
                                              variable=self.var_save_audio)
        self.chk_save_audio.pack(side="left")
        ttk.Label(f, text="逐字稿一律會存", style="Hint.TLabel").pack(side="left", padx=8)

        f = self._row(t, 8, "停頓出字")
        self.var_silence = tk.IntVar()
        self.scl_silence = ttk.Scale(f, from_=300, to=1500, variable=self.var_silence, length=200,
                                     command=lambda _v: self.var_silence.set(round(self.var_silence.get() / 50) * 50))
        self.scl_silence.pack(side="left")
        ttk.Label(f, textvariable=self.var_silence, width=5, anchor="e").pack(side="left")
        ttk.Label(f, text="ms（越短越快出字，但句子容易被切斷）", style="Hint.TLabel").pack(side="left")

        f = self._row(t, 9, "效能")
        ttk.Label(f, text="CPU 執行緒").pack(side="left")
        self.var_threads = tk.IntVar()
        self.spn_threads = ttk.Spinbox(f, from_=0, to=os.cpu_count() or 8, textvariable=self.var_threads, width=4)
        self.spn_threads.pack(side="left", padx=(4, 2))
        ttk.Label(f, text="(0=自動)", style="Hint.TLabel").pack(side="left")
        ttk.Label(f, text="beam").pack(side="left", padx=(14, 0))
        self.var_beam = tk.IntVar()
        self.spn_beam = ttk.Spinbox(f, from_=1, to=5, textvariable=self.var_beam, width=3)
        self.spn_beam.pack(side="left", padx=4)
        ttk.Label(f, text="(1=最快、5=較準)", style="Hint.TLabel").pack(side="left")

        f = self._row(t, 10, "其他")
        self.var_zh = tk.BooleanVar()
        self.chk_zh = ttk.Checkbutton(f, text="中文轉台灣繁體", variable=self.var_zh)
        self.chk_zh.pack(side="left")
        self.var_topmost = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="控制視窗置頂", variable=self.var_topmost,
                        command=lambda: self.root.attributes("-topmost", self.var_topmost.get())).pack(side="left",
                                                                                                    padx=10)
        self.var_minimize = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="開始後縮到最小", variable=self.var_minimize).pack(side="left")

        f = self._row(t, 13, "翻譯模型")
        self.var_mt_model = tk.StringVar()
        self.cmb_mt_model = ttk.Combobox(f, textvariable=self.var_mt_model, state="readonly", width=30,
                                         values=[name for _id, name in MT_MODELS])
        self.cmb_mt_model.pack(side="left", padx=8)
        ttk.Label(f, text="第一次使用會下載模型", style="Hint.TLabel").pack(side="left")

        f = self._row(t, 11, "提示詞")
        self.var_prompt = tk.StringVar()
        self.ent_prompt = ttk.Entry(f, textvariable=self.var_prompt)
        self.ent_prompt.pack(side="left", fill="x", expand=True)
        ttk.Label(t, text="專有名詞用逗號分開，例如：Kubernetes, PostgreSQL, 區間測速。執行中不能改這一頁。",
                  style="Hint.TLabel").grid(row=12, column=1, sticky="w")

    def _build_overlay_tab(self, t: ttk.Frame) -> None:
        t.columnconfigure(1, weight=1)

        f = self._row(t, 0, "大字幕")
        self.var_ov_enabled = tk.BooleanVar()
        ttk.Checkbutton(f, text="執行時顯示大字幕視窗", variable=self.var_ov_enabled).pack(side="left")
        ttk.Label(f, text="（這一頁隨時可改，立即套用）", style="Hint.TLabel").pack(side="left", padx=8)

        f = self._row(t, 1, "樣式")
        self.var_ov_preset = tk.StringVar(value="自訂")
        cmb = ttk.Combobox(f, textvariable=self.var_ov_preset, state="readonly", width=16,
                           values=[*PRESETS, "自訂"])
        cmb.pack(side="left")
        cmb.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())
        self.color_vars: dict[str, tk.StringVar] = {}
        self.color_swatches: dict[str, tk.Label] = {}
        for key, name in (("text_color", "文字"), ("bg_color", "底色"), ("border_color", "外框")):
            ttk.Label(f, text=name).pack(side="left", padx=(12, 2))
            v = tk.StringVar()
            self.color_vars[key] = v
            sw = tk.Label(f, width=3, relief="solid", borderwidth=1, cursor="hand2")
            sw.pack(side="left")
            sw.bind("<Button-1>", lambda _e, k=key: self._pick_color(k))
            self.color_swatches[key] = sw
        ttk.Label(f, text="（點色塊換顏色）", style="Hint.TLabel").pack(side="left", padx=4)

        f = self._row(t, 2, "字型")
        families = sorted({x for x in tkfont.families(self.root) if not x.startswith("@")})
        self.var_ov_font = tk.StringVar()
        ttk.Combobox(f, textvariable=self.var_ov_font, values=families, width=26).pack(side="left")
        ttk.Label(f, text="字級").pack(side="left", padx=(12, 2))
        self.var_ov_size = tk.IntVar()
        ttk.Spinbox(f, from_=12, to=120, increment=2, textvariable=self.var_ov_size, width=4).pack(side="left")
        self.var_ov_bold = tk.BooleanVar()
        ttk.Checkbutton(f, text="粗體", variable=self.var_ov_bold).pack(side="left", padx=8)

        f = self._row(t, 3, "外框粗細")
        self.var_ov_border = tk.IntVar()
        ttk.Spinbox(f, from_=0, to=20, textvariable=self.var_ov_border, width=4).pack(side="left")
        ttk.Label(f, text="像素（0 = 無框）", style="Hint.TLabel").pack(side="left", padx=4)
        ttk.Label(f, text="圓角").pack(side="left", padx=(12, 2))
        self.var_ov_radius = tk.IntVar()
        ttk.Spinbox(f, from_=0, to=60, increment=2, textvariable=self.var_ov_radius, width=4).pack(side="left")
        ttk.Label(f, text="透明").pack(side="left", padx=(12, 2))
        self.var_ov_opacity = tk.DoubleVar()
        ttk.Scale(f, from_=0.3, to=1.0, variable=self.var_ov_opacity, length=120).pack(side="left")
        ttk.Label(f, text="不透明").pack(side="left", padx=(2, 0))

        f = self._row(t, 4, "版面")
        ttk.Label(f, text="同時顯示").pack(side="left")
        self.var_ov_lines = tk.IntVar()
        ttk.Spinbox(f, from_=1, to=6, textvariable=self.var_ov_lines, width=3).pack(side="left", padx=2)
        ttk.Label(f, text="句　寬度").pack(side="left")
        self.var_ov_width = tk.DoubleVar()
        ttk.Scale(f, from_=0.3, to=1.0, variable=self.var_ov_width, length=120).pack(side="left", padx=2)
        ttk.Label(f, text="距底部").pack(side="left", padx=(8, 2))
        self.var_ov_bottom = tk.IntVar()
        ttk.Spinbox(f, from_=0, to=800, increment=10, textvariable=self.var_ov_bottom, width=5).pack(side="left")

        f = self._row(t, 5, "行為")
        ttk.Label(f, text="幾秒沒新句子就清除").pack(side="left")
        self.var_ov_clear = tk.DoubleVar()
        ttk.Spinbox(f, from_=0, to=120, increment=1, textvariable=self.var_ov_clear, width=4).pack(side="left", padx=2)
        ttk.Label(f, text="(0=不清除)", style="Hint.TLabel").pack(side="left")
        self.var_ov_lang = tk.BooleanVar()
        ttk.Checkbutton(f, text="顯示語言標籤", variable=self.var_ov_lang).pack(side="left", padx=(12, 0))
        self.var_ov_source = tk.BooleanVar()
        ttk.Checkbutton(f, text="中文下方小字顯示原文", variable=self.var_ov_source).pack(side="left", padx=(8, 0))
        self.var_ov_through = tk.BooleanVar()
        ttk.Checkbutton(f, text="滑鼠點擊穿透", variable=self.var_ov_through).pack(side="left", padx=8)

        f = self._row(t, 6, "")
        ttk.Button(f, text="預覽字幕", command=self._preview_overlay).pack(side="left")
        ttk.Button(f, text="位置重設", command=lambda: self.overlay.reset_position()).pack(side="left", padx=6)
        ttk.Button(f, text="關閉預覽", command=self._hide_preview).pack(side="left")
        ttk.Label(t, text="字幕視窗可直接拖曳；在字幕上按右鍵可調字級、清除、隱藏；Ctrl + 滾輪也能調字級。",
                  style="Hint.TLabel").grid(row=7, column=1, sticky="w")

        self.overlay_vars: list[tk.Variable] = [
            self.var_ov_enabled, self.var_ov_font, self.var_ov_size, self.var_ov_bold, self.var_ov_border,
            self.var_ov_opacity, self.var_ov_lines, self.var_ov_width, self.var_ov_bottom, self.var_ov_clear,
            self.var_ov_lang, self.var_ov_through, self.var_ov_source, self.var_ov_radius, *self.color_vars.values(),
        ]
        for v in self.overlay_vars:
            v.trace_add("write", lambda *_a: self._overlay_vars_changed())

    # ================================================================== 設定 ↔ 元件
    def _load_into_widgets(self) -> None:
        self._loading_widgets = True
        c = self.cfg
        self.var_model.set(c.asr.model)
        for code, v in self.var_lang.items():
            v.set(code in c.asr.languages)
        self.var_source.set(c.audio.source)
        self.var_save_audio.set(c.output.save_audio)
        self._update_lang_hint()
        self.var_silence.set(c.vad.end_silence_ms)
        self.var_threads.set(c.asr.cpu_threads)
        self.var_beam.set(c.asr.beam_size)
        self.var_zh.set(bool(c.asr.zh_convert))
        self.var_prompt.set(c.asr.initial_prompt)
        self.var_mt_model.set(dict(MT_MODELS).get(c.translate.model, c.translate.model))

        o = c.overlay
        self.var_ov_enabled.set(o.enabled)
        self.var_ov_font.set(o.font_family)
        self.var_ov_size.set(o.font_size)
        self.var_ov_bold.set(o.bold)
        self.var_ov_border.set(o.border_width)
        self.var_ov_radius.set(o.corner_radius)
        self.var_ov_opacity.set(o.opacity)
        self.var_ov_lines.set(o.max_lines)
        self.var_ov_width.set(o.width_ratio)
        self.var_ov_bottom.set(o.bottom_margin)
        self.var_ov_clear.set(o.clear_after_s)
        self.var_ov_lang.set(o.show_language)
        self.var_ov_through.set(o.click_through)
        self.var_ov_source.set(o.show_source)
        for key, v in self.color_vars.items():
            v.set(getattr(o, key))
        self._loading_widgets = False
        self._overlay_vars_changed()

    def _collect(self) -> Config | None:
        cfg = copy.deepcopy(self.cfg)
        model = self.var_model.get().strip()
        if not model:
            messagebox.showwarning("設定", "請選擇模型")
            return None
        cfg.asr.model = model
        chosen = [code for code, v in self.var_lang.items() if v.get()]
        if not chosen:
            messagebox.showwarning("設定", "請至少勾一種講者語言")
            return None
        extra = [x for x in self.cfg.asr.languages if x not in self.var_lang]  # 設定檔裡介面沒列出的語言
        cfg.asr.languages = chosen + extra
        cfg.audio.source = self.var_source.get()
        cfg.audio.mic_device_index = self._selected(self.var_mic, "mic")
        cfg.audio.loopback_device_index = self._selected(self.var_loop, "loopback")
        cfg.output.save_audio = self.var_save_audio.get()
        try:
            cfg.vad.end_silence_ms = int(self.var_silence.get())
            cfg.asr.cpu_threads = max(0, int(self.var_threads.get()))
            cfg.asr.beam_size = min(10, max(1, int(self.var_beam.get())))
        except (tk.TclError, ValueError):
            messagebox.showwarning("設定", "執行緒與 beam 請填整數")
            return None
        cfg.asr.zh_convert = (self.cfg.asr.zh_convert or "s2twp") if self.var_zh.get() else ""
        cfg.asr.initial_prompt = self.var_prompt.get().strip()
        label = self.var_mt_model.get()
        cfg.translate.model = next((mid for mid, name in MT_MODELS if name == label), label or cfg.translate.model)
        cfg.overlay = self.cfg.overlay  # 字幕外觀與介面共用同一份，改了即時生效
        apply_language_policy(cfg)
        return cfg

    def _update_lang_hint(self) -> None:
        chosen = [(code, name) for code, name in SPEAKER_LANGS if self.var_lang[code].get()]
        foreign = [name for code, name in chosen if code != "zh"]
        if not chosen:
            self.var_lang_hint.set("請至少勾一種語言")
            return
        if len(chosen) == 1:
            head = f"只講{chosen[0][1]}：固定用{chosen[0][1]}辨識（最快、最準）"
        else:
            head = "每句在勾選的語言之間自動判斷"
        tail = []
        if any(code == "zh" for code, _n in chosen):
            tail.append("中文與中英夾雜直接顯示")
        if foreign:
            tail.append(f"{'／'.join(foreign)}翻成中文並附原文")
        self.var_lang_hint.set(head + "；" + "，".join(tail))

    def _set_dictation_enabled(self, enabled: bool) -> None:
        for w in self.dictation_widgets:
            if isinstance(w, ttk.Combobox):
                w.configure(state=("normal" if w is self.cmb_model else "readonly") if enabled else "disabled")
            else:
                w.configure(state="normal" if enabled else "disabled")
        if enabled:
            self._update_device_state()

    # ================================================================== 字幕外觀
    def _overlay_vars_changed(self) -> None:
        if self._loading_widgets:
            return
        o = self.cfg.overlay
        try:
            o.enabled = self.var_ov_enabled.get()
            o.font_family = self.var_ov_font.get() or o.font_family
            o.font_size = min(120, max(12, int(self.var_ov_size.get())))
            o.bold = self.var_ov_bold.get()
            o.border_width = min(20, max(0, int(self.var_ov_border.get())))
            o.corner_radius = min(60, max(0, int(self.var_ov_radius.get())))
            o.opacity = round(float(self.var_ov_opacity.get()), 2)
            o.max_lines = min(6, max(1, int(self.var_ov_lines.get())))
            o.width_ratio = round(float(self.var_ov_width.get()), 2)
            o.bottom_margin = max(0, int(self.var_ov_bottom.get()))
            o.clear_after_s = max(0.0, float(self.var_ov_clear.get()))
        except (tk.TclError, ValueError):
            return  # 使用者正在輸入一半（例如清空了數字欄）
        o.show_language = self.var_ov_lang.get()
        o.click_through = self.var_ov_through.get()
        o.show_source = self.var_ov_source.get()
        for key, v in self.color_vars.items():
            color = v.get()
            if color:
                try:
                    self.color_swatches[key].configure(bg=color)
                except tk.TclError:
                    continue  # 不合法的顏色字串
                setattr(o, key, color)
        match = next((name for name, p in PRESETS.items()
                      if all(str(getattr(o, k)).upper() == str(val).upper() for k, val in p.items())), "自訂")
        self.var_ov_preset.set(match)
        if hasattr(self, "overlay"):
            self.overlay.apply(o)
            if self.running:
                (self.overlay.show if o.enabled else self.overlay.hide)()

    def _overlay_changed_outside(self) -> None:
        """在字幕上用右鍵或滾輪改了字級 → 同步回控制視窗。"""
        self._loading_widgets = True
        self.var_ov_size.set(self.overlay.cfg.font_size)
        self._loading_widgets = False

    def _apply_preset(self) -> None:
        p = PRESETS.get(self.var_ov_preset.get())
        if not p:
            return
        self._loading_widgets = True
        for key in ("text_color", "bg_color", "border_color"):
            self.color_vars[key].set(p[key])
        self.var_ov_border.set(p["border_width"])
        self._loading_widgets = False
        self._overlay_vars_changed()

    def _pick_color(self, key: str) -> None:
        _rgb, hexcolor = colorchooser.askcolor(self.color_vars[key].get(), parent=self.root, title="選擇顏色")
        if hexcolor:
            self.color_vars[key].set(hexcolor.upper())

    def _preview_overlay(self) -> None:
        self.overlay.apply(self.cfg.overlay)
        self.overlay.preview()

    def _hide_preview(self) -> None:
        self.overlay.clear()
        if not self.running:
            self.overlay.hide()

    # ================================================================== 裝置
    def _refresh_devices(self) -> None:
        try:
            from .audio import enumerate_devices

            self.devices = enumerate_devices()
        except Exception as exc:  # noqa: BLE001
            self.devices = []
            self._log(f"讀取音訊裝置失敗：{exc}", "err")
        for var, cmb, kind, wanted in (
            (self.var_mic, self.cmb_mic, "mic", self.cfg.audio.mic_device_index),
            (self.var_loop, self.cmb_loop, "loopback", self.cfg.audio.loopback_device_index),
        ):
            items = self._device_items(kind)
            cmb.configure(values=[label for _i, label in items])
            var.set(next((lbl for i, lbl in items if i == wanted), items[0][1]))
        self._update_device_state()

    def _device_items(self, kind: str) -> list[tuple[int, str]]:
        return [(-1, "（系統預設）")] + [(i, f"[{i}] {name}") for i, k, name in self.devices if k == kind]

    def _selected(self, var: tk.StringVar, kind: str) -> int:
        return next((i for i, lbl in self._device_items(kind) if lbl == var.get()), -1)

    def _update_device_state(self) -> None:
        if self.running:
            return
        src = self.var_source.get()
        self.cmb_mic.configure(state="readonly" if src in {"mic", "both"} else "disabled")
        self.cmb_loop.configure(state="readonly" if src in {"loopback", "both"} else "disabled")
        if src == "both":
            self.lbl_both.grid()
        else:
            self.lbl_both.grid_remove()

    # ================================================================== 動作
    def _toggle_run(self) -> None:
        if self.thread and self.thread.is_alive():
            self.btn_start.configure(state="disabled")
            self.app.quit()
            return
        cfg = self._collect()
        if cfg is None:
            return
        from .app import App

        if self.asr is not None and self.asr.can_reuse(cfg.asr):
            self.asr.update_config(cfg.asr)
        else:
            self.asr = None
        if self.translator is not None and self.translator.model_name != cfg.translate.model:
            self.translator = None
        if self.translator is not None:
            self.translator.update_config(cfg.translate)
        self.cfg = cfg
        self.app = App(cfg, on_event=lambda k, p: self.events.put((k, p)), asr=self.asr,
                       translator=self.translator)
        self.asr = self.app.asr
        self.translator = self.app.translator or self.translator
        self.thread = threading.Thread(target=self.app.run, name="voicetype", daemon=True)
        self.thread.start()
        self.running = True
        self._set_dictation_enabled(False)
        self.btn_start.configure(text="■ 停止")
        self.btn_pause.configure(state="normal", text="⏸ 暫停")
        self._set_status("● 啟動中…", "#a60")
        self.overlay.clear()
        if cfg.overlay.enabled:
            self.overlay.show()
        self.txt.focus_set()
        if self.var_minimize.get():
            self.root.after(300, self.root.iconify)

    def _pause(self) -> None:
        if self.app:
            self.app.toggle()

    def _save(self) -> None:
        cfg = self._collect()
        if cfg is None:
            return
        if not self.running:
            self.cfg = cfg
        try:
            path = save_config(cfg, self.config_path)
        except OSError as exc:
            messagebox.showerror("儲存設定", f"寫入失敗：{exc}")
            return
        self._log(f"設定已儲存：{path}", "sys")

    def _open_transcripts(self) -> None:
        d = Path(self.cfg.output.transcript_dir or "transcripts")
        d = d if d.is_absolute() else app_root() / d
        d.mkdir(parents=True, exist_ok=True)
        os.startfile(d)  # type: ignore[attr-defined]  # noqa: S606

    def _benchmark(self) -> None:
        cfg = self._collect()
        if cfg is None:
            return
        self._set_dictation_enabled(False)
        self.btn_start.configure(state="disabled")
        self._log(f"測試 {cfg.asr.model} 在各種執行緒數下的速度（需要一兩分鐘）…", "sys")

        def work() -> None:
            import io
            from contextlib import redirect_stdout

            from .app import run_benchmark

            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    run_benchmark(cfg)
                self.events.put(("bench", buf.getvalue()))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("bench", f"測試失敗：{exc}"))

        threading.Thread(target=work, daemon=True).start()

    # ================================================================== 事件
    def _poll(self) -> None:
        try:
            for _ in range(200):  # 一次最多處理 200 個事件，介面不會被大量事件卡住
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle(kind, payload)
                except Exception:  # noqa: BLE001
                    # 單一事件出錯只記錄下來：不能讓整個更新迴圈停掉（否則介面看起來像當掉）
                    log.exception("處理事件 %s 失敗", kind)
        finally:
            self._poll_job = self.root.after(50, self._poll)

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "result":
            self._log_result(payload)
            self.overlay.add_line(payload.seq, payload.text, payload.language)
        elif kind == "translated":
            self._log_translation(payload)
            self.overlay.set_translation(payload.seq, payload.translation)
        elif kind == "status":
            text = str(payload)
            self._log(text, "sys")
            if text.startswith("載入"):
                self._set_status("● 載入模型中…", "#a60")
            elif text.startswith("🎙"):
                self._set_status("● 聆聽中", "#080")
            elif text.startswith("🔄") and self.running:
                self._set_status("● 聆聽中（已重新連接）", "#080")
        elif kind == "paused":
            self.btn_pause.configure(text="▶ 繼續" if payload else "⏸ 暫停")
            self._set_status("● 已暫停" if payload else "● 聆聽中", "#a60" if payload else "#080")
        elif kind == "level":
            self.level["value"] = max(0.0, min(60.0, float(payload) + 60))
        elif kind == "pending":
            self.var_pending.set(f"辨識中 {payload} 句" if payload else "")
        elif kind == "progress":
            # 第一次下載模型：狀態列顯示進度（空字串 = 下載完成）
            if payload:
                self._set_status("● " + str(payload), "#a60")
        elif kind == "error":
            self._log(str(payload), "err")
            if "無法從網路下載" in str(payload):
                self.root.after(100, lambda m=str(payload): messagebox.showwarning(
                    "需要網路下載模型", m.replace("執行失敗：", "") + "\n\n請連上網路後再按一次「開始字幕」。"))
        elif kind == "stopped":
            self.running = False
            self._set_dictation_enabled(True)
            self.btn_start.configure(text="▶ 開始字幕", state="normal")
            self.btn_pause.configure(state="disabled", text="⏸ 暫停")
            self.level["value"] = 0
            self.var_pending.set("")
            self._set_status("● 未啟動", "#666")
            self.overlay.hide()
        elif kind == "bench":
            self._log(str(payload).strip(), "sys")
            self._set_dictation_enabled(True)
            self.btn_start.configure(state="normal")
            best = [ln for ln in str(payload).splitlines() if "cpu_threads =" in ln and "建議" in ln]
            if best:
                try:
                    self.var_threads.set(int(best[0].rsplit("=", 1)[1]))
                    self._log("已把「CPU 執行緒」填入建議值，按「儲存設定」保留。", "sys")
                except ValueError:
                    pass

    def _set_status(self, text: str, color: str) -> None:
        self.var_status.set(text)
        self.lbl_status.configure(foreground=color)

    def _log(self, text: str, tag: str = "") -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", text + "\n", tag)
        self._trim_log()
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _log_result(self, r: Any) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", f"{datetime.now():%H:%M:%S} [{LANG_TAG.get(r.language, r.language)}] ", "meta")
        self.txt.insert("end", r.text)
        self.txt.insert("end", f"  （辨識 {r.elapsed_s:.1f} 秒）\n", "meta")
        self._trim_log()
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _log_translation(self, r: Any) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", "　　→ ", "meta")
        self.txt.insert("end", r.translation, "zh")
        self.txt.insert("end", f"  （翻譯 {r.translate_s:.1f} 秒）" + chr(10), "meta")
        self._trim_log()
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _trim_log(self) -> None:
        lines = int(self.txt.index("end-1c").split(".")[0])
        if lines > MAX_LOG_LINES:
            self.txt.delete("1.0", f"{lines - MAX_LOG_LINES}.0")

    def _clear_log(self) -> None:
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")

    def _on_close(self) -> None:
        if self.thread and self.thread.is_alive():
            self.app.quit()
            self.thread.join(3)
        try:
            self.root.after_cancel(self._poll_job)  # 關閉後不要再排程更新（否則會留下錯誤訊息）
        except (tk.TclError, AttributeError):
            pass
        self.overlay.destroy()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def run_gui(cfg: Config, config_path: Path | None = None) -> int:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 高 DPI 螢幕上字不模糊
    except (AttributeError, OSError):
        pass
    from .config import CONFIG_WARNINGS

    win = ControlWindow(cfg, config_path)
    if CONFIG_WARNINGS:
        for msg in CONFIG_WARNINGS:
            win._log(msg, "err")
        text = "\n\n".join(CONFIG_WARNINGS[:5])
        win.root.after(300, lambda: messagebox.showwarning("設定檔有問題", text, parent=win.root))
    return win.run()
