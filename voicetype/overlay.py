"""大字幕視窗：無邊框、永遠在最上層、可拖曳；預設黑底白字，顏色／字級／透明度都可設定。

* 不會搶走焦點（WS_EX_NOACTIVATE），點到字幕也不會影響你正在用的視窗。
* 右鍵選單：放大／縮小字級、清除、重設位置、隱藏；Ctrl + 滾輪也能調字級。
* 翻譯時：中文大字在上、原文小字在下（show_source）；原文先出，中文翻好再補上。
* 開啟「點擊穿透」後滑鼠會直接點到後面的視窗（這時改從控制視窗調整）。
"""

from __future__ import annotations

import ctypes
import tkinter as tk
from tkinter import font as tkfont
from collections import deque
from collections.abc import Callable
from typing import Any

from .config import OverlayConfig

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
WS_EX_LAYERED = 0x80000
WS_EX_NOACTIVATE = 0x08000000
GA_ROOT = 2
LANG_TAG = {"en": "英", "ja": "日", "ko": "韓", "zh": "中"}

PRESETS = {
    "灰底白字": {"text_color": "#FFFFFF", "bg_color": "#3C3C3C", "border_color": "#3C3C3C", "border_width": 0},
    "黑底白字": {"text_color": "#FFFFFF", "bg_color": "#000000", "border_color": "#000000", "border_width": 0},
    "白底黑字＋黑框": {"text_color": "#000000", "bg_color": "#FFFFFF", "border_color": "#000000", "border_width": 4},
    "黑底黃字": {"text_color": "#FFE600", "bg_color": "#000000", "border_color": "#000000", "border_width": 0},
    "黑框黑底白字": {"text_color": "#FFFFFF", "bg_color": "#000000", "border_color": "#FFFFFF", "border_width": 3},
}


def _user32() -> ctypes.WinDLL:
    u = ctypes.windll.user32
    u.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    u.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    u.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    u.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    u.GetAncestor.restype = ctypes.c_void_p
    u.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    return u


def _no_activate(win: tk.Misc) -> None:
    """讓視窗不搶焦點、不出現在工作列。"""
    try:
        u = _user32()
        hwnd = u.GetAncestor(win.winfo_id(), GA_ROOT)
        style = u.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        u.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
    except (AttributeError, OSError):
        pass


class _PopupMenu:
    """字幕的右鍵選單（自己畫，不搶焦點）。點選項目立即執行；滑鼠移開或 6 秒沒動作自動關閉。"""

    BG, FG, HOVER, SEP = "#2B2B2B", "#FFFFFF", "#4A6FA5", "#555555"

    def __init__(self, parent: tk.Misc, items: list[tuple[str | None, Callable[[], None] | None]]) -> None:
        self._parent = parent
        self._items = items
        self._win: tk.Toplevel | None = None
        self._jobs: list[str] = []

    @property
    def is_open(self) -> bool:
        return self._win is not None

    def toggle(self, x: int, y: int) -> None:
        if self.is_open:
            self.close()
        else:
            self.open(x, y)

    def open(self, x: int, y: int) -> None:
        self.close()
        win = tk.Toplevel(self._parent)
        win.withdraw()
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=self.SEP)
        body = tk.Frame(win, bg=self.BG)
        body.pack(padx=1, pady=1)
        for label, action in self._items:
            if label is None:
                tk.Frame(body, bg=self.SEP, height=1).pack(fill="x", padx=8, pady=3)
                continue
            item = tk.Label(body, text=label, bg=self.BG, fg=self.FG, anchor="w", padx=16, pady=5,
                            font=("Microsoft JhengHei UI", 11), cursor="hand2")
            item.pack(fill="x")
            item.bind("<Enter>", lambda _e, w=item: w.configure(bg=self.HOVER))
            item.bind("<Leave>", lambda _e, w=item: w.configure(bg=self.BG))
            item.bind("<ButtonRelease-1>", lambda _e, a=action: self._choose(a))
        win.bind("<Enter>", lambda _e: self._cancel_jobs())
        win.bind("<Leave>", lambda _e: self._schedule_close(700))
        win.update_idletasks()
        # 不要超出螢幕右邊／下面
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        x = min(x, win.winfo_screenwidth() - w - 4)
        y = min(y, win.winfo_screenheight() - h - 4)
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.deiconify()
        _no_activate(win)
        self._win = win
        self._schedule_close(6000)

    def _choose(self, action: Callable[[], None] | None) -> None:
        self.close()
        if action:
            # 等選單關掉之後才執行，避免動作（例如重新排版字幕）和關選單同時進行
            self._parent.after(10, action)

    def _schedule_close(self, ms: int) -> None:
        self._cancel_jobs()
        if self._win is not None:
            self._jobs.append(self._win.after(ms, self.close))

    def _cancel_jobs(self) -> None:
        for job in self._jobs:
            try:
                self._parent.after_cancel(job)
            except tk.TclError:
                pass
        self._jobs.clear()

    def close(self) -> None:
        self._cancel_jobs()
        if self._win is not None:
            try:
                self._win.destroy()
            except tk.TclError:
                pass
            self._win = None


class SubtitleOverlay:
    def __init__(self, root: tk.Tk, cfg: OverlayConfig, on_change: Callable[[], None] | None = None) -> None:
        self.cfg = cfg
        self._on_change = on_change or (lambda: None)  # 用右鍵／滾輪改字級時通知控制視窗
        # 每句一筆：[序號, 語言, 原文, 中文翻譯]；中文翻譯晚一點才補上
        self._lines: deque[list[Any]] = deque(maxlen=max(1, cfg.max_lines))
        self._clear_job: str | None = None
        self._user_pos: tuple[int, int] | None = None  # 使用者拖曳過的位置
        self._drag: tuple[int, int] | None = None
        self._visible = False

        self.win = tk.Toplevel(root)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.frame = tk.Frame(self.win)
        self.frame.pack(fill="both", expand=True)
        # 用 Text 而不是 Label：同一個框裡可以有「大字中文＋小字原文」兩種字級
        self.text = tk.Text(self.frame, wrap="word", borderwidth=0, highlightthickness=0, cursor="arrow",
                            padx=24, pady=10, takefocus=0)
        self.text.pack(fill="both", expand=True)

        # 只綁在最外層視窗：裡面的 frame、text 的事件本來就會往上傳到視窗（bindtags），
        # 三個都綁的話同一次點擊會觸發兩次（右鍵選單開了又馬上關、滾一格字級變兩次）
        self.win.bind("<ButtonPress-1>", self._start_drag)
        self.win.bind("<B1-Motion>", self._on_drag)
        self.win.bind("<Button-3>", self._menu)
        self.win.bind("<Control-MouseWheel>", self._wheel)
        # 文字區拿掉 Text 類別的預設行為（拖曳時才不會反白選取文字），事件仍會傳到視窗
        self.text.bindtags((str(self.text), str(self.win), "all"))
        # 右鍵選單：不用 tk.Menu。Windows 原生彈出選單需要搶到前景焦點，
        # 但字幕視窗刻意設成不搶焦點（WS_EX_NOACTIVATE），兩者衝突會讓選單點了沒反應、也關不掉，看起來像當掉。
        # 所以自己用一個同樣不搶焦點的小視窗畫選單。
        self.menu = _PopupMenu(self.win, [
            ("字放大", lambda: self._bump_font(+4)),
            ("字縮小", lambda: self._bump_font(-4)),
            ("清除字幕", self.clear),
            ("位置重設到螢幕下方", self.reset_position),
            (None, None),
            ("暫時隱藏（下一句再出現）", self.clear),
        ])
        self.win.update_idletasks()
        self.apply(cfg)

    # ------------------------------------------------------------------ 外觀
    def apply(self, cfg: OverlayConfig) -> None:
        """套用設定（可在執行中即時修改）。"""
        self.cfg = cfg
        if self._lines.maxlen != max(1, cfg.max_lines):
            self._lines = deque(self._lines, maxlen=max(1, cfg.max_lines))
        weight = "bold" if cfg.bold else "normal"
        # 外框做法：視窗本身塗外框色（圓角），內層 frame 塗底色、四邊內縮 border_width（也是圓角）
        b = max(0, cfg.border_width)
        self.frame.configure(bg=cfg.bg_color, highlightthickness=0)
        self.frame.pack_configure(padx=b, pady=b)
        small = max(10, int(cfg.font_size * cfg.source_font_ratio))
        # 預設字型設成小字：行與行之間的換行字元不會把原文那一行撐成大字的高度
        self.text.configure(bg=cfg.bg_color, fg=cfg.text_color, font=(cfg.font_family, small))
        self.text.tag_configure("main", font=(cfg.font_family, cfg.font_size, weight), justify="center",
                                foreground=cfg.text_color, spacing1=2)
        self.text.tag_configure("src", font=(cfg.font_family, small), justify="center",
                                foreground=cfg.text_color, spacing1=2, spacing3=6)
        self.win.configure(bg=cfg.border_color if b else cfg.bg_color)
        self.win.attributes("-alpha", min(1.0, max(0.3, cfg.opacity)))
        self._apply_exstyle()
        self._render()

    def _apply_round_corners(self, w: int, h: int) -> None:
        """用 Windows 視窗區域（SetWindowRgn）把字幕框裁成圓角；外框那層和底色那層各裁一次。"""
        r = max(0, int(self.cfg.corner_radius))
        b = max(0, self.cfg.border_width)
        try:
            u = _user32()
            gdi = ctypes.windll.gdi32
            gdi.CreateRoundRectRgn.restype = ctypes.c_void_p
            u.SetWindowRgn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            outer = u.GetAncestor(self.win.winfo_id(), GA_ROOT)
            inner = self.frame.winfo_id()
            for hwnd, width, height, radius in ((outer, w, h, r), (inner, w - 2 * b, h - 2 * b, max(0, r - b))):
                if radius > 0:
                    # CreateRoundRectRgn 的右下角不含在內，所以 +1；最後兩個參數是圓角的直徑
                    rgn = gdi.CreateRoundRectRgn(0, 0, width + 1, height + 1, 2 * radius, 2 * radius)
                    u.SetWindowRgn(hwnd, rgn, True)  # 成功後區域由系統接管，不用自己釋放
                else:
                    u.SetWindowRgn(hwnd, None, True)
        except (AttributeError, OSError):
            pass

    def _apply_exstyle(self) -> None:
        try:
            u = _user32()
            hwnd = u.GetAncestor(self.win.winfo_id(), GA_ROOT)
            style = u.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED
            if self.cfg.click_through:
                style |= WS_EX_TRANSPARENT
            else:
                style &= ~WS_EX_TRANSPARENT
            u.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
        except (AttributeError, OSError):
            pass

    def _width(self) -> int:
        return int(self.win.winfo_screenwidth() * min(1.0, max(0.2, self.cfg.width_ratio)))

    # ------------------------------------------------------------------ 內容
    def add_line(self, seq: Any, text: str, language: str = "", translation: str = "") -> None:
        """新增一句（原文先出；翻譯好之後用 set_translation 補上中文）。"""
        self._lines.append([seq, language, text, translation])
        self._render()
        self._restart_clear_timer()

    def set_translation(self, seq: Any, translation: str) -> None:
        for entry in self._lines:
            if entry[0] == seq:
                entry[3] = translation
                self._render()
                self._restart_clear_timer()
                return

    def show_text(self, text: str, language: str = "") -> None:
        self.add_line(None, text, language)

    def _restart_clear_timer(self) -> None:
        if self._clear_job:
            self.win.after_cancel(self._clear_job)
            self._clear_job = None
        if self.cfg.clear_after_s > 0:
            self._clear_job = self.win.after(int(self.cfg.clear_after_s * 1000), self.clear)

    def clear(self) -> None:
        self._lines.clear()
        self._render()

    def _render(self) -> None:
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for i, (_seq, lang, src, zh) in enumerate(self._lines):
            if i:
                t.insert("end", "\n")
            tag = f"[{LANG_TAG.get(lang, lang)}] " if self.cfg.show_language and lang else ""
            if zh:
                t.insert("end", tag + zh, "main")
                if self.cfg.show_source:
                    t.insert("end", "\n" + src, "src")
            else:
                t.insert("end", tag + src, "main")
        t.configure(state="disabled")
        if not self._visible:
            return
        if not self._lines:
            self.win.withdraw()  # 沒字時不留一條空黑框擋畫面
            return
        self.win.deiconify()
        self._place()
        self.win.attributes("-topmost", True)

    def _place(self) -> None:
        w = self._width()
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        x = self._user_pos[0] if self._user_pos else (sw - w) // 2
        # 先只把寬度設好（自動換行要依寬度排版），高度維持原樣，不會閃出一個大框
        self.win.geometry(f"{w}x{max(self.win.winfo_height(), 1)}+{x}+{self.win.winfo_y()}")
        self.win.update_idletasks()
        # 依排版結果算高度（中文自動換行、兩種字級都算進去）：
        # count -ypixels 量到「最後一行的頂端」，再加上最後一行本身的行高
        t = self.text
        px = t.count("1.0", "end-1c", "update", "ypixels")
        top_of_last = int(px[0] if isinstance(px, tuple) else (px or 0))
        tag = "src" if "src" in t.tag_names("end-2c") else "main"
        line_font = tkfont.Font(root=self.win, font=t.tag_cget(tag, "font"))
        spacing = sum(int(t.tag_cget(tag, k) or 0) for k in ("spacing1", "spacing3"))
        h = 10 + top_of_last + line_font.metrics("linespace") + spacing + 10 + 2 * max(0, self.cfg.border_width)
        # 位置以「底端」為準：字變多時往上長，不會長出螢幕底部
        bottom = self._user_pos[1] if self._user_pos else sh - self.cfg.bottom_margin
        y = max(0, bottom - h)  # 太高時頂端貼齊螢幕上緣
        self.win.geometry(f"{w}x{h}+{x}+{y}")
        self.win.update_idletasks()
        self.text.yview_moveto(0)
        self._apply_exstyle()
        self._apply_round_corners(w, h)

    # ------------------------------------------------------------------ 顯示／隱藏
    def show(self) -> None:
        self._visible = True
        self._render()

    def hide(self) -> None:
        self._visible = False
        self.win.withdraw()

    def preview(self) -> None:
        self.show()
        self.add_line("p1", "This is a subtitle preview.", "en", "這是字幕預覽。")
        self.add_line("p2", "今日の会議は三時から始めます。", "ja", "今天的會議三點開始。")

    def reset_position(self) -> None:
        self._user_pos = None
        self._render()

    def destroy(self) -> None:
        if self._clear_job:
            self.win.after_cancel(self._clear_job)
            self._clear_job = None
        self.menu.close()
        self.win.destroy()

    # ------------------------------------------------------------------ 滑鼠
    def _start_drag(self, e: tk.Event) -> None:
        self._drag = (e.x_root - self.win.winfo_x(), e.y_root - self.win.winfo_y())

    def _on_drag(self, e: tk.Event) -> None:
        if self._drag:
            x, y = e.x_root - self._drag[0], e.y_root - self._drag[1]
            # 記住「左緣、底端」：之後字變多會往上長，而不是長出螢幕
            self._user_pos = (x, y + self.win.winfo_height())
            self.win.geometry(f"+{x}+{y}")

    def _menu(self, e: tk.Event) -> None:
        self.menu.toggle(e.x_root, e.y_root)

    def _wheel(self, e: tk.Event) -> None:
        self._bump_font(2 if e.delta > 0 else -2)

    def _bump_font(self, delta: int) -> None:
        self.cfg.font_size = min(120, max(12, self.cfg.font_size + delta))
        self.apply(self.cfg)
        self._on_change()
