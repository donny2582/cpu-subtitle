"""全域快捷鍵（Windows RegisterHotKey），在背景執行緒跑訊息迴圈。"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable
from ctypes import wintypes

log = logging.getLogger(__name__)

MOD = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
NAMED_VK = {
    "space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "pause": 0x13,
    "insert": 0x2D, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "scrolllock": 0x91, "capslock": 0x14,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
}


def parse_hotkey(spec: str) -> tuple[int, int]:
    """"ctrl+alt+space" → (修飾鍵, 虛擬鍵碼)。"""
    mods, vk = 0, 0
    for part in (p.strip().lower() for p in spec.split("+") if p.strip()):
        if part in MOD:
            mods |= MOD[part]
        elif part in NAMED_VK:
            vk = NAMED_VK[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        else:
            raise ValueError(f"看不懂的快捷鍵：{spec}")
    if not vk:
        raise ValueError(f"快捷鍵缺少主要按鍵：{spec}")
    return mods, vk


class HotkeyListener:
    def __init__(self, bindings: dict[str, Callable[[], None]]) -> None:
        self._bindings = bindings
        self._thread_id = 0
        self._thread: threading.Thread | None = None
        self.registered: list[str] = []

    def start(self) -> None:
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="hotkey", daemon=True)
        self._thread.start()
        ready.wait(2)

    def stop(self) -> None:
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(1)  # 等快捷鍵真的取消註冊，馬上重新開始時才註冊得到

    def _run(self, ready: threading.Event) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        actions: dict[int, Callable[[], None]] = {}
        for i, (spec, fn) in enumerate(self._bindings.items(), start=1):
            try:
                mods, vk = parse_hotkey(spec)
            except ValueError as exc:
                log.warning("%s（請檢查 config.toml 的 [hotkey]）", exc)
                continue
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                actions[i] = fn
                self.registered.append(spec)
            else:
                log.warning("快捷鍵 %s 註冊失敗（可能被其他程式占用）", spec)
        ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam in actions:
                    try:
                        actions[msg.wParam]()
                    except Exception:  # noqa: BLE001
                        log.exception("快捷鍵動作失敗")
        finally:
            for i in actions:
                user32.UnregisterHotKey(None, i)
