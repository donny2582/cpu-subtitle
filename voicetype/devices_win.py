"""讀取 Windows 目前的「預設喇叭」與「預設麥克風」（Core Audio：IMMDeviceEnumerator）。

為什麼不用 PyAudio：PortAudio 只在初始化時讀一次裝置清單，執行中換了預設裝置它不會知道。
這裡直接問 Windows，只用 ctypes（不需要額外套件），回傳裝置 ID 字串，用來判斷「預設裝置有沒有換掉」。
"""

from __future__ import annotations

import ctypes
import uuid
from ctypes import POINTER, byref, c_void_p, wintypes

_CLSID_MMDeviceEnumerator = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMMDeviceEnumerator = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_CLSCTX_ALL = 23
E_RENDER, E_CAPTURE = 0, 1   # 喇叭／麥克風
E_CONSOLE = 0                # Windows「預設裝置」
_E_NOTFOUND = 0x80070490     # 沒有任何這類裝置（例如麥克風全拔掉）


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    return _GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


def _method(obj: c_void_p, index: int, *argtypes: object) -> ctypes._CFuncPtr:
    """取得 COM 物件第 index 個方法（vtable）。"""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vtbl[index])


def _release(obj: c_void_p) -> None:
    if obj:
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
        ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(vtbl[2])(obj)


_initialized = __import__("threading").local()


def _ensure_com() -> None:
    """每個執行緒要先初始化 COM 一次。"""
    if not getattr(_initialized, "done", False):
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED；已初始化過也沒關係
        _initialized.done = True


def default_device_id(flow: int) -> str | None:
    """目前 Windows 預設裝置的 ID；沒有這類裝置時回傳 None。flow = E_RENDER（喇叭）或 E_CAPTURE（麥克風）。"""
    _ensure_com()
    ole32 = ctypes.windll.ole32
    enum = c_void_p()
    hr = ole32.CoCreateInstance(byref(_guid(_CLSID_MMDeviceEnumerator)), None, _CLSCTX_ALL,
                                byref(_guid(_IID_IMMDeviceEnumerator)), byref(enum))
    if hr != 0:
        raise OSError(f"CoCreateInstance 失敗：0x{hr & 0xFFFFFFFF:08X}")
    device = c_void_p()
    try:
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint 是第 4 個方法（前 3 個是 IUnknown）
        get_default = _method(enum, 4, ctypes.c_int, ctypes.c_int, POINTER(c_void_p))
        try:
            get_default(enum, flow, E_CONSOLE, byref(device))
        except OSError as exc:  # HRESULT 失敗會變成 OSError
            if (exc.winerror or 0) & 0xFFFFFFFF == _E_NOTFOUND or "0x80070490" in str(exc):
                return None
            raise
        # IMMDevice::GetId 是第 5 個方法
        get_id = _method(device, 5, POINTER(wintypes.LPWSTR))
        text = wintypes.LPWSTR()
        get_id(device, byref(text))
        try:
            return text.value
        finally:
            ole32.CoTaskMemFree(text)
    finally:
        _release(device)
        _release(enum)
