from __future__ import annotations

import ctypes
import sys


class SleepInhibitor:
    def activate(self) -> bool:
        return False

    def release(self) -> bool:
        return False


class NullSleepInhibitor(SleepInhibitor):
    pass


class WindowsSleepInhibitor(SleepInhibitor):
    _ES_CONTINUOUS = 0x80000000
    _ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self._kernel32 = ctypes.windll.kernel32
        self._active = False

    def activate(self) -> bool:
        if self._active:
            return False

        result = self._kernel32.SetThreadExecutionState(
            self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED
        )
        if result == 0:
            raise OSError("SetThreadExecutionState failed while enabling sleep prevention.")
        self._active = True
        return True

    def release(self) -> bool:
        if not self._active:
            return False

        result = self._kernel32.SetThreadExecutionState(self._ES_CONTINUOUS)
        if result == 0:
            raise OSError("SetThreadExecutionState failed while releasing sleep prevention.")
        self._active = False
        return True


def create_sleep_inhibitor() -> SleepInhibitor:
    if sys.platform.startswith("win"):
        return WindowsSleepInhibitor()
    return NullSleepInhibitor()
