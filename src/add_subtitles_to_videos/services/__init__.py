from __future__ import annotations


class OperationCancelledError(RuntimeError):
    """Raised when the user stops the current subtitle job."""


__all__ = ["OperationCancelledError"]
