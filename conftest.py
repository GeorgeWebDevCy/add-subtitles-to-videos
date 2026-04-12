"""Ensure the worktree's own src/ is imported instead of the shared venv editable install."""
import sys
from pathlib import Path

_worktree_src = str(Path(__file__).parent / "src")
if _worktree_src not in sys.path:
    sys.path.insert(0, _worktree_src)
