"""Pytest bootstrap: ensure the repository root (containing the `selftalk`
package) is importable when tests are run from this directory."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
