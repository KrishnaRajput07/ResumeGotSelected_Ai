# conftest.py — root-level pytest configuration
#
# Problem: D:\python-packages is injected into sys.path at interpreter startup
# via the PYTHONPATH environment variable.  That directory contains a broken
# numpy 2.3.5 (missing compiled C extensions) which shadows the venv's working
# numpy 2.4.6 and causes an ImportError on every test that touches numpy.
#
# Fix: remove any sys.path entry that:
#   (a) is not inside the project's .venv, AND
#   (b) contains Python packages (looks like a site-packages or flat packages dir)
#
# This runs at conftest import time, before any test module is collected.

import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
_VENV_ROOT = _PROJECT_ROOT / ".venv"

# Known external paths to always strip (adjust if your setup differs)
_EXTERNAL_BLOCKLIST = {
    pathlib.Path(r"D:\python-packages"),
}

def _strip_external_paths() -> None:
    cleaned = []
    for entry in sys.path:
        if not entry:
            cleaned.append(entry)  # keep "" (cwd placeholder)
            continue
        p = pathlib.Path(entry).resolve()
        # Reject paths that are on the explicit blocklist
        if p in _EXTERNAL_BLOCKLIST:
            continue
        # Reject any site-packages directory that is NOT inside the project venv
        if "site-packages" in p.parts:
            try:
                p.relative_to(_VENV_ROOT)
            except ValueError:
                continue  # not under venv → strip it
        cleaned.append(entry)
    sys.path[:] = cleaned


_strip_external_paths()
