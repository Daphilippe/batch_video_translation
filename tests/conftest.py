"""Shared test configuration — stub optional dependencies.

This conftest is loaded by pytest before any test module, so the
stubs are guaranteed to be in ``sys.modules`` regardless of
collection order or which optional packages are installed.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub optional packages that are not always installed.
#
# The guards (``if ... not in sys.modules``) ensure that:
# - On machines WITHOUT the package → a MagicMock is injected
# - On machines WITH the package    → the real module is kept
#
# This makes the test suite runnable in both environments.
# ---------------------------------------------------------------------------

if "deep_translator" not in sys.modules:
    sys.modules["deep_translator"] = MagicMock()

for _mod_name in ("pywinauto", "pywinauto.keyboard", "win32api", "pyperclip"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
