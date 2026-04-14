"""Pytest session-level setup.

Sets ``KMP_DUPLICATE_LIB_OK=TRUE`` before any test module imports
``torch``. On macOS with miniforge, both ``torch`` and ``numpy``
ship their own libomp, and pytest's test collection triggers a
hard abort on the duplicate-library check. The env var is a
no-op on Linux (the EC2 dev server) where only one libomp is
linked into the process, so this is safe for CI too.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
