"""Thin backwards-compatibility wrapper over ``scripts/run_family.py``.

The Phase 1 mocks pipeline is now one of many families. This script
exists so older invocations (``run_phase1.py --model ... --profile
...``) keep working; under the hood it just delegates to
``run_family.py`` with ``--family mocks --split phase1-v0``.
"""

from __future__ import annotations

import sys

from run_family import main as run_family_main


def main() -> None:
    """Forward to run_family.main with mocks defaults injected."""
    # Inject --family and --split if the caller didn't already pass them.
    args = sys.argv[1:]
    if "--family" not in args:
        args = ["--family", "mocks", *args]
    if "--split" not in args:
        args = ["--split", "phase1-v0", *args]
    sys.argv = [sys.argv[0], *args]
    run_family_main()


if __name__ == "__main__":
    main()
