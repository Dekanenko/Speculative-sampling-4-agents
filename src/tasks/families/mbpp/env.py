"""Per-task filesystem sandbox for the MBPP family.

Each MBPP task runs inside a fresh temporary directory. Only the
sandbox's own path resolver is trusted to turn agent-provided paths
into absolute paths — directly passing through ``open(path)`` would
let the agent escape the sandbox with ``..`` or absolute paths and
read or clobber unrelated files. The resolver therefore rejects both.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


# Name of the hidden pytest file written into each sandbox. The agent
# is told the name but not the contents; ``read_file`` is not blocked
# from reading it (the agent could still try) — hiding is enforced by
# convention in the task prompt, not in code.
HIDDEN_TEST_FILENAME: str = "test_solution.py"


class TempDirSandbox:
    """A per-task temporary directory with path-escape protection.

    Attributes:
        root: Absolute path to the sandbox root directory.
        hidden_test_file: Absolute path to the hidden pytest file.
    """

    def __init__(self) -> None:
        """Create a fresh temp directory for a single task run."""
        self.root: Path = Path(tempfile.mkdtemp(prefix="mbpp_sandbox_"))
        self.hidden_test_file: Path = self.root / HIDDEN_TEST_FILENAME

    def resolve(self, rel_path: str) -> Path:
        """Resolve a relative path inside the sandbox.

        Args:
            rel_path: Agent-provided path. Must be relative and must
                not escape the sandbox root via ``..`` components.

        Returns:
            Absolute resolved path inside the sandbox.

        Raises:
            ValueError: If the path is absolute or escapes the sandbox.
        """
        if not isinstance(rel_path, str) or rel_path == "":
            raise ValueError("path must be a non-empty string")
        candidate = Path(rel_path)
        if candidate.is_absolute():
            raise ValueError("absolute paths are not allowed")
        # Resolve against root and check the result stays inside root.
        resolved = (self.root / candidate).resolve()
        root_resolved = self.root.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError("path escapes sandbox root") from exc
        return resolved

    def teardown(self) -> None:
        """Remove the temp directory tree. Safe to call multiple times."""
        shutil.rmtree(self.root, ignore_errors=True)
