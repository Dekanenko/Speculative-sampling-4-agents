"""MBPP family — Python coding tasks in a sandboxed temp directory.

The MBPP family wraps the Mostly Basic Python Problems dataset behind
the :class:`TaskFamily` interface. Each task asks the agent to write a
Python function that passes a hidden pytest file. Tools expose file IO
inside a per-task temp directory and a ``run_tests`` subprocess runner.
"""

from .env import TempDirSandbox
from .family import MbppFamily
from .tools import build_coding_tools

__all__ = ["MbppFamily", "TempDirSandbox", "build_coding_tools"]
