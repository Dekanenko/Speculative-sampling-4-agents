# Code Quality Rules

These rules apply to every Python file written in this project.

## Docstrings
- All functions and classes must have **Google-style docstrings**.
- Docstrings describe purpose, arguments, returns, and raised exceptions.
- Module-level docstrings explain the module's single responsibility.

## Type Hints
- **Type hints are required on every function signature** (arguments and return).
- Use `from __future__ import annotations` where appropriate.
- Prefer precise types (`list[int]`, `dict[str, Tensor]`) over `Any`.

## Constants and Configuration
- **No magic numbers in code.** All constants live in a dedicated config
  file or a frozen dataclass (e.g., `src/config.py`).
- Hyperparameters, thresholds, and paths must be named and centralised.

## Module Structure
- **One responsibility per module.** If a file grows beyond its stated
  purpose, split it.
- Keep files focused and short enough to hold in one mental model.

## Imports
- **No unused imports committed.** Run a linter before committing.
- Group imports: stdlib, third-party, local — separated by blank lines.

## Tests
- Tests live in `tests/` and **mirror the `src/` structure exactly**.
  For every `src/foo/bar.py`, tests live in `tests/foo/test_bar.py`.
- Every public function should have at least one unit test.

## Function Length
- **Maximum function length: 50 lines.** If exceeded, extract helpers.
- Long functions are a smell; prefer composition over sprawling logic.

## Development Environment
- **Always activate the `sda` conda environment before running any
  code or installing any libraries.** Every shell command that touches
  Python, pip, pytest, or model downloads must be prefixed with
  `conda activate sda &&`.
- Never install packages into the base environment.
- If a command fails because the environment is not active, the fix
  is to activate it — not to work around it.
