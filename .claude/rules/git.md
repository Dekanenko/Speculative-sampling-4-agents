# Git Rules

## Atomic Commits
- **Commits are atomic: one logical change per commit.**
- If a diff contains two unrelated ideas, split it into two commits.

## Commit Messages
- **Imperative mood** ("Add trajectory schema", not "Added trajectory schema").
- Subject line **≤ 72 characters**.
- Blank line after the subject.
- Body explains **WHY**, not WHAT — the diff already shows the what.

## Branch Naming
Use one of the following prefixes:
- `feature/` — new functionality
- `fix/`     — bug fixes
- `experiment/` — exploratory or measurement branches
- `docs/`    — documentation-only changes

## Main Branch Hygiene
- **No commented-out code on main.** Delete it; git remembers.
- **No hardcoded credentials or model paths.** Use environment variables
  and `.env` files (gitignored).

## Experiment Branches
- **Every experiment branch PR description includes the MLflow run ID**
  that produced its results. No exceptions.
