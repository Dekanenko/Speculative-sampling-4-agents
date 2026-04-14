# Documentation Rules

Documentation in this project is **alive**: it is updated in the same
session — and the same commit — as the code it describes.

## Session Discipline
- **Every session ends with the `CLAUDE.md` changelog section updated.**
  Add a new row summarising what was done.
- If the session changed the architecture, update `docs/architecture.md`
  before closing the session.

## Architectural Decisions
- **Every architectural decision is recorded in `docs/decisions.md`**
  in the same session it is made — never retroactively.
- Use the ADR format already established in that file.
- If a decision is reversed, add a new ADR that supersedes the old one;
  do not rewrite history.

## Progress Tracking
- `docs/progress.md` is updated whenever:
  - A phase milestone is reached.
  - A significant implementation decision is made.
  - A deliverable is produced.

## Design Drift
- **If a design changes from what is documented, update the doc in the
  same commit as the code change.** Never leave docs trailing the code.

## Public API
- Public functions get docstrings **before the PR is merged**, not after.

## README
- `README.md` must always reflect the current state of the project —
  phase, how to run it, how to reproduce results.
