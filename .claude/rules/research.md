# Research Notes Rules

The project accumulates findings faster than any single
`progress.md` entry can hold. Every measurement, hypothesis test,
and direction change that a future session needs to know about
goes into `docs/research/`, not into chat or scratch files.

## When to write a research note

- A measurement pass completes (acceptance rates, speedup sim,
  eval distributions, ...).
- A hypothesis is tested (confirmed or refuted) — either
  outcome is worth writing down.
- An analysis reveals something that updates project direction.
- A decision is made that depends on data.

Routine in-progress code changes stay in `CLAUDE.md` changelog
and `docs/progress.md`. Research notes are for durable findings
with numbers behind them.

## Where notes live

`docs/research/YYYY-MM-DD-<slug>.md`. One note per
session-level milestone. `docs/research/README.md` is the
master index — add a one-line entry, newest first, every time
a new note is created.

## Structure

Every note has the following sections, in this order:

- **Context** — what question was being answered and why.
- **What got built** — the specific scripts / measurements /
  datasets that produced the numbers (with file paths).
- **Measurements** — raw numbers as tables. Prose won't do.
- **Findings** — numbered, each with a one-line headline.
- **Interpretation** — what the numbers mean at the project
  level, including uncomfortable truths.
- **Next steps** — what to do next given this data.

## Self-containment

Each note must be readable in isolation. A future session should
not need to scroll through prior notes to understand any single
finding. Repeat the relevant context inside each note; link to
the earlier note only as a pointer, not as a prerequisite.

## Honesty

Research notes include findings that weaken the hypothesis the
project was built around. If a measurement refutes the working
assumption, record that clearly — do not bury it. The research
value of a finding does not depend on which direction it points.
