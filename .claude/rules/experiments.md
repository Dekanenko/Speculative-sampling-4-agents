# Experiment Rules

These rules exist so that every number in this project is reproducible.

## MLflow Logging
- **Every experiment run is logged to MLflow with full hyperparameters.**
- Run parameters include: model pair, seeds, dataset version, Docker tag,
  and the git commit SHA.

## Run IDs
- **No result is reported without the MLflow run ID that produced it.**
  This applies to PR descriptions, blog posts, and the preprint.

## Random Seeds
- **Random seeds are always fixed and logged** for `torch`, `numpy`,
  and Python's `random` module.
- Any additional sources of randomness (e.g. CUDA) must be documented.

## Dataset Versioning
- **Datasets are versioned — never use a mutable "latest" pointer.**
- Tag dataset versions explicitly (e.g., `trajectories-v1`, `v2`).

## Evaluation
- **Evaluation metrics are computed on held-out sets only.**
- Training/validation/test splits are fixed and documented.

## Before/After Comparisons
- **Before/after comparisons must use identical task sets and seeds.**
- If the task set changes, the comparison is invalidated.

## Container Reproducibility
- **The Docker image tag is recorded alongside every set of results.**
- Results produced outside the canonical Docker image are not publishable.
