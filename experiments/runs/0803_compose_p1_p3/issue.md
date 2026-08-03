# Compose P1-P3 Formal Validation

## Goal

Validate Compose: Keyed Functional Expert Composition through P1 composition,
P2 keyed candidate expert pools, and P3 answer-free Query-Key routing, with
automatic stage gates and reproducible artifacts.

## Scope

- Reuse verified Stage 03-07 code, datasets, caches, and expert checkpoints.
- Implement C0-C3 composition exactly as specified, including arithmetic-mean
  layer RMS calibration and validation-only scalar calibration.
- Add a resumable dynamic queue with one large-model task per GPU. Prefer
  physical GPUs 0-3 and use 4-7 only under the user's explicit fallback authorization.
- Produce per-stage metrics, bootstrap intervals, decisions, reports, and final
  reproduction artifacts under `outputs/compose_p1_p3_20260803T090000Z/`.
- Continue to P2/P3 only at the level allowed by the preceding gate.

## Non-scope

Hyperbolic routing, Shadow Update, token routing, expert merging, more than two
candidate slots, end-to-end backbone updates, test-label threshold tuning, and
full CoIN/UCIT continual benchmarks.

## Acceptance criteria

- Repository and asset audit is complete and recorded.
- Required unit/integration/syntax/smoke checks pass before formal execution.
- P1 reports all required metrics across every available verified checkpoint seed.
- P1/P2/P3 decisions use the frozen user-supplied gates without result-driven edits.
- Every task records command, config, seed, CUDA device, checkpoint, stdout,
  stderr, exit code, completeness status, elapsed time, and peak memory.
- Negative and failed results are retained.

## Risks

- GPUs 0-3 are occupied by long-running external processes, so formal runs may
  need the authorized 4-7 fallback.
- P1 can fail scientifically even when implementation and execution are correct.
