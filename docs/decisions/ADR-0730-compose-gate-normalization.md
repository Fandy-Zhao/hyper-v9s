# ADR-0730: Explicit Compose Gate Normalization

- Status: Accepted
- Date: 2026-07-30
- Branch: `exp/0730-compose-task1-oracle`

## Context

`ComposeSelection` previously normalized every sample's gates by their L1 sum. That
made a requested top-2 sum such as `[1.0, 1.0]` silently equivalent to
`[0.5, 0.5]`, obscured experiment semantics, and prevented an inverse-square-root
capacity-preserving combination from being represented directly.

The original `ComposeLinear` also evaluated every selected expert on the full
batch. With heterogeneous per-sample selections this wasted compute and allowed
inactive rows to pass through expert dropout and matrix multiplications.

## Decision

Gate normalization is an explicit `ComposeSelection` contract with exactly three
modes:

- `none`: preserve the supplied non-negative gates exactly.
- `l1`: divide each sample's gates by their sum.
- `l2`: divide each sample's gates by their Euclidean norm.

The default is `none`. Convenience APIs use unit gates when no gates are supplied,
so a default two-expert selection is additive. A capacity-preserving equal top-2
selection is expressed with unit gates and `normalization="l2"`, yielding
`1 / sqrt(2)` for each expert.

Every sample must contain at least one positive finite gate. Top-2 expert IDs must
be distinct within a sample. A zero gate is permitted for padding or an inactive
second slot, but that expert does not execute for the sample.

`ComposeLinear` groups execution by expert, gathers only rows with positive gates
for that expert, runs the expert on that subset, applies the corresponding gates,
and scatters the result back with `index_add_`. The frozen base layer continues to
run once on the full batch.

## Consequences

- Existing callers that relied on implicit L1 normalization must request `l1`
  explicitly.
- Experiment configurations and logs expose `compose_gate_normalization`.
- Top-1 behavior is unchanged for a unit gate.
- Sparse mixed batches avoid full-batch expert computation while retaining
  autograd and bf16 behavior.
- Checkpoint manifests remain unchanged because selection is runtime state, not
  expert weight state.

## Validation

- 30 Compose unit tests pass under Python 3.10 / PyTorch 2.3.1.
- Mixed top-1/top-2 outputs and gradients match a per-sample reference.
- A bf16 CUDA forward/backward smoke on an RTX 4090 produced zero maximum absolute
  difference from the reference for both output and input gradient.
- Forward hooks confirmed expert row counts `{0: 1, 1: 2, 2: 1, 3: 1, 4: 0}`;
  the unselected expert was never called.
