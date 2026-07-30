# ADR-0730: Require a fixed-set Oracle before Set Router work

## Status

Accepted. The Oracle-first workflow is retained and Set Router development is
stopped under stop condition C for the evaluated two-expert pool.

## Context

A learned router cannot establish that composing experts is useful: it can
hide candidate quality, selection, and optimization failures behind one
aggregate metric. The current experts are task-trained functional proxies, so
their usefulness must first be measured with exhaustive empty/single/pair
teacher-forced NLL under a fixed, reproducible gate contract.

## Decision

The project will cache raw per-sample NLL for every stable candidate set before
any router is trained. Pair candidates use explicit L2 normalization. Router
work is allowed only if the fixed-set audit is deterministic and matched
rank-16 controls show that pair gains are not merely additional capacity.

No query encoder, expert key, interaction MLP, router loss, or router training
is implemented by this change.

## Evidence and gate

The two-expert Oracle is deterministic and completed 6,000 samples. It reports
30.60% PairOracleRate, but mean and median synergy are negative. The matched
rank-16 control has exactly the same 39,976,960 adapter parameters, lower mean
NLL (0.11385490 versus 0.16488876), and higher generation accuracy (86.5333%
versus 80.9667%). The pair beats rank-16 NLL on only 22.9500% of samples.

This evidence triggers stop condition C. No Compose Set Router types, query
encoder, expert keys, interaction model, loss, or training entry are added.

If a future experiment seeks to reopen Router work, its supervision must come
from a deterministic fixed-set per-sample NLL cache with immutable checkpoint
and configuration IDs. At minimum it must reproduce positive mean and median
synergy, beat a parameter-matched single adapter on both mean NLL and accuracy,
and show that the pair advantage is not confined to a small task-specific
minority. Until those gates pass, Router work remains out of scope.
