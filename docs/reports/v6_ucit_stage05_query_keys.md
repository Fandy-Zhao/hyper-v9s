# Hyper-LLaVA V6 Stage 05 — Multimodal Query and Learnable Expert Keys

Status: **PASSED**. Git: this report is contained by the atomic Stage 05 commit; source parent `2ab34c6042751744c0b53ef6be173c35b839f54a`.

## Frozen configuration and provenance

- Query: frozen CLIP image projection + question-only text projection, gated fusion, LayerNorm, 128-D L2-normalized output.
- Supervision: historical-only Oracle-Direct; post-task Direct is used only to initialize each expert key on its creation task.
- Scale: 128 train + 32 validation per task (768/192 total), seed 42. The original 32 train rows/task are reused and only 96 missing train + 32 validation rows/task were scored.
- Config SHA256: `3d4e6eef78c649e4539b8ed8c94465bccca27879ab1cbaf21ed179f4ba0b1734`.
- Dataset manifest hash: `2982f01bc8f8d29a3a6ab648286d1ee399f76319c3031649add99db4dd5af3cf`; Oracle cache hash: `779af3046f12b6293d94a329ee837216598a94dbb5f51e0f6fb5874c6f328d5f`.
- Continual checkpoint SHA256: `8b8e3837eef4ac4cf52b20652730d5cf6acfe801f48bec5cbfe8be6836f02439`.
- GPU: preferred 0–3 were occupied by another user; actual 4–7, maximum four GPUs.
- Environment: `{"cuda": "11.8", "peft": "0.4.0", "python": "3.10.20", "pytorch": "2.3.1+cu118", "transformers": "4.33.3"}`.

## Technical validation

- Full Compose tests: 125 passed + 8 subtests; new Stage 05 tests: 19 passed.
- Controlled Query/Key smoke: passed; checkpoint round-trip, deterministic retrieval, DDP state fingerprint, Empty/Single/Pair losses, answer-token truncation and future-key mask all passed.
- Oracle audit: `PASSED`; test data used: false; temporal violations: `0`.
- Historical label cardinalities train/validation: `{'train': {0: 398, 1: 288, 2: 82}, 'validation': {0: 92, 1: 80, 2: 20}}`.

## Continual-anchor validation metrics

```json
{
  "EmptyFalseRecallRate": 0.010869565217391304,
  "MRR": 0.04333333333333333,
  "OracleMemberRecall@1": 0.04,
  "OracleMemberRecall@2": 0.04,
  "OracleMemberRecall@4": 0.05,
  "OracleSetRecall@1": 0.04,
  "OracleSetRecall@2": 0.04,
  "OracleSetRecall@4": 0.05,
  "PairAtLeastOneRecall@4": 0.05,
  "PairBothMembersRecall@4": 0.05,
  "SingleRecall@1": 0.05,
  "average_similarity_margin": 0.07664835420291638,
  "checkpoint": "/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/checkpoints/continual_anchor.pt",
  "config_hash": "cc4ea261d7964f2b2bba0159c4ed76f730f66d3eb564805d56adcfcdb922397b",
  "expert_selection_histogram": {
    "0": 6,
    "1": 6,
    "2": 6,
    "3": 1
  },
  "key_utilization": 0.6666666666666666,
  "one_key_collapse": false,
  "task_ID_collapse_diagnostic": {
    "0": {},
    "1": {},
    "2": {},
    "3": {
      "2": 5
    },
    "4": {
      "3": 1
    },
    "5": {}
  },
  "temporal_mask_violations": 0
}
```

Offline diagnostic OracleSetRecall@4: `0.120000`; continual-minus-offline gap: `-0.070000`. Offline is a diagnostic upper bound, not the continual result.

## Key initialization

```json
{
  "0": {
    "method": "post_task_direct_mean",
    "positive_samples": 106
  },
  "1": {
    "method": "post_task_direct_mean",
    "positive_samples": 113
  },
  "2": {
    "method": "post_task_direct_mean",
    "positive_samples": 118
  },
  "3": {
    "method": "post_task_direct_mean",
    "positive_samples": 104
  },
  "4": {
    "method": "post_task_direct_mean",
    "positive_samples": 112
  },
  "5": {
    "method": "post_task_direct_mean",
    "positive_samples": 93
  }
}
```

## Limitations

Retrieval quality is a method result rather than an engineering pass criterion. Task-ID and one-key collapse are diagnostics only and never Router inputs. The LLaVA backbone, vision tower, text tower, and LoRA experts remain frozen; only Query Encoder and Expert Keys are optimized. No test feature, answer token, task ID, Oracle set, checkpoint name, or test accuracy enters Query.
