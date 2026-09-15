# V9-S UCIT query-cache training adaptation

## Delivered

- `--query-cache-manifest` accepts the V7/UCIT cache manifest or its directory.
- V9 resolves task/split `queries.pt` through the V7 loader before candidate initialization and historical Top-C construction.
- The loader checks manifest-bound split contract, tensor fingerprint, and exact declared-data ID coverage. Manifest mode has no JSON or live-encoder fallback.
- Training receives `--compose_v7_query_tensor`; an explicitly requested tensor that is unavailable is a hard error.
- Calibration resolves the matching manifest `val` split and feeds its tensor directly.
- The task resume contract records manifest SHA-256, tensor value fingerprint, path, task/split, and sample count; a changed source refuses resume.

## Verification

- `python -m py_compile compose/v9/data.py compose/experiments/v9_task_run.py compose/train/train_compose.py` passed.
- The host default Python lacks `pytest` and `transformers`, so GPU-free pytest execution could not run on this host. The new test module is included for the project training environment.
- No query encoder or model was started; no cache artifact was mutated.
