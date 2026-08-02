"""Fixed-seed stratified Residual Buffer cap, not loss-only sampling."""

import random
from collections import defaultdict


def stratified_residual_sample(records, capacity: int = 512, seed: int = 42):
    rows = list(records)
    if len(rows) <= capacity:
        return rows, []
    strata = defaultdict(list)
    for row in rows:
        confidence_bucket = min(3, int(float(row.predicted_sufficiency) * 4))
        gain = max(0.0, float(row.residual_gain))
        gain_bucket = 0 if gain < 0.02 else 1 if gain < 0.10 else 2 if gain < 0.50 else 3
        strata[(confidence_bucket, gain_bucket, row.answer_type, row.question_subtype)].append(row)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
        values.sort(key=lambda row: (row.predicted_sufficiency, -row.residual_gain, row.sample_id))
    retained = []
    keys = sorted(strata, key=str)
    while len(retained) < capacity and any(strata.values()):
        for key in keys:
            if strata[key] and len(retained) < capacity:
                retained.append(strata[key].pop(0))
    retained_ids = {row.sample_id for row in retained}
    discarded = [row for row in rows if row.sample_id not in retained_ids]
    return retained, discarded
