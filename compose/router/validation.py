"""Static and runtime leakage/temporal validation for router batches."""

FORBIDDEN_QUERY_FIELDS = frozenset({
    "answer", "answers", "target", "targets", "labels", "answer_mask", "answer_tokens",
    "task_id", "task_name", "oracle_set", "selected_expert_ids", "test_accuracy", "checkpoint_name",
})


def validate_query_payload(payload) -> None:
    forbidden = sorted(set(payload) & FORBIDDEN_QUERY_FIELDS)
    if forbidden:
        raise ValueError("Query payload contains forbidden answer/task fields: {}".format(forbidden))
    required = {"image_features", "text_features", "image_available", "text_available"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("Query payload missing fields: {}".format(missing))


def validate_temporal_targets(targets, creation_tasks, task_ids) -> None:
    for row, task_id in zip(targets, task_ids):
        future = [expert_id for expert_id in row if int(creation_tasks[expert_id]) >= int(task_id)]
        if future:
            raise ValueError("historical-only target contains current/future experts: {}".format(future))
