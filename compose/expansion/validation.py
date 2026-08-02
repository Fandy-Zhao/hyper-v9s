"""Leakage and chronology audits for sufficiency/buffer construction."""

FORBIDDEN_HEAD_FIELDS = frozenset({"answer", "answer_nll", "target", "teacher_set", "oracle_set", "task_id", "labels"})


def validate_head_payload(payload):
    forbidden = sorted(set(payload) & FORBIDDEN_HEAD_FIELDS)
    if forbidden:
        raise ValueError("Sufficiency Head payload leaks supervision: {}".format(forbidden))


def validate_buffer_record(record, expert_creation_tasks):
    if record.split != "train":
        raise ValueError("buffer contains validation/test data")
    future = [value for value in record.teacher_set if int(expert_creation_tasks[value]) >= int(record.task_id)]
    if future:
        raise ValueError("buffer teacher set contains current/future experts: {}".format(future))
    predicted_future = [value for value in record.predicted_set if int(expert_creation_tasks[value]) >= int(record.task_id)]
    if predicted_future:
        raise ValueError("buffer predicted set contains current/future experts: {}".format(predicted_future))
