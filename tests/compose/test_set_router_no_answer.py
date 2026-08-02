import inspect

from compose.router.set_router import ExpertSetRouter


def test_router_forward_schema_has_no_answer_or_task_id():
    names = set(inspect.signature(ExpertSetRouter.forward).parameters)
    assert not names & {"answer", "labels", "targets", "task_id", "oracle_set"}
