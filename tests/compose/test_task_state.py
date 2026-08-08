"""Compose task-level state machine (spec §17): one-way transitions,
cold-start skip with recorded reason, no-expansion path, persistence."""

import tempfile
import unittest
from pathlib import Path

from compose.experts.task_state import TaskStage, TaskStateMachine


def _full_path(task_id):
    path = [TaskStage.DATA_READY, TaskStage.QUERY_READY]
    if task_id > 0:
        path.append(TaskStage.OLD_TEACHER_READY)
    path.extend(
        [
            TaskStage.RESIDUAL_READY,
            TaskStage.CLUSTERS_READY,
            TaskStage.CLUSTER_EXPERTS_TRAINING,
            TaskStage.CLUSTER_EXPERTS_TRAINED,
            TaskStage.KEYS_TRAINING,
            TaskStage.KEYS_READY,
            TaskStage.EXPERTS_COMMITTED,
            TaskStage.RMS_READY,
            TaskStage.SNAPSHOT_READY,
            TaskStage.EVALUATION_COMPLETE,
            TaskStage.COMPLETED,
        ]
    )
    return path


class TaskStateMachineTest(unittest.TestCase):
    def test_full_official_path_is_legal(self):
        machine = TaskStateMachine(1, "ArxivQA")
        for stage in _full_path(1):
            machine.advance(stage)
        self.assertIs(machine.stage, TaskStage.COMPLETED)
        self.assertEqual(len(machine.history), len(_full_path(1)))

    def test_task0_cold_start_path_is_legal(self):
        # Task 0 skips OLD_TEACHER_READY with a recorded reason; the rest
        # of the path is identical to the full chain.
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.QUERY_READY)
        machine.skip_with_reason(
            TaskStage.OLD_TEACHER_READY,
            reason="task 0: empty registry, no old teacher",
        )
        self.assertEqual(machine.stage, TaskStage.QUERY_READY)
        self.assertEqual(
            machine.history[-1]["note"], "task 0: empty registry, no old teacher"
        )
        # DATA_READY / QUERY_READY are already done; continue from
        # RESIDUAL_READY onward.
        for stage in _full_path(0)[2:]:
            machine.advance(stage)
        self.assertIs(machine.stage, TaskStage.COMPLETED)

    def test_undersized_residual_path_is_legal(self):
        machine = TaskStateMachine(1, "ArxivQA")
        for stage in [
            TaskStage.DATA_READY,
            TaskStage.QUERY_READY,
            TaskStage.OLD_TEACHER_READY,
            TaskStage.RESIDUAL_READY,
            TaskStage.NO_EXPANSION_REQUIRED,  # undersized residual split
            TaskStage.RMS_READY,
            TaskStage.SNAPSHOT_READY,
            TaskStage.EVALUATION_COMPLETE,
            TaskStage.COMPLETED,
        ]:
            machine.advance(stage)
        self.assertIs(machine.stage, TaskStage.COMPLETED)

    def test_undersized_residual_direct_jump_is_illegal(self):
        # RESIDUAL_READY must record WHY via NO_EXPANSION_REQUIRED (or
        # CLUSTERS_READY); jumping straight to the committed state is not
        # in the graph.
        machine = TaskStateMachine(1, "ArxivQA")
        for stage in [
            TaskStage.DATA_READY,
            TaskStage.QUERY_READY,
            TaskStage.OLD_TEACHER_READY,
            TaskStage.RESIDUAL_READY,
        ]:
            machine.advance(stage)
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.EXPERTS_COMMITTED)
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.KEYS_READY)

    def test_backward_transition_rejected(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.NOT_STARTED)
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.COMPLETED)

    def test_skip_transition_rejected(self):
        machine = TaskStateMachine(1, "ArxivQA")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.QUERY_READY)
        # QUERY_READY -> CLUSTERS_READY skips OLD_TEACHER_READY /
        # RESIDUAL_READY and is not in the graph.
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.CLUSTERS_READY)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_state.json"
            machine = TaskStateMachine(1, "ArxivQA")
            machine.advance(TaskStage.DATA_READY, note="data hash verified")
            machine.advance(TaskStage.QUERY_READY)
            machine.save(str(path))

            loaded = TaskStateMachine.load(str(path))
            self.assertEqual(loaded.task_id, 1)
            self.assertEqual(loaded.task_name, "ArxivQA")
            self.assertIs(loaded.stage, TaskStage.QUERY_READY)
            self.assertEqual(len(loaded.history), 2)
            self.assertEqual(loaded.history[0]["note"], "data hash verified")
            # Loading never auto-advances.
            loaded.advance(TaskStage.OLD_TEACHER_READY)
            self.assertIs(loaded.stage, TaskStage.OLD_TEACHER_READY)

    def test_missing_state_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            TaskStateMachine.load("/nonexistent/task_state.json")

    def test_from_state_dict_rejects_unknown_schema(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            TaskStateMachine.from_state_dict({"schema_version": 99})


if __name__ == "__main__":
    unittest.main()
