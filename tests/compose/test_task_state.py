"""V6 Stage E2: task-level state machine (one-way transitions)."""

import tempfile
import unittest
from pathlib import Path

from compose.experts.task_state import TaskStage, TaskStateMachine


class TaskStateMachineTest(unittest.TestCase):
    def test_full_official_path_is_legal(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        path = [
            TaskStage.DATA_READY,
            TaskStage.OLD_TEACHER_RUNNING,
            TaskStage.OLD_TEACHER_READY,
            TaskStage.RESIDUAL_READY,
            TaskStage.CANDIDATE_TRAINING,
            TaskStage.CANDIDATE_TRAINED,
            TaskStage.CANDIDATE_VALIDATED,
            TaskStage.EXPERTS_COMMITTED,
            TaskStage.GLOBAL_TEACHER_READY,
            TaskStage.ROUTER_TRAINING,
            TaskStage.ROUTER_READY,
            TaskStage.RMS_READY,
            TaskStage.SNAPSHOT_READY,
            TaskStage.EVALUATION_COMPLETE,
            TaskStage.COMPLETED,
        ]
        for stage in path:
            machine.advance(stage)
        self.assertIs(machine.stage, TaskStage.COMPLETED)
        self.assertEqual(len(machine.history), len(path))

    def test_task1_cold_start_path_is_legal(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        for stage in [
            TaskStage.DATA_READY,
            TaskStage.CANDIDATE_TRAINING,
            TaskStage.CANDIDATE_TRAINED,
            TaskStage.CANDIDATE_VALIDATED,
            TaskStage.EXPERTS_COMMITTED,
            TaskStage.GLOBAL_TEACHER_READY,
            TaskStage.ROUTER_TRAINING,
            TaskStage.ROUTER_READY,
            TaskStage.RMS_READY,
            TaskStage.SNAPSHOT_READY,
            TaskStage.EVALUATION_COMPLETE,
            TaskStage.COMPLETED,
        ]:
            machine.advance(stage)
        self.assertIs(machine.stage, TaskStage.COMPLETED)

    def test_undersized_residual_path_is_legal(self):
        machine = TaskStateMachine(1, "ArxivQA")
        for stage in [
            TaskStage.DATA_READY,
            TaskStage.OLD_TEACHER_RUNNING,
            TaskStage.OLD_TEACHER_READY,
            TaskStage.RESIDUAL_READY,
            TaskStage.GLOBAL_TEACHER_READY,  # commit_count = 0
            TaskStage.ROUTER_TRAINING,
            TaskStage.ROUTER_READY,
            TaskStage.RMS_READY,
            TaskStage.SNAPSHOT_READY,
            TaskStage.EVALUATION_COMPLETE,
            TaskStage.COMPLETED,
        ]:
            machine.advance(stage)
        self.assertIs(machine.stage, TaskStage.COMPLETED)

    def test_backward_transition_rejected(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.NOT_STARTED)
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.COMPLETED)

    def test_skip_transition_rejected(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.OLD_TEACHER_RUNNING)
        # OLD_TEACHER_RUNNING -> CANDIDATE_TRAINING is not in the graph.
        with self.assertRaisesRegex(ValueError, "illegal task transition"):
            machine.advance(TaskStage.CANDIDATE_TRAINING)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_state.json"
            machine = TaskStateMachine(1, "ArxivQA")
            machine.advance(TaskStage.DATA_READY, note="data hash verified")
            machine.advance(TaskStage.OLD_TEACHER_RUNNING)
            machine.save(str(path))

            loaded = TaskStateMachine.load(str(path))
            self.assertEqual(loaded.task_id, 1)
            self.assertEqual(loaded.task_name, "ArxivQA")
            self.assertIs(loaded.stage, TaskStage.OLD_TEACHER_RUNNING)
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
