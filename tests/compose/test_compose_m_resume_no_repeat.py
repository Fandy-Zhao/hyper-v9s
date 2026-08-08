"""Test M (spec §27): resuming at CLUSTERS_READY / CLUSTER_EXPERTS_TRAINED /
KEYS_READY / EXPERTS_COMMITTED restores the exact stage without repeating
work -- the machine never auto-advances and illegal backward/forward
moves raise."""

import tempfile
import unittest
from pathlib import Path

from compose.experts.task_state import TaskStage, TaskStateMachine


TRANSITION_LIST = [
    (TaskStage.DATA_READY, "data prepared"),
    (TaskStage.QUERY_READY, "queries extracted"),
    (TaskStage.OLD_TEACHER_READY, "teacher searched"),
    (TaskStage.RESIDUAL_READY, "residual split"),
    (TaskStage.CLUSTERS_READY, "clusters formed"),
    (TaskStage.CLUSTER_EXPERTS_TRAINING, "cluster training started"),
    (TaskStage.CLUSTER_EXPERTS_TRAINED, "cluster training finished"),
    (TaskStage.KEYS_TRAINING, "key training started"),
    (TaskStage.KEYS_READY, "keys ready"),
    (TaskStage.EXPERTS_COMMITTED, "committed"),
    (TaskStage.RMS_READY, "rms calibrated"),
    (TaskStage.SNAPSHOT_READY, "snapshot written"),
    (TaskStage.EVALUATION_COMPLETE, "evaluated"),
    (TaskStage.COMPLETED, "done"),
]


def _full_chain(machine):
    for target, note in TRANSITION_LIST:
        machine.advance(target, note=note)


class ResumeNoRepeatTest(unittest.TestCase):
    def test_resume_restores_stage_without_auto_advancing(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        for target, note in TRANSITION_LIST:
            machine.advance(target, note=note)
            if target is TaskStage.SNAPSHOT_READY:
                break  # leave the machine at SNAPSHOT_READY
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_state.json"
            machine.save(path)
            restored = TaskStateMachine.load(path)
        self.assertEqual(restored.stage, TaskStage.SNAPSHOT_READY)
        # The next legal move is a fresh EVALUATION_COMPLETE; nothing
        # auto-advanced on load.
        self.assertEqual(restored.allowable_next_stages(), [TaskStage.EVALUATION_COMPLETE])
        self.assertEqual(len(restored.history), 12)

    @staticmethod
    def _run_to(root, stage):
        machine = TaskStateMachine(1, "ArxivQA")
        for target, note in TRANSITION_LIST:
            machine.advance(target, note=note)
            if target is stage:
                break
        assert machine.stage == stage, "transition list drifted"
        path = Path(root) / "task_state.json"
        machine.save(path)
        return machine

    def test_resume_at_clusters_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            machine = self._run_to(directory, TaskStage.CLUSTERS_READY)
            restored = TaskStateMachine.load(Path(directory) / "task_state.json")
            self.assertEqual(restored.stage, TaskStage.CLUSTERS_READY)
            self.assertEqual(
                restored.allowable_next_stages(), [TaskStage.CLUSTER_EXPERTS_TRAINING]
            )
            # Advancing from the restored machine works exactly once.
            restored.advance(TaskStage.CLUSTER_EXPERTS_TRAINING, note="resumed train")
            self.assertEqual(restored.stage, TaskStage.CLUSTER_EXPERTS_TRAINING)

    def test_resume_at_keys_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            machine = self._run_to(directory, TaskStage.KEYS_READY)
            restored = TaskStateMachine.load(Path(directory) / "task_state.json")
            self.assertEqual(restored.stage, TaskStage.KEYS_READY)
            restored.advance(TaskStage.EXPERTS_COMMITTED, note="commit on resume")
            self.assertEqual(restored.stage, TaskStage.EXPERTS_COMMITTED)
            # One-way: committing twice is illegal.
            with self.assertRaises(ValueError):
                restored.advance(TaskStage.EXPERTS_COMMITTED)

    def test_runner_advance_is_idempotent_on_resume_reentry(self):
        # Regression: a crashed run that entered CLUSTER_EXPERTS_TRAINING
        # before its heavy subprocess finished must re-enter the stage on
        # resume without a machine transition (the machine itself stays
        # strictly one-way; the runner helper absorbs the re-entry).
        from compose.experiments.task_run import _advance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            machine = self._run_to(directory, TaskStage.CLUSTER_EXPERTS_TRAINING)
            history_before = list(machine.history)
            _advance(root, machine, TaskStage.CLUSTER_EXPERTS_TRAINING,
                     note="resumed training")
            self.assertEqual(machine.history, history_before)
            self.assertIs(machine.stage, TaskStage.CLUSTER_EXPERTS_TRAINING)
            # The real transition still works afterwards.
            _advance(root, machine, TaskStage.CLUSTER_EXPERTS_TRAINED,
                     note="training finished")
            self.assertIs(machine.stage, TaskStage.CLUSTER_EXPERTS_TRAINED)
            self.assertTrue((root / "state" / "task_state.json").is_file())

    def test_resume_at_experts_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            machine = self._run_to(directory, TaskStage.EXPERTS_COMMITTED)
            restored = TaskStateMachine.load(Path(directory) / "task_state.json")
            self.assertEqual(restored.stage, TaskStage.EXPERTS_COMMITTED)
            self.assertEqual(restored.allowable_next_stages(), [TaskStage.RMS_READY])

    def test_illegal_moves_raise(self):
        machine = TaskStateMachine(2, "VizWiz")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.QUERY_READY)
        # Task 0's cold start skips OLD_TEACHER_READY; task 2 must not
        # jump RESIDUAL_READY from QUERY_READY illegally -- it may.
        machine.advance(TaskStage.RESIDUAL_READY)
        with self.assertRaises(ValueError):
            machine.advance(TaskStage.QUERY_READY)  # backward move
        with self.assertRaises(ValueError):
            machine.advance(TaskStage.COMPLETED)  # forward jump
        # CLUSTERS_READY IS legal from RESIDUAL_READY (or NO_EXPANSION
        # REQUIRED); a far-forward stage is not.
        with self.assertRaises(ValueError):
            machine.advance(TaskStage.EXPERTS_COMMITTED)

    def test_skip_with_reason_records_without_advancing(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.QUERY_READY)
        machine.skip_with_reason(
            TaskStage.OLD_TEACHER_READY,
            reason="task 0: empty registry, no old teacher",
        )
        self.assertEqual(machine.stage, TaskStage.QUERY_READY)
        # DATA_READY, QUERY_READY, and the recorded skip entry.
        self.assertEqual(len(machine.history), 3)
        self.assertEqual(
            machine.history[-1]["skipped"], TaskStage.OLD_TEACHER_READY.value
        )
        # The skip survives persistence.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_state.json"
            machine.save(path)
            restored = TaskStateMachine.load(path)
        self.assertEqual(restored.stage, TaskStage.QUERY_READY)
        self.assertEqual(
            restored.history[-1]["note"], "task 0: empty registry, no old teacher"
        )

    def test_no_expansion_path_skips_cluster_stages(self):
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.QUERY_READY)
        machine.advance(TaskStage.RESIDUAL_READY)
        machine.advance(TaskStage.NO_EXPANSION_REQUIRED)
        self.assertEqual(machine.stage, TaskStage.NO_EXPANSION_REQUIRED)
        self.assertEqual(machine.allowable_next_stages(), [TaskStage.RMS_READY])
        with self.assertRaises(ValueError):
            machine.advance(TaskStage.CLUSTERS_READY)


if __name__ == "__main__":
    unittest.main()
