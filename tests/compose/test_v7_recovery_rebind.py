"""CPU tests for the operator recovery rebind
(``compose.experiments.v7_task_run.rebind_run_contract``).

2026-09-04 recovery policy: when a fix commit moves git HEAD between S3
launches of the same task root (so ``bind_run_contract`` fail-closes), the
operator may rebind -- only when the stored training recipe is byte-equal --
so a crashed S3 continues from its last checkpoint instead of a full re-run.
"""

import json
import unittest
from pathlib import Path

import pytest

from compose.experiments.v7_task_run import rebind_run_contract


def _contract(git_sha, contract_hash):
    return {
        "git_sha": git_sha,
        "contract_hash": contract_hash,
        "recipe": {
            "world_size": 4,
            "effective_global_batch_size": 64,
            "learning_rate": 2e-5,
        },
    }


class RebindRunContractTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(self._make_tmpdir())

    def _make_tmpdir(self):
        import tempfile

        return tempfile.mkdtemp(prefix="v7_rebind_test_")

    def tearDown(self):
        import shutil

        shutil.rmtree(str(self._tmp), ignore_errors=True)

    def _write_root(self, observed):
        data_dir = self._tmp / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "run_contract.json").write_text(
            json.dumps(observed), encoding="utf-8"
        )
        stages = self._tmp / "stages"
        stages.mkdir(exist_ok=True)
        for stage in ("s0_full_data", "s1_fixed_queries", "s2_candidates"):
            (stages / (stage + ".done")).write_text(
                json.dumps({"stage": stage, "run_contract_hash": observed["contract_hash"]}),
                encoding="utf-8",
            )
        return data_dir, stages

    def test_rebind_refreshes_contract_and_markers_and_audits(self):
        observed = _contract("2d165a4dbd7ba51ca450dbdde7c5be94c648b86d", "oldhash123")
        expected = _contract("de204c6f45ac32efbad9543499254814f45fafcd", "newhash456")
        data_dir, stages = self._write_root(observed)

        result = rebind_run_contract(self._tmp, expected, observed, formal_run=True)

        self.assertEqual(result, "newhash456")
        stored = json.loads((data_dir / "run_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, expected)
        for stage in ("s0_full_data", "s1_fixed_queries", "s2_candidates"):
            payload = json.loads((stages / (stage + ".done")).read_text(encoding="utf-8"))
            self.assertEqual(payload["run_contract_hash"], "newhash456")
        audit = (data_dir / "contract_rebinds.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(audit), 4)  # 1 contract + 3 refreshed markers
        self.assertIn('"new_contract_hash": "newhash456"', audit[0])

    def test_refuses_when_recipe_differs(self):
        observed = _contract("oldsha", "oldhash")
        expected = _contract("newsha", "newhash")
        expected["recipe"] = {"world_size": 4, "effective_global_batch_size": 63}
        self._write_root(observed)
        with pytest.raises(ValueError, match="recipe differs"):
            rebind_run_contract(self._tmp, expected, observed, formal_run=True)

    def test_refuses_for_smoke_launches(self):
        observed = _contract("oldsha", "oldhash")
        expected = _contract("newsha", "newhash")
        self._write_root(observed)
        with pytest.raises(ValueError, match="formal-run recovery only"):
            rebind_run_contract(self._tmp, expected, observed, formal_run=False)

    def test_refuses_when_git_head_did_not_move(self):
        expected = _contract("samesha", "newhash")
        self._write_root(_contract("samesha", "oldhash"))
        with pytest.raises(ValueError, match="did not move"):
            rebind_run_contract(self._tmp, expected, _contract("samesha", "oldhash"), formal_run=True)

    def test_unmarked_s4_stage_is_left_unmarked(self):
        observed = _contract("oldsha", "oldhash")
        expected = _contract("newsha", "newhash")
        self._write_root(observed)
        self.assertFalse((self._tmp / "stages" / "s3_training.done").exists())


if __name__ == "__main__":
    unittest.main()
