"""4-GPU execution contract tests (spec §4/§6/§7/§13/§15/§18).

Unit-level (no GPUs) coverage for the distributed-execution math:

- DistributedSampler partition: every rank draws ceil(n/world) samples;
  explicit divisibility padding is recorded; optimizer steps per epoch
  match the single-GPU steps for every formal cluster size.
- Training contract: global batch and total optimizer steps are equal
  between the single-GPU reference and the 4-GPU recomposition, with
  LR/scheduler/epochs/warmup unchanged.
- Shard slicing: contiguous, disjoint, covering, order-preserving.
- Shard merge: 0 missing / 0 duplicates, canonical order restored.

The GPU-side behaviors (gradient audit, init-hash equality, commit-once,
parity, scaling) are covered by the 4-GPU smoke harness
(scripts/Compose/smoke_4gpu.sh, spec §9/§21/§24/§25).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from compose.eval.sharding import (
    merge_partial_answer_files,
    merge_partial_maps,
    partial_path,
    shard_records,
)
from compose.experiments.task_run import (
    _execution_plan,
    _merge_feature_shards,
    _torchrun_launch,
    _write_distributed_training_contract,
)

#: Formal cluster sizes observed across the seed42 run tasks (manifest
#: lengths that must satisfy the step-count identity ceil(ceil(n/4)/2) ==
#: ceil(n/8)).
FORMAL_CLUSTER_SIZES = [2000, 1074, 926, 1061, 828, 1070, 930, 1, 7, 8, 9]

#: The frozen single-GPU training hyperparameters (configs/compose_ucit.yaml).
SINGLE_PER_DEVICE = 1
SINGLE_ACCUM = 8
EPOCHS = 3
LR = 2.0e-4
WARMUP_RATIO = 0.03


def _config_for(seed=42):
    return {
        "data": {"seed": seed},
        "training": {
            "per_device_train_batch_size": SINGLE_PER_DEVICE,
            "gradient_accumulation_steps": SINGLE_ACCUM,
            "num_train_epochs": EPOCHS,
            "learning_rate": LR,
            "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler_type": "cosine",
        },
    }


def _plan():
    return {
        "mode": "4gpu",
        "gpus": ["4", "5", "6", "7"],
        "world_size": 4,
        "torchrun_prefix": ["python", "-m", "torch.distributed.run", "--module"],
    }


def _sampler_partition(n, world):
    """DistributedSampler semantics: num_samples per rank = ceil(n/world)."""
    import math

    n_rank = int(math.ceil(n / world))
    padding = n_rank * world - n
    return n_rank, padding


class DistributedSamplerPartitionTest(unittest.TestCase):
    def test_steps_identical_for_all_formal_cluster_sizes(self):
        """Optimizer steps per epoch (4-GPU) == single-GPU steps for every
        formal cluster size: ceil(ceil(n/4)/2) == ceil(n/8)."""
        for n in FORMAL_CLUSTER_SIZES:
            n_rank, padding = _sampler_partition(n, 4)
            steps_single = -(-n // (SINGLE_PER_DEVICE * SINGLE_ACCUM))
            steps_four = -(-n_rank // (SINGLE_ACCUM // 4))
            self.assertEqual(
                steps_single,
                steps_four,
                "n={}: single {} vs four {}".format(n, steps_single, steps_four),
            )
            self.assertEqual(steps_four, -(-n // 8))
            # Padding is explicit and small.
            self.assertLessEqual(padding, 3)

    def test_padding_explicit(self):
        n_rank, padding = _sampler_partition(1074, 4)
        self.assertEqual(n_rank, 269)
        self.assertEqual(padding, 2)


class TrainingContractTest(unittest.TestCase):
    def test_contract_assertions_hold_for_all_formal_sizes(self):
        for n in FORMAL_CLUSTER_SIZES:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                contract = _write_distributed_training_contract(
                    root, 3, _config_for(), n, _plan()
                )
                self.assertTrue(all(contract["assertions"].values()), contract)
                reference = contract["single_gpu_reference"]
                four = contract["four_gpu"]
                self.assertEqual(
                    reference["global_batch"], four["global_batch"], n
                )
                self.assertEqual(
                    reference["steps_per_epoch"], four["steps_per_epoch"], n
                )
                self.assertEqual(
                    reference["total_optimizer_steps"],
                    four["total_optimizer_steps"],
                    n,
                )
                self.assertEqual(four["gradient_accumulation_steps"], 2)
                self.assertEqual(four["per_device_train_batch_size"], 1)
                self.assertEqual(reference["learning_rate"], LR)
                self.assertEqual(reference["num_train_epochs"], EPOCHS)
                self.assertEqual(reference["warmup_ratio"], WARMUP_RATIO)
                # File is written with the full contract.
                persisted = json.loads(
                    (root / "distributed_training_contract.json").read_text()
                )
                self.assertEqual(
                    persisted["single_gpu_reference"]["total_optimizer_steps"],
                    four["total_optimizer_steps"],
                )

    def test_contract_rejects_undivisible_accumulation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config_for()
            config["training"]["gradient_accumulation_steps"] = 6
            with self.assertRaises(ValueError):
                _write_distributed_training_contract(
                    Path(directory), 3, config, 2000, _plan()
                )


class ShardSlicingTest(unittest.TestCase):
    def test_shards_are_disjoint_covering_and_order_preserving(self):
        records = [{"id": str(index)} for index in range(17)]
        shards = [shard_records(records, 4, index) for index in range(4)]
        self.assertEqual([len(s) for s in shards], [5, 5, 5, 2])
        seen = []
        for shard in shards:
            self.assertEqual(shard, sorted(shard, key=lambda r: r["id"]))
            seen.extend(record["id"] for record in shard)
        self.assertEqual(seen, [str(index) for index in range(17)])

    def test_invalid_shard_raises(self):
        with self.assertRaises(ValueError):
            shard_records([{"id": "0"}], 4, 4)
        with self.assertRaises(ValueError):
            shard_records([{"id": "0"}], 0, 0)


class ShardMergeTest(unittest.TestCase):
    def _write_partial(self, path, payload):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )

    def test_merge_partial_maps_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._write_partial(
                directory / "a.rank0", {"1": {"empty": 0.5}, "2": {"empty": 0.1}}
            )
            self._write_partial(
                directory / "a.rank1", {"3": {"empty": 0.9}}
            )
            merged = merge_partial_maps(
                [str(directory / "a.rank0"), str(directory / "a.rank1")],
                ["1", "2", "3"],
            )
            self.assertEqual(list(merged), ["1", "2", "3"])
            self.assertEqual(merged["2"], {"empty": 0.1})

    def test_merge_partial_maps_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._write_partial(directory / "b.rank0", {"1": {}})
            self._write_partial(directory / "b.rank1", {"1": {}})
            with self.assertRaises(ValueError):
                merge_partial_maps(
                    [str(directory / "b.rank0"), str(directory / "b.rank1")],
                    ["1"],
                )

    def test_merge_partial_maps_rejects_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._write_partial(directory / "c.rank0", {"1": {}})
            with self.assertRaises(ValueError):
                merge_partial_maps([str(directory / "c.rank0")], ["1", "2"])

    def test_merge_answer_files_restores_order(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "a.rank0.jsonl").write_text('{"q": 0}\n{"q": 1}\n')
            (directory / "a.rank1.jsonl").write_text('{"q": 2}\n')
            merged = merge_partial_answer_files(
                [str(directory / "a.rank0.jsonl"), str(directory / "a.rank1.jsonl")],
                3,
            )
            self.assertEqual(merged, '{"q": 0}\n{"q": 1}\n{"q": 2}\n')
            with self.assertRaises(ValueError):
                merge_partial_answer_files(
                    [str(directory / "a.rank0.jsonl")], 3
                )

    def test_partial_path_convention(self):
        self.assertEqual(partial_path("/x/out.json", 2), "/x/out.json.rank2")


class TorchrunLaunchConstructionTest(unittest.TestCase):
    """The 4-GPU S6/S9 launches must parse as torchrun module launches.

    torchrun's ``--module`` consumes the next positional as the module
    name; embedding the command's own ``[PYTHON, "-m"]`` makes torchrun
    treat the python executable path as the module (regression: verified
    empirically that `--module <python> -m <module>` exits nonzero).
    """

    def test_s6_train_launch(self):
        plan = _execution_plan("4,5,6,7")
        command = [
            "/env/bin/python", "-m", "compose.train.train_compose",
            "--model_name_or_path", "base",
        ]
        launch = _torchrun_launch(plan, command)
        self.assertNotIn("/env/bin/python", launch[7:])  # executable not positional
        module_index = launch.index("--module")
        self.assertEqual(launch[module_index + 1], "compose.train.train_compose")
        self.assertEqual(launch[module_index + 2], "--model_name_or_path")
        # torchrun option region untouched.
        self.assertEqual(launch[:7], plan["torchrun_prefix"])

    def test_s9_rms_launch(self):
        plan = _execution_plan("4,5,6,7")
        command = [
            "/env/bin/python", "-m", "compose.eval.rms_stats",
            "--checkpoint-dir", "pool",
        ]
        launch = _torchrun_launch(plan, command)
        module_index = launch.index("--module")
        self.assertEqual(launch[module_index + 1], "compose.eval.rms_stats")

    def test_single_gpu_plan_has_no_torchrun_prefix(self):
        plan = _execution_plan("4")
        self.assertEqual(plan["mode"], "single")
        self.assertNotIn("torchrun_prefix", plan)

    def test_wrong_gpu_count_is_hard_stop(self):
        with self.assertRaises(ValueError):
            _execution_plan("4,5,6")
        with self.assertRaises(ValueError):
            _execution_plan("4,5,6,7,8")

    def test_merge_feature_shards_reads_worker_partial_names(self):
        """Regression: sharded query_features workers write
        ``<output>.rank{i}`` (partial_path convention); the merge must
        read exactly those names (previously built ``{tag}_{i}_
        features.json.rank{i}`` and raised IndexError)."""
        plan = _execution_plan("4,5,6,7")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features").mkdir()
            for index in range(4):
                payload = {
                    "schema_version": 1,
                    "feature_source": "frozen_clip_l14_336",
                    "query_encoder_provenance": {},
                    "query_encoder_hash": "q",
                    "records": {
                        "sample{}".format(index): {
                            "visual_feature": [1.0],
                            "text_feature": [2.0],
                            "query": [3.0],
                        }
                    },
                }
                (root / "features" / "train_features.json.rank{}".format(index)).write_text(
                    json.dumps(payload)
                )
            records = [{"id": "sample{}".format(index)} for index in range(4)]
            target = _merge_feature_shards(root, "train", records, plan)
            merged = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(sorted(merged["records"]), ["sample0", "sample1", "sample2", "sample3"])
            self.assertTrue((root / "features" / "train_features.json").is_file())


if __name__ == "__main__":
    unittest.main()
