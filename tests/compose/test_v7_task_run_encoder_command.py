"""CPU tests for the legacy live fixed-query encoder command assembly
(``compose/experiments/v7_task_run._live_query_features_command``).

0903 spec §8a root cause: the cache-production / bounded-gate encode paths
run CLIP at batch 32 while the legacy encoder's built-in default is 16, so
live gate twins must be able to override the batch.  With ``batch_size=None``
the assembled command must stay byte-identical to the pre-existing legacy
launchers (no ``--batch-size`` token at all)."""

import pytest

from compose.experiments.v7_task_run import _live_query_features_command


def _base():
    return _live_query_features_command(
        "python", questions="data/train_full.json", images="/data/img",
        output="features/train.json", vision_model="/models/clip",
    )


def test_default_omits_batch_size_entirely():
    # byte-identical legacy shape: no --batch-size token at all (built-in 16)
    command = _base()
    assert command[:4] == ["python", "-m", "compose.eval.query_features",
                           "--questions"]
    assert "--batch-size" not in command
    # exact legacy token stream (order preserved)
    assert command == [
        "python", "-m", "compose.eval.query_features",
        "--questions", "data/train_full.json", "--images", "/data/img",
        "--output", "features/train.json",
        "--query-vision-model", "/models/clip",
        "--query-mode", "v7_fixed", "--device", "cuda:0",
    ]


def test_batch_size_override_appears_once():
    command = _live_query_features_command(
        "python", questions="q.json", images="/img", output="o.json",
        vision_model="/m/clip", batch_size=32,
    )
    assert command.count("--batch-size") == 1
    assert command[command.index("--batch-size") + 1] == "32"


def test_shard_invocation_carries_shard_and_batch_flags():
    command = _live_query_features_command(
        "python", questions="q.json", images="/img", output="p.json",
        vision_model="/m/clip", num_shards=2, shard_index=1, batch_size=32,
    )
    assert "--num-shards" in command and "--shard-index" in command
    assert command[command.index("--num-shards") + 1] == "2"
    assert command[command.index("--shard-index") + 1] == "1"
    assert command[command.index("--batch-size") + 1] == "32"


def test_shard_without_batch_size_keeps_legacy_shape():
    command = _live_query_features_command(
        "python", questions="q.json", images="/img", output="p.json",
        vision_model="/m/clip", num_shards=2, shard_index=1,
    )
    assert "--batch-size" not in command
    assert command.count("--num-shards") == 1


@pytest.mark.parametrize("batch_size", [1, 64])
def test_any_positive_batch_passthrough(batch_size):
    command = _live_query_features_command(
        "python", questions="q.json", images="/img", output="o.json",
        vision_model="/m/clip", batch_size=batch_size,
    )
    assert command[command.index("--batch-size") + 1] == str(batch_size)
