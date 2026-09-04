import pytest
import torch

from compose.eval.v7_formal_ucit_eval import _evaluate_gpu_queue
from compose.train.trainer import ComposeTrainer
from compose.v7.hf_trainer import V7ComposeTrainer
from llava.train.llava_trainer import LengthGroupedSampler


def _indices(size, multiple):
    sampler = LengthGroupedSampler(
        batch_size=1,
        world_size=multiple,
        lengths=[1] * size,
        generator=torch.Generator().manual_seed(7),
        pad_to_multiple=multiple,
    )
    return sampler, list(sampler)


def test_formal_sampler_pads_tail_without_losing_unique_samples():
    sampler, indices = _indices(23_742, 64)

    assert len(sampler) == len(indices) == 23_744
    assert set(indices) == set(range(23_742))
    assert indices[-2:] == indices[:2]

    rank_zero = indices[0::2]
    rank_one = indices[1::2]
    assert len(rank_zero) == len(rank_one) == 11_872
    assert len(rank_zero) % 32 == len(rank_one) % 32 == 0


def test_formal_sampler_does_not_pad_complete_window():
    sampler, indices = _indices(128, 64)

    assert len(sampler) == len(indices) == 128
    assert len(set(indices)) == 128


def test_v7_strict_trainer_enables_complete_global_optimizer_windows(monkeypatch):
    sampler = LengthGroupedSampler(
        batch_size=1, world_size=64, lengths=[1] * 23_742
    )
    monkeypatch.setattr(ComposeTrainer, "_get_train_sampler", lambda _self: sampler)
    trainer = object.__new__(V7ComposeTrainer)
    trainer.v7_require_full_coverage = True

    assert trainer._get_train_sampler() is sampler
    assert sampler.pad_to_multiple == 64
    assert len(sampler) == 23_744


def test_cached_evaluator_worker_accepts_cache_manifest():
    assert "cache_manifest" in __import__("inspect").signature(
        _evaluate_gpu_queue
    ).parameters


def test_sampler_rejects_non_positive_padding_multiple():
    with pytest.raises(ValueError, match="pad_to_multiple must be positive"):
        LengthGroupedSampler(batch_size=1, world_size=2, lengths=[1], pad_to_multiple=0)
