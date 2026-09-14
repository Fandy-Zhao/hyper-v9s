"""The V8-Exact-Accelerated mode must fail closed.

Three properties are checked here:

1. the shipped ``configs/v8_exact_accelerated.yaml`` loads, and its flags land
   on the right arguments without breaking the recipe invariants;
2. every one of the twelve prohibited accelerations is rejected *by name* when
   it appears anywhere in a config, including nested under another key;
3. the recipe guard actually fires -- a config that moves the effective batch,
   the learning rate or the schedule is refused rather than silently applied.
"""

import os
import unittest
from dataclasses import dataclass

from compose.train.v8_flags import (
    EXECUTION_FLAGS,
    PROHIBITED_KEYS,
    ProhibitedFlagError,
    apply_config,
    assert_recipe_invariants,
    check_no_prohibited_keys,
    default_config_path,
    load_config,
)
from compose.train.v8_flags import _unknown_flag_message


@dataclass
class _ModelArgs:
    compose_v7_query_tensor = None
    compose_selection_plan = False
    compose_attn_implementation = ""
    profile_training = False
    profile_sync = True
    profile_flush_every = 25
    compose_rank = 8
    compose_alpha = 16.0


@dataclass
class _TrainingArgs:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 0.0002
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    seed: int = 42
    model_max_length: int = 2048
    bf16: bool = True
    tf32: bool = True
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True
    group_by_modality_length: bool = True
    world_size: int = 2


class ShippedConfigTest(unittest.TestCase):
    def test_config_exists_and_loads(self):
        path = default_config_path()
        self.assertTrue(os.path.isfile(path), path)
        payload = load_config(path)
        self.assertEqual(payload["method"], "v8_exact_accelerated")
        check_no_prohibited_keys(payload, path)

    def test_flags_apply_without_moving_the_recipe(self):
        payload = load_config(default_config_path())
        model_args, training_args = _ModelArgs(), _TrainingArgs()
        resolved = apply_config(payload, model_args, training_args)
        self.assertTrue(model_args.compose_selection_plan)
        self.assertIs(model_args.compose_v7_query_tensor, None)
        self.assertEqual(model_args.compose_attn_implementation, "")
        invariants = assert_recipe_invariants(training_args, model_args)
        self.assertEqual(invariants["effective_batch"], 64)

    def test_cache_queries_path_is_absolutised(self):
        model_args, training_args = _ModelArgs(), _TrainingArgs()
        apply_config(
            {"flags": {"cache_queries": "./queries/task0/train"}},
            model_args,
            training_args,
        )
        self.assertTrue(os.path.isabs(model_args.compose_v7_query_tensor))
        self.assertTrue(model_args.compose_v7_query_tensor.endswith("task0/train"))

    def test_every_flag_name_is_documented(self):
        for name, entry in EXECUTION_FLAGS.items():
            self.assertEqual(len(entry), 3, name)
            self.assertIsInstance(entry[2], str)
            self.assertTrue(entry[2], name)


class AccumulationDerivationTest(unittest.TestCase):
    """``effective_batch`` is declared; accumulation is derived from world size.

    Hard-coding accumulation in the config makes the file correct on exactly one
    world size, which is the failure the invariant gate exists to catch.  These
    tests pin the derivation and, just as importantly, the two ways it can be
    told something contradictory.
    """

    def _apply(self, flags, world_size, micro=1):
        training_args = _TrainingArgs(
            per_device_train_batch_size=micro, gradient_accumulation_steps=0
        )
        return apply_config(
            {"flags": flags}, _ModelArgs(), training_args, world_size=world_size
        )

    def test_accumulation_moves_inversely_with_world_size(self):
        for world_size, expected in ((1, 16), (2, 8), (4, 4)):
            with self.subTest(world_size=world_size):
                resolved = self._apply(
                    {"micro_batch_size": 4, "effective_batch": 64}, world_size
                )
                self.assertEqual(resolved["gradient_accumulation_steps"], expected)
                self.assertEqual(resolved["effective_batch"], 64)

    def test_wider_micro_batch_keeps_the_effective_batch(self):
        for micro, expected in ((1, 32), (2, 16), (4, 8), (8, 4)):
            with self.subTest(micro=micro):
                resolved = self._apply(
                    {"micro_batch_size": micro, "effective_batch": 64}, 2, micro=micro
                )
                self.assertEqual(resolved["gradient_accumulation_steps"], expected)
                self.assertEqual(resolved["effective_batch"], 64)

    def test_indivisible_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            self._apply({"micro_batch_size": 3, "effective_batch": 64}, 2, micro=3)

    def test_contradictory_explicit_accumulation_is_refused(self):
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self._apply(
                {
                    "micro_batch_size": 4,
                    "effective_batch": 64,
                    "gradient_accumulation_steps": 32,
                },
                2,
            )

    def test_agreeing_explicit_accumulation_is_accepted(self):
        resolved = self._apply(
            {
                "micro_batch_size": 4,
                "effective_batch": 64,
                "gradient_accumulation_steps": 8,
            },
            2,
        )
        self.assertEqual(resolved["gradient_accumulation_steps"], 8)

    def test_declared_batch_is_not_leaked_as_an_attribute(self):
        resolved = self._apply({"micro_batch_size": 4, "effective_batch": 64}, 2)
        self.assertNotIn("effective_batch_declared", resolved)
        self.assertEqual(resolved["effective_batch"], 64)

    def test_shipped_config_resolves_on_every_world_size(self):
        payload = load_config(default_config_path())
        for world_size in (1, 2, 4):
            with self.subTest(world_size=world_size):
                model_args = _ModelArgs()
                training_args = _TrainingArgs(gradient_accumulation_steps=0)
                apply_config(
                    payload, model_args, training_args, world_size=world_size
                )
                observed = assert_recipe_invariants(
                    training_args, model_args, world_size=world_size
                )
                self.assertEqual(observed["effective_batch"], 64)


class ProhibitedFlagTest(unittest.TestCase):
    def test_every_prohibited_name_is_rejected(self):
        for name in PROHIBITED_KEYS:
            with self.subTest(name=name):
                with self.assertRaises(ProhibitedFlagError):
                    check_no_prohibited_keys({"flags": {name: 1}})

    def test_nested_prohibited_name_is_rejected(self):
        payload = {"flags": {}, "extra": {"deeper": [{"freeze_teacher": True}]}}
        with self.assertRaisesRegex(ProhibitedFlagError, "freeze_teacher"):
            check_no_prohibited_keys(payload)

    def test_unknown_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown execution flags"):
            apply_config({"flags": {"not_a_real_flag": 1}}, _ModelArgs(), _TrainingArgs())


class RecipeGuardTest(unittest.TestCase):
    def test_effective_batch_change_is_refused(self):
        training_args = _TrainingArgs(gradient_accumulation_steps=16)
        with self.assertRaisesRegex(ValueError, "effective batch 32"):
            assert_recipe_invariants(training_args)

    def test_single_gpu_world_size_is_refused(self):
        # 1 x 32 x 1 = 32 != 64: the config's accumulation assumes world size 2.
        with self.assertRaisesRegex(ValueError, "effective batch 32"):
            assert_recipe_invariants(_TrainingArgs(world_size=1))

    def test_four_gpu_world_size_is_refused_at_accumulation_32(self):
        with self.assertRaisesRegex(ValueError, "effective batch 128"):
            assert_recipe_invariants(_TrainingArgs(world_size=4))

    def test_learning_rate_change_is_refused(self):
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            assert_recipe_invariants(_TrainingArgs(learning_rate=1e-4))

    def test_scheduler_change_is_refused(self):
        with self.assertRaisesRegex(ValueError, "lr_scheduler_type"):
            assert_recipe_invariants(_TrainingArgs(lr_scheduler_type="linear"))

    def test_epoch_change_is_refused(self):
        with self.assertRaisesRegex(ValueError, "num_train_epochs"):
            assert_recipe_invariants(_TrainingArgs(num_train_epochs=3.0))

    def test_gradient_checkpointing_change_is_refused(self):
        with self.assertRaisesRegex(ValueError, "gradient_checkpointing"):
            assert_recipe_invariants(_TrainingArgs(gradient_checkpointing=False))

    def test_shipped_recipe_passes(self):
        observed = assert_recipe_invariants(_TrainingArgs())
        self.assertEqual(observed["effective_batch"], 64)
        self.assertEqual(observed["learning_rate"], 0.0002)

    def test_explicit_world_size_wins_over_the_attribute(self):
        """The launch's real size must be authoritative.

        ``TrainingArguments.world_size`` is a property over the accelerator's
        distributed state, which reads as 1 before the Trainer is built.  If it
        won, a correct two-process launch would be refused with a nonsense
        "effective batch 32" -- so the explicit argument has to take precedence.
        """
        stubbed = _TrainingArgs(world_size=1)  # what the property reports pre-init
        observed = assert_recipe_invariants(stubbed, world_size=2)
        self.assertEqual(observed["effective_batch"], 64)
        self.assertEqual(observed["world_size"], 2)
        # ... and it still refuses a launch that genuinely has one process.
        with self.assertRaisesRegex(ValueError, "effective batch 32"):
            assert_recipe_invariants(stubbed, world_size=1)


class UnknownFlagMessageTest(unittest.TestCase):
    """An unknown flag is rejected -- but the rejection has to be usable.

    The brief names several capabilities this repository deliberately does not
    implement, and the reason for each is already in the shipped config.  A
    reader who copies the brief's flag list into ``flags:`` will trip the
    fail-closed check; if the message only said "unknown flag" they would
    reasonably conclude the capability was overlooked rather than measured and
    declined.
    """

    def _message(self, *names):
        payload = {
            "flags": {n: True for n in names},
            "not_applicable": {
                "cache_historical_nll": {
                    "reason": "Historical-expert answer NLL is not recomputed\n"
                              "in the training loop at all."
                }
            },
        }
        return _unknown_flag_message(sorted(names), payload, "cfg.yaml")

    def test_documented_flag_quotes_its_measured_reason(self):
        msg = self._message("cache_historical_nll")
        self.assertIn("documented as not applicable", msg)
        # Folded onto one line so a multi-line YAML reason stays readable.
        self.assertIn("not recomputed in the training loop at all.", msg)
        self.assertNotIn("\n" + "in the training", msg)

    def test_alias_names_the_flag_that_does_the_job(self):
        self.assertIn("attn_implementation", self._message("use_flash_attention"))
        self.assertIn("micro_batch_size", self._message("microbatch_autotune"))

    def test_undocumented_flag_still_rejects_and_lists_the_legal_set(self):
        msg = self._message("totally_new_flag")
        self.assertIn("unknown execution flags", msg)
        self.assertIn("cache_queries", msg)  # the legal set is still printed
        self.assertNotIn("documented as not applicable", msg)

    def test_the_shipped_config_still_loads(self):
        """The message change must not disturb the real config path."""
        cfg = load_config(default_config_path())
        self.assertIn("flags", cfg)


if __name__ == "__main__":
    unittest.main()
