import os

from llava.train.llava_trainer import LLaVATrainer

from compose.experts.checkpoint import save_expert_checkpoint


class ComposeTrainer(LLaVATrainer):
    """LLaVA Trainer that stores adapter-only Compose checkpoints."""

    def __init__(self, *args, expert_pool=None, **kwargs) -> None:
        if expert_pool is None:
            raise ValueError("expert_pool is required")
        self.expert_pool = expert_pool
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        if not self.args.should_save:
            return
        self.expert_pool.sync_training_step(self.state.global_step)
        os.makedirs(output_dir, exist_ok=True)
        self.model.config.save_pretrained(output_dir)
        save_expert_checkpoint(self.expert_pool, output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
