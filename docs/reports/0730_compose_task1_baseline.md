# Compose Task1 Baseline Report

## Scope

This report records the matched UCIT Task1 validation performed on branch
`exp/0730-compose-task1-oracle`.  Compose and standard PEFT use the same
LLaVA-1.5 7B base, decoder projections, rank, scale, data order, optimizer,
global batch, epoch, maximum length, and seed.

## Configuration

| Method | Rank | Alpha | Scale | Trainable parameters | Target layers | Global batch | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard PEFT LoRA | 8 | 16 | 2 | 19,988,480 | 224 | 64 | 375 |
| Compose Expert 0 | 8 | 16 | 2 | 19,988,480 | 224 | 64 | 375 |

Both methods target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj` in all 32 decoder layers. Neither vision-tower,
projector, nor LM-head parameters are adapted.

## Results

| Method | Rank | Adapter parameters | Target modules | Task1 exact match | Mean train loss | Runtime basis | Peak memory | Checkpoint bytes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Base LLaVA | 0 | 0 | none | 493/3000 (16.4333%) | n/a | 22m16s (6,000-sample mixed evaluation) | 15,099,399,168 bytes (evaluation) | n/a |
| Standard PEFT LoRA | 8 | 19,988,480 | 7 projections | 2705/3000 (90.1667%) | 0.20884625 | 36m53s | 21,072 MiB | 40,047,994 |
| Compose Expert 0 | 8 | 19,988,480 | 7 projections | 2706/3000 (90.2000%) | 0.20838464 | 43m43s | 21,476 MiB | 40,129,746 |
| Hyper-LLaVA Task1 | unavailable | unavailable | unavailable | not reported | not run | not run | not run | unavailable |

The methods differ by one prediction (0.0333 percentage points). Their mean
training losses differ by 0.000462. This is consistent with implementation
parity and passes stop condition A.

## Integrity evidence

- Task1 contains 23,998 training records and 89,375 supervised target tokens.
- Supervised token min/mean/max is 2 / 3.724227 / 7; zero-supervision count is 0.
- Compose injected 224 decoder-only layers and observed finite gradients for
  224/224 LoRA-B tensors.
- Compose saved 448 tensors and PEFT saved the corresponding 448 LoRA tensors.
- Both checkpoints passed two independent strict reloads with no missing or
  unexpected tensors and exact equality across 32,000 fixed-sample logits.
- Compose fixed-sample logits SHA-256 is
  `9f9a4937a8283c0f43b755f62762bd5122943287ded2345df7193220eb41214c`.
- PEFT fixed-sample logits SHA-256 is
  `b9818fc7802ab91bda50e71fd4481600e6d5d52c783932e7abfb1e919ee323ce`.

## Hyper-LLaVA control

No usable UCIT Task1 Hyper-LLaVA checkpoint was found. Existing Hyper
checkpoints cover later UCIT tasks only. No metric is fabricated and no new
full Hyper training is included in this task.

## Experiment paths

- Compose: `/data/ckpt/zhaozhuofan/compose/UCIT/Task1_rank8_seed42`
- PEFT: `/data/ckpt/zhaozhuofan/compose/baselines/peft_task1_rank8_seed42`
- Compose evaluation: `/data/ckpt/zhaozhuofan/compose/eval/task1_compose_rank8_seed42`
- PEFT evaluation: `/data/ckpt/zhaozhuofan/compose/eval/task1_peft_rank8_seed42`
