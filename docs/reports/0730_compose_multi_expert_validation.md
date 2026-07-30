# Compose Multi-expert Validation

## Functional proxy experts

This experiment validates isolation mechanics with two functional expert
proxies; it does not claim that the experts are semantically pure atoms.

| Expert | Origin | Rank | Parameters | Training records | Steps |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | UCIT/ImageNet-R | 8 | 19,988,480 | 23,998 | 375 |
| 1 | UCIT/IconQA | 8 | 19,988,480 | 29,859 | 467 |

Expert 1 completed in 56m39s with mean training loss 0.33701531. Its 94,023
supervised tokens have min/mean/max 2 / 3.148900 / 7 and zero-supervision
count 0. All 224 LoRA-B tensors received finite gradients.

## Pool checkpoint

The resulting pool at
`/data/ckpt/zhaozhuofan/compose/UCIT/Task1_Task4_experts01_rank8_seed42`
contains exactly 896 tensors, 39,976,960 adapter parameters, and 80,259,538
checkpoint bytes. The manifest records both origin task IDs and marks both
experts frozen.

## Isolation gate

- All 448 Expert 0 tensors are exactly equal before and after Expert 1
  training.
- Expert 0 SHA-256 is
  `51d7c2c357522e907c65a916a76281738fd2d13cf6b207882f970b9bd5c55b6a`.
- The old single-expert checkpoint and new pool load 448/448 and 896/896
  tensors respectively, with no missing or unexpected keys.
- Their fixed ImageNet-R sample produces exactly equal 32,000 logits;
  maximum absolute difference is 0.

These checks pass stop condition B and establish that new expert training did
not mutate the old expert.

## Independent fixed-selection evaluation

On the 6,000-sample mixed ImageNet-R/IconQA test set, Expert 0 obtains
2706/3000 (90.2000%) on its ImageNet-R origin task and Expert 1 obtains
2397/3000 (79.9000%) on its IconQA origin task. Their off-task results are
423/3000 and 481/3000 respectively. This confirms that both experts load and
generate independently while also showing that they are strongly
task-specialized functional proxies.
