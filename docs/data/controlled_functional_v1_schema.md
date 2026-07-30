# Controlled Functional Composition V1

This offline synthetic benchmark separates three capabilities:

- A: recognize `circle`, `square`, or `triangle`.
- B: count between one and six objects.
- C: resolve `left` versus `right` relative to a vertical divider.

It contains `A_only`, `B_only`, `C_only`, `A_plus_B`, `B_plus_C`, and the optional `A_plus_C`. Each function has independent train, validation, and test splits. The default sizes are 1,600/200/400, and the default generator seed is 730.

Training records use the existing multimodal conversation schema:

```json
{
  "id": "controlled/A_only/train/0",
  "question_id": "controlled/A_only/train/0",
  "task_id": "Controlled/A_only",
  "template_id": "train:A_only:000000",
  "image": "controlled_functional_v1/images/A_only/train/000000.png",
  "conversations": [
    {"from": "human", "value": "<image>\n..."},
    {"from": "gpt", "value": "circle"}
  ]
}
```

Evaluation records replace `conversations` with `text` and `answer`. Template IDs contain the split and are asserted disjoint across train, validation, and test. Four instruction phrasings are rotated deterministically for every function. Images and instructions are written only below the newly created output root; the generator refuses to reuse an existing root.

The manifest records the seed, split sizes, answer distributions, file paths, and SHA-256 for every instruction and evaluation JSON. Images are deterministic PNGs generated with Pillow; the benchmark performs no network access and modifies no source dataset.
