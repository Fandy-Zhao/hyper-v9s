pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_imagenet.sh hyper-task4 runs/checkpoints/Hyper/UCIT/Task4_llava_lora_ours 1
pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_arxivqa.sh hyper-task4 runs/checkpoints/Hyper/UCIT/Task4_llava_lora_ours 1
pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_vizwiz.sh hyper-task4 runs/checkpoints/Hyper/UCIT/Task4_llava_lora_ours 1
pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_iconqa.sh hyper-task4 runs/checkpoints/Hyper/UCIT/Task4_llava_lora_ours 1

pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_imagenet.sh hyper-task3 runs/checkpoints/Hyper/UCIT/Task3_llava_lora_ours 1
pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_arxivqa.sh hyper-task3 runs/checkpoints/Hyper/UCIT/Task3_llava_lora_ours 1
pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_vizwiz.sh hyper-task3 runs/checkpoints/Hyper/UCIT/Task3_llava_lora_ours 1

pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_imagenet.sh hyper-task2 runs/checkpoints/Hyper/UCIT/Task2_llava_lora_ours 1
pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_arxivqa.sh hyper-task2 runs/checkpoints/Hyper/UCIT/Task2_llava_lora_ours 1

pip install -e .
sh ./scripts/Hyper/Eval_UCIT/eval_imagenet.sh hyper-task1 runs/checkpoints/Hyper/UCIT/Task1_llava_lora_ours 1