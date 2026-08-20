import json
import os
import shutil
import subprocess
from pathlib import Path

PY = '/home/zhaozhuofan/miniconda3/envs/hyper/bin/python'
REPO = Path('/home/zhaozhuofan/Hyper-LlaVA')
ROOT = REPO / 'experiments/runs/task0_multi_r8_seed42'
VROOT = ROOT / 'validation_eval'
VAL_FILE = VROOT / 'validation_200.json'
CONFIGS = {'single_r8': 1, 'two_r8': 2, 'four_r8': 4, 'rank48': 1}


def main():
    VROOT.mkdir(parents=True, exist_ok=True)
    if not VAL_FILE.is_file():
        source = json.loads((ROOT / 'boundary/data/teacher_val.json').read_text(encoding='utf-8'))
        records = []
        for row in source:
            human = row['conversations'][0]['value']
            if human.startswith('<image>\n'):
                human = human[len('<image>\n'):]
            records.append({'question_id': str(row['id']), 'image': row['image'],
                            'text': human, 'answer': row['conversations'][1]['value']})
        VAL_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    pools = VROOT / 'pools'
    if not pools.exists():
        os.symlink(str(ROOT / 'pools'), str(pools))
    eval_dir = VROOT / 'evaluations'
    eval_dir.mkdir(exist_ok=True)
    for config in CONFIGS:
        source = ROOT / 'evaluations' / 'candidates_{}.json'.format(config)
        target = eval_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
    metrics = {}
    for config, count in CONFIGS.items():
        metrics[config] = {}
        for expert in range(count):
            label = 'single:{}'.format(expert)
            command = [PY, '-m', 'compose.experiments.task0_multi_r8_eval', 'gen',
                       '--root', str(VROOT), '--gpus', '4,5,6,7', '--config', config,
                       '--label', label, '--question-file', str(VAL_FILE)]
            subprocess.run(command, cwd=str(REPO), check=True)
            metric_path = VROOT / 'evaluations' / 'answers' / config / label / 'score/metric.json'
            metrics[config][label] = json.loads(metric_path.read_text(encoding='utf-8'))
    out = ROOT / 'evaluations' / 'validation_metrics.json'
    out.write_text(json.dumps({'source': str(VAL_FILE), 'configs': metrics}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('validation metrics written:', out)


if __name__ == '__main__':
    main()
