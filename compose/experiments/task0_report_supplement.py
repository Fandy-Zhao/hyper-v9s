import json
import math
import time
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def record_id(record, index):
    return str(record.get('question_id', record.get('id', index)))


def pct(correct, total):
    return None if not total else 100.0 * correct / total


def metric_cell(metric):
    if not metric:
        return '-', '-'
    value = metric.get('value')
    total = metric.get('samples')
    correct = metric.get('correct')
    if total is not None and correct is not None:
        return '{:.2f}%'.format(float(value)), '{}/{}'.format(correct, total)
    return '{:.2f}%'.format(float(value)) if value is not None else '-', metric.get('correct_total', '-')


def mean_nll(rows, selector):
    values = []
    for row in rows:
        value = selector(row)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def nll_rows(root, config):
    return read_jsonl(Path(root) / 'evaluations' / 'nll' / config / 'nll.jsonl')


def label_nll(rows, label):
    return mean_nll(rows, lambda row: row['set_nll'][row['candidate_labels'].index(label)]
                    if label in row['candidate_labels'] else None)


def routed_nll(root, config, k):
    rows = nll_rows(root, config)
    assignments = read_json(Path(root) / 'clustering' / 'k{}'.format(k) / 'test_assignments.json')['assignments']
    return mean_nll(rows, lambda row: row['set_nll'][row['candidate_labels'].index(
        'single:{}'.format(assignments[str(row['sample_id'])]))])


def oracle_nll(rows, kind):
    return mean_nll(rows, lambda row: row['set_nll'][int(row['best_{}_index'.format(kind)])])


def train_accounting(root, manifest):
    rows = []
    for job in manifest['jobs']:
        config = job['config']
        expert = int(job['expert_id'])
        log_dir = Path(root) / 'logs' / config / 'expert_{}'.format(expert)
        checkpoint_dir = Path(root) / 'checkpoints' / config / 'expert_{}'.format(expert)
        marker_path = log_dir / 'complete.txt'
        marker = read_json(marker_path) if marker_path.is_file() else {}
        trainer_state = checkpoint_dir / 'trainer_state.json'
        actual_steps = None
        if trainer_state.is_file():
            actual_steps = read_json(trainer_state).get('global_step')
        adapter_params = None
        state_path = checkpoint_dir / 'expert_{:04d}.pt'.format(expert)
        if state_path.is_file():
            import torch
            state = torch.load(state_path, map_location='cpu', weights_only=False)['state_dict']
            adapter_params = sum(value.numel() for key, value in state.items()
                                 if key.endswith('.lora_A.weight') or key.endswith('.lora_B.weight'))
        rows.append({
            'config': config, 'expert_id': expert, 'cluster_size': int(job['cluster_size']),
            'manifest_steps': int(job['optimizer_steps_3_epochs']), 'actual_steps': actual_steps,
            'duration_seconds': marker.get('duration_seconds'), 'adapter_parameters': adapter_params,
        })
    out = Path(root) / 'diagnostics' / 'task0_training_accounting.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                               'total_steps': sum(r['actual_steps'] or 0 for r in rows),
                               'rows': rows}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return rows


def cluster_rows(root, manifest, summary, k):
    records = read_json(manifest['dataset'])
    ids = [record_id(record, i) for i, record in enumerate(records)]
    assignments = read_json(Path(root) / 'clustering' / 'k{}'.format(k) / 'test_assignments.json')['assignments']
    base_rows = read_jsonl(Path(root) / 'evaluations' / 'answers' / 'base' / 'base' / 'answers.jsonl')
    base_by_id = {str(row['question_id']): str(row['text']) for row in base_rows}
    answers = {e: {str(row['question_id']): str(row['text']) for row in read_jsonl(
        Path(root) / 'evaluations' / 'answers' / ('two_r8' if k == 2 else 'four_r8') /
        'single:{}'.format(e) / 'answers.jsonl')} for e in range(k)}
    truth = {record_id(record, i): str(record['answer']) for i, record in enumerate(records)}
    matrix = summary.get('cluster_expert_matrices', {}).get('k{}'.format(k), {}).get('matrix', {})
    rows = []
    for cluster in range(k):
        cluster_ids = [sample_id for sample_id in ids if int(assignments[sample_id]) == cluster]
        base_acc = pct(sum(base_by_id.get(sample_id, '').upper() == truth[sample_id].upper()
                           for sample_id in cluster_ids), len(cluster_ids))
        own = matrix.get(str(cluster), {}).get(str(cluster))
        rows.append({'cluster': cluster, 'test_support': len(cluster_ids), 'base_accuracy': base_acc,
                     'own_expert_accuracy': own,
                     'own_gain': None if base_acc is None or own is None else own - base_acc,
                     'expert_accuracy': {str(e): matrix.get(str(e), {}).get(str(cluster)) for e in range(k)}})
    return rows


def supplement(root):
    root = Path(root)
    manifest = read_json(root / 'experiment_manifest.json')
    summary = read_json(root / 'evaluations' / 'summary.json')
    report = root / 'reports' / 'task0_multi_r8_report.md'
    if not report.is_file():
        raise SystemExit('report missing: {}'.format(report))
    accounting = train_accounting(root, manifest)
    out = ['\n## 10. Metric Completeness and Teacher-forcing NLL\n',
           'The accuracy/correct-total columns below use the same exact-match evaluator as `Result.text`. '
           'NLL is the mean teacher-forcing NLL from the frozen evaluator; oracle rows remain diagnostic only.\n',
           '| Model / routing | Accuracy | Correct / Total | Teacher-forcing NLL |',
           '|---|---:|---:|---:|']
    cm = summary.get('candidate_metrics', {})
    nll_cache = {config: nll_rows(root, config) for config in ('single_r8', 'two_r8', 'four_r8', 'rank48')}
    items = [
        ('Base', summary.get('base_accuracy'), None),
        ('1xr8 fixed', cm.get('single_r8', {}).get('single:0'), label_nll(nll_cache['single_r8'], 'single:0')),
        ('2xr8 centroid top-1', summary.get('mode_a', {}).get('k2'), routed_nll(root, 'two_r8', 2)),
        ('2xr8 equal raw', cm.get('two_r8', {}).get('pair:0+1:raw'), None),
        ('2xr8 equal RMS', cm.get('two_r8', {}).get('pair:0+1:rms'), label_nll(nll_cache['two_r8'], 'pair:0+1:rms')),
        ('2xr8 oracle best single', summary.get('mode_c', {}).get('two_r8:single'), oracle_nll(nll_cache['two_r8'], 'single')),
        ('2xr8 oracle best pair', summary.get('mode_c', {}).get('two_r8:pair'), oracle_nll(nll_cache['two_r8'], 'pair')),
        ('2xr8 oracle best overall', summary.get('mode_c', {}).get('two_r8:overall'), oracle_nll(nll_cache['two_r8'], 'overall')),
        ('4xr8 centroid top-1', summary.get('mode_a', {}).get('k4'), routed_nll(root, 'four_r8', 4)),
        ('4xr8 equal raw', cm.get('four_r8', {}).get('equal4:raw'), None),
        ('4xr8 equal RMS', cm.get('four_r8', {}).get('equal4:rms'), label_nll(nll_cache['four_r8'], 'equal4:rms')),
        ('4xr8 oracle best single', summary.get('mode_c', {}).get('four_r8:single'), oracle_nll(nll_cache['four_r8'], 'single')),
        ('4xr8 oracle best pair', summary.get('mode_c', {}).get('four_r8:pair'), oracle_nll(nll_cache['four_r8'], 'pair:0+1:rms')),
        ('4xr8 oracle best overall', summary.get('mode_c', {}).get('four_r8:overall'), oracle_nll(nll_cache['four_r8'], 'overall')),
        ('1xr48 fixed', cm.get('rank48', {}).get('single:0'), label_nll(nll_cache['rank48'], 'single:0')),
    ]
    for label, metric, nll in items:
        accuracy, total = metric_cell(metric)
        out.append('| {} | {} | {} | {} |'.format(label, accuracy, total, '-' if nll is None else '{:.4f}'.format(nll)))
    out += ['', '## 11. Cluster Support, Base Accuracy, and Own-expert Gain', '']
    for k in (2, 4):
        rows = cluster_rows(root, manifest, summary, k)
        out += ['**K={}**'.format(k), '', '| Test cluster | Support | Base Acc. | Own expert Acc. | Own gain |',
                '|---:|---:|---:|---:|---:|']
        for row in rows:
            out.append('| C{} | {} | {} | {} | {} |'.format(
                row['cluster'], row['test_support'],
                '-' if row['base_accuracy'] is None else '{:.2f}%'.format(row['base_accuracy']),
                '-' if row['own_expert_accuracy'] is None else '{:.2f}%'.format(row['own_expert_accuracy']),
                '-' if row['own_gain'] is None else '{:+.2f} pp'.format(row['own_gain'])))
        out.append('')
    out += ['## 12. Training Fairness and Adapter Accounting', '',
            '| Config | Expert | Samples | Manifest steps | Actual steps | Wall clock | LoRA params |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for row in accounting:
        duration = '-' if row['duration_seconds'] is None else '{:.1f} min'.format(row['duration_seconds'] / 60.0)
        out.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            row['config'], row['expert_id'], row['cluster_size'], row['manifest_steps'],
            '-' if row['actual_steps'] is None else row['actual_steps'], duration,
            '-' if row['adapter_parameters'] is None else row['adapter_parameters']))
    out += ['', 'Total train FLOPs were not instrumented by the existing trainer; this is reported as unavailable rather than estimated.', '']
    validation = root / 'evaluations' / 'validation_metrics.json'
    out += ['## 13. Validation Accuracy', '']
    if validation.is_file():
        values = read_json(validation).get('configs', {})
        out += ['| Config | Expert | Accuracy | Correct / Total |', '|---|---:|---:|---:|']
        for config, metrics in values.items():
            for expert, metric in metrics.items():
                accuracy, total = metric_cell(metric)
                out.append('| {} | {} | {} | {} |'.format(config, expert, accuracy, total))
    else:
        out.append('Validation generation was not yet available when this report was assembled.')
    report.write_text(report.read_text(encoding='utf-8').rstrip() + '\n\n' + '\n'.join(out) + '\n', encoding='utf-8')
    print('supplement written:', report)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    supplement(parser.parse_args().root)
