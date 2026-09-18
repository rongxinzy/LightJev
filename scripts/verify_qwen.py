#!/usr/bin/env python3
"""Opt-in real Qwen smoke: one full-parameter update and checkpoint roundtrip.

Downloads ~1.5 GB of pretrained weights; training uses considerably more memory.
This is not a quality benchmark and is intentionally excluded from offline CI.
"""
import argparse
import json
import platform
from pathlib import Path
import time

import torch
from lightjev.defaults import DEFAULT_MODEL, DEFAULT_REVISION
from lightjev.training import train
from lightjev.inference import predict
from lightjev.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='runs/qwen-smoke')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        parser.error('Use a new or empty output directory')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cases = {
        'train': [('I was charged twice for my order.', [1, 0]),
                  ('The app crashes every time it opens.', [0, 1])],
        'dev': [('Please explain the extra fee on this invoice.', [1, 0]),
                ('The settings screen freezes after login.', [0, 1])],
        'test': [('I need a refund for a duplicate payment.', [1, 0]),
                 ('Uploading a file causes an application error.', [0, 1])],
    }
    records = {}
    for split, cases_for_split in cases.items():
        rows = [{'id': f'{split}-{i}', 'group_id': f'{split}-source-{i}',
                 'state': state, 'question': 'Which team should handle this issue?',
                 'kind': 'choice', 'candidates': ['Billing: charges, invoices and refunds',
                                                'Technical: software faults and errors'],
                 'target': target} for i, (state, target) in enumerate(cases_for_split)]
        records[split] = rows
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    started = time.perf_counter()
    summary = train(root / 'train.jsonl', root / 'dev.jsonl', root / 'checkpoint',
                    steps=1, batch_size=1, max_length=192, device=args.device, seed=17)
    predictions = predict(root / 'checkpoint', records['test'], device=args.device)
    report = {'purpose': 'real_pretrained_backbone_integration_only',
              'model': DEFAULT_MODEL, 'revision': DEFAULT_REVISION,
              'device': args.device, 'platform': platform.system() + ' ' + platform.machine(),
              'full_parameter_steps': 1, 'train_questions': 2, 'dev_questions': 2, 'test_questions': 2,
              'versions': summary['versions'], 'best_dev': summary['best_dev'],
              'test_metrics': evaluate(records['test'], predictions),
              'predictions': predictions, 'elapsed_seconds': time.perf_counter() - started,
              'capability_benchmark': False, 'calibration_guaranteed': False,
              'note': 'One update plus save/reload proves integration, not useful decision quality.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
