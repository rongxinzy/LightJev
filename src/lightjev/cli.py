"""LightJev command-line tools."""
import argparse
import json
from pathlib import Path
from .defaults import DEFAULT_MODEL


def _emit(value, output):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


def main():
    parser = argparse.ArgumentParser(description='LightJev: train and evaluate candidate decisions')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Offline tiny random model: train, save, reload, evaluate')
    demo.add_argument('--output-dir', default='runs/demo')
    demo.add_argument('--steps', type=int, default=8)
    train = sub.add_parser('train', help='Fine-tune a backbone and decision head')
    train.add_argument('--train', required=True)
    train.add_argument('--dev', required=True)
    train.add_argument('--output-dir', required=True)
    train.add_argument('--model', default=DEFAULT_MODEL)
    train.add_argument('--revision')
    train.add_argument('--steps', type=int, default=20)
    train.add_argument('--batch-size', type=int, default=2)
    train.add_argument('--lr', type=float, default=2e-5)
    train.add_argument('--loss', choices=['ce', 'brier'], default='ce')
    train.add_argument('--seed', type=int, default=17)
    train.add_argument('--max-length', type=int, default=512)
    train.add_argument('--device', default='cpu')
    train.add_argument('--head-steps', type=int, default=0)
    train.add_argument('--head-lr', type=float)
    train.add_argument('--head-warmup-lr', type=float, default=1e-3)
    train.add_argument('--eval-every', type=int, default=1)
    train.add_argument('--grad-accum-steps', type=int, default=1)
    train.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    train.add_argument('--gradient-checkpointing', action='store_true')
    train.add_argument('--selection-metric', choices=['ce', 'brier'], default='ce')
    pred = sub.add_parser('predict', help='Load a local checkpoint; emit normalized candidate scores')
    pred.add_argument('--checkpoint', required=True)
    pred.add_argument('--input', required=True)
    pred.add_argument('--output')
    pred.add_argument('--device', default='cpu')
    pred.add_argument('--max-length', type=int)
    pred.add_argument('--temperature', type=float, default=1)
    for command in ('evaluate', 'calibrate'):
        p = sub.add_parser(command, help='Use held-out labels with saved predictions')
        p.add_argument('--input', required=True)
        p.add_argument('--predictions', required=True)
        p.add_argument('--output')
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            from .demo import run_demo
            result = run_demo(args.output_dir, args.steps)
        elif args.command == 'train':
            from .training import train as run_train
            result = run_train(args.train, args.dev, args.output_dir, model_name=args.model,
                               steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                               loss=args.loss, seed=args.seed, max_length=args.max_length,
                               device=args.device, revision=args.revision, head_steps=args.head_steps,
                               head_lr=args.head_lr, head_warmup_lr=args.head_warmup_lr,
                               eval_every=args.eval_every, grad_accum_steps=args.grad_accum_steps,
                               precision=args.precision, gradient_checkpointing=args.gradient_checkpointing,
                               selection_metric=args.selection_metric)
        elif args.command == 'predict':
            from .schema import load_records
            from .inference import predict
            result = predict(args.checkpoint, load_records(args.input), max_length=args.max_length,
                             device=args.device, temperature=args.temperature)
        else:
            from .schema import load_records
            from .metrics import evaluate, fit_temperature
            records = load_records(args.input)
            predictions = json.loads(Path(args.predictions).read_text())
            result = (evaluate if args.command == 'evaluate' else fit_temperature)(records, predictions)
        _emit(result, getattr(args, 'output', None))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
