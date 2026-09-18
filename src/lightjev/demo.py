"""Offline plumbing demo. Its random tiny model is not a released language model."""
import json
from pathlib import Path


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_demo(output_dir, steps=8):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from .training import train
    from .inference import predict
    from .metrics import evaluate

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo output must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    torch.set_num_threads(1)
    records = {}
    for split in ("train", "dev", "test"):
        rows = []
        for i in range(12):
            on = i % 2 == 0
            rows.append({"id": f"{split}-{i}", "group_id": f"{split}-group-{i}",
                         "state": f"Sensor {split} {i} is {'active' if on else 'inactive'}.",
                         "question": "Is the sensor active?", "kind": "boolean",
                         "candidates": ["false", "true"], "target": [int(not on), int(on)]})
        records[split] = rows
        (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Explicit small vocabulary; no Hub request, pretrained weights, or external data.
    words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'Evaluate', 'this', 'candidate',
             'for', 'the', 'question', 'state', 'Sensor', 'is', 'active', 'inactive',
             'Is', 'sensor', 'false', 'true', 'Decision', 'train', 'dev', 'test',
             ':', '.', '?', '{', '}', '"', ',', *map(str, range(12))]
    vocab = dict.fromkeys(words)
    vocab = {word: index for index, word in enumerate(vocab)}
    raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                                       unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]')
    base = root / 'tiny-backbone'
    tokenizer.save_pretrained(base)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64,
                        max_position_embeddings=256, hidden_dropout_prob=0,
                        attention_probs_dropout_prob=0, pad_token_id=0)
    BertModel(config).save_pretrained(base)
    summary = train(train_path=root / 'train.jsonl', dev_path=root / 'dev.jsonl',
                    output_dir=root / 'checkpoint', model_name=str(base),
                    steps=steps, batch_size=4, lr=1e-3, loss='ce', seed=17,
                    max_length=128, device='cpu')
    predictions = predict(root / 'checkpoint', records['test'], device='cpu')
    _write(root / 'predictions.json', predictions)
    metrics = evaluate(records['test'], predictions)
    report = {"purpose": "offline_end_to_end_smoke_only", "backbone": "random_tiny_bert",
              "pretrained": False, "semantic_capability_claim": False,
              "note": "Template-shared sensor data validates plumbing, not generalization.",
              "training": summary, "test_metrics": metrics}
    _write(root / 'report.json', report)
    return report
