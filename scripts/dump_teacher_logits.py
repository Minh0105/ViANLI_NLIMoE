"""Save teacher logits on the ViANLI training set for knowledge distillation.

Loads the best checkpoint of each (variant, seed) run and averages their logits:

    python scripts/dump_teacher_logits.py --variants B2 --seeds 42 123 2026 --out results/teacher/B2_large.pt

The student config points to the output file under training.distill.teacher.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_queue import VARIANTS, experiment_prefix
from src.data.dataset import ViANLIDataset, make_dynamic_padding_collate
from src.models.factory import build_moe_model
from src.models.xlmr_baseline import XLMRobertaNLI
from src.utils.training import autocast_context, resolve_precision


def find_checkpoint(variant, seed):
    runs = sorted((PROJECT / "checkpoints").glob(f"{experiment_prefix(variant)}_seed{seed}_*_best.pt"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no checkpoint for {variant} seed {seed} in checkpoints/")
    return runs[-1]


def build(variant, cfg):
    if VARIANTS[variant][1].endswith("run_xlmr.py"):
        model = cfg["model"]
        return XLMRobertaNLI(backbone_name=model["backbone"], num_labels=model.get("num_labels", 3),
                             dropout_prob=model.get("dropout", 0.1), pooling=model.get("pooling", "cls"),
                             encoder_dropout=model.get("encoder_dropout"))
    moe = cfg["moe"]
    return build_moe_model(cfg, moe.get("routing_type"), moe.get("top_k"), moe.get("threshold"))


@torch.no_grad()
def logits_of(model, loader, device, precision):
    model.eval()
    uids, rows, labels = [], [], []
    for batch in loader:
        with autocast_context(device, precision):
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        rows.append(out["logits"].float().cpu())
        labels.append(batch["label"])
        uids.extend(batch["uid"])
    logits, labels = torch.cat(rows), torch.cat(labels)
    return uids, logits, labels, float((logits.argmax(-1) == labels).float().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["B2"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--out", default="results/teacher/B2_large.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--precision", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args.precision, device)
    train = load_dataset("uitnlp/ViANLI")["train"]
    total, uids, sources = None, None, []
    for variant in args.variants:
        for seed in args.seeds:
            checkpoint = find_checkpoint(variant, seed)
            cfg = json.loads(checkpoint.with_suffix(".config.json").read_text(encoding="utf-8"))
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
            loader = DataLoader(ViANLIDataset(train, tokenizer, cfg["data"].get("max_length", 256), dynamic_padding=True),
                                batch_size=args.batch_size, shuffle=False,
                                collate_fn=make_dynamic_padding_collate(tokenizer.pad_token_id))
            model = build(variant, cfg).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            run_uids, logits, labels, accuracy = logits_of(model, loader, device, precision)
            if uids is not None and run_uids != uids:
                raise RuntimeError("training set order differs between runs")
            uids, total = run_uids, logits if total is None else total + logits
            sources.append(checkpoint.name)
            print(f"{variant} seed {seed}: train accuracy {accuracy:.4f}  ({checkpoint.name})")
            del model
            torch.cuda.empty_cache()

    logits = total / len(sources)
    print(f"teacher (mean of {len(sources)} runs): train accuracy {float((logits.argmax(-1) == labels).float().mean()):.4f}")
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"uids": uids, "logits": logits, "sources": sources}, out)
    print(f"saved {len(uids)} teacher logits from {len(sources)} runs to {out}")


if __name__ == "__main__":
    main()
