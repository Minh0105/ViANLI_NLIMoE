"""Size and speed of each variant: parameters, estimated MACs per sentence, model size in fp32 / bf16 /
int8, GPU and CPU latency, and (optionally) test accuracy after int8 dynamic quantization.

    python scripts/efficiency.py --variants B0 B2 B0_base B2_base
    python scripts/efficiency.py --variants B2 B2_base --int8_accuracy   # uses each variant's seed-42 checkpoint

Writes results/efficiency.csv. MACs: encoder linear layers ~ (non-embedding parameters) x tokens, plus
attention 2 x layers x tokens^2 x d; MoE experts ~ expert parameters x experts used per sentence.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.dump_teacher_logits import build, find_checkpoint
from scripts.run_queue import VARIANTS
from src.data.dataset import ViANLIDataset, make_dynamic_padding_collate
from src.utils.viz import load_summary


def count(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 2 ** 20


def loader_for(tokenizer, split, limit=None, batch_size=32):
    data = load_dataset("uitnlp/ViANLI")[split]
    if limit:
        data = data.select(range(min(limit, len(data))))
    return DataLoader(ViANLIDataset(data, tokenizer, 256, dynamic_padding=True), batch_size=batch_size,
                      shuffle=False, collate_fn=make_dynamic_padding_collate(tokenizer.pad_token_id))


@torch.no_grad()
def run(model, loader, device, dtype=None):
    """(accuracy, seconds per sentence)"""
    model.eval()
    correct = seen = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=dtype) if dtype else torch.no_grad():
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        correct += int((out["logits"].argmax(-1).cpu() == batch["label"]).sum())
        seen += len(batch["label"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return correct / seen, (time.perf_counter() - start) / seen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["B0", "B2", "B0_base", "B2_base"], choices=list(VARIANTS))
    parser.add_argument("--latency_sentences", type=int, default=256)
    parser.add_argument("--int8_accuracy", action="store_true", help="evaluate the seed-42 checkpoint after int8 quantization (CPU, slow)")
    parser.add_argument("--out", default="results/efficiency.csv")
    args = parser.parse_args()

    summary = load_summary(PROJECT)
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    cpu = torch.device("cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = []
    for variant in args.variants:
        cfg = yaml.safe_load((PROJECT / VARIANTS[variant][2]).read_text(encoding="utf-8"))
        checkpoint = None
        if args.int8_accuracy:
            checkpoint = find_checkpoint(variant, 42)
            cfg = json.loads(checkpoint.with_suffix(".config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
        model = build(variant, cfg)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()

        encoder = getattr(model, "encoder", None) or getattr(model, "xlmr", None) or getattr(model, "backbone", None)
        experts = getattr(model, "experts", None)
        embeddings = count(getattr(encoder, "embeddings", None))
        config = encoder.config
        test_lengths = [len(tokenizer(p, h)["input_ids"]) for p, h in
                        zip(*[load_dataset("uitnlp/ViANLI")["test"][c] for c in ("premise", "hypothesis")])]
        tokens = float(np.mean(test_lengths))
        used = np.nan
        if experts is not None:
            done = summary[summary["variant"] == variant] if "variant" in summary else summary.iloc[0:0]
            used = float(done["avg_experts_per_sample"].mean()) if len(done) else np.nan
        expert_each = count(experts) / len(experts.experts) if experts is not None else 0
        encoder_macs = (count(encoder) - embeddings) * tokens + 2 * config.num_hidden_layers * tokens ** 2 * config.hidden_size
        expert_macs = expert_each * (used if not np.isnan(used) else 0)

        row = {"variant": variant, "backbone": cfg["model"]["backbone"],
               "params_total_M": count(model) / 1e6, "params_encoder_M": count(encoder) / 1e6,
               "params_experts_M": count(experts) / 1e6, "experts_per_sentence": used,
               "GMACs_per_sentence": (encoder_macs + expert_macs) / 1e9, "expert_share_of_MACs_%": 100 * expert_macs / (encoder_macs + expert_macs),
               "size_fp32_MB": size_mb(model), "size_bf16_MB": size_mb(model) / 2}

        latency = loader_for(tokenizer, "test", args.latency_sentences)
        if gpu is not None:
            model.to(gpu)
            run(model, loader_for(tokenizer, "test", 32), gpu, torch.bfloat16)  # warm-up
            row["gpu_bf16_ms_per_sentence"] = 1000 * run(model, latency, gpu, torch.bfloat16)[1]
            model.to(cpu)
            torch.cuda.empty_cache()
        quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        row["size_int8_MB"] = size_mb(quantized)
        row["cpu_int8_ms_per_sentence"] = 1000 * run(quantized, loader_for(tokenizer, "test", 64, batch_size=8), cpu)[1]
        if checkpoint is not None:
            row["test_acc_int8"] = run(quantized, loader_for(tokenizer, "test", batch_size=16), cpu)[0]
            row["checkpoint"] = checkpoint.name
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        del model, quantized

    frame = pd.DataFrame(rows)
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.round(3).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
