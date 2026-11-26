"""Size and speed of each variant: parameters, estimated MACs per sentence, model size in fp32 / bf16 /
int8, latency of the whole model and of the MoE head alone (sparse vs dense dispatch), peak inference
memory, and (with --int8_accuracy) test accuracy of the seed-42 checkpoint before and after int8.

    python scripts/efficiency.py --variants B0 B2 B0_base B2_base
    python scripts/efficiency.py --variants B0 B1 B2 B3 B3_r384 B4_c002 B4_c01 B0_base B2_base B5_kd --int8_accuracy

Writes results/efficiency.csv. MACs: encoder linear layers ~ (non-embedding parameters) x tokens, plus
attention 2 x layers x tokens^2 x d; MoE experts ~ expert parameters x experts used per sentence.
MoE head = router + experts + LayerNorm + classifier on the pooled sentence vector.
"""
import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_queue import VARIANTS  # noqa: E402

LIGHTWEIGHT = ["B0", "B1", "B2", "B3", "B3_r384", "B4_c002", "B4_c01", "B0_base", "B2_base", "B5_kd"]


def count(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 2 ** 20


def quantize_int8(model):
    """Dynamic int8 quantization of every nn.Linear (CPU inference).

    torch.ao.quantization is deprecated in recent PyTorch releases (moving to torchao) but still works
    in 2.11; the deprecation warning is silenced here so it does not drown the benchmark output.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*quantiz.*")
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def loader_for(split, tokenizer, limit=None, batch_size=32):
    """DataLoader over an already loaded split (load the dataset once, not per variant)."""
    from src.data.dataset import ViANLIDataset, make_dynamic_padding_collate
    data = split.select(range(min(limit, len(split)))) if limit else split
    return DataLoader(ViANLIDataset(data, tokenizer, 256, dynamic_padding=True), batch_size=batch_size,
                      shuffle=False, collate_fn=make_dynamic_padding_collate(tokenizer.pad_token_id))


def _autocast(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype else torch.no_grad()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run(model, loader, device, dtype=None):
    """(accuracy, seconds per sentence)"""
    model.eval()
    correct = seen = 0
    _sync(device)
    start = time.perf_counter()
    for batch in loader:
        with _autocast(device, dtype):
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        correct += int((out["logits"].argmax(-1).cpu() == batch["label"]).sum())
        seen += len(batch["label"])
    _sync(device)
    return correct / seen, (time.perf_counter() - start) / seen


@torch.no_grad()
def pooled_vectors(model, loader, device, dtype=None, limit=256):
    """Pooled sentence vectors h of real sentences, the input of the MoE head."""
    model.eval()
    rows = []
    for batch in loader:
        with _autocast(device, dtype):
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        rows.append(out["pooled_representation"].float())
        if sum(len(r) for r in rows) >= limit:
            break
    return torch.cat(rows)[:limit]


def is_moe(model):
    return all(hasattr(model, name) for name in ("router", "experts", "layer_norm", "classifier"))


@torch.no_grad()
def benchmark_moe_head(model, pooled, device, dtype=None, sparse=True, batch_size=32, repeats=20):
    """Milliseconds per sentence of router + experts + LayerNorm + classifier on pooled vectors.

    sparse=True passes the selected mask to the experts (only chosen experts run); sparse=False runs
    all experts on every sentence (dense dispatch, what training with STE does).
    """
    model.eval()
    pooled = pooled.to(device)

    def once():
        for start in range(0, len(pooled), batch_size):
            x = pooled[start:start + batch_size]
            with _autocast(device, dtype):
                routed = model.router(x)
                vectors = model.experts(x, routed["selected_mask"] if sparse else None)
                combined = (vectors * routed["routing_weights"].unsqueeze(-1).to(vectors.dtype)).sum(1)
                model.classifier(model.layer_norm(combined))

    once()  # warm-up
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        once()
    _sync(device)
    return 1000 * (time.perf_counter() - start) / (repeats * len(pooled))


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from scripts.dump_teacher_logits import build, find_checkpoint
    from src.utils.viz import load_summary

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["B0", "B2", "B0_base", "B2_base"], choices=list(VARIANTS),
                        help=f"variants to benchmark; all lightweight ones: {' '.join(LIGHTWEIGHT)}")
    parser.add_argument("--latency_sentences", type=int, default=256)
    parser.add_argument("--moe_sentences", type=int, default=256, help="pooled vectors used to time the MoE head")
    parser.add_argument("--int8_accuracy", action="store_true",
                        help="load the seed-42 checkpoint and report test accuracy before and after int8 (CPU, slow)")
    parser.add_argument("--out", default="results/efficiency.csv")
    args = parser.parse_args()

    summary = load_summary(PROJECT)
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    cpu = torch.device("cpu")
    test_split = load_dataset("uitnlp/ViANLI")["test"]  # once for all variants
    tokenizers, lengths = {}, {}
    rows = []
    for variant in args.variants:
        cfg = yaml.safe_load((PROJECT / VARIANTS[variant][2]).read_text(encoding="utf-8"))
        checkpoint = None
        if args.int8_accuracy:
            try:
                checkpoint = find_checkpoint(variant, 42)
            except FileNotFoundError as error:
                print(f"{variant}: {error}; copy checkpoints/<name>_best.pt and .config.json from Drive. Skipped.")
                continue
            cfg = json.loads(checkpoint.with_suffix(".config.json").read_text(encoding="utf-8"))
        backbone = cfg["model"]["backbone"]
        if backbone not in tokenizers:
            tokenizers[backbone] = AutoTokenizer.from_pretrained(backbone)
            lengths[backbone] = float(np.mean([len(tokenizers[backbone](p, h)["input_ids"])
                                               for p, h in zip(test_split["premise"], test_split["hypothesis"])]))
        tokenizer, tokens = tokenizers[backbone], lengths[backbone]
        model = build(variant, cfg)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()

        encoder = getattr(model, "encoder", None)
        experts = getattr(model, "experts", None)
        embeddings = count(getattr(encoder, "embeddings", None))
        config = encoder.config
        used = np.nan
        if experts is not None:
            done = summary[summary["variant"] == variant] if "variant" in summary else summary.iloc[0:0]
            used = float(pd.to_numeric(done["avg_experts_per_sample"], errors="coerce").mean()) if len(done) else np.nan
        expert_each = count(experts) / len(experts.experts) if experts is not None else 0
        encoder_macs = (count(encoder) - embeddings) * tokens + 2 * config.num_hidden_layers * tokens ** 2 * config.hidden_size
        expert_macs = expert_each * (used if not np.isnan(used) else 0)

        row = {"variant": variant, "backbone": backbone,
               "params_total_M": count(model) / 1e6, "params_encoder_M": count(encoder) / 1e6,
               "params_experts_M": count(experts) / 1e6, "experts_per_sentence": used, "mean_tokens": tokens,
               "GMACs_per_sentence": (encoder_macs + expert_macs) / 1e9,
               "expert_share_of_MACs_%": 100 * expert_macs / (encoder_macs + expert_macs),
               "size_fp32_MB": size_mb(model), "size_bf16_MB": size_mb(model) / 2}

        device, dtype, prefix = (gpu, torch.bfloat16, "gpu") if gpu is not None else (cpu, None, "cpu_fp32")
        model.to(device)
        latency = loader_for(test_split, tokenizer, args.latency_sentences)
        run(model, loader_for(test_split, tokenizer, 32), device, dtype)  # warm-up
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_ms = 1000 * run(model, latency, device, dtype)[1]
        row[f"{prefix}_ms_per_sentence"] = total_ms
        if device.type == "cuda":
            row["peak_gpu_infer_mb"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        if is_moe(model):
            pooled = pooled_vectors(model, latency, device, dtype, args.moe_sentences)
            sparse_ms = benchmark_moe_head(model, pooled, device, dtype, sparse=True)
            row[f"{prefix}_moe_sparse_ms_per_sentence"] = sparse_ms
            row[f"{prefix}_moe_dense_ms_per_sentence"] = benchmark_moe_head(model, pooled, device, dtype, sparse=False)
            row["moe_time_share_%"] = 100 * sparse_ms / total_ms
        if checkpoint is not None:
            reference = "test_acc_bf16" if device.type == "cuda" else "test_acc_fp32"
            row[reference] = run(model, loader_for(test_split, tokenizer, batch_size=32), device, dtype)[0]
        model.to(cpu)
        if gpu is not None:
            torch.cuda.empty_cache()

        quantized = quantize_int8(model)
        row["size_int8_MB"] = size_mb(quantized)
        row["cpu_int8_ms_per_sentence"] = 1000 * run(quantized, loader_for(test_split, tokenizer, 64, batch_size=8), cpu)[1]
        if checkpoint is not None:
            row["test_acc_int8"] = run(quantized, loader_for(test_split, tokenizer, batch_size=16), cpu)[0]
            row["int8_acc_drop"] = row[reference] - row["test_acc_int8"]
            row["checkpoint"] = checkpoint.name
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        del model, quantized

    if not rows:
        sys.exit("no variant was benchmarked")
    frame = pd.DataFrame(rows)
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.round(3).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
