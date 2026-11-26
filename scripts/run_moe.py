import os
import sys
import time
import json
import argparse
import pandas as pd
from datetime import datetime
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.utils.seed import set_seed
from src.utils.metrics import compute_nli_metrics
from src.utils.routing_diagnostics import compute_routing_diagnostics
from src.utils.logging_utils import append_to_summary, save_predictions
from src.data.dataset import load_vianli_dataloaders, ID2LABEL
from src.models.factory import build_moe_model
from src.utils.training import (train_with_validation, save_run_config, resolve_precision,
                                autocast_context, optimizer_steps_per_epoch, append_history, load_teacher)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate NLIMoE Baseline")
    parser.add_argument("--config", type=str, default="configs/nlimoe_dynamic.yaml", help="YAML config")
    parser.add_argument("--routing_type", type=str, default=None, choices=["dynamic", "top_k", "static", "dense"], help="Override routing type")
    parser.add_argument("--top_k", type=int, default=None, help="Top K experts")
    parser.add_argument("--threshold", type=float, default=None, help="Static threshold tau")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dense_sanity_check", action="store_true", help="Run sanity check with all experts active")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "bf16", "fp16", "auto"], help="Mixed precision (auto = bf16 if supported, else fp16)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps (effective batch = batch_size x grad_accum)")
    parser.add_argument("--no_dynamic_padding", action="store_true", help="Pad every sample to max_length instead of per batch")
    return parser.parse_args()

def evaluate_moe(model, dataloader, device, routing_mode=None, threshold=None):
    model.eval()
    all_preds = []
    all_golds = []
    all_probs = []
    all_uids = []
    all_weights = []
    all_masks = []
    all_complexity = []
    all_threshold = []
    all_pooled = []
    all_moe = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            uids = batch["uid"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                routing_mode_override=routing_mode,
                threshold_override=threshold
            )
            loss = outputs["loss"]
            logits = outputs["logits"]

            if loss is not None:
                total_loss += loss.item() * input_ids.size(0)

            logits = logits.float()
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            golds = labels.cpu().tolist()

            all_preds.extend(preds)
            all_golds.extend(golds)
            all_probs.extend(probs)
            all_uids.extend(uids)
            all_weights.append(outputs["routing_weights"].float().cpu())
            all_masks.append(outputs["selected_mask"].float().cpu())
            if outputs.get("moe_representation") is not None:
                all_pooled.append(outputs["pooled_representation"].float().cpu())
                all_moe.append(outputs["moe_representation"].float().cpu())
            if outputs.get("complexity") is not None:
                all_complexity.append(outputs["complexity"].float().reshape(-1).cpu())
                all_threshold.append(outputs["threshold"].float().reshape(-1).cpu())

    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    metrics = compute_nli_metrics(all_golds, all_preds, label_names=["entailment", "neutral", "contradiction"])
    metrics["loss"] = avg_loss

    cat_weights = torch.cat(all_weights, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)
    cat_golds = torch.tensor(all_golds)
    routing_stats = compute_routing_diagnostics(
        routing_weights=cat_weights,
        selected_mask=cat_masks,
        labels=cat_golds,
        label_names=["entailment", "neutral", "contradiction"]
    )

    metrics["avg_experts"] = routing_stats["avg_experts_per_sample"]
    if all_moe:
        pooled, moe = torch.cat(all_pooled), torch.cat(all_moe)
        metrics["h_std"] = float(pooled.std(0).mean())
        metrics["z_norm"] = float(moe.norm(dim=-1).mean())
        metrics["z_zero_frac"] = float((moe == 0).float().mean())
    metrics["routing_entropy"] = routing_stats["mean_routing_entropy"]
    per_sample = {"weights": cat_weights, "mask": cat_masks,
                  "complexity": torch.cat(all_complexity) if all_complexity else None,
                  "threshold": torch.cat(all_threshold) if all_threshold else None}
    return metrics, routing_stats, (all_uids, all_golds, all_preds, all_probs, per_sample)


def _model_display_name(cfg, paper_mode, ste):
    """Readable model column for results/summary.csv, e.g. 'NLIMoE-paper-v3+STE+bottleneck(r=384)+cost0.02'.

    Display only: experiment_name (used by run_queue.is_done and viz.variant_of) is built separately.
    """
    if not paper_mode:
        return "Lightweight-MoE-experimental" if ste else "NLIMoE"
    moe, training = cfg["moe"], cfg["training"]
    name = "NLIMoE-paper-v3" + ("+STE" if ste else "")
    expert_type = moe.get("expert_type", "linear")
    if expert_type == "bottleneck":
        name += f"+bottleneck(r={moe.get('bottleneck_dim', 256)})"
    elif expert_type == "residual_bottleneck":
        name += f"+res_bottleneck(r={moe.get('bottleneck_dim', 256)})"
    if float(moe.get("expert_cost_weight", 0.0) or 0.0) > 0:
        name += f"+cost{moe['expert_cost_weight']}"
    if moe.get("empty_selection_policy", "zero") != "zero":
        name += f"+empty_{moe['empty_selection_policy']}"
    if training.get("class_weights"):
        name += "+cw" + "/".join(str(w) for w in training["class_weights"])
    if training.get("distill"):
        name += "+KD"
    return name


def save_routing_per_sample(path, bundle):
    """Per-sample routing on the test set."""
    uids, golds, preds, _, per_sample = bundle
    mask, weights = per_sample["mask"].numpy(), per_sample["weights"].numpy()
    frame = pd.DataFrame({"uid": uids, "gold": golds, "pred": preds,
                          "correct": [int(g == p) for g, p in zip(golds, preds)],
                          "n_experts": mask.sum(1).astype(int)})
    if per_sample["complexity"] is not None:
        frame["complexity"] = per_sample["complexity"].numpy()
        frame["threshold"] = per_sample["threshold"].numpy()
    for i in range(mask.shape[1]):
        frame[f"mask_{i}"] = mask[:, i].astype(int)
        frame[f"weight_{i}"] = weights[:, i]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    print(f"saved {path}")

def main():
    args = parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_dir, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["training"].get("seed", 42)
    set_seed(seed)
    grad_accum = args.grad_accum if args.grad_accum is not None else int(cfg["training"].get("gradient_accumulation_steps", 1))
    dynamic_padding = (not args.no_dynamic_padding) and cfg["data"].get("dynamic_padding", True)

    routing_type = args.routing_type or cfg["moe"].get("routing_type", "dynamic")
    if args.dense_sanity_check:
        routing_type = "dense"

    top_k = args.top_k or cfg["moe"].get("top_k", 2)
    threshold = args.threshold if args.threshold is not None else cfg["moe"].get("threshold", 0.25)
    batch_size = args.batch_size or cfg["training"].get("batch_size", 32)
    epochs = args.epochs or cfg["training"].get("epochs", 5)
    backbone = cfg["model"].get("backbone", "xlm-roberta-base")
    num_experts = cfg["moe"].get("num_experts", 4)

    exp_name = f"{cfg['experiment']['name']}_{routing_type}"
    if routing_type == "top_k":
        exp_name += f"_top{top_k}"
    elif routing_type == "static":
        exp_name += f"_tau{int(threshold*100):02d}"
    exp_name += f"_seed{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    paper_mode = cfg['model'].get('architecture') == 'paper_nlimoe_v3'
    ste = cfg["moe"].get("threshold_gradient", "hard") == "straight_through"
    if ste and not paper_mode:
        exp_name = f"lightweight_experimental_{exp_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")  # TF32 for the remaining fp32 matmuls
    precision = resolve_precision(args.precision or cfg["training"].get("precision", "fp32"), device)
    print(exp_name)
    print(f"config {args.config} | encoder_dropout={cfg['model'].get('encoder_dropout')} | "
          f"scheduler={cfg['training'].get('scheduler')} | warmup_ratio={cfg['training'].get('warmup_ratio')}")
    print(f"Routing: {routing_type} | Experts: {num_experts} | Seed: {seed} | Device: {device} | "
          f"Batch: {batch_size} x accum {grad_accum} | Precision: {precision} | Dynamic padding: {dynamic_padding}")

    train_loader, val_loader, test_loader, meta = load_vianli_dataloaders(
        dataset_name=cfg["data"]["dataset_name"],
        tokenizer_name=backbone,
        max_length=cfg["data"].get("max_length", 256),
        batch_size=batch_size,
        dynamic_padding=dynamic_padding
    )

    model = build_moe_model(cfg, routing_type, top_k, threshold).to(device)

    sample_batch = next(iter(train_loader))
    sample_ids = sample_batch["input_ids"][:2].to(device)
    sample_mask = sample_batch["attention_mask"][:2].to(device)
    sample_labels = sample_batch["label"][:2].to(device)
    test_out = model(sample_ids, sample_mask, labels=sample_labels)
    test_out["loss"].backward()
    router_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())
    expert_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.experts.parameters())
    print(f"grad check: router={router_grad} experts={expert_grad}")
    if paper_mode and routing_type == "dynamic":
        gate_grad = sum(float(p.grad.abs().sum()) for p in model.router.complexity_gate.parameters() if p.grad is not None)
        print(f"|dL/dw_c| = {gate_grad:.3e}")
        if ste and gate_grad == 0:
            raise RuntimeError("STE is enabled but the complexity gate received no gradient")
        if not ste:
            print("hard mask: complexity gate gets no gradient")
    if routing_type == "dynamic" and not paper_mode:
        missing = [n for n, p in model.router.dynamic_thresh_head.named_parameters() if p.grad is None]
        if missing:
            raise RuntimeError(
                f"Dynamic threshold has no gradient: {missing}. "
                "Use configs/lightweight_moe_experimental.yaml for the experimental estimator."
            )
    model.zero_grad()

    lr_backbone = float(cfg["training"].get("lr_backbone", 2e-5))
    lr_moe = float(cfg["training"].get("lr_moe", 1e-4))
    weight_decay = cfg["training"].get("weight_decay", 0.01)

    optimizer_groups = [
        {"params": [p for n, p in model.encoder.named_parameters()], "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": [p for n, p in model.router.named_parameters()], "lr": lr_moe, "weight_decay": weight_decay},
        {"params": [p for n, p in model.experts.named_parameters()], "lr": lr_moe, "weight_decay": weight_decay}
    ]
    other_parameters = [p for name, p in model.named_parameters()
                        if not name.startswith(('encoder.', 'router.', 'experts.'))]
    if other_parameters:
        optimizer_groups.append({'params': other_parameters, 'lr': lr_moe, 'weight_decay': weight_decay})
    # fused kernel: same update, fewer GPU launches
    optimizer = AdamW(optimizer_groups, eps=float(cfg['training'].get('adam_epsilon', 1e-8)), fused=device.type == "cuda")
    total_steps = optimizer_steps_per_epoch(train_loader, grad_accum) * epochs
    scheduler = None
    if cfg['training'].get('scheduler', 'linear') == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg['training'].get('warmup_ratio', 0.1)), total_steps)

    best_checkpoint_path = os.path.join(project_dir, "checkpoints", f"{exp_name}_best.pt")
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    start_time = time.time()

    def validation():
        metrics, routing, _ = evaluate_moe(model, val_loader, device, routing_mode=routing_type, threshold=threshold)
        print(f"  experts/sample {routing['avg_experts_per_sample']:.2f}")
        return metrics

    cfg['training'].update(seed=seed, batch_size=batch_size, epochs=epochs,
                           gradient_accumulation_steps=grad_accum, precision=precision)
    cfg['data']['dynamic_padding'] = dynamic_padding
    cfg['moe'].update(routing_type=routing_type, top_k=top_k, threshold=threshold)
    save_run_config(cfg, best_checkpoint_path)
    history_path = os.path.join(project_dir, "logs", f"{exp_name}.jsonl")

    def grad_probe(net):
        gate = getattr(net.router, "complexity_gate", None)
        if gate is None:
            return {}
        grads = [p.grad.float().norm() for p in gate.parameters() if p.grad is not None]
        return {"grad_norm_wc": float(torch.stack(grads).norm()) if grads else 0.0}

    teacher = None
    distill = cfg['training'].get('distill')
    if distill:
        teacher_path = os.path.join(project_dir, distill['teacher'])
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"{teacher_path} not found; create it with scripts/dump_teacher_logits.py")
        teacher = load_teacher(teacher_path)
        print(f"distillation: {len(teacher)} teacher logits from {distill['teacher']}, "
              f"alpha={distill.get('alpha', 0.5)}, T={distill.get('temperature', 2.0)}")

    best_val_metrics = train_with_validation(
        model, train_loader, device, optimizer, scheduler, epochs, cfg['training'],
        validation, best_checkpoint_path, history_path=history_path, grad_probe=grad_probe, teacher=teacher)
    best_val_acc = best_val_metrics['accuracy']

    train_time = round(time.time() - start_time, 2)

    print("\ntest set:")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    inf_start = time.time()
    with autocast_context(device, precision):
        test_metrics, test_routing, test_bundle = evaluate_moe(model, test_loader, device, routing_mode=routing_type, threshold=threshold)
    inference_time = round(time.time() - inf_start, 2)

    print(f"acc {test_metrics['accuracy']:.4f}  macro-F1 {test_metrics['macro_f1']:.4f}")

    pred_path = os.path.join(project_dir, "predictions", f"{exp_name}.csv")
    uids, golds, preds, probs, _ = test_bundle
    save_predictions(pred_path, uids, golds, preds, probs, label_map=ID2LABEL)
    save_routing_per_sample(os.path.join(project_dir, "results", "routing_stats", f"{exp_name}_per_sample.csv"), test_bundle)

    routing_stat_path = os.path.join(project_dir, "results", "routing_stats", f"{exp_name}_diagnostics.json")
    os.makedirs(os.path.dirname(routing_stat_path), exist_ok=True)
    with open(routing_stat_path, "w", encoding="utf-8") as f:
        json.dump(test_routing, f, indent=2, ensure_ascii=False)

    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2)

    summary_csv = os.path.join(project_dir, "results", "summary.csv")
    record = {
        "experiment_name": exp_name,
        "model": _model_display_name(cfg, paper_mode, ste),
        "backbone": backbone,
        "router_type": cfg["moe"].get("router_type", "linear"),
        "routing_type": routing_type,
        "threshold": threshold if routing_type == "static" else "N/A",
        "top_k": top_k if routing_type == "top_k" else "N/A",
        "num_experts": num_experts,
        "seed": seed,
        "val_accuracy": round(best_val_acc, 4),
        "val_macro_f1": round(best_val_metrics["macro_f1"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "avg_experts_per_sample": round(test_routing["avg_experts_per_sample"], 2),
        "routing_entropy": round(test_routing["mean_routing_entropy"], 4),
        "train_time": train_time,
        "inference_time": inference_time,
        "peak_gpu_memory": f"{peak_mem} GB",
        "notes": f"Routing: {routing_type} | {precision} | batch {batch_size}x{grad_accum}"
                 + (f" | distilled from {distill['teacher']}" if distill else "")
    }
    append_to_summary(record, summary_csv)
    append_history(history_path, {"type": "test", "accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"],
                                  "avg_experts": test_routing["avg_experts_per_sample"],
                                  "empty_selection_pct": test_routing["empty_selection_pct"], "train_time": train_time,
                                  "inference_time": inference_time, "gpu_mem_gb": peak_mem,
                                  "expert_params": sum(p.numel() for p in model.experts.parameters())})
    print(f"done: {exp_name}")

if __name__ == "__main__":
    main()
