import os
import sys
import time
import argparse
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
from src.utils.logging_utils import append_to_summary, save_predictions
from src.data.dataset import load_vianli_dataloaders, ID2LABEL
from src.models.xlmr_baseline import XLMRobertaNLI
from src.utils.training import (train_with_validation, save_run_config, resolve_precision,
                                autocast_context, optimizer_steps_per_epoch, append_history)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate XLM-R Dense Baseline")
    parser.add_argument("--config", type=str, default="configs/xlmr_base.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--overfit_sanity_check", action="store_true", help="Run sanity check on 32-64 samples")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "bf16", "fp16", "auto"], help="Mixed precision (auto = bf16 if supported, else fp16)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps (effective batch = batch_size x grad_accum)")
    parser.add_argument("--no_dynamic_padding", action="store_true", help="Pad every sample to max_length instead of per batch")
    return parser.parse_args()

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_golds = []
    all_probs = []
    all_uids = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            uids = batch["uid"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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

    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    metrics = compute_nli_metrics(all_golds, all_preds, label_names=["entailment", "neutral", "contradiction"])
    metrics["loss"] = avg_loss

    return metrics, (all_uids, all_golds, all_preds, all_probs)

def main():
    args = parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_dir, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["training"].get("seed", 42)
    set_seed(seed)

    batch_size = args.batch_size if args.batch_size is not None else cfg["training"].get("batch_size", 32)
    epochs = args.epochs if args.epochs is not None else cfg["training"].get("epochs", 5)
    lr = args.lr if args.lr is not None else float(cfg["training"].get("lr", 2e-5))
    max_length = cfg["data"].get("max_length", 256)
    backbone = cfg["model"].get("backbone", "xlm-roberta-base")
    base_name = cfg['experiment']['name'].rsplit('_seed', 1)[0]
    exp_name = f"{base_name}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if args.overfit_sanity_check:
        exp_name += "_sanity_check"
        epochs = args.epochs if args.epochs is not None else 100
    grad_accum = args.grad_accum if args.grad_accum is not None else int(cfg["training"].get("gradient_accumulation_steps", 1))
    dynamic_padding = (not args.no_dynamic_padding) and cfg["data"].get("dynamic_padding", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args.precision or cfg["training"].get("precision", "fp32"), device)
    print(exp_name)
    print(f"config {args.config} | encoder_dropout={cfg['model'].get('encoder_dropout')} | "
          f"scheduler={cfg['training'].get('scheduler')} | warmup_ratio={cfg['training'].get('warmup_ratio')}")
    print(f"Device: {device} | Seed: {seed} | Backbone: {backbone} | Batch: {batch_size} x accum {grad_accum} | "
          f"Epochs: {epochs} | Precision: {precision} | Dynamic padding: {dynamic_padding}")

    overfit_size = 32 if args.overfit_sanity_check else None
    train_loader, val_loader, test_loader, meta = load_vianli_dataloaders(
        dataset_name=cfg["data"]["dataset_name"],
        tokenizer_name=backbone,
        max_length=max_length,
        batch_size=batch_size,
        overfit_subset_size=overfit_size,
        dynamic_padding=dynamic_padding
    )

    model = XLMRobertaNLI(
        backbone_name=backbone,
        num_labels=cfg["model"].get("num_labels", 3),
        dropout_prob=cfg["model"].get("dropout", 0.1),
        pooling=cfg["model"].get("pooling", "cls"),
        encoder_dropout=cfg["model"].get("encoder_dropout")
    ).to(device)

    sample_batch = next(iter(train_loader))
    sample_ids = sample_batch["input_ids"][:2].to(device)
    sample_mask = sample_batch["attention_mask"][:2].to(device)
    chk_out = model(sample_ids, sample_mask, return_hidden_states=True)
    print(f"[Shape Check] input_ids: {sample_ids.shape}, pooled: {chk_out['pooled_representation'].shape}, logits: {chk_out['logits'].shape}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg["training"].get("weight_decay", 0.01)
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=float(cfg['training'].get('adam_epsilon', 1e-8)))
    total_steps = optimizer_steps_per_epoch(train_loader, grad_accum) * epochs
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.1))
    scheduler = None
    if cfg['training'].get('scheduler', 'linear') == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_checkpoint_path = os.path.join(project_dir, "checkpoints", f"{exp_name}_best.pt")
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    start_time = time.time()

    cfg['training'].update(seed=seed, batch_size=batch_size, epochs=epochs, lr=lr,
                           gradient_accumulation_steps=grad_accum, precision=precision)
    cfg['data']['dynamic_padding'] = dynamic_padding
    save_run_config(cfg, best_checkpoint_path)
    history_path = None if args.overfit_sanity_check else os.path.join(project_dir, "logs", f"{exp_name}.jsonl")
    best_val_metrics = train_with_validation(
        model, train_loader, device, optimizer, scheduler, epochs, cfg['training'],
        lambda: evaluate(model, val_loader, device)[0], best_checkpoint_path,
        sanity=args.overfit_sanity_check, history_path=history_path)
    best_val_acc = best_val_metrics['accuracy']
    sanity_passed = best_val_acc >= 0.95
    if args.overfit_sanity_check and sanity_passed:
        print(f"[Sanity Check Passed] Accuracy on training subset: {best_val_acc:.4f}")

    train_time = round(time.time() - start_time, 2)

    if args.overfit_sanity_check:
        print(f"[Sanity Result] Passed: {sanity_passed} | Best train-subset accuracy: {best_val_acc:.4f}")
        if not sanity_passed:
            raise RuntimeError("Overfit sanity check did not reach 95% training accuracy")
        return

    print("\ntest set:")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    inf_start = time.time()
    with autocast_context(device, precision):
        test_metrics, test_preds_bundle = evaluate(model, test_loader, device)
    inference_time = round(time.time() - inf_start, 2)

    print(f"acc {test_metrics['accuracy']:.4f}  macro-F1 {test_metrics['macro_f1']:.4f}")

    pred_path = os.path.join(project_dir, "predictions", f"{exp_name}.csv")
    uids, golds, preds, probs = test_preds_bundle
    save_predictions(
        save_path=pred_path,
        uids=uids,
        gold_labels=golds,
        pred_labels=preds,
        probabilities=probs,
        label_map=ID2LABEL
    )

    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2)

    summary_csv = os.path.join(project_dir, "results", "summary.csv")
    record = {
        "experiment_name": exp_name,
        "model": "XLM-R",
        "backbone": backbone,
        "router_type": "none",
        "routing_type": "none",
        "threshold": "N/A",
        "top_k": "N/A",
        "num_experts": 1,
        "seed": seed,
        "val_accuracy": round(best_val_acc, 4),
        "val_macro_f1": round(best_val_metrics["macro_f1"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "avg_experts_per_sample": 1.0,
        "routing_entropy": 0.0,
        "train_time": train_time,
        "inference_time": inference_time,
        "peak_gpu_memory": f"{peak_mem} GB",
        "notes": ("Sanity check" if args.overfit_sanity_check else "Dense baseline") + f" | {precision} | batch {batch_size}x{grad_accum}"
    }
    append_to_summary(record, summary_csv)
    append_history(history_path, {"type": "test", "accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"],
                                  "train_time": train_time, "inference_time": inference_time, "gpu_mem_gb": peak_mem})
    print(f"done: {exp_name}")

if __name__ == "__main__":
    main()
