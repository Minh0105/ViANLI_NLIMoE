import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.utils.seed import set_seed
from src.data.dataset import load_vianli_dataloaders
from src.models.factory import build_moe_model
from scripts.run_moe import evaluate_moe

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep static threshold tau on Validation Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained NLIMoE checkpoint")
    parser.add_argument("--config", type=str, default="configs/nlimoe_dynamic.yaml", help="YAML config")
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=0.40)
    parser.add_argument("--tau_step", type=float, default=0.05)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.tau_step <= 0 or not 0 <= args.tau_min <= args.tau_max <= 1:
        raise ValueError("Require tau_step > 0 and 0 <= tau_min <= tau_max <= 1")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_dir, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Threshold Sweep on Validation Set (Rule: Không tune trên test set) ===")

    _, val_loader, test_loader, _ = load_vianli_dataloaders(
        dataset_name=cfg["data"]["dataset_name"],
        tokenizer_name=cfg["model"]["backbone"],
        max_length=cfg["data"].get("max_length", 256),
        batch_size=cfg["training"].get("batch_size", 32)
    )

    if args.checkpoint == "latest":
        candidates = list((Path(project_dir) / 'checkpoints').glob(f"{cfg['experiment']['name']}_*_best.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint for this config; train first or pass --checkpoint explicitly")
        chk_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
    else:
        chk_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(project_dir, args.checkpoint)
    state = torch.load(chk_path, map_location="cpu", weights_only=True)
    paper_mode = cfg['model'].get('architecture') == 'paper_nlimoe_v3'
    paper_checkpoint = any(k.startswith('router.complexity_gate.') for k in state)
    if paper_mode != paper_checkpoint:
        raise ValueError('Checkpoint architecture and YAML do not match; use the saved run config.')
    checkpoint_mode = 'dynamic' if (paper_checkpoint or any(k.startswith('router.dynamic_thresh_head.') for k in state)) else 'static'
    model = build_moe_model(cfg, routing_type=checkpoint_mode).to(device)

    model.load_state_dict(state)
    print(f"Loaded checkpoint from: {chk_path}")

    taus = []
    curr = args.tau_min
    while curr <= args.tau_max + 1e-6:
        taus.append(round(curr, 2))
        curr += args.tau_step

    results = []
    best_val_acc = -1.0
    best_tau = None

    for tau in taus:
        metrics, routing, _ = evaluate_moe(model, val_loader, device, routing_mode="static", threshold=tau)
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        avg_exp = routing["avg_experts_per_sample"]
        entropy = routing["mean_routing_entropy"]
        print(f"tau: {tau:.2f} | Val Acc: {acc:.4f} | Val Macro-F1: {f1:.4f} | Avg Experts: {avg_exp:.2f} | Entropy: {entropy:.3f}")

        results.append({
            "threshold": tau,
            "val_accuracy": acc,
            "val_macro_f1": f1,
            "avg_experts_per_sample": avg_exp,
            "routing_entropy": entropy
        })

        if acc > best_val_acc:
            best_val_acc = acc
            best_tau = tau

    print(f"\n[Validation Result] Best threshold chosen: tau = {best_tau:.2f} (Val Acc: {best_val_acc:.4f})")

    print(f"--- Evaluating Test Set with chosen tau = {best_tau:.2f} ---")
    test_metrics, test_routing, _ = evaluate_moe(model, test_loader, device, routing_mode="static", threshold=best_tau)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(project_dir, "results", "threshold_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved threshold sweep results to {out_csv}")

    try:
        plot_dir = os.path.join(project_dir, "results", "routing_stats")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "threshold_sweep_curve.png")

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'tab:blue'
        ax1.set_xlabel('Threshold (tau)')
        ax1.set_ylabel('Val Accuracy', color=color)
        ax1.plot(df['threshold'], df['val_accuracy'], color=color, marker='o', label='Val Acc')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Avg Experts / Sample', color=color)
        ax2.plot(df['threshold'], df['avg_experts_per_sample'], color=color, marker='s', linestyle='--', label='Avg Experts')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Threshold Sweep on Validation Set (Rule 6: Never tune on Test)')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved threshold curve plot to {plot_path}")
    except Exception as e:
        print(f"[Plot Warning] Could not save plot: {e}")

if __name__ == "__main__":
    main()
