"""Run experiments in sequence, skipping (variant, seed) pairs already in results/summary.csv.

    python scripts/run_queue.py --variants B0 B1 B2 B3 --seeds 42 123 2026
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

# variant -> (description, script, config, name prefix)
VARIANTS = {
    "B0": ("XLM-R Large (no MoE)", "scripts/run_xlmr.py", "configs/xlmr_large.yaml", "xlmr_large_paper_v5"),
    "B1": ("NLIMoE, paper (hard mask)", "scripts/run_moe.py", "configs/nlimoe_dynamic.yaml", "nlimoe_paper_v5_dynamic_dynamic"),
    "B2": ("NLIMoE + STE", "scripts/run_moe.py", "configs/nlimoe_ste.yaml", "nlimoe_paper_v5_ste_dynamic"),
    "B3": ("NLIMoE + STE + bottleneck", "scripts/run_moe.py", "configs/nlimoe_ste_bottleneck.yaml", "nlimoe_paper_v5_ste_bottleneck_dynamic"),
    "A_topk": ("Ablation: paper Top-K (5 experts, K=4)", "scripts/run_moe.py", "configs/nlimoe_topk.yaml", "nlimoe_paper_v5_topk_top_k_top4"),
    "A_nobal": ("Ablation: B2 without balance loss", "scripts/run_moe.py", "configs/nlimoe_ste_nobalance.yaml", "nlimoe_paper_v5_ste_nobalance_dynamic"),
}


def experiment_prefix(variant):
    """Experiment-name prefix used by the run scripts."""
    import yaml
    _, script, config, fallback = VARIANTS[variant]
    cfg = yaml.safe_load((PROJECT / config).read_text(encoding="utf-8"))
    name = cfg["experiment"]["name"]
    if script.endswith("run_xlmr.py"):
        return name.rsplit("_seed", 1)[0]
    routing = cfg["moe"].get("routing_type", "dynamic")
    prefix = f"{name}_{routing}"
    if routing == "top_k":
        prefix += f"_top{cfg['moe'].get('top_k', 2)}"
    return prefix or fallback


def settings(variant):
    """Short description of the training settings, e.g. 'v4 · encoder dropout 0.1 · linear lr, 10% warmup'."""
    import yaml
    cfg = yaml.safe_load((PROJECT / VARIANTS[variant][2]).read_text(encoding="utf-8"))
    version = re.search(r"_v(\d+)", cfg["experiment"]["name"])
    encoder = cfg["model"].get("encoder_dropout")
    training = cfg["training"]
    lr = (f"linear lr, {training.get('warmup_ratio', 0):.0%} warmup" if training.get("scheduler") == "linear"
          else "constant lr")
    clip = training.get("max_grad_norm")
    return " · ".join([f"v{version.group(1)}" if version else "?",
                       f"encoder dropout {encoder if encoder is not None else '0.1'}", lr,
                       f"clip {clip}" if clip else "no clip"])


def finished_runs(summary_csv=PROJECT / "results" / "summary.csv"):
    if not summary_csv.exists():
        return pd.DataFrame(columns=["experiment_name", "seed"])
    frame = pd.read_csv(summary_csv)
    return frame[~frame["experiment_name"].astype(str).str.contains("sanity_check")]


def is_done(variant, seed, finished):
    pattern = re.compile(rf"^{re.escape(experiment_prefix(variant))}_seed{seed}_")
    return any(pattern.match(str(name)) for name in finished["experiment_name"])


def write_status(path, queue, current=None):
    path.write_text(json.dumps({"updated": time.time(), "current": current, "queue": queue},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["B0", "B1", "B2", "B3"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--order", choices=["seed", "variant"], default="seed",
                        help="seed: all variants for seed 42 first (recommended); variant: all seeds of B0 first")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (for quick trials only)")
    parser.add_argument("--dry_run", action="store_true", help="Only print the plan")
    args = parser.parse_args()

    if args.order == "seed":
        plan = [(v, s) for s in args.seeds for v in args.variants]
    else:
        plan = [(v, s) for v in args.variants for s in args.seeds]

    logs = PROJECT / "logs"
    logs.mkdir(exist_ok=True)
    status_path = logs / "queue_status.json"
    finished = finished_runs()
    queue = [{"variant": v, "seed": s, "description": VARIANTS[v][0], "settings": settings(v),
              "state": "done" if is_done(v, s, finished) else "pending"} for v, s in plan]
    for item in queue:
        print(f"  {item['variant']:7s} seed {item['seed']:<5d} {item['state']:8s} {item['settings']:48s} {item['description']}")
    if args.dry_run:
        return
    write_status(status_path, queue)

    for item in queue:
        if item["state"] == "done":
            continue
        variant, seed = item["variant"], item["seed"]
        _, script, config, _ = VARIANTS[variant]
        command = [sys.executable, "-u", script, "--config", config, "--seed", str(seed), "--precision", args.precision]
        if script.endswith("run_moe.py"):
            command += ["--routing_type", "top_k" if variant == "A_topk" else "dynamic"]
        for flag, value in (("--batch_size", args.batch_size), ("--grad_accum", args.grad_accum), ("--epochs", args.epochs)):
            if value is not None:
                command += [flag, str(value)]
        out_path = logs / f"{variant}_seed{seed}.out"
        item.update(state="running", started=time.time(), stdout=str(out_path.relative_to(PROJECT)))
        write_status(status_path, queue, current=item)
        print(f"\n>>> {variant} seed {seed}: {' '.join(command)}", flush=True)
        with open(out_path, "w", encoding="utf-8") as out:
            code = subprocess.call(command, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT)
        item.update(state="done" if code == 0 else "failed", finished=time.time(), exit_code=code)
        write_status(status_path, queue)
        print(f"<<< {variant} seed {seed}: {item['state']} (exit {code})", flush=True)
        if code != 0:
            print(f"    log: {out_path}")
    write_status(status_path, queue)
    print("\nQueue finished.")


if __name__ == "__main__":
    main()
