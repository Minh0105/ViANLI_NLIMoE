"""Run experiments, skipping (variant, seed) pairs already in results/summary.csv.

    python scripts/run_queue.py --variants B0 B1 B2 B3 --seeds 42 123 2026 --parallel 3

--parallel N runs N jobs at once on the same GPU. At batch 16 one job uses ~11 GB and leaves
most of an A100 idle, so 2-3 jobs finish the queue sooner without changing any training setting.
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
    "B2_T005": ("B2 with STE temperature 0.05", "scripts/run_moe.py", "configs/nlimoe_ste_t005.yaml", "nlimoe_paper_v5_ste_t005_dynamic"),
    "B2_T02": ("B2 with STE temperature 0.2", "scripts/run_moe.py", "configs/nlimoe_ste_t02.yaml", "nlimoe_paper_v5_ste_t02_dynamic"),
    "B4_c002": ("B2 + expert cost 0.02", "scripts/run_moe.py", "configs/nlimoe_ste_cost002.yaml", "nlimoe_paper_v5_ste_cost002_dynamic"),
    "B4_c01": ("B2 + expert cost 0.1", "scripts/run_moe.py", "configs/nlimoe_ste_cost01.yaml", "nlimoe_paper_v5_ste_cost01_dynamic"),
    "B3_r384": ("B3 with bottleneck r = 384", "scripts/run_moe.py", "configs/nlimoe_ste_bottleneck384.yaml", "nlimoe_paper_v5_ste_bottleneck384_dynamic"),
    "B0_base": ("XLM-R Base (no MoE)", "scripts/run_xlmr.py", "configs/xlmr_base_v5.yaml", "xlmr_base_paper_v5"),
    "B2_base": ("NLIMoE + STE on XLM-R Base", "scripts/run_moe.py", "configs/nlimoe_ste_base.yaml", "nlimoe_paper_v5_ste_base_dynamic"),
    "B5_kd": ("B2_base distilled from B2 (Large)", "scripts/run_moe.py", "configs/nlimoe_ste_base_kd.yaml", "nlimoe_paper_v5_ste_base_kd_dynamic"),
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


def write_status(path, queue):
    running = [item for item in queue if item["state"] == "running"]
    status = {"updated": time.time(), "current": running[0] if running else None, "running": running, "queue": queue}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# peak GPU memory of one job at batch 16 in bf16 (measured ~11 GB for XLM-R Large on an A100), plus margin
JOB_MEMORY_GB = {"xlm-roberta-large": 12.5, "xlm-roberta-base": 6.5}


def job_memory_gb(variant):
    import yaml
    cfg = yaml.safe_load((PROJECT / VARIANTS[variant][2]).read_text(encoding="utf-8"))
    return JOB_MEMORY_GB.get(cfg["model"]["backbone"], 12.5)


def next_job(pending, active, args):
    """Index of the first pending job that fits next to the running ones, or None.

    With --gpu_memory_gb the jobs are packed by their memory estimate (on an L4, one Large and
    one Base job, or three Base jobs); without it, at most --parallel jobs run.
    """
    if not pending or len(active) >= max(1, args.parallel):
        return None
    if args.gpu_memory_gb is None:
        return 0
    used = sum(job_memory_gb(item["variant"]) for item, _, _ in active)
    for index, item in enumerate(pending):
        if used + job_memory_gb(item["variant"]) <= args.gpu_memory_gb:
            return index
    return 0 if not active else None


def command_for(variant, seed, args):
    _, script, config, _ = VARIANTS[variant]
    command = [sys.executable, "-u", script, "--config", config, "--seed", str(seed), "--precision", args.precision]
    if script.endswith("run_moe.py"):
        command += ["--routing_type", "top_k" if variant == "A_topk" else "dynamic"]
    for flag, value in (("--batch_size", args.batch_size), ("--grad_accum", args.grad_accum), ("--epochs", args.epochs)):
        if value is not None:
            command += [flag, str(value)]
    return command


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
    parser.add_argument("--parallel", type=int, default=1, help="Jobs running at the same time on the GPU")
    parser.add_argument("--gpu_memory_gb", type=float, default=None,
                        help="Pack jobs by their memory estimate instead of counting them (up to --parallel)")
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

    pending = [item for item in queue if item["state"] == "pending"]
    active = []  # (item, process, output file)
    while pending or active:
        while (index := next_job(pending, active, args)) is not None:
            item = pending.pop(index)
            variant, seed = item["variant"], item["seed"]
            command = command_for(variant, seed, args)
            out_path = logs / f"{variant}_seed{seed}.out"
            item.update(state="running", started=time.time(), stdout=str(out_path.relative_to(PROJECT)))
            out = open(out_path, "w", encoding="utf-8")
            active.append((item, subprocess.Popen(command, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT), out))
            write_status(status_path, queue)
            print(f">>> {variant} seed {seed}: {' '.join(command)}", flush=True)
            if next_job(pending, active, args) is not None:
                time.sleep(30)  # stagger model loading
        time.sleep(5)
        for entry in list(active):
            item, process, out = entry
            code = process.poll()
            if code is None:
                continue
            out.close()
            active.remove(entry)
            item.update(state="done" if code == 0 else "failed", finished=time.time(), exit_code=code)
            write_status(status_path, queue)
            print(f"<<< {item['variant']} seed {item['seed']}: {item['state']} (exit {code})", flush=True)
            if code != 0:
                print(f"    log: {item['stdout']}")
    write_status(status_path, queue)
    print("\nQueue finished.")


if __name__ == "__main__":
    main()
