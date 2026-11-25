"""Plots for training logs and results."""
import json
import os
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
LABELS = ["entailment", "neutral", "contradiction"]
VARIANT_ORDER = ["B0", "B1", "B2", "B3", "A_topk", "A_nobal", "B2_T005", "B2_T02", "B4_c002", "B4_c01", "B3_r384",
                 "B0_base", "B2_base", "B5_kd"]
# display codes used in the report: E0, E1 baselines; E2-E6 our experiments; A1-A3 ablations
VARIANT_NAMES = {
    "B0": "E0 · XLM-R Large (baseline)",
    "B1": "E1 · NLIMoE (baseline)",
    "B2": "E2 · + STE",
    "B3": "E3a · bottleneck r = 256",
    "B3_r384": "E3b · bottleneck r = 384",
    "B4_c002": "E4a · phạt chi phí 0,02",
    "B4_c01": "E4b · phạt chi phí 0,1",
    "B0_base": "E5a · XLM-R Base",
    "B2_base": "E5b · XLM-R Base + MoE + STE",
    "B5_kd": "E6 · E5b chưng cất từ E2",
    "A_topk": "A1 · Top-K",
    "A_nobal": "A2 · bỏ L_bal",
    "B2_T005": "A3a · T = 0,05",
    "B2_T02": "A3b · T = 0,2",
}
COLORS = {"B0": "#5A6474", "B1": "#2350A8", "B2": "#C2551B", "B3": "#2E8B57",
          "A_topk": "#8E6CC4", "A_nobal": "#B8860B", "B2_T005": "#E08A5A", "B2_T02": "#8C3A10",
          "B4_c002": "#4FA3A5", "B4_c01": "#1F6E70", "B3_r384": "#7BBF8E",
          "B0_base": "#9AA6BF", "B2_base": "#F2A65A", "B5_kd": "#D9480F"}
PAPER = {"B0": 0.455, "B1": 0.473}


def _prefixes():
    # longest prefix first
    import sys
    sys.path.insert(0, str(PROJECT))
    from scripts.run_queue import VARIANTS, experiment_prefix
    pairs = [(experiment_prefix(v), v) for v in VARIANTS]
    return sorted(pairs, key=lambda pair: -len(pair[0]))


def variant_of(experiment_name):
    for prefix, variant in _prefixes():
        if re.match(rf"^{re.escape(prefix)}_seed\d+_", str(experiment_name)):
            return variant
    return None


def seed_of(experiment_name):
    match = re.search(r"_seed(\d+)_", str(experiment_name))
    return int(match.group(1)) if match else None


def load_history(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # partially written line
    return pd.DataFrame(rows)


def all_histories(project=PROJECT):
    """Histories of all runs in logs/."""
    return {p.stem: load_history(p) for p in sorted((Path(project) / "logs").glob("*.jsonl"))}


def latest_history(project=PROJECT):
    files = sorted((Path(project) / "logs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return (files[-1].stem, load_history(files[-1])) if files else (None, pd.DataFrame())


def load_summary(project=PROJECT):
    path = Path(project) / "results" / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[~frame["experiment_name"].astype(str).str.contains("sanity_check")].copy()
    frame["variant"] = frame["experiment_name"].map(variant_of)
    return frame[frame["variant"].notna()]


def _fmt_duration(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def version_of(experiment_name):
    match = re.search(r"_v(\d+)_", str(experiment_name))
    return f"v{match.group(1)}" if match else "?"


def progress_text(name, history):
    if history.empty:
        return f"{name}: chưa có dữ liệu"
    start = history[history["type"] == "start"]
    train = history[history["type"] == "train"]
    val = history[history["type"] == "val"]
    test = history[history["type"] == "test"]
    total = int(start["total_steps"].iloc[-1]) if not start.empty else 0
    step = int(train["step"].iloc[-1]) if not train.empty else 0
    elapsed = history["time"].iloc[-1] - history["time"].iloc[0]
    eta = elapsed / step * (total - step) if step else float("nan")
    parts = [f"[{version_of(name)}] {name}", f"bước {step}/{total} ({100 * step / max(1, total):.0f}%)",
             f"đã chạy {_fmt_duration(elapsed)}"]
    if not test.empty:
        parts.append(f"XONG · test acc {test['accuracy'].iloc[-1]:.4f}")
    elif step and not math.isnan(eta):
        parts.append(f"còn khoảng {_fmt_duration(eta)}")
    if not val.empty:
        parts.append(f"val acc tốt nhất {val['accuracy'].max():.4f}")
    if "gpu_mem_gb" in train and not train.empty:
        parts.append(f"GPU {train['gpu_mem_gb'].max():.1f} GB")
    return " · ".join(parts)


def plot_live(name, history, figsize=(16, 8)):
    """Loss, validation, routing, C/rho, grad of w_c and GPU memory for one run."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(progress_text(name, history), fontsize=11, ha="left", x=0.01)
    train = history[history["type"] == "train"] if not history.empty else history
    val = history[history["type"] == "val"] if not history.empty else history

    ax = axes[0, 0]
    if not train.empty:
        ax.plot(train["step"], train["loss"], color="#BFC8DA", lw=1, label="loss")
        ax.plot(train["step"], train["loss"].rolling(10, min_periods=1).mean(), color="#14213D", lw=2, label="trung bình trượt")
        ax.legend(fontsize=8)
    ax.set_title("Loss huấn luyện")
    ax.set_xlabel("bước optimizer")

    ax = axes[0, 1]
    if not val.empty:
        ax.plot(val["step"], val["accuracy"], "o-", color="#2350A8", label="val accuracy")
        ax.plot(val["step"], val["macro_f1"], "s--", color="#C2551B", label="val macro-F1")
        best = val.loc[val["accuracy"].idxmax()]
        ax.annotate(f"{best['accuracy']:.3f}", (best["step"], best["accuracy"]), textcoords="offset points", xytext=(0, 8), ha="center")
        ax.axhline(1 / 3, color="grey", ls=":", lw=1, label="ngẫu nhiên (33%)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "chờ lần đánh giá đầu tiên\n(sau mỗi eval_frequency bước)", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("Tập kiểm định")
    ax.set_xlabel("bước optimizer")

    ax = axes[0, 2]
    if "avg_experts" in train:
        ax.plot(train["step"], train["avg_experts"], color="#2E8B57", label="train")
        if "avg_experts" in val:
            ax.plot(val["step"], val["avg_experts"], "o", color="#14213D", label="val")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "không có MoE", ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_title("Số chuyên gia trung bình mỗi câu")

    ax = axes[1, 0]
    if "mean_complexity" in train:
        ax.plot(train["step"], train["mean_complexity"], color="#C2551B", label="C trung bình")
        ax2 = ax.twinx()
        ax2.plot(train["step"], train["mean_threshold"], color="#2350A8", ls="--", label="ρ trung bình")
        ax2.set_ylabel("ρ", color="#2350A8")
        ax.set_ylabel("C", color="#C2551B")
    else:
        ax.text(0.5, 0.5, "không có ngưỡng động", ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_title("Độ khó C và ngưỡng ρ")

    ax = axes[1, 1]
    if "grad_norm_wc" in train:
        values = train["grad_norm_wc"].clip(lower=1e-12)
        ax.semilogy(train["step"], values, color="#A3430F")
        ax.set_title("‖∂L/∂w_c‖ (log) · bằng 0 nghĩa là ngưỡng không học")
        if (train["grad_norm_wc"] == 0).all():
            ax.text(0.5, 0.5, "= 0 ở mọi bước\n(mặt nạ cứng chặn gradient)", ha="center", va="center",
                    transform=ax.transAxes, color="#A3430F")
    else:
        ax.text(0.5, 0.5, "không áp dụng", ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("‖∂L/∂w_c‖")

    ax = axes[1, 2]
    if "h_std" in train or "h_std" in val:
        if "h_std" in train:
            ax.plot(train["step"], train["h_std"], color="#2350A8", label="độ đa dạng h (train)")
        if "h_std" in val:
            ax.plot(val["step"], val["h_std"], "o", color="#14213D", label="độ đa dạng h (val)")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("std của h giữa các câu")
        if "z_zero_frac" in train or "z_zero_frac" in val:
            ax2 = ax.twinx()
            if "z_zero_frac" in train:
                ax2.plot(train["step"], train["z_zero_frac"] * 100, color="#C2551B", ls="--", label="% z = 0 (train)")
            if "z_zero_frac" in val:
                ax2.plot(val["step"], val["z_zero_frac"] * 100, "s", color="#A3430F", label="% z = 0 (val)")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("% đầu ra MoE bằng 0", color="#C2551B")
            lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
            ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper right")
        else:
            ax.legend(fontsize=7)
        ax.set_title("Chẩn đoán sụp · h về 0 hoặc z = 0 gần 100% là sụp")
    else:
        ax.text(0.5, 0.5, "chưa có số liệu chẩn đoán", ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Chẩn đoán sụp")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def queue_table(project=PROJECT):
    path = Path(project) / "logs" / "queue_status.json"
    if not path.exists():
        return pd.DataFrame()
    status = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(status["queue"])
    now = time.time()
    if "started" in frame:
        end = frame["finished"] if "finished" in frame else pd.Series(np.nan, index=frame.index)
        end = end.fillna(now)
        frame["thời gian"] = [(_fmt_duration(e - s) if not pd.isna(s) else "") for s, e in zip(frame["started"], end)]
    keep = [c for c in ["variant", "seed", "settings", "description", "state", "thời gian"] if c in frame]
    return frame[keep]


def queue_alive(project=PROJECT):
    """True if logs/queue.pid names a live run_queue.py process (a new Colab VM may reuse the number)."""
    try:
        pid = int((Path(project) / "logs" / "queue.pid").read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    return "run_queue" in cmdline.read_bytes().decode(errors="ignore") if cmdline.exists() else True


def running_jobs(project=PROJECT):
    """[(status item, (name, history) or None)] for every running job."""
    path = Path(project) / "logs" / "queue_status.json"
    if not path.exists():
        return []
    status = json.loads(path.read_text(encoding="utf-8"))
    items = status.get("running") or ([status["current"]] if status.get("current") else [])
    jobs = []
    for item in items:
        if item.get("state") != "running":
            continue
        files = [p for p in (Path(project) / "logs").glob("*.jsonl")
                 if p.stat().st_mtime >= item.get("started", 0)
                 and variant_of(p.stem) == item["variant"] and seed_of(p.stem) == item["seed"]]
        newest = max(files, key=lambda p: p.stat().st_mtime) if files else None
        jobs.append((item, (newest.stem, load_history(newest)) if newest else None))
    return jobs


def current_run(project=PROJECT):
    """(status of the first running item, its history) or (None, None)."""
    jobs = running_jobs(project)
    return jobs[0] if jobs else (None, None)


def monitor(refresh_seconds=60, project=PROJECT, stop_when_idle=True):
    """Refresh the dashboard of the running job until the queue finishes."""
    from IPython.display import clear_output, display
    try:
        while True:
            clear_output(wait=True)
            table = queue_table(project)
            if not table.empty:
                display(table)
            status_path = Path(project) / "logs" / "queue_status.json"
            if stop_when_idle and status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if all(entry["state"] in ("done", "failed") for entry in status["queue"]):
                    print("Hàng đợi đã chạy xong.")
                    return
            if not queue_alive(project):
                print("Hàng đợi không chạy. Chạy ô khởi động hàng đợi; nếu lỗi, xem logs/queue.out.")
                return
            jobs = running_jobs(project)
            if not jobs:
                print("Hàng đợi đang khởi động...")
            for item, run in jobs:
                if run is None:
                    print(f"{item['variant']} seed {item['seed']}: đang tải mô hình và dữ liệu...")
                else:
                    plot_live(*run)
                    plt.show()
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("Đã dừng theo dõi (huấn luyện vẫn tiếp tục chạy nền).")


def results_table(summary):
    """Mean and std over seeds per variant."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for variant in [v for v in VARIANT_ORDER if v in set(summary["variant"])]:
        group = summary[summary["variant"] == variant]
        acc, f1 = group["test_accuracy"] * 100, group["test_macro_f1"] * 100
        experts = pd.to_numeric(group["avg_experts_per_sample"], errors="coerce")
        rows.append({
            "Biến thể": VARIANT_NAMES[variant],
            "Số seed": len(group),
            "Seeds": ", ".join(str(s) for s in sorted(group["seed"])),
            "Test acc (%)": f"{acc.mean():.2f} ± {acc.std(ddof=1):.2f}" if len(group) > 1 else f"{acc.mean():.2f}",
            "Macro-F1 (%)": f"{f1.mean():.2f} ± {f1.std(ddof=1):.2f}" if len(group) > 1 else f"{f1.mean():.2f}",
            "Val acc (%)": f"{group['val_accuracy'].mean() * 100:.2f}",
            "Chuyên gia/câu": f"{experts.mean():.2f}" if variant != "B0" else "—",
            "Bài báo (%)": f"{PAPER[variant] * 100:.1f}" if variant in PAPER else "—",
        })
    return pd.DataFrame(rows)


def plot_results_bar(summary, figsize=(10, 5)):
    """Test accuracy per variant (mean, seeds, paper value)."""
    fig, ax = plt.subplots(figsize=figsize)
    variants = [v for v in VARIANT_ORDER if v in set(summary["variant"])]
    for i, variant in enumerate(variants):
        values = summary.loc[summary["variant"] == variant, "test_accuracy"] * 100
        ax.bar(i, values.mean(), color=COLORS[variant], alpha=0.85,
               yerr=values.std(ddof=1) if len(values) > 1 else None, capsize=6)
        ax.scatter([i] * len(values), values, color="black", s=18, zorder=3)
        ax.text(i, values.mean() + 0.6, f"{values.mean():.1f}", ha="center", fontsize=9)
        if variant in PAPER:
            ax.scatter(i, PAPER[variant] * 100, marker="*", s=180, color="gold", edgecolor="black", zorder=4)
    ax.axhline(100 / 3, color="grey", ls=":", lw=1)
    ax.set_xticks(range(len(variants)), [VARIANT_NAMES[v] for v in variants], rotation=15, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    low = summary["test_accuracy"].min() * 100 if not summary.empty else 30
    ax.set_ylim(max(0, min(low, 33) - 5), None)
    ax.set_title("Độ chính xác trên tập kiểm thử (cột: trung bình, chấm: từng seed, ★: số bài báo)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_curves_compare(histories, metric="accuracy", event="val", seed=42, figsize=(10, 5)):
    """One metric over training steps, one line per variant."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, history in histories.items():
        variant = variant_of(name)
        if variant is None or seed_of(name) != seed or history.empty or metric not in history:
            continue
        part = history[history["type"] == event]
        if part.empty:
            continue
        y = part[metric] if event == "val" else part[metric].rolling(10, min_periods=1).mean()
        ax.plot(part["step"], y, "o-" if event == "val" else "-", ms=3, color=COLORS[variant], label=VARIANT_NAMES[variant])
    ax.set_xlabel("bước optimizer")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} ({'kiểm định' if event == 'val' else 'huấn luyện'}) theo bước, seed {seed}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gradnorm_compare(histories, seed=42, figsize=(10, 4)):
    """Norm of dL/dw_c over training steps."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, history in histories.items():
        variant = variant_of(name)
        if variant is None or seed_of(name) != seed or "grad_norm_wc" not in history:
            continue
        train = history[history["type"] == "train"]
        ax.semilogy(train["step"], train["grad_norm_wc"].clip(lower=1e-12), color=COLORS[variant], label=VARIANT_NAMES[variant])
    ax.axhline(1e-12, color="grey", ls=":", lw=1)
    ax.text(0.01, 0.04, "đường đáy = gradient bằng 0", transform=ax.transAxes, color="grey", fontsize=8)
    ax.set_title(f"‖∂L/∂w_c‖ theo bước (seed {seed}): ngưỡng chỉ học được khi khác 0")
    ax.set_xlabel("bước optimizer")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def routing_frame(experiment_name, project=PROJECT):
    path = Path(project) / "results" / "routing_stats" / f"{experiment_name}_per_sample.csv"
    return pd.read_csv(path) if path.exists() else None


def spearman(x, y):
    from scipy.stats import spearmanr
    result = spearmanr(x, y)
    return float(result.statistic if hasattr(result, "statistic") else result.correlation), float(result.pvalue)


def plot_routing(experiment_name, project=PROJECT, figsize=(16, 4.5)):
    """Experts per sample, C for correct/wrong, expert usage per label."""
    frame = routing_frame(experiment_name, project)
    if frame is None:
        raise FileNotFoundError(f"No per-sample routing file for {experiment_name}")
    experts = sorted(int(c.split("_")[1]) for c in frame.columns if c.startswith("mask_"))
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"{VARIANT_NAMES.get(variant_of(experiment_name), experiment_name)} · seed {seed_of(experiment_name)}",
                 fontsize=11, ha="left", x=0.01)

    ax = axes[0]
    counts = frame["n_experts"].value_counts().reindex(range(0, len(experts) + 1), fill_value=0)
    ax.bar(counts.index, counts.values, color="#2350A8")
    ax.set_title(f"Số chuyên gia mỗi câu (TB {frame['n_experts'].mean():.2f}; {int((frame['n_experts'] == 0).sum())} câu không chọn được)")
    ax.set_xlabel("số chuyên gia")
    ax.set_ylabel("số câu")

    ax = axes[1]
    if "complexity" in frame:
        wrong = 1 - frame["correct"]
        rho, p = spearman(frame["complexity"], wrong)
        data = [frame.loc[frame["correct"] == 1, "complexity"], frame.loc[frame["correct"] == 0, "complexity"]]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks([1, 2], ["đoán đúng", "đoán sai"])
        ax.set_title(f"Độ khó C theo đúng/sai · Spearman ρ_s = {rho:.3f} (p = {p:.3g})")
        ax.set_ylabel("C")
    else:
        ax.text(0.5, 0.5, "không có C", ha="center", va="center", transform=ax.transAxes)

    ax = axes[2]
    usage = np.array([[frame.loc[frame["gold"] == label, f"mask_{e}"].mean() * 100 for e in experts]
                      for label in range(len(LABELS))])
    image = ax.imshow(usage, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for (r, c), value in np.ndenumerate(usage):
        ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=8, color="white" if value > 60 else "black")
    ax.set_xticks(range(len(experts)), [f"E{e + 1}" for e in experts])
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_title("% câu kích hoạt từng chuyên gia, theo nhãn")
    fig.colorbar(image, ax=ax, fraction=0.046)
    for a in axes[:2]:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def predictions(experiment_name, project=PROJECT):
    return pd.read_csv(Path(project) / "predictions" / f"{experiment_name}.csv")


def mcnemar(name_a, name_b, project=PROJECT):
    """Exact McNemar test between two runs on the same test set."""
    from scipy.stats import binomtest
    a, b = predictions(name_a, project), predictions(name_b, project)
    merged = a.merge(b, on="uid", suffixes=("_a", "_b"))
    right_a = merged["gold_label_a"] == merged["pred_label_a"]
    right_b = merged["gold_label_b"] == merged["pred_label_b"]
    only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"A": name_a, "B": name_b, "acc_A": right_a.mean(), "acc_B": right_b.mean(),
            "chỉ A đúng": only_a, "chỉ B đúng": only_b, "p_value": p}


def mcnemar_table(summary, pairs=(("B0", "B1"), ("B1", "B2"), ("B2", "B3")), project=PROJECT):
    rows = []
    for seed in sorted(summary["seed"].unique()):
        names = summary[summary["seed"] == seed].set_index("variant")["experiment_name"]
        for left, right in pairs:
            if left in names and right in names:
                result = mcnemar(names[left], names[right], project)
                rows.append({"Seed": seed, "So sánh": f"{left} → {right}",
                             "Acc trái (%)": f"{result['acc_A'] * 100:.1f}", "Acc phải (%)": f"{result['acc_B'] * 100:.1f}",
                             "Chỉ trái đúng": result["chỉ A đúng"], "Chỉ phải đúng": result["chỉ B đúng"],
                             "p (McNemar)": round(result["p_value"], 4),
                             "Kết luận": "khác biệt có ý nghĩa" if result["p_value"] < 0.05 else "chưa đủ bằng chứng"})
    return pd.DataFrame(rows)


def plot_confusion(experiment_name, project=PROJECT, ax=None):
    frame = predictions(experiment_name, project)
    matrix = pd.crosstab(frame["gold_label"], frame["pred_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix.values, cmap="Blues")
    for (r, c), value in np.ndenumerate(matrix.values):
        ax.text(c, r, value, ha="center", va="center", color="white" if value > matrix.values.max() / 2 else "black")
    ax.set_xticks(range(3), ["E", "N", "C"])
    ax.set_yticks(range(3), ["E", "N", "C"])
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("nhãn đúng")
    ax.set_title(VARIANT_NAMES.get(variant_of(experiment_name), experiment_name), fontsize=10)
    return ax.figure


def expert_parameter_table(hidden=1024, experts=7, bottleneck=256):
    linear = hidden * hidden + hidden
    narrow = (hidden * bottleneck + bottleneck) + (bottleneck * hidden + hidden)
    return pd.DataFrame([
        {"Chuyên gia": "Linear 1024→1024 (B1, B2)", "Tham số / chuyên gia": f"{linear:,}", "× 7": f"{linear * experts:,}", "Thay đổi": "—"},
        {"Chuyên gia": "Bottleneck 1024→256→1024 (B3)", "Tham số / chuyên gia": f"{narrow:,}", "× 7": f"{narrow * experts:,}",
         "Thay đổi": f"{(narrow / linear - 1) * 100:.1f}%"},
    ])
