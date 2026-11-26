"""Tables, significance tests, error-annotation sheet and figures for the report, in one command.

    python scripts/analyze_results.py --out_dir reports/generated
    python scripts/analyze_results.py --annotated_errors reports/generated/error_annotation_sheet_filled.csv

Inputs (copy them from Google Drive: /content/drive/MyDrive/ViANLI_NLIMoE/):
    results/summary.csv                       one row per finished run
    predictions/<experiment>.csv              test predictions with class probabilities
    results/routing_stats/<experiment>_per_sample.csv   C, rho, mask and weights per test sentence (MoE runs)
    logs/<experiment>.jsonl                   training history (grad_norm_wc, mean_complexity, ...)
    results/efficiency.csv                    optional, from scripts/efficiency.py
The test sentences (premise, hypothesis) come from the Hugging Face dataset uitnlp/ViANLI, or from
--test_csv (columns uid, premise, hypothesis); without either, the text columns stay empty.

Outputs in --out_dir (default reports/generated) and --figures_dir (default reports/figures, 300 DPI):
    input_inventory.csv, main_results.{csv,md}, statistical_tests.{csv,md}, complexity_and_routing.{csv,md},
    expert_usage.csv, cross_run_analysis.json, unanimously_wrong.csv, error_annotation_sheet.csv,
    analysis_log.md, and with --annotated_errors: trap_type_counts.{csv,md}, trap_type_tests.json.
A section whose inputs are missing is skipped and listed in analysis_log.md with the files it needs.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.utils import stats, viz  # noqa: E402
from src.utils.viz import LABELS, MissingData, code_of  # noqa: E402

TRAP_TYPES = ("numerical", "overgeneralization", "temporal", "world_knowledge", "lexical_overlap",
              "negation", "annotation_noise")
# heuristic cues to speed up manual annotation; not used for any reported statistic
NEGATION_CUES = ("không", "chẳng", "chả", "chưa", "chớ", "đừng", "đâu có", "không hề", "chẳng hề", "không phải")
NUMBER_WORDS = ("mười", "mươi", "chục", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "phần trăm")
PAPER = {"B0": 45.5, "B1": 47.3}
SHORT = {"entailment": "E", "neutral": "N", "contradiction": "C"}


# ------------------------------------------------------------------------------------------------ helpers

class Log:
    """Done / skipped sections, written to analysis_log.md."""

    def __init__(self):
        self.done, self.skipped = [], []

    def ok(self, name, detail=""):
        self.done.append((name, detail))
        print(f"[ok] {name}" + (f": {detail}" if detail else ""))

    def skip(self, name, reason):
        self.skipped.append((name, str(reason)))
        print(f"[skipped] {name}: {reason}")

    def write(self, path):
        lines = ["# analyze_results.py", "", "## Done", ""]
        lines += [f"- {name}" + (f": {detail}" if detail else "") for name, detail in self.done] or ["- (none)"]
        lines += ["", "## Skipped", ""]
        lines += [f"- {name}: {reason}" for name, reason in self.skipped] or ["- (none)"]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def section(log, name, function, *args, **kwargs):
    """Run one section; a MissingData error skips it with the missing file named."""
    try:
        return function(*args, **kwargs)
    except MissingData as error:
        log.skip(name, error)
        return None


def to_markdown(frame, floatfmt=".2f"):
    """Markdown table without the optional tabulate dependency."""
    if frame is None or frame.empty:
        return "_(no rows)_\n"

    def cell(value):
        if isinstance(value, float):
            return "" if np.isnan(value) else format(value, floatfmt)
        return str(value).replace("|", "\\|")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def write_table(frame, out_dir, stem, title, note=""):
    frame.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    text = f"# {title}\n\n" + (note + "\n\n" if note else "") + to_markdown(frame)
    (out_dir / f"{stem}.md").write_text(text, encoding="utf-8")


def mean_std(values, scale=1.0):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna() * scale
    if values.empty:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def pm(mean, std):
    return "" if np.isnan(mean) else f"{mean:.1f} ± {std:.1f}"


def present_variants(summary, wanted=None):
    have = set(summary["variant"])
    order = [v for v in viz.REPORT_ORDER if v in have]
    return [v for v in order if wanted is None or v in wanted]


# --------------------------------------------------------------------------------------------- inventory

def inventory(summary, project):
    rows = []
    for row in summary.itertuples():
        name = row.experiment_name
        moe = row.variant not in viz.NO_MOE_VARIANTS
        rows.append({"variant": row.variant, "code": code_of(row.variant), "seed": row.seed, "experiment_name": name,
                     "prediction": (project / "predictions" / f"{name}.csv").exists(),
                     "routing_per_sample": (project / "results" / "routing_stats" / f"{name}_per_sample.csv").exists() if moe else None,
                     "log": (project / "logs" / f"{name}.jsonl").exists()})
    return pd.DataFrame(rows)


def print_inventory(frame):
    print(f"runs in results/summary.csv: {len(frame)} ({frame['variant'].nunique()} variants)")
    for column, folder, suffix in (("prediction", "predictions/", ".csv"),
                                   ("routing_per_sample", "results/routing_stats/", "_per_sample.csv"),
                                   ("log", "logs/", ".jsonl")):
        values = frame[column].dropna()
        missing = frame.loc[values.index[~values.astype(bool)], "experiment_name"].tolist()
        print(f"  {column:18s} {int(values.astype(bool).sum())}/{len(values)} present"
              + (f"; missing e.g. {folder}{missing[0]}{suffix}" if missing else ""))


# ------------------------------------------------------------------------------------------ main results

def main_results(summary, project, variants):
    rows = []
    for variant in variants:
        group = summary[summary["variant"] == variant].sort_values("seed")
        acc_m, acc_s = mean_std(group["test_accuracy"], 100)
        f1_m, f1_s = mean_std(group["test_macro_f1"], 100)
        row = {"code": code_of(variant), "variant": variant, "name": viz.VARIANT_NAMES.get(variant, variant),
               "seeds": len(group),
               "test_acc": pm(acc_m, acc_s), "test_acc_mean": acc_m, "test_acc_std": acc_s,
               "test_acc_per_seed": ", ".join(f"{s}: {a * 100:.1f}" for s, a in zip(group["seed"], group["test_accuracy"])),
               "macro_f1": pm(f1_m, f1_s), "macro_f1_mean": f1_m, "macro_f1_std": f1_s,
               "val_acc_mean": mean_std(group["val_accuracy"], 100)[0],
               "paper_acc": PAPER.get(variant, float("nan")),
               "avg_experts": float("nan") if variant in viz.NO_MOE_VARIANTS
               else mean_std(group["avg_experts_per_sample"])[0],
               "router_entropy": float("nan") if variant in viz.NO_MOE_VARIANTS
               else mean_std(group["routing_entropy"])[0],
               "train_time_min": mean_std(group["train_time"])[0] / 60,
               "inference_time_s": mean_std(group["inference_time"])[0]}
        try:
            recall = viz.per_class_recall_table(summary, project, [variant]).iloc[0]
            for label in LABELS:
                row[f"recall_{SHORT[label]}"] = recall[f"recall_{label}"]
            row["collapsed_seeds"] = int(recall["collapsed_seeds"])
        except MissingData:
            for label in LABELS:
                row[f"recall_{SHORT[label]}"] = float("nan")
            row["collapsed_seeds"] = float("nan")
        row["empty_selection_pct"] = float("nan")
        if variant not in viz.NO_MOE_VARIANTS:
            shares = []
            for name in viz.runs_of(summary, variant).values():
                frame = viz.routing_frame(name, project)
                if frame is not None:
                    shares.append(float((frame["n_experts"] == 0).mean() * 100))
            if shares:
                row["empty_selection_pct"] = float(np.mean(shares))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- statistical tests

def statistical_tests(summary, project, pairs, n_boot, seed):
    rows = []
    have = set(summary["variant"])
    for a, b in pairs:
        if a not in have or b not in have:
            continue
        acc = lambda v: summary[summary["variant"] == v].set_index("seed")["test_accuracy"]
        acc_a, acc_b = acc(a), acc(b)
        shared = sorted(set(acc_a.index) & set(acc_b.index))
        row = {"comparison": f"{code_of(b)} vs {code_of(a)}", "variant_a": a, "variant_b": b,
               "acc_a_mean": acc_a.mean() * 100, "acc_b_mean": acc_b.mean() * 100,
               "delta_acc_mean": (acc_b.mean() - acc_a.mean()) * 100,
               "wins_over_seeds": f"{sum(acc_b[s] > acc_a[s] for s in shared)}/{len(shared)}"}
        try:
            boot = viz.clustered_bootstrap_test(summary, a, b, n_boot=n_boot, seed=seed, project=project)
            pooled = viz.pooled_mcnemar(summary, a, b, project)
            ensemble = viz.ensemble_mcnemar(summary, a, b, project)
            row.update(bootstrap_diff=boot["mean_diff"] * 100,
                       bootstrap_ci95=f"[{boot['ci_95_low'] * 100:+.1f}, {boot['ci_95_high'] * 100:+.1f}]",
                       bootstrap_ci95_low=boot["ci_95_low"] * 100, bootstrap_ci95_high=boot["ci_95_high"] * 100,
                       bootstrap_p=boot["p_value_two_sided"], bootstrap_p_percentile=boot["p_bootstrap_two_sided"],
                       n_sentences=boot["n_sentences"],
                       ensemble_acc_a=ensemble["ensemble_acc_a"] * 100, ensemble_acc_b=ensemble["ensemble_acc_b"] * 100,
                       ensemble_mcnemar_p=ensemble["p_value"],
                       mcnemar_pooled_p=pooled["p_value"], mcnemar_pooled_pairs=pooled["n_pairs"], note="")
        except MissingData as error:
            row["note"] = str(error)
        rows.append(row)
    return pd.DataFrame(rows)


STAT_NOTE = """\
Difference = B - A in accuracy points. Official test: **clustered bootstrap** over the test sentences
(per sentence, accuracy averaged over the seeds of each variant; sentences resampled with replacement;
p from a sign-flip permutation test). **Ensemble McNemar**: exact McNemar between the probability-averaged
ensembles of the two variants (one pair per sentence, independent). **Pooled McNemar** over seeds x
sentences is a reference number only: the same sentence appears once per seed, so its pairs are not
independent and its p-value is too small."""


# ---------------------------------------------------------------------------------- complexity and routing

def gradient_stats(summary, project, variants):
    rows = []
    for variant in variants:
        if variant in viz.NO_MOE_VARIANTS:
            continue
        for seed, name in sorted(viz.runs_of(summary, variant).items()):
            path = project / "logs" / f"{name}.jsonl"
            if not path.exists():
                raise MissingData(f"missing logs/{name}.jsonl")
            history = viz.load_history(path)
            train = history[history["type"] == "train"] if "type" in history else history.iloc[0:0]
            row = {"variant": variant, "code": code_of(variant), "seed": seed}
            if "grad_norm_wc" in train:
                grad = pd.to_numeric(train["grad_norm_wc"], errors="coerce").dropna()
                row.update(grad_logged=len(grad), grad_zero=int((grad == 0).sum()),
                           grad_median=float(grad.median()) if len(grad) else float("nan"),
                           grad_max=float(grad.max()) if len(grad) else float("nan"))
            if "mean_complexity" in train and train["mean_complexity"].notna().any():
                values = train["mean_complexity"].dropna()
                row.update(C_first=float(values.iloc[0]), C_last=float(values.iloc[-1]))
            if "mean_threshold" in train and train["mean_threshold"].notna().any():
                values = train["mean_threshold"].dropna()
                row.update(rho_first=float(values.iloc[0]), rho_last=float(values.iloc[-1]))
            rows.append(row)
    return pd.DataFrame(rows)


def expert_usage(summary, project, variants):
    from src.utils.routing_diagnostics import compute_routing_diagnostics
    rows = []
    for variant in variants:
        if variant in viz.NO_MOE_VARIANTS:
            continue
        for seed, name in sorted(viz.runs_of(summary, variant).items()):
            frame = viz.routing_frame(name, project)
            if frame is None:
                raise MissingData(f"missing results/routing_stats/{name}_per_sample.csv")
            experts = sorted(int(c.split("_")[1]) for c in frame.columns if c.startswith("mask_"))
            masks = frame[[f"mask_{e}" for e in experts]].to_numpy()
            weights = frame[[f"weight_{e}" for e in experts]].to_numpy()
            diag = compute_routing_diagnostics(weights, masks, frame["gold"].to_numpy(), LABELS)
            base = {"variant": variant, "code": code_of(variant), "seed": seed,
                    "collapse_warning": diag["collapse_warning"], "empty_selection_pct": diag["empty_selection_pct"],
                    "min_experts": diag["min_experts_per_sample"], "max_experts": diag["max_experts_per_sample"]}
            rows.append({**base, "scope": "all", **diag["expert_usage_pct"]})
            for label in LABELS:
                rows.append({**base, "scope": label, **diag["expert_distribution_per_label"][label]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- cross-run

def all_runs(summary, project, variants):
    """Predictions of every run aligned on the sentences shared by all runs."""
    arrays, names = [], []
    for variant in variants:
        try:
            data = viz.prediction_arrays(summary, variant, project)
        except MissingData:
            continue
        arrays.append(data)
        names += [(variant, s) for s in data["seeds"]]
    if not arrays:
        raise MissingData("no prediction files in predictions/")
    common = set(arrays[0]["uids"])
    for data in arrays[1:]:
        common &= set(data["uids"])
    uids = [u for u in arrays[0]["uids"] if u in common]
    pred, probs, gold = [], [], None
    for data in arrays:
        index = {u: i for i, u in enumerate(data["uids"])}
        idx = [index[u] for u in uids]
        pred.append(data["pred"][:, idx])
        probs.append(data["probs"][:, idx])
        gold = data["gold"][idx] if gold is None else gold
    return {"uids": np.array(uids), "gold": gold, "pred": np.concatenate(pred), "probs": np.concatenate(probs),
            "runs": names}


def cross_run_analysis(summary, project, variants, texts):
    data = all_runs(summary, project, variants)
    pred, probs, gold = data["pred"], data["probs"], data["gold"]
    correct = pred == gold
    votes = np.stack([(pred == c).sum(0) for c in range(3)], -1)
    ensembles = {}
    for variant in variants:
        try:
            arrays = viz.prediction_arrays(summary, variant, project)
        except MissingData:
            continue
        ensembles[variant] = float((stats.ensemble_predictions(arrays["probs"]) == arrays["gold"]).mean() * 100)
    share_right = correct.mean(0)
    result = {
        "runs": len(data["runs"]), "variants": [code_of(v) for v in sorted({v for v, _ in data["runs"]}, key=viz.REPORT_ORDER.index)],
        "sentences": int(len(gold)),
        "pairwise_agreement_pct": stats.pairwise_agreement(pred) * 100,
        "oracle_coverage_pct": float(correct.any(0).mean() * 100),
        "never_correct": int((~correct.any(0)).sum()),
        "never_correct_pct": float((~correct.any(0)).mean() * 100),
        "correct_in_under_20pct_of_runs_pct": float((share_right < 0.2).mean() * 100),
        "majority_vote_acc_pct": float((stats.majority_vote(pred, probs) == gold).mean() * 100),
        "mean_probability_ensemble_acc_pct": float((stats.ensemble_predictions(probs) == gold).mean() * 100),
        "ensemble_acc_pct_by_variant": {code_of(v): a for v, a in ensembles.items()},
        "mean_run_accuracy_pct": float(correct.mean() * 100),
        "recall_pct_over_all_runs": {label: float(correct[:, gold == c].mean() * 100) for c, label in enumerate(LABELS)},
    }
    wrong = np.where(~correct.any(0))[0]
    table = pd.DataFrame({
        "uid": data["uids"][wrong], "gold_label": [LABELS[g] for g in gold[wrong]],
        "vote_E": votes[wrong, 0], "vote_N": votes[wrong, 1], "vote_C": votes[wrong, 2],
        "mean_prob_gold": [float(probs[:, i, gold[i]].mean()) for i in wrong], "n_runs": len(data["runs"])})
    table.insert(1, "premise", table["uid"].map(lambda u: texts.get(u, ("", ""))[0]))
    table.insert(2, "hypothesis", table["uid"].map(lambda u: texts.get(u, ("", ""))[1]))
    return result, table


# --------------------------------------------------------------------------------------- error annotation

def words(text):
    return set(re.findall(r"\w+", str(text).lower()))


def jaccard(a, b):
    a, b = words(a), words(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def has_negation(text):
    text = f" {str(text).lower()} "
    return any(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text) for cue in NEGATION_CUES)


def has_number(text):
    text = str(text).lower()
    return bool(re.search(r"\d", text)) or any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text) for w in NUMBER_WORDS)


def error_sheet(summary, project, texts, variant="B2", seed=42, per_class=40, compare=("B1", "B3_r384", "B5_kd"),
                sample_seed=42):
    runs = viz.runs_of(summary, variant)
    if seed not in runs:
        raise MissingData(f"no run of {variant} seed {seed} in results/summary.csv")
    name = runs[seed]
    main = viz.load_predictions(name, project).drop_duplicates("uid").set_index("uid")
    routing = viz.routing_frame(name, project)
    if routing is None:
        raise MissingData(f"missing results/routing_stats/{name}_per_sample.csv")
    routing = routing.drop_duplicates("uid").set_index("uid")
    errors = main[main["gold_label"] != main["pred_label"]]
    rng = np.random.default_rng(sample_seed)
    chosen = []
    for label in LABELS:
        pool = sorted(errors.index[errors["gold_label"] == label])
        take = min(per_class, len(pool))
        chosen += sorted(rng.choice(pool, take, replace=False).tolist()) if take else []
    others = {}
    for other in compare:
        other_runs = viz.runs_of(summary, other)
        if seed in other_runs:
            try:
                others[other] = viz.load_predictions(other_runs[seed], project).drop_duplicates("uid").set_index("uid")
            except MissingData:
                pass
    experts = sorted(int(c.split("_")[1]) for c in routing.columns if c.startswith("mask_"))
    rows = []
    for uid in chosen:
        premise, hypothesis = texts.get(uid, ("", ""))
        r = routing.loc[uid] if uid in routing.index else None
        row = {"uid": uid, "premise": premise, "hypothesis": hypothesis, "gold": main.at[uid, "gold_label"],
               f"pred_{code_of(variant)}": main.at[uid, "pred_label"]}
        for other, frame in others.items():
            if uid in frame.index:
                row[f"pred_{code_of(other)}"] = frame.at[uid, "pred_label"]
                row[f"wrong_in_{code_of(other)}"] = int(frame.at[uid, "pred_label"] != frame.at[uid, "gold_label"])
        row.update(
            complexity_C=float(r["complexity"]) if r is not None and "complexity" in r else float("nan"),
            threshold_rho=float(r["threshold"]) if r is not None and "threshold" in r else float("nan"),
            n_experts=int(r["n_experts"]) if r is not None else -1,
            selected_experts=",".join(str(e) for e in experts if r is not None and int(r[f"mask_{e}"])),
            lexical_overlap_jaccard=jaccard(premise, hypothesis) if premise else float("nan"),
            has_negation_cue=int(has_negation(hypothesis)) if hypothesis else -1,
            has_number_cue=int(has_number(hypothesis)) if hypothesis else -1,
            trap_type="", notes="")
        rows.append(row)
    return pd.DataFrame(rows)


def annotated_summary(path, figures_dir, out_dir, log):
    from scipy.stats import chi2_contingency, kruskal
    frame = pd.read_csv(path)
    if "trap_type" not in frame:
        raise ValueError(f"{path} has no trap_type column")
    frame["trap_type"] = frame["trap_type"].fillna("").astype(str).str.strip().str.lower()
    labelled = frame[frame["trap_type"] != ""]
    unknown = sorted(set(labelled["trap_type"]) - set(TRAP_TYPES))
    if unknown:
        print(f"warning: trap types outside {TRAP_TYPES}: {unknown} (kept as given)")
    counts = labelled["trap_type"].value_counts().rename_axis("trap_type").reset_index(name="count")
    counts["pct"] = counts["count"] / max(1, len(labelled)) * 100
    by_gold = pd.crosstab(labelled["trap_type"], labelled["gold"]).reindex(columns=LABELS, fill_value=0)
    counts = counts.merge(by_gold.reset_index(), on="trap_type", how="left")
    write_table(counts, out_dir, "trap_type_counts", "Error types (manual annotation)",
                f"{len(labelled)} of {len(frame)} sampled errors annotated.")
    max_expert = 0
    for value in labelled["selected_experts"].fillna(""):
        for e in str(value).split(","):
            if e.strip().isdigit():
                max_expert = max(max_expert, int(e) + 1)
    types = counts["trap_type"].tolist()
    activation = np.zeros((len(types), max_expert))
    for i, trap in enumerate(types):
        rows = labelled[labelled["trap_type"] == trap]
        for value in rows["selected_experts"].fillna(""):
            for e in str(value).split(","):
                if e.strip().isdigit():
                    activation[i, int(e)] += 1
    tests = {"annotated": int(len(labelled))}
    table = activation[activation.sum(1) > 0][:, activation.sum(0) > 0]
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p, dof, _ = chi2_contingency(table)
        tests["chi2_trap_type_x_expert"] = {"chi2": float(chi2), "dof": int(dof), "p_value": float(p),
                                            "note": "activation counts; one sentence contributes one count per selected expert"}
    for column in ("n_experts", "complexity_C"):
        groups = [pd.to_numeric(labelled.loc[labelled["trap_type"] == t, column], errors="coerce").dropna()
                  for t in types]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            h, p = kruskal(*groups)
            tests[f"kruskal_{column}_by_trap_type"] = {"H": float(h), "p_value": float(p), "groups": len(groups)}
    (out_dir / "trap_type_tests.json").write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")
    if max_expert:
        import matplotlib.pyplot as plt
        share = activation / np.maximum(1, counts["count"].to_numpy()[:, None]) * 100
        fig, ax = plt.subplots(figsize=(1.1 * max_expert + 3, 0.5 * len(types) + 1.5))
        image = ax.imshow(share, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for (r, c), value in np.ndenumerate(share):
            ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=8, color="white" if value > 60 else "black")
        ax.set_xticks(range(max_expert), [f"E{e + 1}" for e in range(max_expert)])
        ax.set_yticks(range(len(types)), [f"{t} ({n})" for t, n in zip(types, counts["count"])])
        ax.set_title("% lỗi kích hoạt từng chuyên gia, theo kiểu bẫy")
        fig.colorbar(image, ax=ax, fraction=0.046)
        save(fig, figures_dir / "fig5_3b_trap_type_expert_heatmap.png")
    log.ok("annotated errors", f"{len(labelled)} labelled, tests in trap_type_tests.json")


# ---------------------------------------------------------------------------------------------- figures

def save(fig, path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_length_hist(splits, tokenizer_name, max_length, path):
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    fig, ax = plt.subplots(figsize=(8, 4))
    test_p95 = None
    for name, split in splits.items():
        lengths = [len(tokenizer(p, h)["input_ids"]) for p, h in zip(split["premise"], split["hypothesis"])]
        ax.hist(lengths, bins=60, alpha=0.5, label=f"{name} (n = {len(lengths)})", density=True)
        if name == "test":
            test_p95 = float(np.percentile(lengths, 95))
    if test_p95 is not None:
        ax.axvline(test_p95, color="#C2551B", ls="--", label=f"P95 test = {test_p95:.0f}")
    ax.axvline(max_length, color="black", ls=":", label=f"max_length = {max_length}")
    ax.set_xlabel(f"số subword ({tokenizer_name}, tiền đề + giả thuyết)")
    ax.set_ylabel("mật độ")
    ax.legend()
    ax.set_title("Độ dài cặp câu theo tập")
    save(fig, path)
    return test_p95


def fig_main(table, summary, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.9 * len(table) + 4), 5), sharey=False)
    for ax, column, title in ((axes[0], "test_accuracy", "Test accuracy (%)"), (axes[1], "test_macro_f1", "Macro-F1 (%)")):
        for i, variant in enumerate(table["variant"]):
            values = summary.loc[summary["variant"] == variant, column] * 100
            ax.bar(i, values.mean(), color=viz.COLORS.get(variant, "#888888"), alpha=0.85,
                   yerr=values.std(ddof=1) if len(values) > 1 else None, capsize=4)
            ax.scatter([i] * len(values), values, color="black", s=12, zorder=3)
            if column == "test_accuracy" and variant in PAPER:
                ax.scatter(i, PAPER[variant], marker="*", s=160, color="gold", edgecolor="black", zorder=4)
        ax.axhline(100 / 3, color="grey", ls=":", lw=1)
        ax.set_xticks(range(len(table)), table["code"], rotation=45, ha="right")
        ax.set_ylim(25, None)
        ax.set_title(title + ("  (★ bài báo)" if column == "test_accuracy" else ""))
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, path)


def fig_grad_complexity(summary, project, path, seed=42):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    drawn = False
    for variant in ("B1", "B2"):
        name = viz.runs_of(summary, variant).get(seed)
        log = project / "logs" / f"{name}.jsonl" if name else None
        if log is None or not log.exists():
            continue
        history = viz.load_history(log)
        train = history[history["type"] == "train"]
        if "grad_norm_wc" in train:
            axes[0].semilogy(train["step"], train["grad_norm_wc"].clip(lower=1e-12), color=viz.COLORS[variant],
                             label=viz.VARIANT_NAMES[variant])
            drawn = True
    axes[0].set_title(f"(a) ‖∂L/∂w_c‖ theo bước, seed {seed} (đáy 1e-12 = 0)")
    axes[0].set_xlabel("bước optimizer")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)
    frames = []
    for s, name in sorted(viz.runs_of(summary, "B2").items()):
        frame = viz.routing_frame(name, project)
        if frame is not None and "complexity" in frame:
            frames.append(frame.assign(seed=s))
    if frames:
        data = pd.concat(frames)
        axes[1].boxplot([data.loc[data["correct"] == 1, "complexity"], data.loc[data["correct"] == 0, "complexity"]],
                        showmeans=True)
        axes[1].set_xticks([1, 2], ["đoán đúng", "đoán sai"])
        axes[1].set_title(f"(b) C của E2 theo đúng/sai ({len(frames)} seed)")
        rows = viz.spearman_summary_table(summary, project, ["B2"])
        parts = ["low", "mid", "high"]
        axes[2].plot(parts, [rows[f"acc_C_{p}"].mean() for p in parts], "o-", color="#2350A8", label="accuracy (%)")
        twin = axes[2].twinx()
        twin.plot(parts, [rows[f"experts_C_{p}"].mean() for p in parts], "s--", color="#C2551B", label="chuyên gia/câu")
        axes[2].set_xticks(range(3), ["C thấp", "C giữa", "C cao"])
        axes[2].set_ylabel("accuracy (%)", color="#2350A8")
        twin.set_ylabel("chuyên gia/câu", color="#C2551B")
        axes[2].set_title("(c) E2 theo tam phân vị C")
        drawn = True
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if not drawn:
        import matplotlib.pyplot as plt
        plt.close(fig)
        raise MissingData("needs logs/*.jsonl or results/routing_stats/*_per_sample.csv of E1/E2")
    save(fig, path)


def fig_routing_heatmap(summary, project, path, variants=("B1", "B2", "B4_c002", "A_nobal"), seed=42):
    import matplotlib.pyplot as plt
    frames = {}
    for variant in variants:
        name = viz.runs_of(summary, variant).get(seed)
        frame = viz.routing_frame(name, project) if name else None
        if frame is not None:
            frames[variant] = frame
    if not frames:
        raise MissingData("needs results/routing_stats/*_per_sample.csv")
    fig, axes = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 3.4), squeeze=False)
    for ax, (variant, frame) in zip(axes[0], frames.items()):
        experts = sorted(int(c.split("_")[1]) for c in frame.columns if c.startswith("mask_"))
        usage = np.array([[frame.loc[frame["gold"] == c, f"mask_{e}"].mean() * 100 for e in experts] for c in range(3)])
        ax.imshow(usage, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for (r, c), value in np.ndenumerate(usage):
            ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=7, color="white" if value > 60 else "black")
        ax.set_xticks(range(len(experts)), [f"E{e + 1}" for e in experts], fontsize=8)
        ax.set_yticks(range(3), ["E", "N", "C"])
        ax.set_title(f"{viz.VARIANT_NAMES[variant]}\nTB {frame['n_experts'].mean():.1f} chuyên gia/câu", fontsize=9)
    fig.suptitle(f"% câu kích hoạt từng chuyên gia theo nhãn (seed {seed})")
    fig.tight_layout()
    save(fig, path)


def fig_confusions(summary, project, path, variants=("B0", "B1", "B2", "B4_c002", "B2_base", "B5_kd"), seed=42):
    import matplotlib.pyplot as plt
    names = [(v, viz.runs_of(summary, v).get(seed)) for v in variants]
    names = [(v, n) for v, n in names if n and (project / "predictions" / f"{n}.csv").exists()]
    if not names:
        raise MissingData("needs predictions/*.csv")
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), squeeze=False)
    for ax, (_, name) in zip(axes[0], names):
        viz.plot_confusion(name, project, ax=ax)
    fig.tight_layout()
    save(fig, path)


def fig_collapse(project, path):
    import matplotlib.pyplot as plt
    histories = {}
    for log in sorted((project / "logs").glob("*.jsonl")):
        version = viz.version_of(log.stem)
        if version in ("v3", "v4", "v5") and log.stem.startswith("nlimoe_paper_") and "_seed42_" in log.stem \
                and "_dynamic_dynamic_" in log.stem and version not in histories:
            histories[version] = viz.load_history(log)
    if not {"v3", "v4"} & set(histories):
        raise MissingData("needs v3/v4 NLIMoE logs (logs/nlimoe_paper_v3_dynamic_dynamic_seed42_*.jsonl)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"v3": "#A3430F", "v4": "#B8860B", "v5": "#2350A8"}
    for version, history in sorted(histories.items()):
        train, val = history[history["type"] == "train"], history[history["type"] == "val"]
        if "h_std" in train:
            axes[0].plot(train["step"], train["h_std"], color=colors[version], label=version)
        if "accuracy" in val:
            axes[1].plot(val["step"], val["accuracy"], "o-", color=colors[version], label=version)
    axes[0].set_title("Độ đa dạng h giữa các câu (về 0 = sụp)")
    axes[1].axhline(1 / 3, color="grey", ls=":")
    axes[1].set_title("Val accuracy")
    for ax in axes:
        ax.set_xlabel("bước optimizer")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, path)


def fig_efficiency(project, table, path):
    import matplotlib.pyplot as plt
    efficiency_path = project / "results" / "efficiency.csv"
    if not efficiency_path.exists():
        raise MissingData("missing results/efficiency.csv (run scripts/efficiency.py)")
    eff = pd.read_csv(efficiency_path).merge(table[["variant", "code", "test_acc_mean", "macro_f1_mean"]], on="variant")
    if eff.empty:
        raise MissingData("results/efficiency.csv has no variant with test results")
    columns = [("GMACs_per_sentence", "GMACs / câu (ước tính)")]
    for column, label in (("gpu_ms_per_sentence", "GPU bf16 ms/câu"), ("cpu_int8_ms_per_sentence", "CPU int8 ms/câu")):
        if column in eff and eff[column].notna().any():
            columns.append((column, label))
    fig, axes = plt.subplots(1, len(columns), figsize=(5.5 * len(columns), 4.5), squeeze=False)
    for ax, (column, label) in zip(axes[0], columns):
        for row in eff.itertuples():
            ax.scatter(getattr(row, column), row.test_acc_mean, s=60, color=viz.COLORS.get(row.variant, "#888"))
            ax.annotate(row.code, (getattr(row, column), row.test_acc_mean), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy theo chi phí suy luận")
    fig.tight_layout()
    save(fig, path)


# ------------------------------------------------------------------------------------------------- texts

def load_texts(test_csv=None, skip_dataset=False):
    """{uid: (premise, hypothesis)} of the test set, plus the splits for the length histogram."""
    if test_csv:
        frame = pd.read_csv(test_csv)
        return {u: (p, h) for u, p, h in zip(frame["uid"], frame["premise"], frame["hypothesis"])}, None
    if skip_dataset:
        return {}, None
    try:
        from datasets import load_dataset
        dataset = load_dataset("uitnlp/ViANLI")
    except Exception as error:  # offline, no cache, missing token
        print(f"warning: could not load uitnlp/ViANLI ({type(error).__name__}: {error}); text columns stay empty")
        return {}, None
    test = dataset["test"]
    uids = test["uid"] if "uid" in test.column_names else list(range(len(test)))
    return {u: (p, h) for u, p, h in zip(uids, test["premise"], test["hypothesis"])}, dataset


# -------------------------------------------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=str(PROJECT), help="folder with results/, predictions/, logs/")
    parser.add_argument("--out_dir", default="reports/generated")
    parser.add_argument("--figures_dir", default="reports/figures")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="variants to include (default: every variant found in results/summary.csv)")
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42, help="random seed of the bootstrap and of the error sample")
    parser.add_argument("--error_variant", default="B2")
    parser.add_argument("--error_seed", type=int, default=42)
    parser.add_argument("--errors_per_class", type=int, default=40)
    parser.add_argument("--test_csv", default=None, help="CSV with uid, premise, hypothesis instead of the HF dataset")
    parser.add_argument("--skip_dataset", action="store_true", help="do not load uitnlp/ViANLI (no texts, no Fig. 3.1)")
    parser.add_argument("--length_tokenizer", default="xlm-roberta-large")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--annotated_errors", default=None,
                        help="filled copy of error_annotation_sheet.csv: error-type table, heatmap, chi-square / Kruskal-Wallis")
    return parser.parse_args(argv)


def main(argv=None):
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    args = parse_args(argv)
    project = Path(args.project).resolve()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else project / p
    out_dir, figures_dir = resolve(args.out_dir), resolve(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Log()

    if args.annotated_errors:
        annotated_summary(resolve(args.annotated_errors), figures_dir, out_dir, log)
        log.write(out_dir / "analysis_log.md")
        return 0

    summary = viz.load_summary(project)
    if summary.empty:
        print(f"No finished run found in {project / 'results' / 'summary.csv'}.\n"
              "Copy results/summary.csv, predictions/, results/routing_stats/ and logs/ from Google Drive "
              "(/content/drive/MyDrive/ViANLI_NLIMoE/) into this folder, then run again.")
        return 2
    variants = present_variants(summary, set(args.variants) if args.variants else None)
    summary = summary[summary["variant"].isin(variants)]

    found = inventory(summary, project)
    found.to_csv(out_dir / "input_inventory.csv", index=False)
    print_inventory(found)
    texts, dataset = load_texts(args.test_csv, args.skip_dataset)

    table = main_results(summary, project, variants)
    write_table(table, out_dir, "main_results", "Main results (test set, mean ± std over seeds)",
                "Accuracy and macro-F1 in %. recall_E/N/C: mean over seeds. collapsed_seeds: seeds in which a class "
                "is predicted for < 5% of the sentences. empty_selection_pct: sentences with no expert selected.")
    log.ok("main results", f"{len(table)} variants")

    pairs = [p for p in viz.DEFAULT_PAIRS + viz.P2_PAIRS]
    tests = statistical_tests(summary, project, pairs, args.n_boot, args.seed)
    if not tests.empty:
        write_table(tests, out_dir, "statistical_tests", "Paired comparisons", STAT_NOTE)
        missing = tests["note"].fillna("").astype(bool).sum() if "note" in tests else 0
        log.ok("statistical tests", f"{len(tests)} pairs" + (f", {missing} without prediction files" if missing else ""))
    else:
        log.skip("statistical tests", "no pair of variants with results")

    grad = section(log, "gradient of w_c (logs)", gradient_stats, summary, project, variants)
    spear = section(log, "Spearman C (routing files)", viz.spearman_summary_table, summary, project, variants)
    parts = [f for f in (grad, spear) if f is not None and not f.empty]
    if parts:
        merged = parts[0] if len(parts) == 1 else parts[0].merge(parts[1], on=["variant", "code", "seed"], how="outer")
        write_table(merged, out_dir, "complexity_and_routing", "Complexity C, threshold gradient and routing",
                    "grad_zero / grad_logged: logged steps with ||dL/dw_c|| = 0. rho_C_wrong < 0: sentences with a "
                    "higher C are answered correctly more often, i.e. C acts as an easiness score (rho = 0.1 + 0.1 C, "
                    "so a higher C means fewer experts).")
        log.ok("complexity and routing", f"{len(merged)} runs")
    usage = section(log, "expert usage (routing files)", expert_usage, summary, project, variants)
    if usage is not None and not usage.empty:
        usage.to_csv(out_dir / "expert_usage.csv", index=False)
        log.ok("expert usage", f"{usage['seed'].count()} rows")

    cross = section(log, "cross-run analysis", cross_run_analysis, summary, project, variants, texts)
    if cross is not None:
        result, wrong = cross
        (out_dir / "cross_run_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        wrong.to_csv(out_dir / "unanimously_wrong.csv", index=False, encoding="utf-8-sig")
        log.ok("cross-run analysis", f"{result['runs']} runs, {result['never_correct']} sentences never correct")

    sheet = section(log, "error annotation sheet", error_sheet, summary, project, texts, args.error_variant,
                    args.error_seed, args.errors_per_class, sample_seed=args.seed)
    if sheet is not None:
        sheet.to_csv(out_dir / "error_annotation_sheet.csv", index=False, encoding="utf-8-sig")
        log.ok("error annotation sheet", f"{len(sheet)} errors of {code_of(args.error_variant)} seed {args.error_seed}; "
                                         f"fill trap_type with one of {', '.join(TRAP_TYPES)}")

    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        figures = [
            ("fig5_1_main_accuracy_f1", lambda p: fig_main(table, summary, p)),
            ("fig5_2_grad_and_complexity", lambda p: fig_grad_complexity(summary, project, p)),
            ("fig5_3_routing_heatmap", lambda p: fig_routing_heatmap(summary, project, p)),
            ("fig5_4_confusion_matrices", lambda p: fig_confusions(summary, project, p)),
            ("fig5_5_collapse_v3_v4_v5", lambda p: fig_collapse(project, p)),
            ("fig5_6_efficiency_tradeoff", lambda p: fig_efficiency(project, table, p)),
        ]
        if dataset is not None:
            splits = {"train": dataset["train"], "validation": dataset["validation"], "test": dataset["test"]}
            figures.insert(0, ("fig3_1_length_hist", lambda p: fig_length_hist(splits, args.length_tokenizer, 256, p)))
        else:
            log.skip("fig3_1_length_hist", "dataset not loaded (needs uitnlp/ViANLI)")
        for stem, draw in figures:
            path = figures_dir / f"{stem}.png"
            if section(log, stem, lambda p, draw=draw: draw(p) or True, path):
                log.ok(stem, str(path.relative_to(project)) if path.is_relative_to(project) else str(path))

    log.write(out_dir / "analysis_log.md")
    print(f"\nwrote {out_dir}" + ("" if args.no_figures else f" and {figures_dir}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
