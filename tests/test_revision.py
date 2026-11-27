"""REVISION_PLAN.md: empty-selection policies, main.py, efficiency, diagnostics, statistics, KD blending,
P2 variants (residual bottleneck, mean + LayerNorm, single expert, class weights) and analyze_results.py."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import XLMRobertaConfig, XLMRobertaModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from src.models.paper_nlimoe import PaperRouter, ResidualBottleneckExpert, build_expert  # noqa: E402
from src.models.factory import build_moe_model  # noqa: E402
from src.utils import stats, viz  # noqa: E402

TINY = XLMRobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
load = lambda name: yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def tiny_model(config_name, training=None, **moe_overrides):
    cfg = load(config_name)
    cfg["model"].update(expected_hidden_size=16, dropout=0.0, encoder_dropout=0.0)
    cfg["moe"].update(moe_overrides)
    cfg["training"].update(training or {})
    with patch("src.models.paper_nlimoe.AutoConfig.from_pretrained", return_value=TINY), \
            patch("src.models.paper_nlimoe.AutoModel.from_pretrained", side_effect=lambda *a, **k: XLMRobertaModel(TINY)):
        return build_moe_model(cfg)


def tiny_xlmr(**kwargs):
    from src.models.xlmr_baseline import XLMRobertaNLI
    with patch("src.models.xlmr_baseline.AutoConfig.from_pretrained", return_value=TINY), \
            patch("src.models.xlmr_baseline.AutoModel.from_pretrained", side_effect=lambda *a, **k: XLMRobertaModel(TINY)):
        return XLMRobertaNLI(backbone_name="tiny", dropout_prob=0.0, **kwargs)


def batch(size=8, length=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 32, (size, length), generator=generator)
    mask = torch.ones(size, length, dtype=torch.long)
    mask[0, -4:] = 0
    return ids, mask, torch.randint(0, 3, (size,), generator=generator)


def empty_router_model(policy):
    """Model whose router selects no expert for any sentence: P = 1/7 < rho = 0.15."""
    torch.manual_seed(0)
    model = tiny_model("nlimoe_ste", empty_selection_policy=policy)
    with torch.no_grad():
        for layer in (model.router.gate, model.router.complexity_gate):
            layer.weight.zero_()
            layer.bias.zero_()
    return model


def gate_grad_norm(model):
    ids, mask, labels = batch()
    model.train()
    out = model(ids, mask, labels=labels)
    out["loss"].backward()
    grads = [p.grad for p in model.router.complexity_gate.parameters()]
    return out, grads, float(torch.stack([g.norm() for g in grads]).norm())


class EmptySelectionTests(unittest.TestCase):
    def test_v5_configs_keep_the_zero_policy(self):
        for name in ("nlimoe_dynamic", "nlimoe_ste", "nlimoe_ste_bottleneck", "nlimoe_ste_base_kd", "nlimoe_ste_cost002"):
            self.assertEqual(load(name)["moe"]["empty_selection_policy"], "zero", name)
            self.assertEqual(tiny_model(name, bottleneck_dim=4).router.empty_selection_policy, "zero", name)
        with self.assertRaises(ValueError):
            PaperRouter(8, empty_selection_policy="fallback")

    def test_ste_empty_selection_has_bounded_gradient(self):
        out, grads, norm = gate_grad_norm(empty_router_model("zero_safe"))
        self.assertTrue(out["empty_selection"].all())
        self.assertEqual(float(out["routing_weights"].detach().abs().sum()), 0.0)  # forward: zero MoE vector, as in the paper
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.0)
        self.assertLess(norm, 1e3)

    def test_zero_policy_keeps_the_v5_gradient_spike(self):
        # documents why zero_safe exists: the v5 normalisation divides by 1e-9 on empty rows
        _, _, legacy = gate_grad_norm(empty_router_model("zero"))
        _, _, safe = gate_grad_norm(empty_router_model("zero_safe"))
        self.assertGreater(legacy, 1e6)
        self.assertGreater(legacy / safe, 1e6)

    def test_zero_safe_equals_zero_when_no_row_is_empty(self):
        torch.manual_seed(0)
        zero = tiny_model("nlimoe_ste")
        safe = tiny_model("nlimoe_ste", empty_selection_policy="zero_safe")
        safe.load_state_dict(zero.state_dict())
        with torch.no_grad():  # threshold 0.1 < max P: every row selects at least one expert
            for model in (zero, safe):
                model.router.complexity_gate.weight.zero_()
                model.router.complexity_gate.bias.fill_(-30.0)
        ids, mask, labels = batch()
        zero.train(), safe.train()
        a, b = zero(ids, mask, labels=labels), safe(ids, mask, labels=labels)
        self.assertFalse(a["empty_selection"].any())
        torch.testing.assert_close(a["logits"], b["logits"])
        a["loss"].backward(), b["loss"].backward()
        for (name, p), q in zip(zero.named_parameters(), safe.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, msg=name)

    def test_empty_selection_policy_top1(self):
        router = PaperRouter(8, empty_selection_policy="top1")
        with torch.no_grad():
            router.complexity_gate.weight.zero_()
            router.complexity_gate.bias.zero_()
        x = torch.randn(32, 8)
        with torch.no_grad():
            router.gate.weight.mul_(0.05)  # P close to 1/7 < 0.15: many rows empty before the fallback
            router.gate.bias.zero_()
        for train_mode in (True, False):
            router.train(train_mode)
            out = router(x)
            self.assertTrue(out["empty_selection"].any())  # reported before the fallback
            self.assertTrue((out["selected_mask"].sum(-1) >= 1).all())
            torch.testing.assert_close(out["routing_weights"].sum(-1), torch.ones(32))
            chosen = out["selected_mask"][out["empty_selection"]].float().argmax(-1)
            torch.testing.assert_close(chosen, out["raw_probs"][out["empty_selection"]].argmax(-1))


class MainAndScriptTests(unittest.TestCase):
    def test_main_menu_is_sorted_by_number_and_forwards_configs(self):
        import main
        keys = [k for k, _ in main.ordered_tasks()]
        self.assertEqual(keys, [str(i) for i in range(1, len(main.TASKS) + 1)])
        self.assertIn("scripts/efficiency.py", main.TASKS[main.ALIASES["efficiency"]][1])
        self.assertIn("scripts/analyze_results.py", main.TASKS[main.ALIASES["analyze"]][1])
        self.assertEqual(main.config_command("configs/nlimoe_ste.yaml", 42, ["--precision", "auto"]),
                         ["scripts/run_moe.py", "--config", "configs/nlimoe_ste.yaml", "--seed", "42", "--precision", "auto"])
        self.assertEqual(main.config_command("configs/xlmr_large.yaml")[0], "scripts/run_xlmr.py")
        with patch.object(main, "run_command") as run, patch.object(sys, "argv", ["main.py", "--config",
                                                                                   "configs/nlimoe_ste.yaml", "--seed", "7"]):
            main.main()
        self.assertEqual(run.call_args[0][1][-2:], ["--seed", "7"])

    def test_benchmark_moe_head_sparse_and_dense(self):
        from scripts.efficiency import benchmark_moe_head, is_moe, quantize_int8
        torch.manual_seed(0)
        model = tiny_model("nlimoe_ste").eval()
        self.assertTrue(is_moe(model))
        pooled = torch.randn(40, 16)
        for sparse in (True, False):
            self.assertGreater(benchmark_moe_head(model, pooled, torch.device("cpu"), sparse=sparse, repeats=2), 0.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quantized = quantize_int8(model)
        self.assertFalse([w for w in caught if "quantiz" in str(w.message).lower()])
        self.assertGreater(benchmark_moe_head(quantized, pooled, torch.device("cpu"), repeats=1), 0.0)

    def test_routing_diagnostics_report_empty_selection(self):
        from src.utils.routing_diagnostics import compute_routing_diagnostics
        mask = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float)
        out = compute_routing_diagnostics(mask / mask.sum(-1, keepdim=True).clamp_min(1), mask)
        self.assertEqual(out["empty_selection_pct"], 50.0)
        self.assertEqual((out["min_experts_per_sample"], out["max_experts_per_sample"]), (0, 2))

    def test_model_display_name(self):
        from scripts.run_moe import _model_display_name
        name = lambda config: _model_display_name(load(config), True, load(config)["moe"]["threshold_gradient"] == "straight_through")
        self.assertEqual(name("nlimoe_dynamic"), "NLIMoE-paper-v3")
        self.assertEqual(name("nlimoe_ste"), "NLIMoE-paper-v3+STE")
        self.assertEqual(name("nlimoe_ste_bottleneck"), "NLIMoE-paper-v3+STE+bottleneck(r=256)")
        self.assertEqual(name("nlimoe_ste_bottleneck384"), "NLIMoE-paper-v3+STE+bottleneck(r=384)")
        self.assertEqual(name("nlimoe_ste_res_bottleneck256"), "NLIMoE-paper-v3+STE+res_bottleneck(r=256)")
        self.assertEqual(name("nlimoe_ste_cost002"), "NLIMoE-paper-v3+STE+cost0.02")
        self.assertEqual(name("nlimoe_ste_base_kd"), "NLIMoE-paper-v3+STE+KD")
        self.assertEqual(name("nlimoe_ste_safe_empty"), "NLIMoE-paper-v3+STE+empty_zero_safe")

    def test_blend_aux(self):
        from src.utils.training import blend_distillation
        loss, task, kd = torch.tensor(2.0), torch.tensor(1.5), torch.tensor(0.4)
        self.assertAlmostEqual(float(blend_distillation(loss, task, kd, 0.5)), 0.5 * 2.0 + 0.5 * 0.4)
        self.assertAlmostEqual(float(blend_distillation(loss, task, kd, 0.5, blend_aux=False)), 0.5 * 1.5 + 0.5 * 0.4 + 0.5)
        self.assertAlmostEqual(float(blend_distillation(loss, None, kd, 0.5, blend_aux=False)), 1.2)


class MonitorTests(unittest.TestCase):
    def test_queue_state_waits_for_the_process_not_the_old_status_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "logs").mkdir()
            status = lambda *states: (root / "logs" / "queue_status.json").write_text(
                json.dumps({"queue": [{"variant": "B0", "seed": 42, "state": s} for s in states]}), encoding="utf-8")
            status("done", "done")  # left by the previous queue
            with patch.object(viz, "queue_alive", return_value=True):
                self.assertEqual(viz.queue_state(root), "running")  # new queue still planning
            with patch.object(viz, "queue_alive", return_value=False):
                self.assertEqual(viz.queue_state(root), "done")
                status("done", "pending")
                self.assertEqual(viz.queue_state(root), "stopped")
                (root / "logs" / "queue_status.json").unlink()
                self.assertEqual(viz.queue_state(root), "stopped")


class StatisticsTests(unittest.TestCase):
    def test_clustered_bootstrap(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, (400, 3))
        same = stats.clustered_bootstrap(a, a, n_boot=2000)
        self.assertEqual(same["mean_diff"], 0.0)
        self.assertEqual(same["p_value_two_sided"], 1.0)
        self.assertLessEqual(same["ci_95_low"], 0.0)
        self.assertGreaterEqual(same["ci_95_high"], 0.0)
        better = a.copy()
        better[:80] = 1  # B right on 80 sentences where A may be wrong
        result = stats.clustered_bootstrap(a, better, n_boot=2000)
        self.assertAlmostEqual(result["mean_diff"], (better.mean(1) - a.mean(1)).mean())
        self.assertGreater(result["ci_95_low"], 0.0)
        self.assertLess(result["p_value_two_sided"], 0.01)
        self.assertEqual(result, stats.clustered_bootstrap(a, better, n_boot=2000))  # seeded
        # the unit is the sentence: S identical seeds give the same CI as one seed
        one = stats.clustered_bootstrap(a[:, :1], better[:, :1], n_boot=2000)
        three = stats.clustered_bootstrap(np.repeat(a[:, :1], 3, 1), np.repeat(better[:, :1], 3, 1), n_boot=2000)
        self.assertEqual((one["ci_95_low"], one["ci_95_high"]), (three["ci_95_low"], three["ci_95_high"]))

    def test_mcnemar_vote_and_agreement(self):
        self.assertEqual(stats.exact_mcnemar([1, 1, 0, 0], [1, 1, 0, 0])["p_value"], 1.0)
        result = stats.exact_mcnemar([0] * 20, [1] * 20)
        self.assertEqual((result["only_a"], result["only_b"]), (0, 20))
        self.assertLess(result["p_value"], 1e-4)
        pred = np.array([[0, 1, 2, 1], [0, 1, 1, 1], [2, 1, 0, 1]])
        self.assertAlmostEqual(stats.pairwise_agreement(pred), np.mean([3 / 4, 2 / 4, 2 / 4]))
        np.testing.assert_array_equal(stats.majority_vote(pred)[[0, 1, 3]], [0, 1, 1])
        self.assertEqual(stats.collapsed_classes(np.array([0] * 98 + [1, 1])), [1, 2])


def make_project(root, n=90, variants=("B0", "B1", "B2", "B0_base"), seeds=(42, 123, 2026), with_routing=True,
                 with_logs=True, accuracy=None):
    """Synthetic results folder: summary.csv, predictions, routing files and logs."""
    from scripts.run_queue import experiment_prefix
    accuracy = accuracy or {"B0": 0.45, "B1": 0.45, "B2": 0.6, "B0_base": 0.4}
    rng = np.random.default_rng(1)
    gold = np.repeat(np.arange(3), n // 3)
    uids = [f"u{i:03d}" for i in range(n)]
    for folder in ("results/routing_stats", "predictions", "logs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for v_index, variant in enumerate(variants):
        for s_index, seed in enumerate(seeds):
            name = f"{experiment_prefix(variant)}_seed{seed}_20261120_000000_{v_index}{s_index:05d}"
            right = rng.random(n) < accuracy[variant]
            right[:3] = False  # three sentences nobody gets right
            pred = np.where(right, gold, (gold + 1 + rng.integers(0, 2, n)) % 3)
            probs = np.full((n, 3), 0.1)
            probs[np.arange(n), pred] = 0.8
            pd.DataFrame({"uid": uids, "gold_label": [viz.LABELS[g] for g in gold],
                          "pred_label": [viz.LABELS[p] for p in pred], "prob_entailment": probs[:, 0],
                          "prob_neutral": probs[:, 1], "prob_contradiction": probs[:, 2]}).to_csv(
                root / "predictions" / f"{name}.csv", index=False)
            moe = variant not in viz.NO_MOE_VARIANTS
            if moe and with_routing:
                mask = (rng.random((n, 7)) < 0.4).astype(int)
                mask[:5] = 0
                complexity = rng.random(n) * 0.5 + 0.3 * (1 - right)
                frame = pd.DataFrame({"uid": uids, "gold": gold, "pred": pred, "correct": right.astype(int),
                                      "n_experts": mask.sum(1), "complexity": complexity,
                                      "threshold": 0.1 + 0.1 * complexity})
                for e in range(7):
                    frame[f"mask_{e}"] = mask[:, e]
                    frame[f"weight_{e}"] = mask[:, e] / np.maximum(1, mask.sum(1))
                frame.to_csv(root / "results" / "routing_stats" / f"{name}_per_sample.csv", index=False)
            if with_logs:
                events = [{"type": "start", "total_steps": 4}]
                for step in range(1, 5):
                    event = {"type": "train", "step": step, "loss": 1.0, "h_std": 0.5}
                    if moe:
                        event.update(grad_norm_wc=0.0 if variant == "B1" else 0.01 * step,
                                     mean_complexity=0.5 - 0.05 * step, mean_threshold=0.15 - 0.005 * step,
                                     avg_experts=3.0)
                    events.append(event)
                events.append({"type": "val", "step": 4, "accuracy": 0.4, "macro_f1": 0.4})
                (root / "logs" / f"{name}.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            rows.append({"experiment_name": name, "seed": seed, "val_accuracy": 0.4, "val_macro_f1": 0.4,
                         "test_accuracy": right.mean(), "test_macro_f1": right.mean() - 0.02,
                         "avg_experts_per_sample": 2.8 if moe else 1.0, "routing_entropy": 0.9 if moe else 0.0,
                         "train_time": 600.0, "inference_time": 3.0})
    pd.DataFrame(rows).to_csv(root / "results" / "summary.csv", index=False)
    return pd.DataFrame({"uid": uids, "premise": [f"Có {i} trường học ở đây." for i in range(n)],
                         "hypothesis": [("Không có " if i % 2 else "Có ") + f"{i} trường." for i in range(n)]})


class ProjectStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.texts = make_project(self.root)
        self.summary = viz.load_summary(self.root)

    def tearDown(self):
        self.folder.cleanup()

    def test_viz_statistics_on_a_synthetic_project(self):
        self.assertEqual(sorted(self.summary["variant"].unique()), ["B0", "B0_base", "B1", "B2"])
        boot = viz.clustered_bootstrap_test(self.summary, "B1", "B2", n_boot=1000, project=self.root)
        self.assertGreater(boot["mean_diff"], 0)
        self.assertEqual(boot["n_sentences"], 90)
        self.assertLessEqual(boot["ci_95_low"], boot["mean_diff"])
        pooled = viz.pooled_mcnemar(self.summary, "B1", "B2", self.root)
        self.assertEqual(pooled["n_pairs"], 270)
        self.assertIn("reference only", pooled["note"])
        ensemble = viz.ensemble_mcnemar(self.summary, "B1", "B2", self.root)
        self.assertEqual(ensemble["n_pairs"], 90)
        recall = viz.per_class_recall_table(self.summary, self.root)
        self.assertEqual(list(recall["variant"]), ["B0", "B1", "B2", "B0_base"])
        spear = viz.spearman_summary_table(self.summary, self.root)
        self.assertEqual(set(spear["variant"]), {"B1", "B2"})  # E0 / E5a have no C
        self.assertTrue((spear["rho_C_wrong"] > 0).all())  # synthetic C is higher on errors
        self.assertTrue((spear["empty_selection_pct"] > 0).all())
        table = viz.results_table(self.summary)
        self.assertEqual(table.set_index("Biến thể").loc[viz.VARIANT_NAMES["B0_base"], "Chuyên gia/câu"], "—")
        with self.assertRaises(viz.MissingData):
            os.remove(next((self.root / "predictions").glob("nlimoe_paper_v5_ste_dynamic_seed42_*.csv")))
            viz.clustered_bootstrap_test(self.summary, "B1", "B2", n_boot=10, project=self.root)

    def test_analyze_results_writes_tables_figures_and_error_sheet(self):
        import scripts.analyze_results as ar
        test_csv = self.root / "test.csv"
        self.texts.to_csv(test_csv, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            code = ar.main(["--project", str(self.root), "--test_csv", str(test_csv), "--n_boot", "500",
                            "--errors_per_class", "5"])
        self.assertEqual(code, 0)
        out, figures = self.root / "reports" / "generated", self.root / "reports" / "figures"
        for name in ("main_results.csv", "main_results.md", "statistical_tests.csv", "complexity_and_routing.csv",
                     "expert_usage.csv", "cross_run_analysis.json", "unanimously_wrong.csv",
                     "error_annotation_sheet.csv", "analysis_log.md", "input_inventory.csv"):
            self.assertTrue((out / name).exists(), name)
        for name in ("fig5_1_main_accuracy_f1", "fig5_2_grad_and_complexity", "fig5_3_routing_heatmap",
                     "fig5_4_confusion_matrices"):
            self.assertTrue((figures / f"{name}.png").exists(), name)
        main_table = pd.read_csv(out / "main_results.csv")
        self.assertEqual(list(main_table["code"]), ["E0", "E1", "E2", "E5a"])
        tests = pd.read_csv(out / "statistical_tests.csv")
        self.assertIn("E2 vs E1", set(tests["comparison"]))
        self.assertTrue({"bootstrap_ci95", "bootstrap_p", "ensemble_mcnemar_p", "mcnemar_pooled_p"} <= set(tests))
        routing = pd.read_csv(out / "complexity_and_routing.csv")
        b1 = routing[routing["variant"] == "B1"]
        self.assertTrue((b1["grad_zero"] == b1["grad_logged"]).all())
        cross = json.loads((out / "cross_run_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(cross["runs"], 12)
        self.assertGreaterEqual(cross["never_correct"], 3)
        wrong = pd.read_csv(out / "unanimously_wrong.csv")
        self.assertTrue({"u000", "u001", "u002"} <= set(wrong["uid"]))
        self.assertEqual((wrong[["vote_E", "vote_N", "vote_C"]].sum(axis=1) == 12).all(), True)
        self.assertTrue(wrong["premise"].str.startswith("Có").all())
        sheet = pd.read_csv(out / "error_annotation_sheet.csv")
        self.assertEqual(sheet["gold"].value_counts().to_dict(), {label: 5 for label in viz.LABELS})
        self.assertTrue({"pred_E2", "pred_E1", "wrong_in_E1", "complexity_C", "threshold_rho", "n_experts",
                         "selected_experts", "lexical_overlap_jaccard", "has_negation_cue", "has_number_cue",
                         "trap_type", "notes"} <= set(sheet))
        self.assertTrue((sheet["has_number_cue"] == 1).all())
        self.assertTrue(sheet["trap_type"].isna().all())

        # annotated mode: error-type table, tests and heatmap
        types = list(ar.TRAP_TYPES)
        sheet["trap_type"] = [types[i % 3] for i in range(len(sheet))]
        sheet["selected_experts"] = [",".join(str(e) for e in range(i % 4 + 1)) for i in range(len(sheet))]
        filled = out / "filled.csv"
        sheet.to_csv(filled, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ar.main(["--project", str(self.root), "--annotated_errors", str(filled)]), 0)
        counts = pd.read_csv(out / "trap_type_counts.csv")
        self.assertEqual(int(counts["count"].sum()), len(sheet))
        result = json.loads((out / "trap_type_tests.json").read_text(encoding="utf-8"))
        self.assertIn("chi2_trap_type_x_expert", result)
        self.assertIn("kruskal_n_experts_by_trap_type", result)
        self.assertTrue((figures / "fig5_3b_trap_type_expert_heatmap.png").exists())

    def test_analyze_results_reports_missing_inputs(self):
        import scripts.analyze_results as ar
        with tempfile.TemporaryDirectory() as folder:
            bare = Path(folder)
            make_project(bare, variants=("B1", "B2"), with_routing=False, with_logs=False)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(ar.main(["--project", str(bare), "--skip_dataset", "--n_boot", "100",
                                          "--no_figures"]), 0)
            log = (bare / "reports" / "generated" / "analysis_log.md").read_text(encoding="utf-8")
            self.assertIn("missing logs/", log)
            self.assertIn("_per_sample.csv", log)
            self.assertIn("routing_per_sample 0/6 present", buffer.getvalue())
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ar.main(["--project", folder, "--skip_dataset"]), 2)  # no summary.csv


class P2VariantTests(unittest.TestCase):
    def test_residual_bottleneck(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        expert = build_expert(1024, dropout=0.0, expert_type="residual_bottleneck", bottleneck_dim=256)
        self.assertIsInstance(expert, ResidualBottleneckExpert)
        self.assertEqual(count(expert), 525_568)
        self.assertEqual(count(expert), count(build_expert(1024, expert_type="bottleneck", bottleneck_dim=256)))
        small = ResidualBottleneckExpert(16, 4, dropout=0.0)
        with torch.no_grad():
            small.up.weight.zero_()
            small.up.bias.zero_()
        x = torch.randn(5, 16, requires_grad=True)
        torch.testing.assert_close(small(x), x)  # identity when the up-projection is zero
        torch.manual_seed(0)
        model = tiny_model("nlimoe_ste_res_bottleneck256", bottleneck_dim=4)
        self.assertEqual(model.experts.expert_type, "residual_bottleneck")
        ids, mask, labels = batch()
        model.train()
        model(ids, mask, labels=labels)["loss"].backward()
        self.assertGreater(sum(float(e.down.weight.grad.abs().sum()) for e in model.experts.experts), 0.0)
        self.assertGreater(float(model.router.complexity_gate.weight.grad.abs().sum()), 0.0)

    def test_xlmr_mean_ln(self):
        torch.manual_seed(0)
        plain = tiny_xlmr(pooling="mean")
        torch.manual_seed(0)
        normed = tiny_xlmr(pooling="mean", use_layer_norm=True)
        self.assertFalse(hasattr(plain, "layer_norm"))
        self.assertEqual(set(normed.state_dict()) - set(plain.state_dict()), {"layer_norm.weight", "layer_norm.bias"})
        ids, mask, labels = batch()
        out = normed.eval()(ids, mask, labels=labels)
        expected = normed.classifier(normed.layer_norm(out["pooled_representation"]))
        torch.testing.assert_close(out["logits"], expected)
        cfg = load("xlmr_large_mean_ln")
        self.assertEqual((cfg["model"]["pooling"], cfg["model"]["use_layer_norm"]), ("mean", True))

    def test_class_weights(self):
        ids, mask, labels = batch()
        torch.manual_seed(0)
        weighted = tiny_model("nlimoe_ste_cw")
        torch.testing.assert_close(weighted.class_weights, torch.tensor([1.0, 1.0, 1.2]))
        self.assertNotIn("class_weights", weighted.state_dict())  # checkpoints stay loadable by plain models
        weighted.eval()
        out = weighted(ids, mask, labels=labels)
        torch.testing.assert_close(out["task_loss"], torch.nn.functional.cross_entropy(
            out["logits"], labels, weight=torch.tensor([1.0, 1.0, 1.2])))
        torch.manual_seed(0)
        plain = tiny_model("nlimoe_ste")
        self.assertIsNone(plain.class_weights)
        xlmr = tiny_xlmr(class_weights=[1.0, 1.0, 1.2]).eval()
        out = xlmr(ids, mask, labels=labels)
        torch.testing.assert_close(out["loss"], torch.nn.functional.cross_entropy(
            out["logits"], labels, weight=torch.tensor([1.0, 1.0, 1.2])))
        with self.assertRaises(ValueError):
            tiny_xlmr(class_weights=[1.0, 2.0])

    def test_single_expert_dense(self):
        torch.manual_seed(0)
        model = tiny_model("nlimoe_single_expert")
        self.assertEqual(len(model.experts.experts), 1)
        ids, mask, labels = batch()
        for train_mode in (True, False):
            model.train(train_mode)
            out = model(ids, mask, labels=labels)
            self.assertTrue(out["selected_mask"].all())
            torch.testing.assert_close(out["routing_weights"], torch.ones(8, 1))

    def test_p2_configs_change_only_one_setting(self):
        def diff(a, b, path=""):
            keys = set(a) | set(b) if isinstance(a, dict) and isinstance(b, dict) else None
            if keys is None:
                return set() if a == b else {path}
            return set().union(*(diff(a.get(k), b.get(k), f"{path}.{k}".strip(".")) for k in keys)) if keys else set()
        cases = {
            "xlmr_large_mean_ln": ("xlmr_large", {"model.pooling", "model.use_layer_norm"}),
            "nlimoe_single_expert": ("nlimoe_dynamic", {"moe.num_experts", "moe.routing_type", "moe.top_k"}),
            "nlimoe_ste_safe_empty": ("nlimoe_ste", {"moe.empty_selection_policy"}),
            "nlimoe_ste_res_bottleneck256": ("nlimoe_ste_bottleneck", {"moe.expert_type"}),
            "nlimoe_ste_cw": ("nlimoe_ste", {"training.class_weights"}),
            "nlimoe_ste_base_kd_ens": ("nlimoe_ste_base_kd", {"training.distill.teacher"}),
            "nlimoe_ste_large_kd_ens": ("nlimoe_ste", {"training.distill"}),
        }
        for name, (parent, expected) in cases.items():
            child, base = load(name), load(parent)
            changed = diff(child, base) - {"experiment.name", "experiment.description"}
            self.assertEqual(changed, expected, name)
            self.assertNotEqual(child["experiment"]["name"], base["experiment"]["name"])
        self.assertEqual(load("nlimoe_ste_large_kd_ens")["training"]["distill"],
                         {**load("nlimoe_ste_base_kd")["training"]["distill"], "teacher": "results/teacher/B0_B2_large.pt"})

    def test_p2_variants_are_registered(self):
        import scripts.run_queue as rq
        names = {
            "A_mean_ln": "xlmr_large_mean_ln_paper_v5_seed42_20261127_101010_000001",
            "A_1exp": "nlimoe_paper_v5_single_expert_dense_seed42_20261127_101010_000002",
            "B2_safe": "nlimoe_paper_v5_ste_safe_empty_dynamic_seed42_20261127_101010_000003",
            "B3_res256": "nlimoe_paper_v5_ste_res_bottleneck256_dynamic_seed42_20261127_101010_000004",
            "B2_cw": "nlimoe_paper_v5_ste_cw_dynamic_seed42_20261127_101010_000005",
            "B5_kd_ens": "nlimoe_paper_v5_ste_base_kd_ens_dynamic_seed42_20261127_101010_000006",
            "B2_kd_ens": "nlimoe_paper_v5_ste_large_kd_ens_dynamic_seed42_20261127_101010_000007",
        }
        for variant, name in names.items():
            self.assertIn(variant, rq.P2_VARIANTS)
            self.assertEqual(viz.variant_of(name), variant)
            self.assertIn(variant, viz.VARIANT_NAMES)
            self.assertIn(variant, viz.COLORS)
            self.assertTrue(rq.is_done(variant, 42, pd.DataFrame({"experiment_name": [name]})))
        # existing v5 names still map to their variant
        self.assertEqual(viz.variant_of("nlimoe_paper_v5_ste_base_kd_dynamic_seed42_20261126_101010_000007"), "B5_kd")
        self.assertEqual(viz.variant_of("xlmr_large_paper_v5_seed42_20261117_040330_385566"), "B0")
        self.assertEqual(rq.V5_VARIANTS, viz.V5_VARIANTS)
        args = type("Args", (), {"precision": "auto", "batch_size": None, "grad_accum": None, "epochs": None})()
        routing = lambda v: rq.command_for(v, 42, args)[rq.command_for(v, 42, args).index("--routing_type") + 1]
        self.assertEqual(routing("A_1exp"), "dense")
        self.assertEqual(routing("A_topk"), "top_k")
        self.assertEqual(routing("B2"), "dynamic")
        self.assertNotIn("--routing_type", rq.command_for("A_mean_ln", 42, args))
        self.assertEqual(rq.job_memory_gb("B5_kd_ens"), 6.5)
        self.assertEqual(rq.TEACHERS["B2_kd_ens"], "results/teacher/B0_B2_large.pt")


if __name__ == "__main__":
    unittest.main()
