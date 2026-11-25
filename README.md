# ViANLI / NLIMoE

Lightweight Mixture-of-Experts with learnable routing thresholds for Vietnamese adversarial NLI (ViANLI).
Baseline: NLIMoE, [arXiv:2406.17716v3](https://arxiv.org/html/2406.17716v3). Unreported details and our
assumptions are listed in [analysis/paper_extraction.md](analysis/paper_extraction.md).

## Models

| Config | Model |
|---|---|
| `xlmr_large.yaml` | B0 · XLM-R Large + linear head |
| `nlimoe_dynamic.yaml` | B1 · NLIMoE as in the paper (7 experts, dynamic threshold, hard mask) |
| `nlimoe_ste.yaml` | B2 · B1 + straight-through estimator for the threshold |
| `nlimoe_ste_bottleneck.yaml` | B3 · B2 + bottleneck experts 1024→256→1024 |
| `nlimoe_topk.yaml` | ablation · paper Top-K (5 experts, K=4) |
| `nlimoe_ste_nobalance.yaml` | ablation · B2 without balance loss |
| `nlimoe_ste_t005.yaml`, `nlimoe_ste_t02.yaml` | B2 with STE temperature 0.05 / 0.2 |
| `nlimoe_ste_cost002.yaml`, `nlimoe_ste_cost01.yaml` | B4 · B2 + λ · experts per sample in the loss (λ = 0.02 / 0.1) |
| `nlimoe_ste_bottleneck384.yaml` | B3 with r = 384 |
| `xlmr_base_v5.yaml`, `nlimoe_ste_base.yaml` | B0_base, B2_base · same settings on XLM-R Base (~half the compute of Large) |
| `nlimoe_ste_base_kd.yaml` | B5_kd · B2_base distilled from the B2 Large models (`scripts/dump_teacher_logits.py`) |
| `xlmr_base.yaml`, `lightweight_moe_experimental.yaml` | early experiments |

Training follows Sec. 6.2: 7 epochs, batch 16, lr 1e-5, max length 256, dropout 0.4 (experts/head),
validation every 400 optimizer steps; linear schedule with 10% warmup and gradient clipping 1.0
(not reported). Earlier settings are kept in `configs/v3/` and `configs/v4/`
(see [analysis/paper_extraction.md](analysis/paper_extraction.md)).

With the paper's hard mask `1(P > rho)` the complexity gate gets no gradient, so the threshold
is never trained. B2 keeps the hard mask in the forward pass and uses the gradient of
`sigmoid((P - rho) / T)` in the backward pass. B3 replaces each expert by
`Linear(1024,256) → ReLU → Linear(256,1024) → Dropout → ReLU` (525,568 parameters, −49.9%).

The expert-cost penalty only reaches the threshold through the STE: with the hard mask the number of
selected experts is a constant for autograd. Results so far: [analysis/v5_results.md](analysis/v5_results.md).

## Usage

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/run_xlmr.py --config configs/xlmr_large.yaml --seed 42 --precision auto
python scripts/run_moe.py --config configs/nlimoe_ste.yaml --routing_type dynamic --seed 42 --precision auto
python scripts/run_queue.py --variants B0 B1 B2 B3 --seeds 42 123 2026
```

Options: `--precision {fp32,bf16,fp16,auto}`, `--batch_size`, `--grad_accum` (e.g. 8 × 2 = 16),
`--no_dynamic_padding`. `run_queue.py` skips runs already in `results/summary.csv`.

Colab: [colab/ViANLI_NLIMoE_colab.ipynb](colab/ViANLI_NLIMoE_colab.ipynb).

## Outputs

| Path | Content |
|---|---|
| `results/summary.csv` | one row per run |
| `predictions/<run>.csv` | test predictions |
| `results/routing_stats/<run>_per_sample.csv` | C, rho and selected experts per test sample |
| `logs/<run>.jsonl` | training history |
| `checkpoints/<run>_best.pt` | best validation checkpoint and its config |

Plots: `src/utils/viz.py`. Size and speed (parameters, MACs, fp32/bf16/int8 size, GPU/CPU latency,
int8 accuracy): `python scripts/efficiency.py --variants B0 B2 B0_base B2_base B5_kd` → `results/efficiency.csv`.
