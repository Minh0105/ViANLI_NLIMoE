# Kế hoạch Điều chỉnh & Hoàn thiện Dự án `ViANLI_NLIMoE` (Detailed Revision Spec)

> **Mục đích tệp này:** Đặc tả kỹ thuật, khoa học và tài liệu chi tiết đến từng tệp, số dòng, công thức và bài kiểm thử để kỹ sư / AI coding agent (Opus 5.5) thực thi trực tiếp nhằm nâng chất lượng dự án lên mức chặt chẽ tuyệt đối trước khi nộp báo cáo cuối kỳ ($\ge 30$ trang) và bảo vệ đồ án CS2311.11.
>
> **Nguyên tắc bất di bất dịch khi chỉnh sửa:**
> 1. **Không phá vỡ khả năng tái lập của 42 lần chạy v5 đã hoàn tất:** Mọi cải tiến kiến trúc mới (ví dụ `residual_bottleneck`, `empty_selection_policy="top1"`, `class_weights`, `use_layer_norm`) đều phải là **opt-in qua YAML config**, giữ nguyên mặc định cũ cho các config v5 hiện hữu.
> 2. **Giữ toàn bộ test hiện có pass 100%** và bổ sung unit test mới cho mọi thay đổi mã nguồn. Sau mọi đợt sửa code, chạy:
>    ```bash
>    python -m unittest discover -s tests -v
>    ```
> 3. **Mọi tuyên bố trong báo cáo, slide, đề cương phải khớp 100% với mã nguồn thực tế và số liệu lưu trên đĩa.**

---

## Mục lục Nhanh theo Độ ưu tiên

| Mức | Nhóm công việc | Mục tiêu chính |
|---|---|---|
| **P0** | Sửa lỗi kỹ thuật & bất nhất trong mã nguồn | Vá lỗ hổng gradient spike $10^9\times$ ở STE empty selection, sửa lỗi CLI `main.py`, tối ưu `efficiency.py`, hoàn thiện `viz.py` & `routing_diagnostics.py` |
| **P1** | Đồng bộ dữ liệu Colab, kiểm định thống kê chuẩn & hoàn thiện MT4 | Kéo kết quả 42 runs về local, bổ sung kiểm định **clustered bootstrap theo 1.000 câu**, đo latency/int8 thật, tạo script giải phẫu 100+ lỗi & 36 câu sai tuyệt đối |
| **P2** | Thí nghiệm bổ sung bịt lỗ hổng phản biện (chạy GPU) | Ablation tách biến nhiễu `Mean+LayerNorm` (`E0_mean`), sửa bottleneck có residual (`E3_res256`), thử class weighting cho lớp `contradiction` & ensemble KD |
| **P3** | Đồng bộ toàn bộ tài liệu (`.md`, `.docx`, `.pptx`) | Cập nhật đồng nhất từ v3 $\to$ v5, ký hiệu `E0–E6 / A1–A3`, xóa mọi placeholder, đảm bảo Báo cáo cuối kỳ $\ge 30$ trang và Slide khớp 31 slide của `script_video.md` |

---

## P0 — Sửa lỗi Kỹ thuật & Bất nhất trong Mã nguồn

### P0.1. Vá lỗi phóng đại gradient $10^9\times$ khi `empty_selection` trong STE (`src/models/paper_nlimoe.py`)

* **Tệp & dòng:** `src/models/paper_nlimoe.py` (dòng 34–66, đặc biệt dòng 51–57) và `src/models/factory.py` (dòng 10–20).
* **Hiện trạng code:**
  ```python
  selection = mask.to(probabilities.dtype)
  if mode == "dynamic" and self.training and self.threshold_gradient == "straight_through":
      soft = torch.sigmoid((probabilities - threshold) / self.threshold_temperature)
      selection = selection + (soft - soft.detach())
  masked = probabilities * selection
  weights = masked / masked.sum(-1, keepdim=True).clamp_min(1e-9)
  ```
* **Phân tích lỗi toán học:**
  * Khi một mẫu không có chuyên gia nào vượt ngưỡng (`mask.any(-1) == False`), ở lượt truyền xuôi `selection = 0` $\Rightarrow$ `masked = 0` $\Rightarrow$ `masked.sum(-1) = 0`.
  * Phép `.clamp_min(1e-9)` trả về hằng số `1e-9` (đạo hàm theo mẫu số $= 0$), nhưng tử số `masked = probabilities * selection` **vẫn truyền đạo hàm qua `soft`**.
  * Kết quả: $\frac{\partial \text{weights}}{\partial \text{soft}} = \frac{P}{10^{-9}} \sim 10^8\text{–}10^9$.
  * Hậu quả: Gây nổ gradient trên cấu hình không có clipping (v4) và làm gradient clip (`max_grad_norm = 1.0` ở v5) bị chi phối hoàn toàn bởi 1 mẫu rỗng, triệt tiêu tín hiệu cập nhật của 15 mẫu còn lại trong batch.
  * Ngoài ra, khóa YAML `moe.empty_selection_policy` (ví dụ `configs/nlimoe_dynamic.yaml` dòng 32) hiện **chưa được đọc** trong `factory.py` và `PaperRouter`.
* **Yêu cầu chỉnh sửa:**
  1. Thêm tham số `empty_selection_policy: str = "zero"` vào `PaperRouter.__init__` và `PaperNLIMoEModel.__init__`, truyền từ `cfg['moe'].get('empty_selection_policy', 'zero')` trong `src/models/factory.py`. Chỉ chấp nhận `"zero"` hoặc `"top1"`.
  2. Xử lý an toàn số học & gradient trong `PaperRouter.forward`:
     * Nếu `self.empty_selection_policy == "top1"`: trước bước tạo `selection`, với các hàng `empty = ~mask.any(-1)`, bật top-1 bằng `mask = mask.clone()` rồi `scatter(1, probabilities[empty].argmax(-1, keepdim=True), True)`. Lưu ý vẫn ghi `"empty_selection": raw_empty` (trạng thái trước fallback) để theo dõi thống kê.
     * Nếu `self.empty_selection_policy == "zero"` (mặc định, giữ nguyên hành vi paper):
       ```python
       has_any = mask.any(-1, keepdim=True)
       denom = masked.sum(-1, keepdim=True)
       # Khi mask toàn 0: forward của masked = 0 nên weights = 0 bất kể mẫu số > 0.
       # Dùng mẫu số 1.0 thay cho 1e-9 ở hàng rỗng để tránh khuếch đại gradient STE lên 10^9 lần.
       safe_denom = torch.where(has_any, denom.clamp_min(1e-9), torch.ones_like(denom))
       weights = masked / safe_denom
       ```
       *(Kiểm chứng toán: khi `has_any == False`, forward `weights = 0 / 1.0 = 0.0` giữ nguyên 100% hành vi bài báo và `test_empty_selection_convention_is_finite_and_not_top1_fallback`; backward $\frac{\partial \text{weights}}{\partial \text{soft}} = P \approx 0.14$ hữu hạn và đúng tỷ lệ!)*
  3. **Bổ sung unit test** trong `tests/test_improvements.py`:
     * `test_ste_empty_selection_has_bounded_gradient`: Khởi tạo `gate` và `complexity_gate` về 0 (giống `test_paper_alignment.py` L40–46 để `1/7 < 0.15` gây 100% empty selection), bật `threshold_gradient="straight_through"`, chạy `backward()`, xác nhận:
       * `output["routing_weights"]` forward toàn 0;
       * `complexity_gate.weight.grad` và `bias.grad` hữu hạn (`isfinite`), khác 0, và có norm `< 1e3` (trước khi sửa, norm sẽ $> 10^6$).
     * `test_empty_selection_policy_top1`: xác nhận khi cấu hình `empty_selection_policy="top1"`, mọi mẫu chọn ít nhất 1 chuyên gia và tổng `routing_weights == 1`.

---

### P0.2. Sửa các lỗi nhỏ & bất nhất trong `main.py`

* **Tệp & dòng:** `main.py` (dòng 56–60, 70–97).
* **Các lỗi hiện tại:**
  1. Dòng 56: `sorted(TASKS.items())` sắp xếp chuỗi nên in thứ tự sai: `[1], [10], [11], [12], [2], [3]...`.
  2. Dòng 60: `input("Nhập số tương ứng [0-8]: ")` sai phạm vi vì có tới 12 tác vụ (`[0-12]`).
  3. Phụ lục B của `Bao_cao_cuoi_ky_LightweightMoE_ViANLI.docx` (dòng 690) ghi lệnh `python main.py --config configs/nlimoe_dynamic.yaml --seed 42`, nhưng `main.py` chỉ có `--task`, không có `--config` hay `--seed`.
* **Yêu cầu chỉnh sửa:**
  1. Sửa dòng 56 thành `sorted(TASKS.items(), key=lambda kv: int(kv[0]))`.
  2. Sửa dòng 60 thành `input(f"Nhập số tương ứng [0-{len(TASKS)}]: ")`.
  3. Cập nhật `TASKS` cho nhất quán với các biến thể hiện có (thêm tuỳ chọn chạy `efficiency.py` và `analyze_results.py`), đồng thời giữ lệnh trong Phụ lục B báo cáo dùng trực tiếp `scripts/run_moe.py` / `scripts/run_xlmr.py` / `scripts/run_queue.py` (hoặc hỗ trợ chuyển tiếp `--config`/`--seed` trong `main.py`).

---

### P0.3. Nâng cấp `scripts/efficiency.py` (tránh tải dataset lặp + đo latency riêng khối MoE & sparse/dense)

* **Tệp & dòng:** `scripts/efficiency.py` (dòng 43–48, 70–138).
* **Vấn đề hiện tại:**
  1. Trong vòng lặp `for variant in args.variants:` (dòng 82–128), `load_dataset("uitnlp/ViANLI")["test"]` bị gọi lặp lại 4–5 lần cho **mỗi biến thể** (dòng 99 và các lời gọi `loader_for`).
  2. Mới đo độ trễ toàn mô hình (`gpu_bf16_ms_per_sentence`, `cpu_int8_ms_per_sentence`), **chưa tách riêng độ trễ khối MoE** và **chưa so sánh sparse dispatch vs dense dispatch** — trong khi Báo cáo mục 5.6 yêu cầu rõ *"báo cáo riêng thời gian của khối MoE"* và so sánh `B3` với `B3 + sparse dispatch`.
  3. Mặc định `--variants` chỉ có 4 biến thể `["B0", "B2", "B0_base", "B2_base"]`; nên bổ sung khả năng benchmark đủ các biến thể lightweight: `B0 B1 B2 B3 B3_r384 B4_c002 B4_c01 B0_base B2_base B5_kd`.
* **Yêu cầu chỉnh sửa:**
  1. Tải `test_split = load_dataset("uitnlp/ViANLI")["test"]` **một lần duy nhất** trước vòng lặp, truyền `test_split` vào `loader_for(test_split, tokenizer, ...)`.
  2. Bổ sung hàm `benchmark_moe_head(model, pooled_samples, device, dtype, sparse=True)` đo riêng thời gian (ms/câu) của `router + experts + layer_norm + classifier` trên `N = 256` vector câu thật (so sánh `sparse=True` vs `sparse=False` bằng cách truyền `selected_mask` vs `None` vào `model.experts`).
  3. Ghi thêm các cột vào `results/efficiency.csv`:
     * `gpu_moe_sparse_ms_per_sentence`, `gpu_moe_dense_ms_per_sentence`, `moe_time_share_%`
     * `peak_gpu_infer_mb` (dùng `torch.cuda.reset_peak_memory_stats` + `max_memory_allocated`)
     * `test_acc_bf16` (hoặc `fp32`) song song với `test_acc_int8` để tính trực tiếp `int8_acc_drop` trên cùng checkpoint seed 42!
  4. Kiểm tra cảnh báo deprecation của `torch.ao.quantization.quantize_dynamic` trên PyTorch 2.11 và đảm bảo test `test_int8_quantized_moe_runs_on_cpu` tiếp tục pass.

---

### P0.4. Hoàn thiện `src/utils/routing_diagnostics.py` và `scripts/run_moe.py`

* **Tệp & dòng:**
  * `src/utils/routing_diagnostics.py` (dòng 5–65)
  * `scripts/run_moe.py` (dòng 307–330)
* **Yêu cầu chỉnh sửa:**
  1. Trong `compute_routing_diagnostics(...)`, bổ sung trường `"empty_selection_pct": float(np.mean(experts_per_sample == 0) * 100.0)` và `"min_experts_per_sample"`, `"max_experts_per_sample"` vào dictionary `diagnostics`.
  2. Trong `scripts/run_moe.py` dòng 309–310, tạo hàm nhỏ `_model_display_name(cfg, paper_mode, ste)` phản ánh đủ biến thể:
     * Phân biệt `bottleneck(r=256)` vs `bottleneck(r=384)` vs `res_bottleneck(r=256)`.
     * Gắn `+cost{lambda}` nếu `expert_cost_weight > 0`.
     * Gắn `+KD` nếu `training.distill` có cấu hình.
     *(Chú ý: KHÔNG đổi quy tắc đặt tên `experiment_name` để không làm hỏng `run_queue.is_done()` và `viz.variant_of()`!)*

---

### P0.5. Sửa lỗi và mở rộng `src/utils/viz.py`

* **Tệp & dòng:** `src/utils/viz.py` (dòng 303–323, 448–500).
* **Yêu cầu chỉnh sửa:**
  1. **Sửa bug dòng 319:** Đổi `if variant != "B0"` thành `if variant not in ("B0", "B0_base")` để `E5a (XLM-R Base)` hiện `—` ở cột `Chuyên gia/câu` thay vì `1.00`.
  2. **Sửa `expert_parameter_table` (dòng 493–500):** Hỗ trợ danh sách `bottlenecks=(256, 384)` và thêm cột tỷ lệ so với toàn mô hình XLM-R Large (~560M) và XLM-R Base (~278M) để bảng minh bạch cả % khối MoE lẫn % toàn mô hình.
  3. **Bổ sung hàm thống kê chuẩn** vào `src/utils/viz.py`:
     * `clustered_bootstrap_test(summary, variant_a, variant_b, n_boot=10000, seed=42, project=PROJECT)`:
       * Với mỗi biến thể, lấy dự đoán trên 3 seeds theo đúng `uid` (1.000 câu).
       * Tính độ chính xác trung bình theo từng câu $s_i^{(A)} = \frac{1}{S}\sum_{k=1}^S \mathbb{1}(\hat{y}_{i,k}^{(A)} = y_i)$ và hiệu $d_i = s_i^{(B)} - s_i^{(A)}$ cho $i = 1 \dots 1000$.
       * Lấy mẫu lại (bootstrap) 1.000 chỉ số câu có hoàn lại `n_boot = 10,000` lần (bảo toàn cấu trúc phụ thuộc giữa 3 seed trên cùng một câu!).
       * Trả về: `mean_diff`, `ci_95_low`, `ci_95_high`, `p_value_two_sided` (paired bootstrap / permutation trên vector $d_i$ độc lập giữa 1.000 câu).
     * `ensemble_mcnemar(summary, variant_a, variant_b, project=PROJECT)`: Trung bình xác suất softmax qua 3 seed thành 1 dự đoán ensemble/câu, rồi chạy exact McNemar trên đúng 1.000 cặp độc lập.
     * `pooled_mcnemar(summary, variant_a, variant_b, project=PROJECT)`: Giữ để đối chiếu với bảng trong `v5_results.md`, nhưng gắn chú thích rõ là chỉ số tham khảo (gộp 3.000 cặp không độc lập).
     * `per_class_recall_table(summary, project=PROJECT)`: Tính recall trung bình ± std cho từng lớp `entailment`, `neutral`, `contradiction` và đếm số seed bị sụp lớp (tỷ lệ dự đoán lớp $< 5\%$).
     * `spearman_summary_table(summary, project=PROJECT)`: Tự động tổng hợp bảng Spearman $\rho_s(C, \text{đoán sai})$, $\rho_s(C, \text{số chuyên gia})$, $p$-value, accuracy và số chuyên gia theo 3 nhóm tam phân vị $C$ (thấp / giữa / cao), cùng tỷ lệ `empty_selection` cho tất cả biến thể có ngưỡng động.
  4. **Cập nhật `mcnemar_table`** nhận danh sách cặp mặc định đầy đủ cho các so sánh chính:
     ```python
     DEFAULT_PAIRS = (
         ("B0", "B1"), ("B1", "B2"), ("B0", "B2"),
         ("B2", "B3"), ("B2", "B3_r384"), ("B3", "B3_r384"),
         ("B2", "B4_c002"), ("B2", "B4_c01"),
         ("B0_base", "B2_base"), ("B1", "B2_base"), ("B0", "B2_base"),
         ("B2_base", "B5_kd"), ("B0", "B5_kd"),
         ("B1", "A_topk"), ("B2", "A_nobal"),
         ("B2", "B2_T005"), ("B2", "B2_T02"),
     )
     ```

---

### P0.6. Làm rõ công thức Knowledge Distillation trong `src/utils/training.py`

* **Tệp & dòng:** `src/utils/training.py` (dòng 78–79, 145–150).
* **Hiện trạng:**
  ```python
  loss = output['loss']
  if teacher is not None:
      targets = torch.stack([teacher[uid] for uid in batch['uid']]).to(device)
      kd = distillation_loss(output['logits'], targets, kd_temperature)
      loss = (1 - alpha) * loss + alpha * kd
  ```
* **Điểm cần lưu ý:** Vì `output['loss'] = task_loss + λ_d * dynamic_loss + λ_b * balance_loss`, nhân `(1 - alpha)` với `output['loss']` cũng nhân `(1 - alpha)` vào cả `dynamic_loss` và `balance_loss`.
* **Yêu cầu chỉnh sửa:**
  1. **Giữ hành vi mặc định** (`blend_aux: true`) để không sai khác với kết quả E6 đã chạy, nhưng thêm tuỳ chọn `distill.blend_aux` (mặc định `True`): nếu `False` thì tính `(1 - alpha) * task_loss + alpha * kd + (output['loss'] - task_loss)`.
  2. Ghi đúng công thức thực tế $\mathcal{L}_{\text{KD-total}} = (1 - \alpha)\mathcal{L}_{\text{NLIMoE}} + \alpha T^2 \text{KL}(\sigma(z_T/T) \Vert \sigma(z_S/T))$ vào Chương 4 báo cáo để code và công thức khớp tuyệt đối.

---

## P1 — Đồng bộ Dữ liệu, Kiểm định Thống kê & Tự động hóa Phân tích Lỗi (MT4)

### P1.1. Đồng bộ kết quả 42 lần chạy v5 từ Google Drive/Colab về repo local

* **Hiện trạng:** Local hiện chỉ có 3 dòng thử nghiệm cũ từ ngày 01/10 trong `results/summary.csv`; thư mục `logs/`, `predictions/`, `results/routing_stats/` hầu như rỗng.
* **Hành động cần kiểm tra / hướng dẫn rõ:**
  * Copy từ `/content/drive/MyDrive/ViANLI_NLIMoE/` về workspace local:
    * `results/summary.csv` (đủ 42 dòng v5 cho 14 biến thể $\times$ 3 seeds)
    * `predictions/*.csv` (42 tệp dự đoán trên tập test)
    * `results/routing_stats/*_per_sample.csv` và `*_diagnostics.json` (36 tệp của 12 biến thể MoE $\times$ 3 seeds)
    * `logs/*.jsonl` (42 tệp lịch sử huấn luyện để vẽ đường cong hội tụ & `grad_norm_wc`)
    * *(Tùy chọn, không commit vào git)* `checkpoints/*_best.pt` và `*.config.json` (ít nhất seed 42 cho mỗi biến thể để chạy `efficiency.py --int8_accuracy`).
  * Xóa hoặc tách riêng 3 dòng chạy thử ngày 01/10 (`xlmr_dense_seed42_*`) sang `results/legacy_summary.csv` để `results/summary.csv` sạch 100% các lần chạy v5 (hoặc giữ nguyên vì `viz.load_summary()` đã lọc theo `variant_of`, nhưng nên dọn sạch bảng `results/summary.md`).

### P1.2. Tạo script phân tích toàn diện `scripts/analyze_results.py`

Tạo một script chạy một lệnh:
```bash
python scripts/analyze_results.py --out_dir reports/generated
```
Script đọc toàn bộ `results/summary.csv`, `predictions/*.csv`, `results/routing_stats/*_per_sample.csv`, `logs/*.jsonl` và bộ test `uitnlp/ViANLI`, rồi tự động sinh ra:

1. **`reports/generated/main_results.csv` & `.md`:** Bảng tổng hợp 14 biến thể (E0–E6, A1–A3): Test Acc (mean ± std, từng seed), Macro-F1, Val Acc, Recall từng lớp E/N/C, số seed bị sụp lớp, Avg Experts, Router Entropy, Empty-Selection %, thời gian train/infer.
2. **`reports/generated/statistical_tests.csv` & `.md`:** Cho đủ 17 cặp so sánh quan trọng, xuất song song 4 cột:
   * `delta_acc_mean` (điểm %)
   * `wins_over_seeds` (ví dụ `3/3`)
   * `mcnemar_pooled_p` (3.000 cặp — ghi chú chỉ để tham khảo do lặp mẫu)
   * **`bootstrap_ci95` và `bootstrap_p`** (10.000 lần lấy mẫu lại theo cụm 1.000 câu test độc lập — **chỉ số chính thức cho báo cáo**)
   * **`ensemble_mcnemar_p`** (exact McNemar trên ensemble xác suất trung bình 3 seed — 1.000 cặp độc lập).
3. **`reports/generated/complexity_and_routing.csv` & `.md`:**
   * Thống kê `‖∂L/∂w_c‖` từ `logs/*.jsonl` (số lần $= 0$, trung vị, max, $C$ đầu $\to$ cuối huấn luyện) cho B1, B2 và các biến thể STE.
   * Bảng Spearman $\rho_s(C, \text{wrong})$ và $\rho_s(C, \text{n\_experts})$ cho từng seed, chia 3 tam phân vị $C$ (thấp / giữa / cao).
   * Tần suất kích hoạt 7 chuyên gia (`expert_0`..`expert_6`), `collapse_warning`, và phân bố chuyên gia theo 3 nhãn E/N/C.
4. **`reports/generated/cross_run_analysis.json` & `unanimously_wrong_36.csv`:**
   * Tính chính xác: tỷ lệ đồng thuận cặp (pairwise agreement), độ bao phủ oracle (tỷ lệ câu được $\ge 1/42$ lần chạy đoán đúng), độ chính xác bỏ phiếu đa số (42 runs) và ensemble trung bình xác suất 3 seed của từng biến thể.
   * Xuất bảng chi tiết **tất cả các câu bị 100% mô hình đoán sai** (khoảng 36 câu): `uid`, `premise`, `hypothesis`, `gold_label`, phân bố nhãn mà 42 lần chạy dự đoán (`vote_E`, `vote_N`, `vote_C`), để nhóm đọc thủ công đánh giá tỷ lệ **annotation noise** vs **suy luận siêu khó**.
5. **`reports/generated/error_annotation_sheet.csv`:**
   * Lấy mẫu phân tầng **120 câu đoán sai** của `E2 (B2, seed 42)` (40 câu mỗi nhãn gold E/N/C, có cờ đánh dấu trùng với các lỗi của `E1`, `E3b`, `E6`), kèm đầy đủ cột: `uid`, `premise`, `hypothesis`, `gold`, `pred_E2`, `pred_E1`, `pred_E6`, `complexity_C`, `threshold_rho`, `n_experts`, `selected_experts`, `lexical_overlap_jaccard`, `has_negation_cue`, `has_number_cue`, và 2 cột trống `trap_type` (trong 7 loại: `numerical`, `overgeneralization`, `temporal`, `world_knowledge`, `lexical_overlap`, `negation`, `annotation_noise`) + `notes` để người dùng gán nhãn thủ công nhanh.
   * Thêm chế độ `--annotated_errors reports/generated/error_annotation_sheet_filled.csv`: sau khi có file gán nhãn, script tự động tính bảng tần suất lỗi (Bảng 5.7 báo cáo), vẽ **heatmap `trap_type × expert`** (Hình 5.3 báo cáo), và chạy **kiểm định $\chi^2$ / Kruskal-Wallis** xem các chuyên gia có chuyên môn hóa có ý nghĩa thống kê theo kiểu bẫy hay không (trả lời định lượng cho **RQ4 / MT4**!).
6. **Sinh tự động toàn bộ hình chuẩn xuất bản (`reports/figures/*.png`, 300 DPI):**
   * `fig3_1_length_hist.png`: Histogram độ dài subword trên Train/Val/Test với đường dọc P95 = 95 và `max_length = 256`.
   * `fig5_1_main_accuracy_f1.png`: Biểu đồ so sánh Test Accuracy & Macro-F1 (kèm điểm từng seed, thanh sai số std, dấu ★ bài báo).
   * `fig5_2_grad_and_complexity.png`: (a) `‖∂L/∂w_c‖` theo bước huấn luyện (E1 vs E2, trục log); (b) Boxplot $C$ giữa đoán đúng vs đoán sai và đường accuracy/số chuyên gia theo 3 nhóm $C$.
   * `fig5_3_routing_heatmap.png`: Tần suất sử dụng 7 chuyên gia theo nhãn E/N/C (so sánh E1, E2, E4a, A2) và theo kiểu bẫy suy luận (khi có file gán nhãn).
   * `fig5_4_confusion_matrices.png`: Ma trận nhầm lẫn 3×3 của E0, E1, E2, E4a, E5b, E6 (thể hiện rõ hiện tượng sụp lớp Contradiction ở E0/E5b và sự phục hồi ở E2/E6).
   * `fig5_5_collapse_v3_v4_v5.png`: Nếu còn log v3/v4, vẽ minh họa quá trình gỡ lỗi sụp biểu diễn (`h_std`, `val_acc`).
   * `fig5_6_efficiency_tradeoff.png`: Biểu đồ Pareto (MACs / Latency GPU & CPU int8 vs Accuracy / Macro-F1) đọc từ `results/efficiency.csv`.

### P1.3. Chạy benchmark hiệu năng thật

```bash
python scripts/efficiency.py --variants B0 B1 B2 B3 B3_r384 B4_c002 B4_c01 B0_base B2_base B5_kd --int8_accuracy --out results/efficiency.csv
```
Đảm bảo có số đo thật về: tham số tổng / encoder / MoE, GMACs, `% MACs của MoE`, kích thước fp32 / bf16 / int8 (MB), độ trễ GPU bf16 (tổng + riêng MoE sparse vs dense), độ trễ CPU int8, và **Test Accuracy sau lượng tử hóa int8**.

---

## P2 — Thí nghiệm Bổ sung Bịt lỗ hổng Phản biện (Chạy trên GPU Colab)

> Các thí nghiệm này rất gọn (mỗi cấu hình 3 seeds chỉ tốn ~35–45 phút trên L4/A100), nhưng nâng tầm phản biện và tính thuyết phục của luận văn/báo cáo lên rõ rệt.

### P2.1. Ablation kiểm soát biến nhiễu `Mean Pooling + LayerNorm` (Bắt buộc khuyên làm!)

* **Lý do:** Hiện tại `E0` (`configs/xlmr_large.yaml`) dùng `pooling: cls`, không `LayerNorm`, còn `E1/E2` dùng `pooling: mean` + `LayerNorm` + 7 experts. Chưa tách được đóng góp của `Mean + LayerNorm` khỏi `MoE + STE`.
* **Thay đổi code:**
  1. Trong `src/models/xlmr_baseline.py`, thêm tham số `use_layer_norm: bool = False`. Nếu `True`, chèn `self.layer_norm = nn.LayerNorm(self.hidden_size)` trước `self.dropout` / `self.classifier`.
  2. Trong `scripts/run_xlmr.py`, truyền `use_layer_norm=cfg["model"].get("use_layer_norm", False)`.
  3. Tạo 2 config mới (kế thừa y nguyên siêu tham số v5 của `xlmr_large.yaml` / `nlimoe_dynamic.yaml`):
     * `configs/xlmr_large_mean_ln.yaml` (mã `A4` / `A_mean_ln`): XLM-R Large + `pooling: mean` + `use_layer_norm: true`, không MoE.
     * `configs/nlimoe_single_expert.yaml` (mã `A5` / `A_1exp`): `PaperNLIMoEModel` với `num_experts: 1, routing_type: dense` (1 expert FFN 1024→1024 + Mean + LayerNorm, không định tuyến).
  4. Đăng ký vào `scripts/run_queue.py` (`VARIANTS`), `src/utils/viz.py` (`VARIANT_ORDER`, `VARIANT_NAMES`, `COLORS`), và thêm test kiểm tra độ cô lập biến trong `tests/test_improvements.py`.

### P2.2. Thử nghiệm Residual Bottleneck (Kiểm chứng nguyên nhân thất bại của E3a $r=256$)

* **Lý do:** Bottleneck hiện tại (`Linear(1024,256) → ReLU → Linear(256,1024) → Dropout → ReLU`) có 2 lớp ReLU và không có skip connection $h + f(h)$ như Houlsby Adapter [13], dễ gây chết nơ-ron.
* **Thay đổi code:**
  1. Trong `src/models/paper_nlimoe.py`, thêm class `ResidualBottleneckExpert(nn.Module)`:
     ```python
     class ResidualBottleneckExpert(nn.Module):
         """Adapter-style bottleneck with residual skip: h + Dropout(W_up(ReLU(W_down(h))))."""
         def __init__(self, hidden_size, bottleneck_dim=256, dropout=0.4):
             super().__init__()
             self.down = nn.Linear(hidden_size, bottleneck_dim)
             self.act = nn.ReLU()
             self.up = nn.Linear(bottleneck_dim, hidden_size)
             self.drop = nn.Dropout(dropout)

         def forward(self, x):
             return x + self.drop(self.up(self.act(self.down(x))))
     ```
     Hỗ trợ `expert_type="residual_bottleneck"` trong `build_expert(...)` (số tham số y hệt `bottleneck` = 525.568, −49,9%).
  2. Tạo config `configs/nlimoe_ste_res_bottleneck256.yaml` (mã `E3c` / `B3_res256`), đăng ký vào `run_queue.py` & `viz.py`, viết unit test kiểm tra số tham số và forward/backward.

### P2.3. Xử lý sụp lớp `contradiction` & Chưng cất từ Ensemble mạnh (Tùy chọn nếu muốn đẩy mô hình đơn vượt 47,3%)

* **Thay đổi code:**
  1. Hỗ trợ `training.class_weights` (ví dụ `[1.0, 1.0, 1.2]` cho `[entailment, neutral, contradiction]`) trong `PaperNLIMoEModel` / `XLMRobertaNLI` (truyền `weight` vào `F.cross_entropy`). Mặc định `None`.
  2. Tạo config `configs/nlimoe_ste_cw.yaml` (mã `E7_cw` / `B2_cw`) với `class_weights: [1.0, 1.0, 1.2]`.
  3. Tạo teacher logits ensemble từ `B0 + B2` (6 checkpoints: 3 seed E0 + 3 seed E2, vốn đạt ~49%+ trên test) bằng lệnh đã có sẵn trong `scripts/dump_teacher_logits.py`:
     ```bash
     python scripts/dump_teacher_logits.py --variants B0 B2 --seeds 42 123 2026 --out results/teacher/B0_B2_large.pt
     ```
     và tạo config `configs/nlimoe_ste_base_kd_ens.yaml` (`B5_kd_ens`) + `configs/nlimoe_ste_large_kd_ens.yaml` (`B2_kd_ens`) để xem một mô hình đơn có vượt mốc 47,3% của bài báo hay không.

### P2.4. Cập nhật Colab Notebook

* Cập nhật `colab/build_notebook.py` (thêm biến thể mới vào `VARIANTS`, cập nhật cell `Efficiency` chạy đầy đủ biến thể kèm `--int8_accuracy`, thêm cell chạy `scripts/analyze_results.py` và export toàn bộ bảng/hình mới).
* Chạy `python colab/build_notebook.py` để tái sinh `colab/ViANLI_NLIMoE_colab.ipynb`.

---

## P3 — Đồng bộ Toàn bộ Tài liệu, Báo cáo Cuối kỳ & Slide

### P3.1. Cập nhật các tệp Markdown trong dự án

1. **`analysis/discrepancy_analysis.md`:**
   * Điền đầy đủ Bảng 1 (Published vs Reproduced v5 cho E0, E1, A1 Top-K, $\Delta$, SE).
   * Tích hoàn tất Checklist 2.1–2.4 dựa trên `paper_extraction.md` và quá trình v3 $\to$ v4 $\to$ v5.
   * Hoàn thành Mục 3 (Kết luận điều tra: 7 chi tiết `NOT_REPORTED`, dao động seed $\sigma \approx 2\text{–}4$ điểm, đánh giá độ tin cậy tái hiện `MEDIUM-HIGH`).
2. **`analysis/routing_analysis.md`:**
   * Đổi khung bảng từ 4 chuyên gia thành **7 chuyên gia** (`Expert 0..6`).
   * Điền số liệu thực tế từ `reports/generated/complexity_and_routing.csv` (so sánh E1, E2, E4a, E4b, A2 `A_nobal`).
   * Ghi rõ phát hiện về **độ dễ $C$** ($\rho_s(C, \text{đoán sai}) < 0$ trên cả 3 seed E2), số chuyên gia theo 3 nhóm $C$, và tỷ lệ `empty_selection`.
3. **`analysis/error_analysis.md`:**
   * Điền bảng phân loại $\ge 100$ mẫu lỗi theo 6 kiểu bẫy + 1 mục `Ambiguous / Annotation Noise` (kết nối trực tiếp với 36 câu sai tuyệt đối và các mẫu nghi ngờ nhiễu nhãn như mẫu #9, #30 trong `analysis/data_analysis.md`).
   * Điền chi tiết 6–10 mẫu lỗi điển hình (đầy đủ `Sample ID`, `Premise`, `Hypothesis`, `Gold`, `Pred`, giá trị `C`, `n_experts`, `selected_experts`, phân tích bản chất lỗi).
4. **`analysis/v5_results.md`:**
   * Bổ sung cột `Bootstrap 95% CI` và `Bootstrap p (1.000 câu)` bên cạnh McNemar gộp 3.000 cặp, giải thích rõ lý do thống kê.
   * Bổ sung bảng kết quả từ `results/efficiency.csv` (thay cho dòng ước tính tham số ở dòng 125) và kết quả các thí nghiệm P2 (nếu đã chạy).
5. **`results/summary.md` & `reports/baseline_reproduction_report.md` & `README.md`:**
   * `results/summary.md`: Tự động đồng bộ bảng 14 biến thể v5 (xóa dòng sanity check duy nhất ngày 01/10).
   * `reports/baseline_reproduction_report.md`: Cập nhật Bảng 4 bằng kết quả v5 của B0/B1/A1, sửa dòng 17 (`xlm-roberta-base` $\to$ `xlm-roberta-large`), và ghi chú rõ trạng thái hoàn tất.
   * `README.md`: Thêm bảng ánh xạ mã báo cáo (`E0–E6`, `A1–A5`) $\leftrightarrow$ mã code (`B0–B5_kd`), cập nhật lệnh chạy `scripts/analyze_results.py` và các biến thể P2.

---

### P3.2. Cập nhật Đề cương (`deliverables/De_cuong_Mau2_LightweightMoE_ViANLI.docx`)

Dùng `python-docx` cập nhật giữ nguyên định dạng Mẫu 2 của UIT:
* **Thông tin hành chính (Mục 2–3, cuối Mục 8):** Đánh dấu nổi bật `[CẦN NHÓM ĐIỀN]` rõ ràng cho mã ngành, tên nhóm, họ tên/MSHV thành viên, GVHD, lịch gặp (nếu người dùng chưa cung cấp).
* **Mục 4.6 & 5 (Tính mới):** Cập nhật thêm cơ chế phạt chi phí chuyên gia ($\lambda \cdot \text{experts}$) và làm nhẹ cấp độ backbone (`XLM-R Base + MoE + STE + KD`), phản ánh sự mở rộng hợp lý từ giả thuyết ban đầu.
* **Mục 6.1 (MT2 & MT3):**
  * Sửa MT2: bỏ giả định cứng *"C có tương quan dương với tỷ lệ đoán sai"*, thay bằng *"C có tương quan đơn điệu, nhất quán qua các seed với độ khó/độ dễ mẫu (phù hợp với dấu của $\rho = 0.1 + 0.1C$)"*.
  * Sửa MT3: ghi nhận cả hai cấp độ lightweight — cấp khối MoE (bottleneck $r=256/384$ và phạt số chuyên gia) và cấp toàn mô hình (XLM-R Base + chưng cất tri thức + lượng tử hóa int8).
* **Mục 6.3 (Phạm vi):** Mở rộng phạm vi từ *"giữ nguyên backbone XLM-R Large"* sang *"backbone chính XLM-R Large và đối sánh phiên bản nhẹ XLM-R Base"* + phần cứng thực nghiệm gồm RTX 5080 Laptop 16GB và Google Colab A100 40GB / L4 24GB.
* **Mục 7 (Bước 1–4):**
  * Cập nhật cấu hình huấn luyện sang **v5** (linear LR + 10% warmup, gradient clipping 1,0, encoder dropout 0,1) thay cho v3 (`learning rate không đổi`).
  * Sửa thứ tự lớp bottleneck ở Bước 3 khớp với `paper_nlimoe.py`: `Linear(1024, r) → ReLU → Linear(r, 1024) → Dropout → ReLU` (và bổ sung residual bottleneck nếu chạy P2.2).
  * Cập nhật Bảng 2 với đầy đủ cấu trúc E0–E6 / A1–A3.

---

### P3.3. Hoàn thiện Báo cáo Cuối kỳ (`deliverables/Bao_cao_cuoi_ky_LightweightMoE_ViANLI.docx`) — Đảm bảo $\ge 30$ trang & đúng Rubric

> **Rubric chấm điểm từ `Projects information.docx`:** Abstract ($\le 300$ từ) | Introduction (10%) | Related Work (10%) | Data (10%) | Methods (30%) | Experiments (30%) | Conclusion (5%) | Writing/Formatting (5%) | Tối thiểu **30 trang** + nộp kèm Supplementary Material (mã nguồn, video, demo).

Chỉnh sửa chi tiết từng chương (dùng script `python-docx` sinh/cập nhật sạch sẽ, **xóa sạch 100% các hộp `▸ Hướng dẫn (xoá khi nộp)`** và mọi dấu `[ ]` trừ thông tin cá nhân cần nhóm tự điền):

1. **Trang bìa, Lời cảm ơn, Tóm tắt ($\le 300$ từ), Danh mục bảng/hình/từ viết tắt:**
   * Viết lại **Tóm tắt** đúng cấu trúc 5 câu với số liệu thực tế v5: bối cảnh ViANLI (47,3%) $\to$ phát hiện lỗi gradient $= 0$ ở nhánh độ khó NLIMoE $\to$ phương pháp (tái hiện 3 seed, STE, mở rộng lightweight: bottleneck $r=256/384$, phạt chi phí chuyên gia, XLM-R Base + KD + int8) $\to$ kết quả chính (E2 đạt 45,8% Acc / 45,6% Macro-F1 cao nhất, thắng E1 cả 3/3 seed; $C$ học điểm dễ với $\rho_s < 0$ có ý nghĩa cả 3 seed; E4a giảm 55% chuyên gia không giảm Acc có ý nghĩa; E5b/E6 đạt ngang E1 với chi phí encoder giảm ~3,5× và KD tăng +3,3 F1) $\to$ ý nghĩa & công bố mã nguồn.
   * Bổ sung từ viết tắt: `KD` (Knowledge Distillation), `MACs` (Multiply-Accumulate Operations), `INT8`, `VRAM`, `CI` (Confidence Interval), `F1`.
2. **Chương 1 — Giới thiệu (~3–4 trang):**
   * Cập nhật RQ3 thành câu hỏi bao quát 2 cấp độ nhẹ (bottleneck/phạt chuyên gia ở khối MoE và thu gọn backbone + KD).
   * Cập nhật Mục 1.5 (Đóng góp) theo đúng 4 đóng góp thực tế đã đạt được (42 runs + kiểm định thống kê; chứng minh & sửa lỗi gradient bằng STE kèm phát hiện vai trò "điểm dễ" của $C$; phân tích định lượng bài toán nhẹ trên hai cấp độ khối MoE vs backbone; giải phẫu lỗi 100+ mẫu & 36 câu sai tuyệt đối).
3. **Chương 2 — Các nghiên cứu liên quan (~4–5 trang):**
   * Xác minh kích thước trong Bảng 2.1 (SNLI ~570k, MNLI ~433k, ANLI ~163k, ViNLI ~30k, ViANLI 10.012).
   * Sửa ô `[kiểm tra]` ở Bảng 2.2 cho Huang et al. [8] (kiểm tra từ paper arXiv:2403.07652: đánh giá trên LLM benchmark chuẩn, 1 lần chạy).
   * Bổ sung tiểu mục **2.6b. Chưng cất tri thức (Knowledge Distillation) và Lượng tử hóa** (trích dẫn Hinton et al., 2015; Sanh et al., 2019 DistilBERT) để nối mạch trực tiếp sang E6 và int8.
   * Trích dẫn đầy đủ mã nguồn / thư viện nền tảng theo yêu cầu bắt buộc của `Projects information.docx`: PyTorch, Hugging Face Transformers, Datasets, scikit-learn, `uitnlp/ViANLI`, `xlm-roberta-large` / `xlm-roberta-base`.
4. **Chương 3 — Dữ liệu (~4 trang):**
   * Điền chi tiết quy trình xây dựng ViANLI ở Mục 3.1 từ Section 3 của [11] (quy trình HAMLET / human-and-model-in-the-loop, nguồn báo tin tức tiếng Việt, các vòng đối kháng và bước xác minh người gán nhãn).
   * Khẳng định ánh xạ nhãn chuẩn theo `src/data/dataset.py` (`0: entailment, 1: neutral, 2: contradiction`), bỏ thẻ `[kiểm tra]`.
   * Chèn `fig3_1_length_hist.png` (Hình 3.1).
   * Thêm phân tích thống kê ngắn về **lối tắt từ vựng** (ví dụ phân bố từ phủ định *"không"* gần đều giữa 3 nhãn, cho thấy ViANLI ít lối tắt bề mặt hơn SNLI/MNLI) và ghi nhận tồn tại một tỷ lệ mẫu mơ hồ/nghi nhiễu nhãn (dẫn ví dụ từ `data_analysis.md`).
5. **Chương 4 — Phương pháp (~8–9 trang, 30% điểm):**
   * Vẽ lại **Hình 4.1** sắc nét (sơ đồ kiến trúc đầy đủ: XLM-R $\to$ Mean Pooling $\to$ Router + Complexity Gate $C$ $\to$ $\rho = 0.1 + 0.1C$ $\to$ STE Mask $\to$ Experts [Linear / Bottleneck / Residual] + Expert Cost Penalty $\to$ Sparse Dispatch $\to$ LayerNorm $\to$ Linear $\to$ E/N/C + nhánh Teacher KD).
   * Bổ sung **Ví dụ tính tay chi tiết** (có thể mang trực tiếp các phép tính minh họa từ Slide 8–10 & 18 của `script_video.md` vào thành một box/tiểu mục minh họa luồng xuôi và luồng ngược STE — rất trực quan và giúp mở rộng chiều sâu phương pháp).
   * **Sửa Mục 4.6 (Bottleneck):**
     * Sửa công thức (4.16) khớp chính xác với `paper_nlimoe.py` L76–77: `ReLU(Dropout(W_up · ReLU(W_down · h + b_1) + b_2))`.
     * Điền đầy đủ Bảng 4.1 cho `Linear (1024→1024)`, `Bottleneck r=384` (787.840/expert, 5.514.880 cho 7 experts, −24,9%), `Bottleneck r=256` (525.568/expert, 3.678.976 cho 7 experts, −49,9%), kèm cột **% tổng tham số XLM-R Large** (~1,31% $\to$ 0,99% $\to$ 0,66%).
     * Trình bày thêm **Residual Bottleneck** nếu thực hiện P2.2.
   * **Bổ sung Mục 4.6b — Phân tích giới hạn nhẹ ở cấp khối MoE & Các cải tiến Lightweight mở rộng:**
     * Chỉ rõ khối MoE sau mean pooling chỉ chiếm ~1% tổng MACs, do đó đề xuất 3 cơ chế bổ trợ: (i) **Phạt chi phí chuyên gia** $\mathcal{L}_{\text{cost}} = \lambda \cdot \frac{1}{B}\sum_{b=1}^B \sum_{i=1}^N m_{b,i}^{\text{STE}}$ (nhấn mạnh: chỉ hoạt động được nhờ STE!), (ii) **Thu gọn backbone XLM-R Base** (~3,5× ít phép tính encoder ngoài embedding hơn), và (iii) **Chưng cất tri thức từ E2 Large sang E5b Base** + **Lượng tử hóa động INT8**.
   * **Cập nhật Mục 4.8 (Bảng 4.2 — Giả định & Tiến hóa v3 $\to$ v4 $\to$ v5):**
     * Sửa Bảng 4.2 từ v3 sang **v5** (`encoder_dropout = 0.1`, `expert/head dropout = 0.4`, `linear lr + 10% warmup`, `max_grad_norm = 1.0`).
     * Viết hẳn một tiểu mục ngắn giải thích **hành trình chẩn đoán sụp biểu diễn v3 $\to$ v4 $\to$ v5** (v3 sụp về 1 nhãn ngay từ đầu; v4 đạt 0,43 ở bước 400 rồi sụp ở bước 800; v5 thêm `max_grad_norm = 1.0` ổn định hoàn toàn — và phân tích cả vai trò của việc xử lý gradient khi `empty_selection`). Đây là điểm cộng rất lớn cho phần Methods!
   * **Sửa Mục 4.9 (Bảng cấu trúc mã nguồn):**
     * Sửa đúng vai trò tệp: `src/models/paper_nlimoe.py` cài đặt toàn bộ NLIMoE v3, STE, bottleneck, expert cost; `src/models/nlimoe.py`, `router.py`, `moe_layers.py` là bản prototype thử nghiệm ban đầu; bổ sung `scripts/run_queue.py`, `scripts/dump_teacher_logits.py`, `scripts/efficiency.py`, `scripts/analyze_results.py`, `src/utils/viz.py`, và 20+ kiểm thử tự động trong `tests/`.
6. **Chương 5 — Thực nghiệm và Đánh giá (~10–12 trang, 30% điểm):**
   * **Bảng ánh xạ mã thực nghiệm** (`E0–E6`, `A1–A5` $\leftrightarrow$ tên code `B0–B5_kd`).
   * **5.1 Thiết lập (Bảng 5.1):** Cập nhật đúng cấu hình v5, nhiệt độ STE $T=0.1$ (kèm kết quả dò $T \in \{0.05, 0.1, 0.2\}$), phần cứng thực tế (Colab A100 40GB / L4 24GB + RTX 5080 Laptop 16GB).
   * **5.3 & 5.4 Kết quả chính & Tái hiện (Bảng 5.2, 5.3, Hình 5.1):** Điền toàn bộ 14 biến thể v5 (+ biến thể P2 nếu có), từng seed, so sánh với bài báo, chèn bảng kiểm định thống kê (**Clustered Bootstrap 1.000 câu + Ensemble McNemar + Pooled McNemar tham khảo**).
   * **5.5 Phân tích nhánh độ khó $C$ và STE (Bảng 5.4, Hình 5.2):** Điền đầy đủ `‖∂L/∂w_c‖` (525/525 bằng 0 vs 0/525 bằng 0), quỹ đạo $C$ đầu $\to$ cuối, bảng Spearman $\rho_s(C, \text{đoán sai})$ theo từng seed và 3 nhóm $C$. Trả lời thẳng thắn RQ2: dấu tương quan âm chứng tỏ $C$ học vai trò **điểm "dễ"** nhất quán với công thức $\rho = 0.1 + 0.1C$.
   * **5.6 Hiệu quả Lightweight & Benchmark tốc độ (Bảng 5.5, Hình 5.6):** Điền đầy đủ E3a/E3b/(E3c), E4a/E4b, E5a/E5b, E6 và toàn bộ số đo thật từ `results/efficiency.csv` (MACs, kích thước fp32/bf16/int8, latency GPU tổng & riêng khối MoE sparse vs dense, CPU int8, int8 test accuracy).
   * **5.7 Ablation (Bảng 5.6):** Top-K (A1), bỏ $L_{\text{bal}}$ (A2), nhiệt độ STE $T \in \{0.05, 0.1, 0.2\}$ (A3a/A3b), và Mean+LN / 1-expert (A4/A5 nếu chạy P2.1).
   * **5.8 & 5.9 Phân tích định tuyến & Giải phẫu lỗi (Bảng 5.7, Hình 5.3, Hình 5.4):**
     * Sử dụng chuyên gia theo nhãn E/N/C và theo 6 kiểu bẫy (MT4).
     * Hiện tượng sụp lớp `contradiction` (recall 27,9%, cơ chế cứu lớp của E2 và KD E6).
     * Phân tích độ bất ổn cross-run (60,7% agreement, 96,4% oracle, 48,7% majority vote, 49,2% E0 ensemble) và mổ xẻ **36 câu sai tuyệt đối** (tách câu siêu khó vs nghi ngờ nhiễu nhãn).
   * **5.10 Thảo luận:** Trả lời dứt khoát từng câu hỏi RQ1–RQ4 + phân tích **Threats to Validity** (cỡ tập test 1.000 câu với SE $\approx 1.6\%$, phụ thuộc mẫu khi gộp 3 seed, phạm vi 1 dataset, biến nhiễu pooling/LayerNorm, giới hạn ~1% FLOPs của MoE sau pooling).
7. **Chương 6 — Kết luận và Hướng phát triển (~1,5–2 trang):**
   * Tóm tắt đúng số liệu đã chứng minh, nêu trung thực giới hạn, và đề xuất 4 hướng phát triển cụ thể: (1) MoE dạng LoRA đặt **bên trong các tầng encoder Transformer** (nơi chiếm 99% chi phí tính toán), (2) Fine-tune trung gian trên ViNLI rồi chuyển giao sang ViANLI, (3) Xử lý mất cân bằng độ khó lớp `contradiction` & chưng cất từ ensemble 49%+, (4) Mở rộng đa ngôn ngữ sang XNLI / ANLI và backbone tiếng Việt như PhoBERT-large.
8. **Tài liệu tham khảo & Phụ lục A–D:**
   * Bổ sung trích dẫn cho Hinton et al. (2015) KD, DistilBERT / quantization, `datasets` (Lhoest et al., 2021), `scikit-learn` (Pedregosa et al., 2011), kiểm tra lại năm/hội nghị theo chuẩn IEEE thống nhất với Slide 30–31.
   * **Phụ lục A:** Thay YAML v3 (`scheduler: "constant"`) bằng YAML **v5** thực tế (`configs/nlimoe_ste.yaml` + bảng diff cho các biến thể khác).
   * **Phụ lục B:** Sửa lệnh sai:
     * Đổi `python -m pytest tests` $\to$ `python -m unittest discover -s tests -v` (`pytest` không nằm trong `requirements.txt`).
     * Đổi `python main.py --config ...` $\to$ `python scripts/run_moe.py --config configs/nlimoe_ste.yaml --routing_type dynamic --seed 42 --precision auto` và `python scripts/run_queue.py ...`.
   * **Phụ lục C:** Bảng phân công thành viên rõ ràng.
   * **Phụ lục D:** 10 mẫu kiểm thử thật từ `predictions/` và `routing_stats/` (Premise, Hypothesis, Gold, dự đoán E1/E2/E6, giá trị $C$, $\rho$, tập chuyên gia được chọn).

---

### P3.4. Đồng bộ Slide thuyết trình (`.pptx`) với `presentation/script_video.md`

* **Hiện trạng:** Cả hai tệp `Proposal Lightweight MoE cho ViANLI.pptx` và `Proposal_Lightweight_MoE_ViANLI_Redesigned.pptx` đều chỉ có **13 slides cũ** (trước v5: dùng mã B0–B3, nói "đang chạy B0/B1", Slide 9 ghi sai "bật Top-1 fallback", Slide 8 giữ giả thuyết cũ "Spearman > 0"), trong khi `presentation/script_video.md` đã được viết hoàn chỉnh cho **31 slides** với toàn bộ 42 kết quả v5!
* **Yêu cầu chỉnh sửa:**
  1. Tạo tệp slide chuẩn mới (ví dụ `presentation/Slides_Lightweight_MoE_ViANLI_v5_31slides.pptx`, bằng `python-pptx` giữ tone thiết kế sạch của bản Redesigned) gồm **đúng 31 slides khớp 1–1 với `presentation/script_video.md`**, chèn trực tiếp các hình xuất từ `reports/figures/*.png`.
  2. Rà soát & chỉnh nhỏ 3 chi tiết trong `presentation/script_video.md`:
     * **Slide 12 & 17:** Đảm bảo không nhắc "Top-1 fallback" nếu vẫn dùng `empty_selection_policy="zero"` (ghi rõ dùng quy ước $z=0$ an toàn gradient để khớp paper).
     * **Slide 25 (Kiểm định thống kê):** Cập nhật thêm kết quả **clustered bootstrap theo 1.000 câu** (tránh bị giảng viên bắt bẻ về giả định độc lập khi gộp 3.000 cặp McNemar).
     * **Slide 27 (Kế hoạch):** Cập nhật tiến độ thực tế sau khi chạy xong `efficiency.py` và phân tích lỗi MT4.
  3. Chuyển 2 file `.pptx` cũ (13 slides) vào thư mục `presentation/archive/` (hoặc ghi chú rõ phiên bản cũ) để tránh nhầm lẫn khi nộp.

---

## Danh sách Kiểm tra Nghiệm thu (Verification Checklist cho Opus 5.5)

Sau khi hoàn tất các thay đổi, bắt buộc chạy và xác nhận đạt toàn bộ các mốc sau:

- [ ] **Unit Tests:** `python -m unittest discover -s tests -v` — 100% pass (gồm cả các test mới: `test_ste_empty_selection_has_bounded_gradient`, `test_empty_selection_policy_top1`, `test_residual_bottleneck`, `test_xlmr_mean_ln`, `test_clustered_bootstrap`).
- [ ] **Syntax & CLI Sanity:**
  - `python main.py --help` và kiểm tra menu `[0-12]` sắp xếp đúng số thứ tự `1..12`.
  - `python scripts/run_queue.py --variants B0 B1 B2 B3 B3_r384 B4_c002 B4_c01 B0_base B2_base B5_kd --seeds 42 123 2026 --dry_run` không lỗi.
  - `python scripts/efficiency.py --variants B0 B2 B0_base B2_base --latency_sentences 16` chạy sạch không lỗi.
  - `python scripts/analyze_results.py --out_dir reports/generated` sinh đủ bảng và hình 300 DPI.
- [ ] **Colab Notebook:** `python colab/build_notebook.py` sinh `colab/ViANLI_NLIMoE_colab.ipynb` hợp lệ JSON.
- [ ] **Kiểm tra sạch tài liệu (`.docx`, `.md`, `.pptx`):**
  - Không còn dòng `▸ Hướng dẫn (xoá khi nộp)` trong `deliverables/Bao_cao_cuoi_ky_LightweightMoE_ViANLI.docx`.
  - Không còn số liệu để trống `[ ]`, `[ ± ]`, `[chạy]`, `[tính]`, `[kiểm tra]` trong bảng/văn bản (chỉ giữ duy nhất thông tin cá nhân/nhóm nếu chưa điền, đánh dấu rõ `[THÔNG TIN NHÓM]`).
  - Khớp 100% giữa số liệu trong `results/summary.csv` $\leftrightarrow$ `analysis/v5_results.md` $\leftrightarrow$ `Bao_cao_cuoi_ky_LightweightMoE_ViANLI.docx` $\leftrightarrow$ `presentation/script_video.md` $\leftrightarrow$ file `.pptx` 31 slides.
  - Kiểm tra tổng số trang khi xuất PDF của `Bao_cao_cuoi_ky_LightweightMoE_ViANLI.docx` đạt **$\ge 30$ trang** theo yêu cầu `Projects information.docx`.
