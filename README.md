# ViANLI / NLIMoE

Dự án nghiên cứu Vietnamese adversarial NLI và Lightweight MoE.
Đã chỉnh baseline theo các giá trị công bố và công thức của
[arXiv:2406.17716v3](https://arxiv.org/html/2406.17716v3).
Chưa đóng băng baseline; các giả định còn lại nằm trong
[paper_extraction](analysis/paper_extraction.md).

## Cấu hình

| File | Mục đích |
|---|---|
| configs/xlmr_base.yaml | XLM-R Base, thông số huấn luyện Section 6.2 |
| configs/xlmr_large.yaml | XLM-R Large với cùng thông số huấn luyện |
| configs/nlimoe_dynamic.yaml | Kiến trúc paper, 7 expert, dynamic threshold |
| configs/nlimoe_topk.yaml | Kiến trúc paper, 5 expert, Top-4 cho ViANLI |
| configs/nlimoe_ste.yaml | **B2**: như nlimoe_dynamic, chỉ thêm STE cho ngưỡng |
| configs/nlimoe_ste_bottleneck.yaml | **B3**: như B2, chỉ đổi chuyên gia sang bottleneck 1024→256→1024 |
| configs/nlimoe_ste_nobalance.yaml | Ablation: B2 bỏ loss cân bằng |
| configs/lightweight_moe_experimental.yaml | Prototype cũ, không phải paper baseline |

Baseline dùng 7 epoch, batch 16, lr 1e-5, max_length 256, dropout 0.4,
weight decay 0, epsilon 1e-8, validation mỗi 400 optimizer step và cuối run.
Expert paper: Linear 1024→1024 → Dropout → ReLU; aggregation rồi
LayerNorm và classifier 1024→3. Ngưỡng: 0.1 + 0.1 × sigmoid(Linear(S)).

Hard mask của công thức không truyền gradient về complexity gate. Script
in rõ hạn chế này và không âm thầm dùng estimator của prototype. Head này
không được tối ưu bằng các loss đã công bố khi dùng autograd thông thường.
Chi tiết cần được xác minh với code tác giả trước khi khẳng định reproduction.

## Chạy trong PyCharm

Interpreter: `D:/Learning/NLP/ViANLI_NLIMoE/.venv/Scripts/python.exe`.
Working directory: thư mục gốc dự án. Các cấu hình có sẵn:

- `3_Sanity_Check_XLMR`: ghi nhớ 32 mẫu train, tối đa 100 epoch diagnostic.
- `4_Train_XLMR_Baseline`: XLM-R Base, 7 epoch theo cấu hình mới.
- `5_Train_NLIMoE_Dynamic`: paper equations, XLM-R Large và 7 expert.
- `6_Train_NLIMoE_TopK_Paper`: paper equations, N=5, K=4.
- `7_Sweep_Threshold`: checkpoint paper dynamic mới nhất, tune trên validation.
- `8_Train_Lightweight_Experimental`: prototype riêng.

Các file XML có thể cần được PyCharm reload để cập nhật danh sách.
NLIMoE mới dùng backbone Large nên cần tải trọng số mới và nhiều VRAM hơn.
Nếu gặp OOM, giảm batch hoặc dùng mixed precision/checkpointing là thay đổi
runtime cần ghi lại; không tự coi batch khác 16 là cấu hình paper nguyên bản.

## CLI

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe main.py --task sanity_check
.venv/Scripts/python.exe main.py --task train_xlmr
.venv/Scripts/python.exe main.py --task train_dynamic
.venv/Scripts/python.exe main.py --task train_topk
.venv/Scripts/python.exe main.py --task sweep
.venv/Scripts/python.exe main.py --task train_lightweight
```

## Cải tiến của nhóm (B2, B3)

Cài trong `src/models/paper_nlimoe.py`, chỉ bật khi config yêu cầu:

- `threshold_gradient: straight_through` (B2): lượt xuôi giữ mặt nạ cứng của bài báo;
  lượt ngược dùng đạo hàm của `sigmoid((P - rho) / T)`, nên complexity gate có gradient.
  Khi huấn luyện, mọi chuyên gia đều được tính để chuyên gia chưa được chọn vẫn nhận gradient.
- `expert_type: bottleneck` (B3): `Linear(1024,256) → ReLU → Linear(256,1024) → Dropout → ReLU`,
  525.568 tham số mỗi chuyên gia (−49,9%).

## Mixed precision, tích luỹ gradient, padding động

`--precision {fp32,bf16,fp16,auto}`, `--grad_accum N`, `--batch_size B`,
`--no_dynamic_padding` cho cả `run_xlmr.py` và `run_moe.py`. Ví dụ batch 8 × tích luỹ 2
= batch hiệu dụng 16 như bài báo. Precision và batch thực tế được ghi vào
`notes` của `results/summary.csv` và file `.config.json` cạnh checkpoint.

## Theo dõi và Google Colab

- Mỗi lần chạy ghi `logs/<experiment>.jsonl` (loss, val, số chuyên gia, C, ρ, ‖∂L/∂w_c‖, GPU).
- MoE ghi thêm `results/routing_stats/<experiment>_per_sample.csv` (C, ρ, chuyên gia từng câu).
- `scripts/run_queue.py` chạy lần lượt nhiều biến thể/seed và bỏ qua lần chạy đã xong.
- `src/utils/viz.py` vẽ biểu đồ theo dõi và phân tích; notebook `colab/ViANLI_NLIMoE_colab.ipynb` dùng các hàm này; notebook lấy code bằng `git clone`/`git pull` từ GitHub và được tạo bởi `colab/build_notebook.py`.

## Kết quả và checkpoint

Checkpoint mới có tên timestamp và file `.config.json` bên cạnh; không tự
nạp checkpoint cũ khác kiến trúc. Kết quả 44.40% của lần XLM-R trước thuộc
cấu hình cũ (5 epoch, batch 32, lr 2e-5, dropout 0.1), chưa được đổi thành
kết quả của cấu hình mới. Không cần xóa dữ liệu thực nghiệm cũ.

19 kiểm tra offline đã đạt (13 kiểm tra đối chiếu bài báo + 6 cho B2/B3, log, hàng đợi), gồm công thức router/loss, expert, checkpoint,
sparse dispatch và evaluation cadence. Chưa chạy train/test đầy đủ với
cấu hình paper mới. Báo cáo prototype trước đây chỉ có giá trị cho prototype.
