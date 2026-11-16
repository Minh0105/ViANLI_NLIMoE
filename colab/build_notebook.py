"""Build colab/ViANLI_NLIMoE_colab.ipynb. The notebook gets the code from GitHub.

Usage: python colab/build_notebook.py   (writes colab/ViANLI_NLIMoE_colab.ipynb)
"""
import json
import sys
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# ViANLI · Lightweight MoE — chạy thí nghiệm trên Google Colab

Notebook này chạy toàn bộ thí nghiệm của đồ án, theo dõi trực tiếp bằng biểu đồ, rồi vẽ sẵn các bảng và hình cho báo cáo.

| Mã | Biến thể | Thay đổi so với dòng trên | Câu hỏi |
|---|---|---|---|
| **B0** | XLM-R Large | không có MoE | mốc so sánh (bài báo: 45,5%) |
| **B1** | NLIMoE (bài báo) | + MoE, mặt nạ cứng | tái hiện được 47,3%? |
| **B2** | NLIMoE + STE | mặt nạ cứng → STE | ngưỡng học được có giúp? |
| **B3** | + bottleneck | chuyên gia 1024→256→1024 | nhỏ 50% có giữ độ chính xác? |
| A_topk | Ablation | top-K của bài báo (5 chuyên gia, K=4) | so với ngưỡng động |
| A_nobal | Ablation | B2 bỏ loss cân bằng | L_bal có cần không? |

**Cách dùng:** mở notebook trên Colab (File → Open notebook → GitHub → `Minh0105/ViANLI_NLIMoE`, hoặc File → Upload notebook), chọn GPU, rồi chạy lần lượt từ trên xuống.
- Bước 3 khởi động hàng đợi **chạy nền**. Bước 4 hiển thị biểu đồ cập nhật mỗi phút; bấm ■ để dừng *xem*, việc huấn luyện vẫn tiếp tục.
- **Nếu Colab bị ngắt:** chạy lại bước 1 → 3. Hàng đợi tự bỏ qua các lần chạy đã xong (đọc từ `results/summary.csv`). Lần chạy đang dở sẽ chạy lại từ đầu.
- Mọi kết quả nằm trên Google Drive, nên không mất khi Colab ngắt.
""")

md(r"""
## 1. Chuẩn bị môi trường

Code lấy từ GitHub: **https://github.com/Minh0105/ViANLI_NLIMoE**. Lần đầu notebook `git clone`, các lần sau `git pull` để nhận bản code mới nhất.
- `USE_DRIVE = True` (khuyên dùng): code và kết quả nằm trên Google Drive, nên không mất khi Colab ngắt.
- `USE_DRIVE = False`: chạy trong `/content`; kết quả **mất khi phiên Colab kết thúc**, nhớ tải `reports/` về trước.
- **Repo riêng tư (private):** tạo GitHub token (Settings → Developer settings → Fine-grained token, quyền *Contents: Read*) rồi thêm vào **Secrets** của Colab với tên `GITHUB_TOKEN`. Repo công khai thì không cần.
""")
code(r"""
# Đổi Runtime → Change runtime type → GPU trước khi chạy ô này.
import os, sys, subprocess
REPO = 'github.com/Minh0105/ViANLI_NLIMoE.git'
BRANCH = 'main'
USE_DRIVE = True

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/ViANLI_NLIMoE'
else:
    PROJECT = '/content/ViANLI_NLIMoE'

try:
    from google.colab import userdata
    token = userdata.get('GITHUB_TOKEN')
except Exception:
    token = None
url = f'https://{token}@{REPO}' if token else f'https://{REPO}'

def git(*args):
    result = subprocess.run(['git', *args], cwd=PROJECT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if token:
        output = output.replace(token, '***')
    if result.returncode:
        raise RuntimeError(f'git {args[0]} lỗi: {output}')
    return output.strip()

os.makedirs(PROJECT, exist_ok=True)
if not os.path.isdir(os.path.join(PROJECT, '.git')):
    # Lần đầu (kể cả khi thư mục đã có results/ từ trước): gắn repo vào, giữ nguyên các file kết quả.
    git('init', '-q')
    git('remote', 'add', 'origin', f'https://{REPO}')
# Token chỉ dùng cho lệnh fetch, không được lưu vào .git/config trên Drive.
git('fetch', '-q', url, BRANCH)
git('checkout', '-q', '-f', '-B', BRANCH, 'FETCH_HEAD')
print('Code ở commit:', git('log', '-1', '--format=%h · %s · %cr'))

os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
print('Thư mục làm việc:', os.getcwd())
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
code(r"""
!pip install -q -r requirements.txt
# (Tuỳ chọn) nếu tải dữ liệu bị lỗi quyền: thêm HF_TOKEN vào mục Secrets (biểu tượng chìa khoá bên trái).
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
!python -m unittest discover -s tests 2>&1 | tail -3
""")

md(r"""
## 2. Chọn thí nghiệm

- `SEEDS = [42]` cho proposal; thêm `123, 2026` cho báo cáo cuối kỳ.
- Batch được chọn theo bộ nhớ GPU: ≥ 40 GB (A100) chạy batch 16 như bài báo; GPU nhỏ hơn chạy **8 × tích luỹ 2** (batch hiệu dụng vẫn là 16).
- `precision='auto'`: bf16 trên A100/L4, fp16 trên T4.
""")
code(r"""
import torch
VARIANTS = ['B0', 'B1', 'B2', 'B3']          # thêm 'A_topk', 'A_nobal' khi cần ablation
SEEDS = [42]                                  # báo cáo cuối kỳ: [42, 123, 2026]
PRECISION = 'auto'

memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
BATCH, ACCUM = (16, 1) if memory_gb >= 39 else (8, 2)
print(f'GPU {torch.cuda.get_device_name(0)} · {memory_gb:.0f} GB → batch {BATCH} × tích luỹ {ACCUM}')

QUEUE_ARGS = ['--variants', *VARIANTS, '--seeds', *map(str, SEEDS), '--precision', PRECISION,
              '--batch_size', str(BATCH), '--grad_accum', str(ACCUM)]
!python scripts/run_queue.py {' '.join(QUEUE_ARGS)} --dry_run
""")

md("## 3. Khởi động hàng đợi (chạy nền)")
code(r"""
PID_FILE = 'logs/queue.pid'
os.makedirs('logs', exist_ok=True)

def queue_running():
    try:
        pid = int(open(PID_FILE).read())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None

if queue_running():
    print('Hàng đợi đang chạy (pid', queue_running(), ') — không khởi động thêm.')
else:
    log = open('logs/queue.out', 'a')
    process = subprocess.Popen([sys.executable, '-u', 'scripts/run_queue.py', *QUEUE_ARGS],
                               cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    open(PID_FILE, 'w').write(str(process.pid))
    print('Đã khởi động hàng đợi, pid', process.pid, '· log: logs/queue.out')
""")

md(r"""
## 4. Theo dõi trực tiếp

Bảng trên cùng là tiến độ hàng đợi. Sáu biểu đồ là của lần chạy **mới nhất**:

| Biểu đồ | Đọc thế nào |
|---|---|
| Loss huấn luyện | phải giảm dần; nếu đi ngang ở ~1,10 (= ln 3) là mô hình chưa học |
| Tập kiểm định | accuracy/macro-F1 sau mỗi 400 bước; điểm cao nhất được lưu checkpoint |
| Số chuyên gia/câu | MoE: bao nhiêu chuyên gia vượt ngưỡng; dồn về 1 hoặc 7 là dấu hiệu bất thường |
| Độ khó C và ngưỡng ρ | B1: gần như **đứng yên** (không học); B2/B3: thay đổi trong quá trình học |
| ‖∂L/∂w_c‖ | B1 **bằng 0** (hạn chế của bài báo); B2/B3 **khác 0** (cải tiến 1 hoạt động) |
| Bộ nhớ GPU | gần sát mức tối đa của GPU thì giảm batch |
""")
code(r"""
from src.utils import viz
viz.monitor(refresh_seconds=60)   # bấm ■ để dừng xem; huấn luyện vẫn chạy nền
""")
code(r"""
# Xem 30 dòng log cuối của lần chạy hiện tại (khi cần kiểm tra lỗi)
import json, glob
status = json.load(open('logs/queue_status.json'))
current = status.get('current') or {}
print('Đang chạy:', current.get('variant'), 'seed', current.get('seed'))
if current.get('stdout'):
    !tail -n 30 "{current['stdout']}"
""")

md(r"""
## 5. Kết quả tổng hợp

Chạy được ngay cả khi hàng đợi chưa xong (chỉ dùng các lần chạy đã hoàn tất).
""")
code(r"""
import matplotlib.pyplot as plt
from IPython.display import display
summary = viz.load_summary()
table = viz.results_table(summary)
display(table)
fig_bar = viz.plot_results_bar(summary); plt.show()
""")

md(r"""
### Kiểm định ghép cặp (McNemar)

Cùng một seed, hai mô hình chấm trên cùng 1.000 câu. *Chỉ trái đúng* / *chỉ phải đúng* là số câu mà chỉ một mô hình đoán đúng. p < 0,05 mới kết luận được là khác biệt thật; chênh 1–2 điểm thường **chưa đủ bằng chứng**.
""")
code(r"""
display(viz.mcnemar_table(summary))
""")

md("## 6. Quá trình huấn luyện của các biến thể (seed 42)")
code(r"""
histories = viz.all_histories()
fig_val = viz.plot_curves_compare(histories, 'accuracy', 'val'); plt.show()
fig_loss = viz.plot_curves_compare(histories, 'loss', 'train'); plt.show()
fig_experts = viz.plot_curves_compare(histories, 'avg_experts', 'train'); plt.show()
""")

md(r"""
## 7. Cải tiến 1: ngưỡng có thật sự học được không?

Đây là bằng chứng chính cho hạn chế ở slide 6 và cải tiến ở slide 9: B1 phải nằm ở đáy (gradient = 0), B2/B3 phải khác 0.
""")
code(r"""
fig_grad = viz.plot_gradnorm_compare(histories); plt.show()
""")

md(r"""
## 8. Phân tích định tuyến trên tập kiểm thử

Với mỗi mô hình MoE (seed 42):
1. **Số chuyên gia mỗi câu** — phân bố và số câu không chọn được chuyên gia nào.
2. **C theo đúng/sai** — nếu ngưỡng phản ánh độ khó thì câu đoán sai có C cao hơn (Spearman ρ_s > 0, p < 0,05). Đây là tiêu chí thành công của cải tiến 1.
3. **Chuyên gia theo nhãn** — chuyên gia nào hay được dùng cho entailment / neutral / contradiction.
""")
code(r"""
routing_figs = {}
seed42 = summary[summary['seed'] == 42].set_index('variant')['experiment_name']
for variant in ['B1', 'B2', 'B3', 'A_nobal']:
    if variant in seed42 and viz.routing_frame(seed42[variant]) is not None:
        routing_figs[variant] = viz.plot_routing(seed42[variant]); plt.show()
""")

md("## 9. Ma trận nhầm lẫn (seed 42)")
code(r"""
names = [seed42[v] for v in viz.VARIANT_ORDER if v in seed42]
fig_cm, axes = plt.subplots(1, max(1, len(names)), figsize=(4.2 * max(1, len(names)), 4))
for ax, name in zip(getattr(axes, 'flat', [axes]), names):
    viz.plot_confusion(name, ax=ax)
fig_cm.tight_layout(); plt.show()
""")

md("## 10. Số tham số khối chuyên gia (cải tiến 2)")
code(r"""
display(viz.expert_parameter_table())
""")

md(r"""
## 11. Xuất hình và bảng cho báo cáo

Lưu vào `reports/figures/` và `reports/tables/` trên Drive. Tên file khớp với các vị trí `[CHÈN HÌNH]` / Bảng 5.x trong khung báo cáo.
""")
code(r"""
os.makedirs('reports/figures', exist_ok=True); os.makedirs('reports/tables', exist_ok=True)
figures = {'hinh_5_1_test_accuracy': fig_bar, 'hinh_5_2_val_accuracy_theo_buoc': fig_val,
           'hinh_5_3_gradient_wc': fig_grad, 'hinh_5_x_loss': fig_loss, 'hinh_5_x_so_chuyen_gia': fig_experts,
           'hinh_5_x_confusion': fig_cm}
figures.update({f'hinh_5_4_routing_{v}': f for v, f in routing_figs.items()})
for name, figure in figures.items():
    figure.savefig(f'reports/figures/{name}.png', dpi=200, bbox_inches='tight')
table.to_csv('reports/tables/bang_5_ket_qua_chinh.csv', index=False, encoding='utf-8-sig')
viz.mcnemar_table(summary).to_csv('reports/tables/bang_5_mcnemar.csv', index=False, encoding='utf-8-sig')
summary.to_csv('reports/tables/tat_ca_lan_chay.csv', index=False, encoding='utf-8-sig')
print('Đã lưu', len(figures), 'hình và 3 bảng vào reports/')
print(table.to_string(index=False))

if not USE_DRIVE:   # chạy trong /content thì tải kết quả về máy trước khi phiên kết thúc
    import shutil
    from google.colab import files
    shutil.make_archive('/content/ket_qua_ViANLI', 'zip', PROJECT, '.')
    files.download('/content/ket_qua_ViANLI.zip')
""")

md(r"""
## Ghi chú

- **Số cho slide 12 / đề cương:** lấy cột *Test acc* của B0, B1 (và B2 nếu có) ở bước 5.
- **Phần cứng:** nhớ sửa dòng phần cứng trên slide 12 thành GPU Colab bạn đã dùng.
- **Dừng hàng đợi hẳn:** chạy `!kill $(cat logs/queue.pid)`.
- **Checkpoint** mỗi lần chạy khoảng 1,1–2,2 GB; xoá các file không cần trong `checkpoints/` nếu Drive gần đầy.
""")

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                          "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                          "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("ViANLI_NLIMoE_colab.ipynb")
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
