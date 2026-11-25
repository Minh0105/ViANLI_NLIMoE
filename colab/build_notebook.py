"""Generate colab/ViANLI_NLIMoE_colab.ipynb."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# ViANLI · Lightweight MoE

B0 XLM-R Large · B1 NLIMoE · B2 + STE · B3 + bottleneck · ablations A_topk, A_nobal
""")

md("## Setup")
code(r"""
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
        raise RuntimeError(f'git {args[0]}: {output}')
    return output.strip()

os.makedirs(PROJECT, exist_ok=True)
if not os.path.isdir(os.path.join(PROJECT, '.git')):
    git('init', '-q')
    git('remote', 'add', 'origin', f'https://{REPO}')
git('fetch', '-q', url, BRANCH)  # token is not stored in .git/config
git('checkout', '-q', '-f', '-B', BRANCH, 'FETCH_HEAD')
print(git('log', '-1', '--format=%h %s'))

os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
code(r"""
!pip install -q -r requirements.txt
try:
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
!python -m unittest discover -s tests 2>&1 | tail -3
""")

md("## Runs")
code(r"""
import torch
VARIANTS = ['B0_base', 'B2_base', 'B5_kd',                  # lightweight: XLM-R Base, distillation
            'B4_c002', 'B4_c01', 'B2_T005', 'B2_T02', 'B3_r384',
            'B0', 'B1', 'B2', 'B3', 'A_topk', 'A_nobal']   # finished runs are skipped
SEEDS = [42, 123, 2026]
PRECISION = 'auto'

memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
# one XLM-R Large job at batch 16 peaks at ~11 GB, so 16 x 1 fits from an L4 (22 GB) up
BATCH, ACCUM = (16, 1) if memory_gb >= 20 else (8, 2)
BUDGET_GB = memory_gb - 1   # jobs are packed by memory: Large ~12.5 GB, Base ~6.5 GB
print(f'{torch.cuda.get_device_name(0)}, {memory_gb:.0f} GB -> batch {BATCH} x {ACCUM}, '
      f'jobs packed into {BUDGET_GB:.0f} GB (A100: 3 Large; L4: 1 Large + 1 Base or 3 Base)')

QUEUE_ARGS = ['--variants', *VARIANTS, '--seeds', *map(str, SEEDS), '--precision', PRECISION,
              '--batch_size', str(BATCH), '--grad_accum', str(ACCUM), '--parallel', '3',
              '--gpu_memory_gb', f'{BUDGET_GB:.1f}']
!python scripts/run_queue.py {' '.join(QUEUE_ARGS)} --dry_run
""")
code(r"""
# B5_kd learns from the B2 (XLM-R Large) models: save their logits on the training set once
TEACHER = 'results/teacher/B2_large.pt'
if 'B5_kd' in VARIANTS and not os.path.exists(TEACHER):
    !python scripts/dump_teacher_logits.py --variants B2 --seeds 42 123 2026 --out {TEACHER}
""")
code(r"""
PID_FILE = 'logs/queue.pid'
os.makedirs('logs', exist_ok=True)

def queue_pid():
    # the pid must still be run_queue.py: after a runtime change the old number can belong to another process
    try:
        pid = int(open(PID_FILE).read())
        os.kill(pid, 0)
        with open(f'/proc/{pid}/cmdline', 'rb') as handle:
            return pid if b'run_queue' in handle.read() else None
    except Exception:
        return None

if queue_pid():
    print('queue already running, pid', queue_pid())
else:
    log = open('logs/queue.out', 'a')
    process = subprocess.Popen([sys.executable, '-u', 'scripts/run_queue.py', *QUEUE_ARGS],
                               cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    open(PID_FILE, 'w').write(str(process.pid))
    print('started, pid', process.pid)
""")
code(r"""
# stop the queue and its running jobs (run only when needed; unfinished runs restart from scratch)
import signal
pid = queue_pid()
if pid:
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    print('stopped queue', pid)
else:
    print('queue is not running')
""")

md("## Monitor")
code(r"""
from src.utils import viz
viz.monitor(refresh_seconds=60)
""")
code(r"""
import json
current = json.load(open('logs/queue_status.json')).get('current') or {}
print(current.get('variant'), current.get('seed'))
if current.get('stdout'):
    !tail -n 30 "{current['stdout']}"
""")

md("## Results")
code(r"""
import matplotlib.pyplot as plt
from IPython.display import display
summary = viz.load_summary()
table = viz.results_table(summary)
display(table)
fig_bar = viz.plot_results_bar(summary); plt.show()
display(viz.mcnemar_table(summary))
""")
code(r"""
histories = viz.all_histories()
fig_val = viz.plot_curves_compare(histories, 'accuracy', 'val'); plt.show()
fig_loss = viz.plot_curves_compare(histories, 'loss', 'train'); plt.show()
fig_experts = viz.plot_curves_compare(histories, 'avg_experts', 'train'); plt.show()
fig_grad = viz.plot_gradnorm_compare(histories); plt.show()
""")

md("## Routing")
code(r"""
routing_figs = {}
seed42 = summary[summary['seed'] == 42].set_index('variant')['experiment_name']
for variant in ['B1', 'B2', 'B3', 'A_nobal']:
    if variant in seed42 and viz.routing_frame(seed42[variant]) is not None:
        routing_figs[variant] = viz.plot_routing(seed42[variant]); plt.show()

names = [seed42[v] for v in viz.VARIANT_ORDER if v in seed42]
fig_cm, axes = plt.subplots(1, max(1, len(names)), figsize=(4.2 * max(1, len(names)), 4))
for ax, name in zip(getattr(axes, 'flat', [axes]), names):
    viz.plot_confusion(name, ax=ax)
fig_cm.tight_layout(); plt.show()
display(viz.expert_parameter_table())
""")

md("## Efficiency")
code(r"""
# parameters, MACs per sentence, size fp32 / bf16 / int8, GPU and CPU latency; --int8_accuracy adds a slow CPU test pass
import pandas as pd
from IPython.display import display
!python scripts/efficiency.py --variants B0 B2 B0_base B2_base B5_kd
display(pd.read_csv('results/efficiency.csv').round(2))
""")

md("## Export")
code(r"""
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports/tables', exist_ok=True)
figures = {'test_accuracy': fig_bar, 'val_accuracy': fig_val, 'train_loss': fig_loss,
           'experts_per_sample': fig_experts, 'grad_wc': fig_grad, 'confusion': fig_cm}
figures.update({f'routing_{v}': f for v, f in routing_figs.items()})
for name, figure in figures.items():
    figure.savefig(f'reports/figures/{name}.png', dpi=200, bbox_inches='tight')
table.to_csv('reports/tables/results.csv', index=False, encoding='utf-8-sig')
viz.mcnemar_table(summary).to_csv('reports/tables/mcnemar.csv', index=False, encoding='utf-8-sig')
summary.to_csv('reports/tables/all_runs.csv', index=False, encoding='utf-8-sig')

if not USE_DRIVE:
    import shutil
    from google.colab import files
    shutil.make_archive('/content/results', 'zip', PROJECT, '.')
    files.download('/content/results.zip')
""")

notebook = {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("ViANLI_NLIMoE_colab.ipynb")
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
