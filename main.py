import os
import sys
import subprocess
import argparse

# Thiết lập UTF-8 chuẩn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

TASKS = {
    "1": ("Kiểm tra Phần cứng & Môi trường GPU", ["scripts/check_env.py"]),
    "2": ("Khảo sát Dữ liệu ViANLI (Phase C)", ["scripts/check_vianli.py"]),
    "3": ("Chạy Sanity Check XLM-R (32 mẫu)", ["scripts/run_xlmr.py", "--overfit_sanity_check"]),
    "4": ("Huấn luyện XLM-R Dense Baseline (Seed 42)", ["scripts/run_xlmr.py", "--config", "configs/xlmr_base.yaml", "--seed", "42"]),
    "5": ("Huấn luyện NLIMoE Dynamic Routing (Seed 42)", ["scripts/run_moe.py", "--config", "configs/nlimoe_dynamic.yaml", "--routing_type", "dynamic", "--seed", "42"]),
    "6": ("Huấn luyện NLIMoE Top-K bài báo (5 expert, K=4)", ["scripts/run_moe.py", "--config", "configs/nlimoe_topk.yaml", "--seed", "42"]),
    "7": ("Dò quét Static Threshold (checkpoint paper mới nhất)", ["scripts/sweep_threshold.py", "--checkpoint", "latest"]),
    "8": ("Huấn luyện Lightweight MoE thử nghiệm (không phải paper baseline)", ["scripts/run_moe.py", "--config", "configs/lightweight_moe_experimental.yaml", "--seed", "42"]),
    "9": ("B0 · Huấn luyện XLM-R Large (Seed 42, bf16, 8x2)", ["scripts/run_xlmr.py", "--config", "configs/xlmr_large.yaml", "--seed", "42", "--precision", "auto", "--batch_size", "8", "--grad_accum", "2"]),
    "10": ("B2 · NLIMoE + STE (Seed 42, bf16, 8x2)", ["scripts/run_moe.py", "--config", "configs/nlimoe_ste.yaml", "--routing_type", "dynamic", "--seed", "42", "--precision", "auto", "--batch_size", "8", "--grad_accum", "2"]),
    "11": ("B3 · NLIMoE + STE + bottleneck (Seed 42, bf16, 8x2)", ["scripts/run_moe.py", "--config", "configs/nlimoe_ste_bottleneck.yaml", "--routing_type", "dynamic", "--seed", "42", "--precision", "auto", "--batch_size", "8", "--grad_accum", "2"]),
    "12": ("Hàng đợi B0 → B3, seed 42 (bỏ qua lần chạy đã xong)", ["scripts/run_queue.py", "--variants", "B0", "B1", "B2", "B3", "--seeds", "42", "--precision", "auto", "--batch_size", "8", "--grad_accum", "2"])
}

def run_task(task_key: str):
    if task_key not in TASKS:
        print(f"[Lỗi] Không tìm thấy tác vụ tương ứng với lựa chọn: {task_key}")
        return
    desc, cmd_args = TASKS[task_key]
    print("\n" + "=" * 80)
    print(f" Đang thực thi: {desc}")
    print(f" Lệnh: {PYTHON_EXE} {' '.join(cmd_args)}")
    print("=" * 80 + "\n")
    
    full_cmd = [PYTHON_EXE] + [os.path.join(PROJECT_DIR, arg) if arg.endswith(".py") else arg for arg in cmd_args]
    try:
        subprocess.run(full_cmd, cwd=PROJECT_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Lỗi] Tiến trình kết thúc với mã lỗi: {e.returncode}")
    except KeyboardInterrupt:
        print("\n[Hủy] Người dùng đã dừng tiến trình.")

def interactive_menu():
    while True:
        print("\n" + "=" * 80)
        print(" ViANLI NLIMoE - Research Reproduction Pipeline")
        print(" Thư mục dự án: " + PROJECT_DIR)
        print(" Python runtime: " + PYTHON_EXE)
        print("=" * 80)
        for k, (desc, _) in sorted(TASKS.items()):
            print(f"  [{k}] {desc}")
        print("  [0] Thoát")
        print("=" * 80)
        choice = input("Nhập số tương ứng [0-8]: ").strip()
        if choice == "0":
            print("Tạm biệt!")
            break
        elif choice in TASKS:
            run_task(choice)
            input("\nNhấn Enter để quay lại menu chính...")
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại.")

def main():
    parser = argparse.ArgumentParser(description="ViANLI NLIMoE Reproduction Master Entrypoint")
    parser.add_argument("--task", type=str, choices=list(TASKS.keys()) + ["check_env", "check_data", "sanity_check", "train_xlmr", "train_dynamic", "train_topk", "sweep", "train_lightweight", "train_xlmr_large", "train_ste", "train_bottleneck", "queue"], help="Tên hoặc mã số tác vụ cần chạy")
    args = parser.parse_args()

    alias_map = {
        "check_env": "1",
        "check_data": "2",
        "sanity_check": "3",
        "train_xlmr": "4",
        "train_dynamic": "5",
        "train_topk": "6",
        "sweep": "7",
        "train_lightweight": "8",
        "train_xlmr_large": "9",
        "train_ste": "10",
        "train_bottleneck": "11",
        "queue": "12"
    }

    if args.task:
        task_id = alias_map.get(args.task, args.task)
        run_task(task_id)
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
