import os
import sys
import platform
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def get_env_info():
    info = {}
    info["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    info["python"] = sys.version.replace("\n", " ")
    
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
        else:
            info["cuda_version"] = "N/A"
            info["gpu_name"] = "None (CPU only)"
            info["gpu_memory_gb"] = 0
    except ImportError:
        info["torch"] = "Not installed"
        info["cuda_available"] = False
        info["cuda_version"] = "N/A"
        info["gpu_name"] = "N/A"

    try:
        smi_out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True)
        info["hardware_gpu"] = smi_out.strip()
    except Exception:
        info["hardware_gpu"] = "nvidia-smi not available"

    for pkg in ["transformers", "datasets", "sklearn", "pandas", "numpy"]:
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "Installed")
        except ImportError:
            info[pkg] = "Not installed"

    return info

def write_environment_md(info, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = f"""# Thông tin Môi trường Thực nghiệm (Environment)

*Cập nhật tự động bởi `scripts/check_env.py`*

- **Hệ điều hành**: `{info['os']}`
- **Python**: `{info['python']}`
- **Phần cứng GPU (Hệ thống)**: `{info['hardware_gpu']}`
- **PyTorch**: `{info['torch']}`
- **PyTorch CUDA Available**: `{info['cuda_available']}`
- **CUDA Version (Torch)**: `{info['cuda_version']}`
- **Tên GPU (Torch nhận diện)**: `{info['gpu_name']}`
- **VRAM khả dụng**: `{info.get('gpu_memory_gb', 0)} GB`

### Phiên bản các thư viện chính:
- `transformers`: `{info['transformers']}`
- `datasets`: `{info['datasets']}`
- `scikit-learn`: `{info['sklearn']}`
- `pandas`: `{info['pandas']}`
- `numpy`: `{info['numpy']}`
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[CheckEnv] Environment report written to {output_path}")

if __name__ == "__main__":
    env_info = get_env_info()
    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "environment.md")
    write_environment_md(env_info, out_file)
    print("=== Environment Summary ===")
    for k, v in env_info.items():
        print(f"  {k}: {v}")
