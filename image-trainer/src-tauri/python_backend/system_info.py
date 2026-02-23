import torch
import platform
import psutil
import sys
import json
import os

def get_system_info():
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.3)
    disk = psutil.disk_usage('/')

    info = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "hardware": {
            "cpu": platform.processor(),
            "architecture": platform.machine(),
            "cores": psutil.cpu_count(logical=True),
            "cpu_usage_percent": cpu_percent,
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,

            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_used_gb": round((vm.total - vm.available) / (1024**3), 2),
            "ram_used_percent": vm.percent,

            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_used_percent": disk.percent,
        },
        "platform": {
            "os": platform.system(),
            "release": platform.release(),
            "hostname": platform.node(),
        }
    }

    print(json.dumps(info))


if __name__ == "__main__":
    get_system_info()