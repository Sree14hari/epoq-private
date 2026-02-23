import torch
import sys
import json

def get_gpu_info() -> dict:
    cuda_available = torch.cuda.is_available()

    info = {
        "python_version": sys.version.split(" ")[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }

    return info

if __name__ == "__main__":
    print(json.dumps(get_gpu_info()))
