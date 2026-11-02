#!/usr/bin/env bash
set -euo pipefail

env_name="${1:-s2enet-rs}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v conda >/dev/null 2>&1 || { printf 'Conda was not found. Install Miniconda/Miniforge first.\n' >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "$env_name"; then
  printf 'Conda environment already exists: %s\nChoose another name or install requirements manually.\n' "$env_name" >&2
  exit 1
fi

conda create -y -n "$env_name" python=3.10 pip
conda activate "$env_name"
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$repo_root/requirements_server.txt"
python -c 'from torch.hub import load_state_dict_from_url; load_state_dict_from_url("https://download.pytorch.org/models/resnet18-5c106cde.pth", map_location="cpu"); print("ResNet18 ImageNet checkpoint cached")'
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
