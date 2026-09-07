#!/usr/bin/env bash
# Create/refresh the `selftalk` conda environment and install all dependencies.
#
# Usage:
#     bash setup_env.sh                       # CUDA torch (cu130 wheels, RTX 5090 OK)
#     TORCH_INDEX=cpu bash setup_env.sh       # CPU-only torch
#     INSTALL_GSPLAT=1 bash setup_env.sh      # also build the gsplat rasterizer
#
# pip mirror: export PIP_INDEX_URL=... before calling if the default PyPI is
# slow; conda already honors ~/.condarc mirrors.
set -euo pipefail

ENV_NAME=${ENV_NAME:-selftalk}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
INSTALL_GSPLAT=${INSTALL_GSPLAT:-0}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found on PATH" >&2
    exit 1
fi
# conda's shell hooks are not `set -u` clean; source/activate with it disabled.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"

# An empty env shell (created by `conda create -n selftalk` without python)
# has no bin/python; only (re)create when the interpreter is missing.
ENV_PREFIX="$(conda info --base)/envs/${ENV_NAME}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"
set -u

python -m pip install --upgrade pip
# torch first (CUDA wheels from the pytorch index), then the rest.
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"
python -m pip install -r "${HERE}/requirements.txt"

if [[ "${INSTALL_GSPLAT}" == "1" ]]; then
    python -m pip install ninja
    python -m pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 \
        --no-build-isolation
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "[setup] running CPU-only unit tests..."
cd "${HERE}"
python -m pytest -q
echo "[setup] done. Next: bash run_pipeline_check.sh"
