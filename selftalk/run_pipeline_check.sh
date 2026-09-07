#!/usr/bin/env bash
# End-to-end pipeline check on synthetic data: stages 1 -> 7 chained through
# their checkpoints (no OpenFace / FLAME / cameras needed).
#
# Usage:
#     bash run_pipeline_check.sh              # CPU, small iteration counts
#     DEVICE=cuda bash run_pipeline_check.sh  # GPU
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

python -c "import torch" 2>/dev/null || {
    echo "error: torch not importable — run 'bash setup_env.sh' or 'conda activate selftalk' first" >&2
    exit 1
}

DEVICE=${DEVICE:-cpu}
OUT=${OUT:-runs/selfcheck}
ITERS=${ITERS:-50}
DEFORM_ITERS=${DEFORM_ITERS:-10}

echo "[pipeline-check] device=${DEVICE} out=${OUT}"

# Stage 1: canonical Gaussians (synthetic scene).
python -m selftalk.training.train_canonical --selfcheck \
    --iterations "${ITERS}" --num_gaussians 128 --device "${DEVICE}" \
    --out "${OUT}/canonical"

# Stage 2: Audio-to-AU (synthetic sequences; 64-dim pseudo-features).
python -m selftalk.training.train_audio2au --selfcheck --epochs 2 \
    --device "${DEVICE}" --out "${OUT}/audio2au"

# Stages 3-7: deformation (motion-token variant) on a synthetic binary dir
# written to ${OUT}/deform/selfcheck_binary.
python -m selftalk.training.train_deform --selfcheck \
    --canonical_ckpt "${OUT}/canonical/canonical.pt" \
    --audio2au_ckpt "${OUT}/audio2au/audio2au.pt" \
    --variant motion_token --iterations "${DEFORM_ITERS}" \
    --device "${DEVICE}" --out "${OUT}/deform"

# Stage 7: AU editing inference on the selfcheck models.
python -m selftalk.training.au_editing \
    --ckpt "${OUT}/deform/deform_motion_token.pt" \
    --canonical_ckpt "${OUT}/canonical/canonical.pt" \
    --binary "${OUT}/deform/selfcheck_binary" \
    --au AU26 --scale 1.5 --device "${DEVICE}"

echo "[pipeline-check] OK (stages 1-7 ran end-to-end)"
