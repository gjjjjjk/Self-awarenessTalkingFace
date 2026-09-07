#!/usr/bin/env bash
# Stage 0 for one subject: extract wav -> AU extraction -> selftalk binary.
#
# Usage (from selftalk/):
#     bash preprocess_subject.sh Jae-in                        # OpenFace backend
#     AU_BACKEND=pyfeat bash preprocess_subject.sh Jae-in      # py-feat backend (pip)
#     OPENFACE_BIN=/opt/OpenFace/build/bin/FeatureExtraction bash preprocess_subject.sh Jae-in
#
# AU_BACKEND=openface requires the OpenFace FeatureExtraction binary.
# AU_BACKEND=pyfeat requires `pip install py-feat` (models download on first run).
# ffmpeg is used when available (system, else the imageio-ffmpeg wheel's binary);
# wav2vec2 features need `transformers + librosa` and network access on first run.
set -euo pipefail

SUBJECT=${1:?usage: bash preprocess_subject.sh <subject>}
AU_BACKEND=${AU_BACKEND:-openface}
VIDEO=${VIDEO:-../dataset/raw/videos/${SUBJECT}.mp4}
OUT=${OUT:-../dataset/binary/${SUBJECT}}
OPENFACE_BIN=${OPENFACE_BIN:-FeatureExtraction}

if [[ ! -f "${VIDEO}" ]]; then
    echo "error: video not found: ${VIDEO}" >&2
    exit 1
fi

python -c "import torch" 2>/dev/null || {
    echo "error: torch not importable — run 'bash setup_env.sh' or 'conda activate selftalk' first" >&2
    exit 1
}

# 1) mono 16 kHz wav for wav2vec2 (fall back to the video track if no ffmpeg).
FFMPEG_BIN=$(command -v ffmpeg || true)
if [[ -z "${FFMPEG_BIN}" ]]; then
    FFMPEG_BIN=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" \
        2>/dev/null || true)
fi
AUDIO_ARG="${VIDEO}"
if [[ -n "${FFMPEG_BIN}" ]]; then
    mkdir -p "${OUT}"
    "${FFMPEG_BIN}" -y -loglevel error -i "${VIDEO}" -vn -ac 1 -ar 16000 \
        "${OUT}/audio16k.wav"
    AUDIO_ARG="${OUT}/audio16k.wav"
fi

if [[ "${AU_BACKEND}" == "pyfeat" ]]; then
    # 2a) frames + wav2vec2 features first (no AU yet).
    python -m selftalk.data.preprocess \
        --video "${VIDEO}" \
        --audio "${AUDIO_ARG}" \
        --out "${OUT}"
    # 2b) AU/pose/landmarks/blink from py-feat over the extracted frames.
    python -m selftalk.data.au_pyfeat \
        --frames_dir "${OUT}/frames" \
        --out_dir "${OUT}"
else
    if ! command -v "${OPENFACE_BIN}" >/dev/null 2>&1; then
        echo "error: OpenFace '${OPENFACE_BIN}' not on PATH; set OPENFACE_BIN=<path>/FeatureExtraction" >&2
        echo "   ...or switch backend: AU_BACKEND=pyfeat bash $0 ${SUBJECT}" >&2
        exit 1
    fi
    # 2) OpenFace on the video (one CSV row per decoded frame).
    mkdir -p "${OUT}/openface"
    "${OPENFACE_BIN}" -f "${VIDEO}" -out_dir "${OUT}/openface" \
        -of "${SUBJECT}" -q
    CSV="${OUT}/openface/${SUBJECT}.csv"
    if [[ ! -f "${CSV}" ]]; then
        echo "error: OpenFace did not produce ${CSV}" >&2
        exit 1
    fi

    # 3) selftalk Stage 0: frames + au/pose/landmarks/blink + wav2vec2 features.
    python -m selftalk.data.preprocess \
        --video "${VIDEO}" \
        --audio "${AUDIO_ARG}" \
        --au_csv "${CSV}" \
        --out "${OUT}"
fi

echo "[stage0:${SUBJECT}] done -> ${OUT}"
