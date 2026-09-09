# Self-Awareness:Face Animaton Drived by Self-Awareness Audio with 3D Gaussian Splatting 

## Installation
We implemented & tested **Self-Awareness** with NVIDIA RTX 5090 

Run the below codes for the environment setting. ( details are in requirements.txt )
```bash
conda create -n GaussianTalker5090 python=3.10.19 -y
conda activate GaussianTalker5090

python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130

cd /data/Projects/self-awarenessTalkingFace/GaussianTalker1
python -m pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST="12.0"

python -m pip install --no-build-isolation \
  -e submodules/custom-bg-depth-diff-gaussian-rasterization

python -m pip install --no-build-isolation \
  -e submodules/simple-knn

cd selftalk

python -m src.training.train_audio2au \
  --data ../GaussianTalker1/data/May \
  --device cuda \
  --num_workers 4
```