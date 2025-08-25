# Triton

TBU

```bash
conda create --name triton-client python=3.9 -y
conda activate triton-client
pip install tritonclient numpy onnx onnxruntime-gpu psutil gputil tqdm matplotlib

docker pull nvcr.io/nvidia/tritonserver:23.08-py3
bash run_triton_server.sh

python check_preformance.py
```