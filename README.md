# Triton

TBU

```bash
conda create --name triton-client python=3.9 -y
conda activate triton-client
pip install tritonclient[grpc] numpy onnx onnxruntime-gpu
```