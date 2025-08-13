conda create --name triton-client python=3.9 -y
conda activate triton-client
pip install tritonclient[http] numpy
pip install onnx onnxruntime-gpu