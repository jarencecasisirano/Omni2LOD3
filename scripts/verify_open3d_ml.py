import open3d.ml.torch as ml3d
import torch
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    print("Open3D-ML torch module imported successfully.")
except ImportError as e:
    print(f"Failed to import Open3D-ML: {e}")
