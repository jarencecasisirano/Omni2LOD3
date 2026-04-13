import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available: ",len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True
)

DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

mesh = trimesh.load(os.path.join(DATA_DIR), "ModelNet10")