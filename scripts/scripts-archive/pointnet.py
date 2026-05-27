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

DATA_DIR = "C:\\Users\\Sky Torneros\\Omni2LOD3\\PointNet\\ModelNet10\\ModelNet10"

mesh = trimesh.load("C:\\Users\\Sky Torneros\\Omni2LOD3\\PointNet\\ModelNet10\\ModelNet10\\chair\\train\\chair_0001.off")


points = mesh.sample(2048)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()

def parse_dataset(num_points=2048):
    
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = []
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        
        class_map[i] = folder.split("/")[-1]

        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)
        
        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points), 
        np.array(train_labels), 
        np.array(test_points), 
        np.array(test_labels), 
        class_map
    )


NUM_POINTS = 2048
NUM_CLASSES = 10

train_points, train_labels, test_points, test_labels, class_map = parse_dataset(NUM_POINTS)

print("Train points shape: ", train_points.shape)
print("Train labels shape: ", train_labels.shape)
print("Test points shape: ", test_points.shape)
print("Test labels shape: ", test_labels.shape)
print("Class map: ", class_map)