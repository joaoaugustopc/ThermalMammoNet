import pandas
import tensorflow as tf
import os

gpu_list = tf.config.list_physical_devices('GPU')

if not os.path.exists("output"):
    os.makedirs("output")

with open("output/GPUs.txt", "w") as f:
    f.write("\n".join([str(gpu) for gpu in gpu_list]))