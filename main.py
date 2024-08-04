import pandas
import numpy as np
import tensorflow as tf
import os
from src.models import ResNet34

def load_data():

    list = ["frontal", "Left45", "Left90", "Right90","Right45" ]

    for angulo in list:
        imagens_train = np.load(f"dataset_np/imagens_train_{angulo}.npy")
        labels_train = np.load(f"dataset_np/labels_train_{angulo}.npy")
        imagens_valid = np.load(f"dataset_np/imagens_valid_{angulo}.npy")
        labels_valid = np.load(f"dataset_np/labels_valid_{angulo}.npy")

    print("Imagens_train :",imagens_train.shape)
    print("Labels_train:",labels_train.shape)
    print("Imagens valid:",imagens_valid.shape)
    print("labels_valid:",labels_valid.shape)

    return imagens_train, labels_train, imagens_valid, labels_valid

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)

    imagens_train, labels_train, imagens_valid, labels_valid = load_data()

    data_train = tf.data.Dataset.from_tensor_slices((imagens_train, labels_train))
    data_train = data_train.shuffle(buffer_size=len(labels_train)).batch(2).prefetch(tf.data.experimental.AUTOTUNE)

    data_valid = tf.data.Dataset.from_tensor_slices((imagens_valid, labels_valid))
    data_valid = data_valid.shuffle(buffer_size=len(labels_valid)).batch(2).prefetch(tf.data.experimental.AUTOTUNE)

    list = ["frontal", "Left45", "Left90", "Right90","Right45" ]

    for angulo in list:

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/ResNet34_{angulo}.keras", save_best_only=True, monitor = "val_accuracy")

        model = ResNet34()
        history = model.fit(data_train, epochs = 50, validation_data= data_valid, callbacks= [checkpoint])

        model.save(f"modelos/ResNet34_{angulo}_model.keras")

