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
        imagens_test = np.load(f"dataset_np/imagens_test_{angulo}.npy")
        labels_test = np.load(f"dataset_np/labels_test_{angulo}.npy")

    print("Imagens_train :",imagens_train.shape)
    print("Labels_train:",labels_train.shape)
    print("Imagens valid:",imagens_valid.shape)
    print("labels_valid:",labels_valid.shape)
    print("Imagens_test:",imagens_test.shape)
    print("Labels_test:",labels_test.shape)

    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)

    imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data()

    list = ["frontal", "Left45", "Left90", "Right90","Right45" ]

    for angulo in list:

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/ResNet34_{angulo}.keras", save_best_only=True, monitor = "val_accuracy")

        model = ResNet34()
        history = model.fit(imagens_train, labels_train, epochs = 50, validation_data= (imagens_valid, labels_valid), callbacks= [checkpoint], batch_size = 8)

        model.save(f"modelos/ResNet34_{angulo}_model.keras")

