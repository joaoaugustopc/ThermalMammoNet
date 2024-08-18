import pandas
import numpy as np
import tensorflow as tf
import os
from src.models import ResNet34
from src.models import VGG_16
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def plot_visualization(history, model_name, angulo, i):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy for {model_name} - {angulo} - Iteration {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"modelos/graficos/{model_name}_{angulo}_accuracy_plot_{i+1}.png")
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss for {model_name} - {angulo} - Iteration {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"modelos/graficos/{model_name}_{angulo}_loss_plot_{i+1}.png")
    plt.close()
    

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
    models = [ResNet34, VGG_16]

    for angulo in list:

        for model in models:

            model_name = model.__name__

            for i in range(10):

                checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/{model_name}_{angulo}_{i}.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto', period=1)
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

                model = model()

                history = model.fit(imagens_train, labels_train, epochs = 100, validation_data= (imagens_valid, labels_valid), 
                                    callbacks= [checkpoint,earlystop], batch_size = 6, verbose = 1, shuffle = True)

                plot_visualization(history, model_name, angulo, i)