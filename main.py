import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from src.models.resNet_34 import ResNet34
from src.models import googleLenet
from src.models.Vgg_16 import VGG_16
from src.models.googleLenet import googleLenet
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time 
import shutil
from src.models.alexNet import alexnet
    

#Arrays Numpy
def load_data(angulo):

    imagens_train = np.load(f"dataset_np/imagens_train_{angulo}.npy")
    labels_train = np.load(f"dataset_np/labels_train_{angulo}.npy")
    imagens_valid = np.load(f"dataset_np/imagens_valid_{angulo}.npy")
    labels_valid = np.load(f"dataset_np/labels_valid_{angulo}.npy")
    imagens_test = np.load(f"dataset_np/imagens_test_{angulo}.npy")
    labels_test = np.load(f"dataset_np/labels_test_{angulo}.npy")

    """
    print("Imagens_train :",imagens_train.shape)
    print("Labels_train:",labels_train.shape)
    print("Imagens valid:",imagens_valid.shape)
    print("labels_valid:",labels_valid.shape)
    print("Imagens_test:",imagens_test.shape)
    print("Labels_test:",labels_test.shape)
    """

    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test


def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalizar a imagem
    return image, label

# text_dataset_from_directory, retorna um tf.Dataset
def load_tf_data(path):
    train_path = os.path.join(path, "train")

    train_ds = keras.utils.text_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        shuffle=True,
        seed=123,
        batch_size = None
    )

    test_path = os.path.join(path, "test")

    val_ds, test_ds = keras.utils.text_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        shuffle=True,
        seed=123,
        validation_split=0.5,
        subset="both",
        batch_size = None
    )

    return train_ds, val_ds, test_ds

#nao esta sendo usada
def data_generator(imagens, labels):
    for img, label in zip(imagens, labels):
        img = np.expand_dims(img, axis = -1)
        yield img, label

#nao esta sendo usada
def crate_dataset(imagens, labels, batch_size = 2, image_size = (480, 640)):

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(imagens, labels), 
            output_signature=( tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32), 
            tf.TensorSpec(shape=(), dtype=tf.int32))
    )

    dataset = dataset.map(lambda x, y: preprocess_image(x, y))

    dataset = dataset.batch(batch_size)

    return dataset

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Pasta {folder_path} deletada.")
    else:
        print(f"Pasta {folder_path} não encontrada.")

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Arquivo {file_path} deletado.")
    else:
        print(f"Arquivo {file_path} não encontrado.")
    

if __name__ == "__main__":

    """ 
    for i in range(3):
        delete_file(f"VGG_16_frontal_{i}_time.txt")

    delete_folder("modelos")
    """

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    list = ["frontal", "Left45", "Left90", "Right90","Right45" ]
    models = [alexnet]

    for angulo in list:

        print(f"ANGULO: {angulo}")

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        
        imagens_train = np.expand_dims(imagens_train, axis = -1)
        imagens_valid = np.expand_dims(imagens_valid, axis = -1)
        imagens_test = np.expand_dims(imagens_test, axis = -1)

        imagens_train = tf.image.resize(imagens_train, (227, 227))
        imagens_valid = tf.image.resize(imagens_valid, (227, 227))
        imagens_test = tf.image.resize(imagens_test, (227, 227))
        


        for model_func in models:

            model_name = model_func.__name__

            for i in range(10):

                start_time = time.time()

                checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/{model_name}_{angulo}_{i}.h5", monitor='loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False)
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=50, verbose=1, mode='auto')

                model = model_func()

                history = model.fit(imagens_train, labels_train, epochs = 100, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint, earlystop], batch_size = 1, verbose = 1, shuffle = True)
                
                end_time = time.time()

                with open(f"{model_name}_{angulo}_{i}_time.txt", "w") as f:
                    f.write(f"Modelo: {model_name}\n")
                    f.write(f"Tempo de execução: {end_time - start_time}\n")
                    f.write(f"Loss: {history.history['loss']}\n")
                    f.write(f"Val_loss: {history.history['val_loss']}\n")
                    f.write(f"Accuracy: {history.history['accuracy']}\n")
                    f.write(f"Val_accuracy: {history.history['val_accuracy']}\n")
                    f.write("\n")
