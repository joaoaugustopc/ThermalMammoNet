import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from src.models.resNet_34 import ResNet34
from src.models import googleLenet
from src.models.Vgg_16 import VGG_16
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time 
    

#Arrays Numpy
def load_data(angulo):

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


def data_generator(imagens, labels):
    for img, label in zip(imagens, labels):
        img = np.expand_dims(img, axis = -1)
        yield img, label


def crate_dataset(imagens, labels, batch_size = 2, image_size = (480, 640)):

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(imagens, labels), 
            output_signature=( tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32), 
            tf.TensorSpec(shape=(), dtype=tf.int32))
    )

    dataset = dataset.map(lambda x, y: preprocess_image(x, y))

    dataset = dataset.batch(batch_size)

    return dataset

    

if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    list = ["frontal", "Left45", "Left90", "Right90","Right45" ]
    models = [VGG_16]

    for angulo in list:

        #train_ds, val_ds, test_ds = load_tf_data(f"dataset/{angulo}")
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)

        data_train = crate_dataset(imagens_train, labels_train)
        data_valid = crate_dataset(imagens_valid, labels_valid)
        data_test = crate_dataset(imagens_test, labels_test)

        print("Train dataset: ", type(data_train))
        print("Validation dataset: ", type(data_valid))
        print("Test dataset: ", type(data_test))

        """
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

        print("Train dataset: ", type(train_ds))
        print("Validation dataset: ", type(val_ds))
        print("Test dataset: ", type(test_ds))
        """

        for model in models:

            model_name = model.__name__

            for i in range(10):

                start_time = time.time()

                checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/{model_name}_{angulo}_{i}.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto', period=1)
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

                model = model()

                history = model.fit(data_train, epochs = 100, validation_data= data_valid, 
                                    callbacks= [checkpoint,earlystop], batch_size = 1, verbose = 1, shuffle = True)
                
                end_time = time.time()

                with open(f"{model_name}_{angulo}_{i}_time.txt", "a") as f:
                    f.write(f"Tempo de execução: {end_time - start_time}\n")

                #plot_visualization(history, model_name, angulo, i)
