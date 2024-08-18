import tensorflow as tf
from tensorflow import keras
import numpy as np


def VGG_16():
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(64, 3, input_shape=[480, 640, 1],padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(keras.layers.flatten())

    model.add(keras.layers.Dense(units=4096, activation='relu'))
    model.add(keras.layers.Dense(units=4096, activation='relu'))
    model.add(keras.layers.Dense(units=2, activation='softmax'))

    optimize = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimize, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print (model.summary())

    return model