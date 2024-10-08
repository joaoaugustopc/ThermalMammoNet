import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import numpy as np

mixed_precision.set_global_policy('mixed_float16')


def VGG_16():

    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(64, 3, input_shape=[240, 320, 1],padding="same", activation="relu"))
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

    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(units=4096, activation='relu')) # 4096 -> Reduzi por conta da memoria (TESTE)
    model.add(keras.layers.Dense(units=4096, activation='relu')) # 4096 -> Reduzi por conta da memoria (TESTE)
    model.add(keras.layers.Dense(units=2, activation='softmax', dtype='float32'))

    optimize = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimize, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model