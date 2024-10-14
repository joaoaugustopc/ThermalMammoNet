import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import mixed_precision

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding="same", use_bias=True, kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=True, kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding="same", use_bias=True),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def ResNet34():
  mixed_precision.set_global_policy('mixed_float16')

  model = keras.models.Sequential()
  model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224,1],
	                              padding="same", use_bias=True))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
  prev_filters = 64
  for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
  model.add(keras.layers.GlobalAvgPool2D())
  model.add(keras.layers.Flatten())
  #model.add(keras.layers.Dense(4096, activation="relu"))
  #model.add(keras.layers.Dropout(0.5))
  #model.add(keras.layers.Dense(4096, activation="relu"))
  #model.add(keras.layers.Dropout(0.5))

  model.add(keras.layers.Dense(1, activation = "sigmoid", dtype='float32'))

  opt = keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
  
  return model