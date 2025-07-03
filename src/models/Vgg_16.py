from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision


class Vgg_16:

    def __init__(self, input_shape=(224, 224, 1), num_classes=1, learning_rate=0.0001):
    
        mixed_precision.set_global_policy('mixed_float16')
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            # conv1
            keras.layers.Conv2D(64, 3, input_shape=self.input_shape, padding="same", activation="relu" ,kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.MaxPool2D(pool_size=2, strides=2),
            keras.layers.BatchNormalization(),

            # conv2
            keras.layers.Conv2D(128, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.MaxPool2D(pool_size=2, strides=2),
            keras.layers.BatchNormalization(),

            # conv3
            keras.layers.Conv2D(256, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(256, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(256, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.MaxPool2D(pool_size=2, strides=2),
            keras.layers.BatchNormalization(),

            # conv4
            keras.layers.Conv2D(512, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(512, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(512, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.MaxPool2D(pool_size=2, strides=2),
            keras.layers.BatchNormalization(),

            # conv5
            keras.layers.Conv2D(512, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(512, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Conv2D(512, 3, padding="same", activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.MaxPool2D(pool_size=2, strides=2),
            keras.layers.BatchNormalization(),

            # Fully connected layers
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.num_classes, activation="sigmoid", dtype="float32")
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model
