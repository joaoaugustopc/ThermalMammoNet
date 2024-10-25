from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision

"""
Criando AlexNet (para usar a rede, criar o objeto AlexNet e usar .model)

Params: (para criar o modelo):
input_shape: especificar entrada da rede
num_classes: quantidade de classes para classificação
learning_rate: taxa de aprendizado 

"""

class AlexNet:
    
    def __init__(self, input_shape=(227, 227, 1), num_classes=2, learning_rate=0.00001):
        # Configura a política de precisão misturada
        mixed_precision.set_global_policy('mixed_float16')
        
        # Inicializa os atributos
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(96, input_shape=self.input_shape, kernel_size=(11, 11), strides=(4, 4), 
                                padding="valid", activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(256, kernel_size=(5, 5), padding="same", activation="relu", 
                                kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu", 
                                kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu", 
                                kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu", 
                                kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.num_classes, activation="sigmoid", dtype="float32", 
                               kernel_regularizer=keras.regularizers.l2(0.001))
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model