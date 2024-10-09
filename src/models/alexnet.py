from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision

def alexnet():
    
    #TODO: olhar a distribuição dos valid e same
    #add mixed precision: layers use float16 computations and float32 variables.
    mixed_precision.set_global_policy('mixed_float16')
    
    model = keras.Sequential([
    #primeiro bloco : 96 neurônios 
    #entrada da rede: 227 x 227               
    #fix canais
    keras.layers.Conv2D(96, input_shape = (227, 227, 1), kernel_size=(11, 11), 
    strides=(4, 4), padding = "valid", activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
    
    #segundo bloco
    keras.layers.Conv2D(256, kernel_size = (5, 5), padding = "same", activation = "relu", 
                        kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
    
    #terceiro bloco: 3 camadas convolucionais e batch normalization
    keras.layers.Conv2D(384, kernel_size = (3, 3), padding = "same", 
                        activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.Conv2D(384, kernel_size = (3, 3), padding = "same", 
                        activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.Conv2D(256, kernel_size = (3, 3), padding = "same", 
                        activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'), 
    
    #quarto bloco
    keras.layers.Flatten(), #transformando em vetor
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)), 
    keras.layers.Dropout(0.5),  
    #sexto bloco
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),  #TODO: olhar os regularizadores
    keras.layers.Dropout(0.5),  
    #camada de saída: especificar para float32 -> output
    keras.layers.Dense(2, activation = "softmax", dtype="float32", kernel_regularizer = keras.regularizers.l2(0.001))])
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001) #melhorar
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model
     