from tensorflow import keras
import tensorflow as tf

def alexnet():
    model = keras.Sequential([
    #primeiro bloco : 96 neurônios 
    #entrada da rede: 227 x 227               
    #fix canais     
    keras.layers.Conv2D(96, input_shape = (227, 227, 3), kernel_size=(11, 11), 
    strides=(4, 4), padding = "valid", activation="relu", kernel_regularizer = keras.regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    #keras.layers.Dropout(0.25),   
    
    #segundo bloco
    keras.layers.Conv2D(256, kernel_size = (5, 5), padding = "same", activation = "relu", 
                        kernel_regularizer = keras.regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    #keras.layers.Dropout(0.25),
    
    #terceiro bloco: 3 camadas convolucionais e batch normalization
    keras.layers.Conv2D(384, kernel_size = (3, 3), padding = "same", 
                        activation = "relu", kernel_regularizer = keras.regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(384, kernel_size = (3, 3), padding = "same", 
                        activation = "relu", kernel_regularizer = keras.regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, kernel_size = (3, 3), padding = "same", 
                        activation = "relu", kernel_regularizer = keras.regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(), 
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)), 
    #keras.layers.Dropout(0.25),    
    
    #quarto bloco
    keras.layers.Flatten(), #transformando em vetor
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.0001)), 
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),  
    #sexto bloco
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.0001)), 
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),  
    #camada de saída
    keras.layers.Dense(2, activation = "softmax", kernel_regularizer = keras.regularizers.l2(0.0001))])
    
    
    optimizer = tf.keras.optimizers.Adam() #melhorar
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model
     