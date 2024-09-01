from tensorflow import keras
from tensorflow.keras.applications import VGG16


def VGG16_trained():
    model_trained = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(480, 640, 3),
                pooling='max')
    
    model = keras.models.Sequential()
    model.add(model_trained)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    

    return model

