import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# Importe o modelo e a função de pré-processamento
from tensorflow.keras.applications import ResNet50

mixed_precision.set_global_policy('mixed_float16')

def resnet50_pre_trained(input_shape=(224,224,3), num_classes=1):
    # Carrega o modelo base pré-treinado no ImageNet
    # include_top=False: remove a camada de saída (a que fazia a classificação em 1000 classes)
    # weights='imagenet': carrega os pesos pré-treinados
    # input_shape: define o formato da sua imagem de entrada
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape)

    # Congela as camadas do modelo base. 
    # Isso impede que os pesos pré-treinados sejam alterados durante o treino.
    base_model.trainable = False

    # Constrói o novo modelo usando a API funcional
    inputs = keras.Input(shape=input_shape)

    # Aplica o pré-processamento específico do ImageNet
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='sigmoid', dtype='float32')(x)
    
    model = keras.Model(inputs, x)

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
