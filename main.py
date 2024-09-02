import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from src.models.resNet_34 import ResNet34, ResidualUnit
from src.models import googleLenet
from src.models.Vgg_16 import VGG_16
from src.models.googleLenet import googleLenet
from src.models.vgg_16_trained import VGG16_trained
from src.models.resNet_152 import ResNet152_trained
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import time 
import shutil
from tensorflow.keras.utils import custom_object_scope
from utils.data_prep import to_array, load_data, get_boxPlot, move_files_to_folder,delete_folder, delete_file
    
def main_func(models_list):
    list = ["Frontal","Left90","Left45","Right90","Right45" ]
    models = models_list

    for angulo in list:

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        
        imagens_train = np.expand_dims(imagens_train, axis = -1)  # Add uma dimensão para o canal de cor
        imagens_valid = np.expand_dims(imagens_valid, axis = -1) 
        imagens_test = np.expand_dims(imagens_test, axis = -1)
        

        imagens_train = tf.image.resize(imagens_train, (120,160), method='bicubic')
        imagens_valid = tf.image.resize(imagens_valid, (120,160), method='bicubic')

        print(f"Imagens de treino: {imagens_train.shape}")
        print(f"Imagens de validação: {imagens_valid.shape}")
        
        for model_func in models:

            model_name = model_func.__name__

            for i in range(10):

                start_time = time.time()

                checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/{model_name}_{angulo}_{i}.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

                model = model_func()

                model.summary()

                history = model.fit(imagens_train, labels_train, epochs = 100, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint,earlystop], batch_size = 1, verbose = 1, shuffle = True)
                
                end_time = time.time()

                with open(f"{model_name}_{angulo}_{i}_time.txt", "w") as f:
                    f.write(f"Modelo: {model_name}\n")
                    f.write(f"Tempo de execução: {end_time - start_time}\n")
                    f.write(f"Loss: {history.history['loss']}\n")
                    f.write(f"Val_loss: {history.history['val_loss']}\n")
                    f.write(f"Accuracy: {history.history['accuracy']}\n")
                    f.write(f"Val_accuracy: {history.history['val_accuracy']}\n")
                    f.write("\n") 

        """
        """


    

if __name__ == "__main__":

    #main_func([VGG_16])
    #get_boxPlot("ResNet34")

    """
    data_train = np.load("np_dataset/imagens_train_Frontal.npy")

    

    data_aug = tf.keras.Sequential([
          #keras.layers.RandomFlip("horizontal_and_vertical"),
          #keras.layers.RandomRotation(0.2),
          #keras.layers.RandomZoom(0.2),
          #keras.layers.RandomContrast(0.2),
          #keras.layers.RandomBrightness(0.2),
          keras.layers.RandomTranslation(0.2, 0.2),
          #keras.layers.RandomRotation(0.01),
        
        ])
    
    data_aug = data_aug(data_train)

    plt.figure(figsize=(10, 10))

    for i in range(6):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(data_aug[i].numpy().reshape(480,640), cmap='gray')
        plt.axis("off")

    plt.savefig("teste2.png")
    """
    
    
    