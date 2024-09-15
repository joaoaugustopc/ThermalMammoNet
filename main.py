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
    

def plot_convergence(history, model_name, angulo, i):
    # Gráfico de perda de treinamento
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'Training Loss Convergence for {model_name} - {angulo} - Run {i}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_{angulo}_{i}_training_loss_convergence.png")
    plt.close()

    # Gráfico de perda de validação
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Validation Loss Convergence for {model_name} - {angulo} - Run {i}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_{angulo}_{i}_validation_loss_convergence.png")
    plt.close()


def main_func(models_list):
    
    list = ["Frontal","Left90","Left45","Right90","Right45" ]
    models = models_list

    for angulo in list:

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        
        """
        imagens_train = np.expand_dims(imagens_train, axis = -1)  # Add uma dimensão para o canal de cor
        imagens_valid = np.expand_dims(imagens_valid, axis = -1) 
        imagens_test = np.expand_dims(imagens_test, axis = -1)
        

        imagens_train = tf.image.resize(imagens_train, (120,160), method='bicubic')
        imagens_valid = tf.image.resize(imagens_valid, (120,160), method='bicubic')

        print(f"Imagens de treino: {imagens_train.shape}")
        print(f"Imagens de validação: {imagens_valid.shape}")

        """
        
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
                
                plot_convergence(history, model_name, angulo, i)
        


def apply_augmentation(train, labels):
       # Definir a sequência de augmentação de dados
    simple_aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(factor=0.02),
        keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ])
    
    #pode alterar para usar o dataset todo de uma vez 
    BATCH_SIZE = 128
    
    train_aug = tf.data.Dataset.from_tensor_slices((train, labels))
    
    train_aug = (
        train_aug.shuffle(BATCH_SIZE*100).batch(BATCH_SIZE).map(lambda x, y: (simple_aug(x),y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    )
    
    # Juntar todas as imagens em um único array
    all_images = []
    all_labels = []

    for images, lbls in train_aug:
        all_images.append(images.numpy())
        all_labels.append(lbls.numpy())

    # Concatenar todas as imagens e rótulos
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("Shape of all images:", all_images.shape)
    print("Shape of all labels:", all_labels.shape)
    
    
    return all_images, all_labels
   


def visualize_augmentation(original_images, augmented_dataset, num_images=5):
    augmented_images, labels = next(iter(augmented_dataset))
    augmented_images = augmented_images[:num_images]
    
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        # Imagem Original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].astype("uint8"))
        plt.title("Original")
        plt.axis("off")
        
        # Imagem Augmentada
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title("Augmentada")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    
    #main_func([ResNet34])


    """
    list = ["Frontal","Left90","Left45","Right90","Right45" ]

    for angulo in list:
        files = [f"ResNet34_{angulo}_{i}_time.txt" for i in range(10)]

        move_files_to_folder(files, f"history/ResNet34/{angulo}")
    """
    
    
    """
    train_original = np.load("np_dataset/imagens_train_Frontal.npy")
    labels_original = np.load("np_dataset/labels_train_Frontal.npy")
    
    print(train_original.shape)
    print(labels_original.shape)
    
                    
    train, labels = apply_augmentation(train_original, labels_original)
    """