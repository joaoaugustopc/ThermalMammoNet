import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from src.models.resNet_34 import ResNet34, ResidualUnit
from src.models import googleLenet
from src.models.Vgg_16 import VGG_16
from src.models.googleLenet import googleLenet
from src.models.alexnet import alexnet
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
    
    list = ["Frontal","Left45","Right90","Right45" ]
    models = models_list

    for angulo in list:

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        imagens_train, labels_train = apply_augmentation_and_expand(imagens_train, labels_train, 2, resize=True, target_size=227)
        
        imagens_valid = np.expand_dims(imagens_valid, axis=-1)
        imagens_valid = tf.image.resize_with_pad(imagens_valid, 227, 227, method="bicubic")
        imagens_valid = np.squeeze(imagens_valid, axis=-1)
        print(imagens_valid.shape)
        
        imagens_test = np.expand_dims(imagens_test, axis=-1)
        imagens_test = tf.image.resize_with_pad(imagens_test, 227, 227, method="bicubic")
        imagens_test = np.squeeze(imagens_test, axis=-1)
        print(imagens_test.shape)
        
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
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto') #alterei patience

                model = model_func()

                model.summary()

                history = model.fit(imagens_train, labels_train, epochs = 100, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint,earlystop], batch_size = 1, verbose = 1, shuffle = True)
                
                end_time = time.time()

                # Avaliação do modelo com conjunto de teste
                test_loss, test_accuracy = model.evaluate(imagens_test, labels_test, verbose=1)

                with open(f"{model_name}_{angulo}_{i}_time.txt", "w") as f:
                    f.write(f"Modelo: {model_name}\n")
                    f.write(f"Tempo de execução: {end_time - start_time}\n")
                    f.write(f"Loss: {history.history['loss']}\n")
                    f.write(f"Val_loss: {history.history['val_loss']}\n")
                    f.write(f"Accuracy: {history.history['accuracy']}\n")
                    f.write(f"Val_accuracy: {history.history['val_accuracy']}\n")
                    f.write(f"Test Loss: {test_loss}\n")
                    f.write(f"Test Accuracy: {test_accuracy}\n")
                    f.write("\n")

                    
                plot_convergence(history, model_name, angulo, i)

#TODO: olhar a direção do treino - métrica 

"""
    Params:
    train: conjuntos de imagens de treino original
    labels: conjunto de labels de treino original
    num_augmented_copies: números de vezes para expandir o dataset
    resize: bool -> para redimensionar
    targert_size: int -> novo tamanho para redimensionar
    
    return:
    all_imagens, all_labels : imagens e labels originais + aug
"""
def apply_augmentation_and_expand(train, labels, num_augmented_copies, resize=False, target_size=0):
    train = np.expand_dims(train, axis=-1)
    
    #TODO: olhar como foi utilizado - usar separado -> melhorar o dataset
    simple_aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.4, 0.4),
        keras.layers.RandomBrightness(factor=(0.1), value_range=(0.0, 1.0)),  # ajuste conforme necessário
        keras.layers.RandomContrast(factor=0.3)  # ajuste conforme necessário
        #TODO: arrumar o fundo - azul
        #TODO: olhar o opencv - transformar em 3 canais (comparar)
    ])
    
    # listas para armazenar as imagens e labels
    all_images = []
    all_labels = []
    
    #adicionar as imagens e labels originais
    for image, label in zip(train, labels):
        all_images.append(image)
        all_labels.append(label)
    
    # copias aug de acordo com a entrada num
    for _ in range(num_augmented_copies):
        for image, label in zip(train, labels):
            augmented_image = simple_aug(image)
            all_images.append(augmented_image)
            all_labels.append(label)
    
    #convertendo em np
    all_images = np.array([img for img in all_images])
    all_labels = np.array(all_labels)
    
    #se passado como parametro
    if resize:
        all_images = tf.image.resize_with_pad(all_images, target_size, target_size, method="bicubic")
    
    
    all_images = np.squeeze(all_images, axis=-1)    
    print(all_images.shape)
    
    #teste
    visualize_augmentation(all_images[:10], all_images[156:166], 10)
    
    return all_images, all_labels    

"""
Params:
    original_img: dataset original
    aug_img: dataset alterado (aug e resize)
    num_images: imagens visualizadas
    
    Return:
    void -> mas as imagens são salvas
"""
def visualize_augmentation(original_img, aug_img, num_images=5):
    
    plt.figure(figsize=(15, 6))         
    for i in range(num_images):
        # Imagem Original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_img[i].astype('float32'))
        plt.title("Original")
        plt.axis("off")
        
        # Imagem Augmentada
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(aug_img[i].astype("float32"))
        plt.title("Augmentada")
        plt.axis("off")
    plt.savefig("foto_aug/")


if __name__ == "__main__":
        
    
    main_func([alexnet])


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