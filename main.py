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


def get_boxPlot():
    list = ["frontal","Left90","Left45","Right90","Right45" ]

    for angulo in list:
        acc = []
        loss = []
        for i in range(10):
            #with custom_object_scope({'ResidualUnit': ResidualUnit}):
            model = tf.keras.models.load_model(f"modelos/VGG16_trained_{angulo}_{i}.h5")
            
            imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data("frontal")

            imagens_test = np.expand_dims(imagens_test, axis = -1)

            imagens_test = np.repeat(imagens_test, 3, axis=-1)

            #imagens_test = tf.image.resize(imagens_test, (200, 200))

            loss_, acc_ = test_model(model, imagens_test, labels_test)

            acc.append(acc_)
            loss.append(loss_)
        
        bloxPlot(acc, loss, "ResNet34", f"VGG_trained_{angulo}.png")

        print(f"Acurácia média: {np.mean(acc)}")
        print(f"Loss médio: {np.mean(loss)}")
        print(f"Desvio padrão da acurácia: {np.std(acc)}")
        print(f"Desvio padrão do loss: {np.std(loss)}")
        print(f"Mediana da acurácia: {np.median(acc)}")
        print(f"Mediana do loss: {np.median(loss)}")
        
def apply_augmentation(train, labels):
    # definir a sequência de aug de dados
    simple_aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.02),
        keras.layers.RandomZoom(0.2, 0.2)
    ])
    
    # Usar um batch size fixo
    BATCH_SIZE = len(train)
    
    # Criar o dataset e aplicar a augmentação
    train_aug = tf.data.Dataset.from_tensor_slices((train, labels)) \
        .shuffle(len(train)).batch(BATCH_SIZE) \
        .map(lambda x, y: (simple_aug(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
        .prefetch(tf.data.AUTOTUNE)
    
    # Juntar todas as imagens e rótulos em arrays únicos
    all_images, all_labels = zip(*[(img.numpy(), lbl.numpy()) for img, lbl in train_aug])

    # Concatenar as listas em arrays finais
    return np.concatenate(all_images), np.concatenate(all_labels)


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
    
    train_original = np.load("np_dataset/imagens_train_Frontal.npy")
    labels_original = np.load("np_dataset/labels_train_Frontal.npy")
    
    print(train_original.shape)
    print(labels_original.shape)
    
                    
    train, labels = apply_augmentation(train_original, labels_original)
    
    
    """
    #main_func([VGG16_trained])
    for angle in ["Frontal", "Right45", "Right90", "Left45", "Left90"]:

        print("ANGLE:",angle)
        print("Train shape:",imagens_train.shape)
        print(labels_train.shape)
        print("valid shape:",imagens_valid.shape)
        print(labels_valid.shape)
        print("test shape:",imagens_test.shape)
        print(labels_test.shape)

        print("Train Healthy:",len(labels_train[labels_train == 0]))
        print("Train Sick:",len(labels_train[labels_train == 1]))
        print("Valid Healthy:",len(labels_valid[labels_valid == 0]))
        print("Valid Sick:",len(labels_valid[labels_valid == 1]))
        print("Test Healthy:",len(labels_test[labels_test == 0]))
        print("Test Sick:",len(labels_test[labels_test == 1]))

        if not os.path.exists("np_dataset"):
            os.makedirs("np_dataset")

        np.save(f"np_dataset/imagens_train_{angle}.npy", imagens_train)
        np.save(f"np_dataset/labels_train_{angle}.npy", labels_train)
        np.save(f"np_dataset/imagens_valid_{angle}.npy", imagens_valid)
        np.save(f"np_dataset/labels_valid_{angle}.npy", labels_valid)
        np.save(f"np_dataset/imagens_test_{angle}.npy", imagens_test)
        np.save(f"np_dataset/labels_test_{angle}.npy", labels_test)
    
    
    """