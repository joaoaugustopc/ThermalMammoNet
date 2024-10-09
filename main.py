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
import re
import shutil
from tensorflow.keras.utils import custom_object_scope
from utils.data_prep import to_array, load_data, get_boxPlot, move_files_to_folder,delete_folder, delete_file
from PIL import Image


def apply_mask():
    masks = os.listdir('masks')

    regex = re.compile(r'.*1\.S.*')

    mascaras = [mask for mask in masks if regex.match(mask)]

    max_value = 0
    img_max = None

    for mask in mascaras:
        img = Image.open(os.path.join('masks', mask))
        img = np.array(img)

        n_true = np.count_nonzero(img)

        if n_true > max_value:
            max_value = n_true
            img_max = img


    imagens_train = np.load('np_dataset/imagens_train_Frontal.npy')

    lista = []

    for imagem in imagens_train:

        masked = np.ma.masked_array(imagem, ~img_max)
        filled = np.ma.filled(masked, 0)
        lista.append(filled)

    masked_array = np.array(lista)

    np.save('imagens_train_Frontal_masked.npy', masked_array)


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
    
    list = ["Frontal"]
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
        imagens_train, labels_train = apply_augmentation_and_expand(imagens_train, labels_train, 2, resize=True, target_size=227)
        
        imagens_valid = np.expand_dims(imagens_valid, axis=-1)
        imagens_valid = tf.image.resize_with_pad(imagens_valid, 227, 227, method="bicubic")
        imagens_valid = np.squeeze(imagens_valid, axis=-1)
        print(imagens_valid.shape)
        
        imagens_test = np.expand_dims(imagens_test, axis=-1)
        imagens_test = tf.image.resize_with_pad(imagens_test, 227, 227, method="bicubic")
        imagens_test = np.squeeze(imagens_test, axis=-1)
        print(imagens_test.shape)
        
        imagens_train = np.expand_dims(imagens_train, axis = -1)  # Add uma dimensão para o canal de cor
        imagens_valid = np.expand_dims(imagens_valid, axis = -1) 
        imagens_test = np.expand_dims(imagens_test, axis = -1)

        
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
                                    callbacks= [checkpoint,earlystop], batch_size = 4, verbose = 1, shuffle = True)
                
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


# Funções lambda para cada transformação separada
random_flip = lambda: keras.layers.RandomFlip("horizontal")
random_rotation = lambda: keras.layers.RandomRotation(0.05)
random_zoom = lambda: keras.layers.RandomZoom(0.4, 0.4)
random_brightness = lambda: keras.layers.RandomBrightness(factor=0.3, value_range=(0.0, 1.0))
random_contrast = lambda: keras.layers.RandomContrast(factor=0.3)

# Função para aplicar as transformações individualmente
def apply_transformation(image, transformation):
    return transformation()(image)


def resize_images(all_images, target_size, resize=False):
    if resize:
        # Usando keras.layers.Resizing
        resizing_layer = keras.layers.Resizing(
            height=target_size,
            width=target_size,
            interpolation="bicubic",
            crop_to_aspect_ratio=False,
            pad_to_aspect_ratio=True,  # Equivalente ao comportamento de resize_with_pad
            fill_mode="constant",
            fill_value=0.0,
            data_format=None
        )
        all_images = resizing_layer(all_images)
    return all_images

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
#TODO: olhar como foi utilizado - usar separado -> melhorar o dataset
#TODO: arrumar o fundo - azul
#TODO: olhar o opencv - transformar em 3 canais (comparar)
def apply_augmentation_and_expand(train, labels, num_augmented_copies, resize=False, target_size=0):
    train = np.expand_dims(train, axis=-1)
    
    # listas para armazenar as imagens e labels
    all_images = []
    all_labels = []
    
    #adicionar as imagens e labels originais
    for image, label in zip(train, labels):
        all_images.append(image)
        all_labels.append(label)
    transformations = [random_rotation, random_zoom, random_brightness, random_contrast]

    #aplicar cada transformação separado
    for _ in range(num_augmented_copies):
        for image, label in zip(train, labels):
            for transformation in transformations:  # Aplicar cada transformação separadamente
                augmented_image = apply_transformation(image, transformation)
                
                # Aleatoriamente decidir se vai aplicar random_flip
                if random.random() > 0.5:  # 50% de chance de aplicar o flip
                    augmented_image = apply_transformation(augmented_image, random_flip)
                
                
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
    #apply_mask()

    imagens_train = np.load("RGB/imagens_train_Frontal.npy")
    labels_train = np.load("RGB/labels_train_Frontal.npy")
    imagens_valid = np.load("RGB/imagens_valid_Frontal.npy")
    labels_valid = np.load("RGB/labels_valid_Frontal.npy")
    imagens_test = np.load("RGB/imagens_test_Frontal.npy")
    labels_test = np.load("RGB/labels_test_Frontal.npy")

    imagens_test = tf.image.resize(imagens_test, (224,224), method='bicubic')

    model = VGG16( weights='imagenet', include_top=True, classes = 1000)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    loss, acc = model.evaluate(imagens_test, labels_test, verbose=2)

    print("Model, accuracy: {:5.2f}%".format(100*acc))









    #main_func([VGG16_trained])

        
    

    """
    for angle in ["Frontal", "Right45", "Right90", "Left45", "Left90"]:
        
        train = np.load(f"np_dataset/imagens_train_{angle}.npy")
        valid = np.load(f"np_dataset/imagens_valid_{angle}.npy")
        test = np.load(f"np_dataset/imagens_test_{angle}.npy")

        labels_train = np.load(f"np_dataset/labels_train_{angle}.npy")
        labels_valid = np.load(f"np_dataset/labels_valid_{angle}.npy")
        labels_test = np.load(f"np_dataset/labels_test_{angle}.npy")

        print("ANGLE:",angle)
        print("Train shape:",train.shape)
        print(labels_train.shape)
        print("valid shape:",valid.shape)
        print(labels_valid.shape)
        print("test shape:",test.shape)
        print(labels_test.shape)

        print("Train Healthy:",len(labels_train[labels_train == 0]))
        print("Train Sick:",len(labels_train[labels_train == 1]))

    #main_func([alexnet])

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