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
from utils.data_prep import to_array
    

#Arrays Numpy
def load_data(angulo):

    imagens_train = np.load(f"dataset_np/imagens_train_{angulo}.npy")
    labels_train = np.load(f"dataset_np/labels_train_{angulo}.npy")
    imagens_valid = np.load(f"dataset_np/imagens_valid_{angulo}.npy")
    labels_valid = np.load(f"dataset_np/labels_valid_{angulo}.npy")
    imagens_test = np.load(f"dataset_np/imagens_test_{angulo}.npy")
    labels_test = np.load(f"dataset_np/labels_test_{angulo}.npy")


    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test

#nao esta sendo usada
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalizar a imagem
    return image, label

# nao esta sendo usada text_dataset_from_directory, retorna um tf.Dataset
def load_tf_data(path):
    train_path = os.path.join(path, "train")

    train_ds = keras.utils.text_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        shuffle=True,
        seed=123,
        batch_size = None
    )

    test_path = os.path.join(path, "test")

    val_ds, test_ds = keras.utils.text_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        shuffle=True,
        seed=123,
        validation_split=0.5,
        subset="both",
        batch_size = None
    )

    return train_ds, val_ds, test_ds

#nao esta sendo usada
def data_generator(imagens, labels):
    for img, label in zip(imagens, labels):
        img = np.expand_dims(img, axis = -1)
        yield img, label

#nao esta sendo usada
def crate_dataset(imagens, labels, batch_size = 2, image_size = (480, 640)):

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(imagens, labels), 
            output_signature=( tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32), 
            tf.TensorSpec(shape=(), dtype=tf.int32))
    )

    dataset = dataset.map(lambda x, y: preprocess_image(x, y))

    dataset = dataset.batch(batch_size)

    return dataset


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Pasta {folder_path} deletada.")
    else:
        print(f"Pasta {folder_path} não encontrada.")


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Arquivo {file_path} deletado.")
    else:
        print(f"Arquivo {file_path} não encontrado.")


def move_files_to_folder(file_list, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file in file_list:
        if os.path.exists(file):
            shutil.move(file, destination_folder)

def test_model(model, imagens_test, labels_test):
    """
    imagens_test = np.expand_dims(imagens_test, axis = -1)
    imagens_test = np.repeat(imagens_test, 3, axis=-1)
    #imagens_test = tf.image.resize(imagens_test, (200, 200))
    """
   
    loss, acc = model.evaluate(imagens_test, labels_test)

    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")

    return loss, acc

def bloxPlot(acc_data, loss_data, title, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Boxplot para acurácia
    axs[0].boxplot(acc_data)
    axs[0].set_title('Acurácia dos modelos')
    axs[0].set_xlabel('Modelo')
    axs[0].set_ylabel('Acurácia')

    # Boxplot para loss
    axs[1].boxplot(loss_data)
    axs[1].set_title('Loss do modelo')
    axs[1].set_xlabel('Modelo')
    axs[1].set_ylabel('Loss')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def main_func(models_list):
    list = ["frontal","Left90","Left45","Right90","Right45" ]
    models = models_list

    for angulo in list:

        #train_ds, val_ds, test_ds = load_tf_data(f"dataset/{angulo}")
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        imagens_train = np.expand_dims(imagens_train, axis = -1)  # Add uma dimensão para o canal de cor
        imagens_valid = np.expand_dims(imagens_valid, axis = -1) #
        imagens_test = np.expand_dims(imagens_test, axis = -1)  # 

        imagens_train = np.repeat(imagens_train, 3, axis=-1) # Repete a imagem para 3 canais ( Não fazer isso )
        imagens_valid = np.repeat(imagens_valid, 3, axis=-1)
        imagens_test = np.repeat(imagens_test, 3, axis=-1)
        
        """
        imagens_train = tf.image.resize(imagens_train, (200, 200))
        imagens_valid = tf.image.resize(imagens_valid, (200, 200)) #Resize das imagens ( fazer para tamanhos proporcionais )
        imagens_test = tf.image.resize(imagens_test, (200, 200))
        """
        for model_func in models:

            model_name = model_func.__name__

            for i in range(10):

                start_time = time.time()

                checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/{model_name}_{angulo}_{i}.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto')

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
      # Definir a sequência de augmentação de dados
    simple_aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(factor=0.02),
        keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ])
    
    BATCH_SIZE = 128
    AUTO = tf.data.AUTOTUNE
    
    # Criar o dataset e aplicar as transformações
    train_original = tf.data.Dataset.from_tensor_slices((train, labels))
    train_dataset = train_original.shuffle(BATCH_SIZE * 100)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.map(lambda x, y: (simple_aug(x), y), num_parallel_calls=AUTO)
    train_dataset = train_dataset.prefetch(AUTO)
    
    train_original.concatenete(train_dataset)
    
    return train_original.concatenate(train_dataset)


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
    
    train = np.load("np_dataset/imagens_train_Frontal.npy")
    labels = np.load("np_dataset/labels_train_Frontal.npy")
    
    print(train.shape)
    print(labels.shape)
    
                    
    train_dataset = apply_augmentation(train, labels)
    
     # Verifique o formato do dataset
    for images, labels in train_dataset.take(1):
        
        print(images.shape)
        print(labels.shape)
    
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