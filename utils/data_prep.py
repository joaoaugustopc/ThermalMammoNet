from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import custom_object_scope
from src.models.resNet_34 import ResNet34, ResidualUnit


#Arrays Numpy
def load_data(angulo):

    imagens_train = np.load(f"np_dataset/imagens_train_{angulo}.npy")
    labels_train = np.load(f"np_dataset/labels_train_{angulo}.npy")
    imagens_valid = np.load(f"np_dataset/imagens_valid_{angulo}.npy")
    labels_valid = np.load(f"np_dataset/labels_valid_{angulo}.npy")
    imagens_test = np.load(f"np_dataset/imagens_test_{angulo}.npy")
    labels_test = np.load(f"np_dataset/labels_test_{angulo}.npy")


    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test


def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalizar a imagem
    return image, label

# text_dataset_from_directory, retorna um tf.Dataset
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

def get_boxPlot(modelo):
    list = ["Frontal","Left90","Left45","Right90","Right45" ]

    for angulo in list:
        acc = []
        loss = []
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        for i in range(10):
            with custom_object_scope({'ResidualUnit': ResidualUnit}):
              model = tf.keras.models.load_model(f"modelos/{modelo}_{angulo}_{i}.h5")
            
            #imagens_test = np.expand_dims(imagens_test, axis = -1)

            #imagens_test = np.repeat(imagens_test, 3, axis=-1)

            #imagens_test = tf.image.resize(imagens_test, (200, 200))

            loss_, acc_ = test_model(model, imagens_test, labels_test)

            acc.append(acc_)
            loss.append(loss_)
        
        bloxPlot(acc, loss, "ResNet34", f"ResNEt34_{angulo}.png")

        print(f"Acurácia média: {np.mean(acc)}")
        print(f"Loss médio: {np.mean(loss)}")
        print(f"Desvio padrão da acurácia: {np.std(acc)}")
        print(f"Desvio padrão do loss: {np.std(loss)}")
        print(f"Mediana da acurácia: {np.median(acc)}")
        print(f"Mediana do loss: {np.median(loss)}")

def preprocess(image):
    # Normalizar a imagem
    max = np.max(image)
    min = np.min(image)
    image = (image - min) / (max - min)
    return image

def extract_id(filename):
    # Extrair o ID a partir do nome do arquivo
    return filename.split('_')[0]


def to_array(directory):
    arquivos = os.listdir(directory)

    print(arquivos)

    sick_path = os.path.join(directory, arquivos[1])
    healthy_path = os.path.join(directory, arquivos[0])

    healthy = os.listdir(healthy_path)
    sick = os.listdir(sick_path)

    imagens = []
    labels = []
    ids = []

    for arquivo in healthy:
        path = os.path.join(healthy_path, arquivo)
        try:
          with open(path, 'r') as f:
            primeira_linha = f.readline()
            if ';' in primeira_linha:
              delimiter = ';'
            else:
              delimiter = ' '
          imagem = np.loadtxt(path, delimiter=delimiter)
          imagem = preprocess(imagem)
          imagens.append(imagem)
          labels.append(0)
          ids.append(extract_id(arquivo))
        except ValueError as e:
          print(e)
          print(arquivo)
          continue

    for arquivo in sick:
        path = os.path.join(sick_path, arquivo)
        try:
          with open(path, 'r') as f:
            primeira_linha = f.readline()
            if ';' in primeira_linha:
              delimiter = ';'
            else:
              delimiter = ' '
          imagem = np.loadtxt(path, delimiter=delimiter)
          imagem = preprocess(imagem)
          imagens.append(imagem)
          labels.append(1)
          ids.append(extract_id(arquivo))
        except ValueError as e:
          print(e)
          print(arquivo)
          continue

    ids_unicos = []
    imagens_unicas = []
    labels_unicos = []

    consulta_count = {}

    mult_consultas_img = []
    mult_consultas_label = []
    mult_consultas_id = []

    #contar o número de consultas por paciente  
    for id in ids:
        if id in consulta_count:
            consulta_count[id] += 1
        else:
            consulta_count[id] = 1
    

    for imagem, label, id in zip(imagens, labels, ids):
        if consulta_count[id] == 1:
            imagens_unicas.append(imagem)
            labels_unicos.append(label)
            ids_unicos.append(id)
        else:
            mult_consultas_img.append(imagem)
            mult_consultas_label.append(label)
            mult_consultas_id.append(id)
      
    total_imgs = len(imagens)
    total_imgs_unicas = len(imagens_unicas)
    total_mult_imgs = len(mult_consultas_img)

    mult_porcent = (total_mult_imgs / total_imgs)

    train_percent = 0.6 - mult_porcent

    imagens_unicas = np.array(imagens_unicas)
    labels_unicos = np.array(labels_unicos)

    # Primeiro split: 60% treino, 40% restante
    imagens_train, imagens_rest, labels_train, labels_rest = train_test_split(imagens_unicas, labels_unicos, test_size= 1 - train_percent, shuffle = True)

    imagens_train = np.concatenate((imagens_train, mult_consultas_img), axis=0)
    labels_train = np.concatenate((labels_train, mult_consultas_label), axis=0)

    # Embaralhar os dados de treino
    permutation = np.random.permutation(len(imagens_train))
    imagens_train = imagens_train[permutation]
    labels_train = labels_train[permutation]

    # Segundo split: 50% validação, 50% teste do restante (que é 20% cada do total original)

    imagens_valid, imagens_test, labels_valid, labels_test = train_test_split(imagens_rest, labels_rest, test_size=0.5, shuffle = True)

    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test


def format_data(directory_raw):
    for angle in ["Frontal", "Right45", "Right90", "Left45", "Left90"]:
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = to_array(f"raw_dataset/{angle}")

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

    