from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil


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

    unique_ids = set()
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

    if not os.path.exists("dataset_np"):
        os.makedirs("dataset_np")

    directorys = [f"{directory_raw}/Right45", f"{directory_raw}/Left45", f"{directory_raw}/Right90", f"{directory_raw}/Left90",f"{directory_raw}/frontal"]

    for directory in directorys:
        imagens_train, labels_train, imagens_valid, labels_valid,imagens_test, labels_test = to_array(directory)
        np.save(f"dataset_np/imagens_train_{directory.split('/')[1]}.npy", imagens_train)
        np.save(f"dataset_np/labels_train_{directory.split('/')[1]}.npy", labels_train)
        np.save(f"dataset_np/imagens_valid_{directory.split('/')[1]}.npy", imagens_valid)
        np.save(f"dataset_np/labels_valid_{directory.split('/')[1]}.npy", labels_valid)
        np.save(f"dataset_np/imagens_test_{directory.split('/')[1]}.npy", imagens_test)
        np.save(f"dataset_np/labels_test_{directory.split('/')[1]}.npy", labels_test)


        