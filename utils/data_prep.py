from sklearn.model_selection import train_test_split
import numpy as np
import os


def preprocess(image):
    image = image / 255.0
    image -= np.mean(image)
    return image


def to_array(directory):
    arquivos = os.listdir(directory)

    print(arquivos)

    sick_path = os.path.join(directory, arquivos[1])
    healthy_path = os.path.join(directory, arquivos[0])

    healthy = os.listdir(healthy_path)
    sick = os.listdir(sick_path)

    imagens = []
    labels = []

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
        except ValueError as e:
          print(e)
          print(arquivo)
          continue

    imagens = np.array(imagens)
    labels = np.array(labels)

    imagens_train, imagens_valid, labels_train, labels_valid = train_test_split(imagens, labels, test_size=0.3, shuffle = True)



    return imagens_train, labels_train, imagens_valid, labels_valid

def format_data(directory_raw):

    if not os.path.exists("dataset_np"):
        os.makedirs("dataset_np")

    directorys = [f"{directory_raw}\\Right45", f"{directory_raw}\\Left45", f"{directory_raw}\\Right90", f"{directory_raw}\\Left90",f"{directory_raw}\\frontal"]

    for directory in directorys:
        imagens_train, labels_train, imagens_valid, labels_valid = to_array(directory)
        np.save(f"dataset_np/imagens_train_{directory.split("\\")[1]}.npy", imagens_train)
        np.save(f"dataset_np/labels_train_{directory.split("\\")[1]}.npy", labels_train)
        np.save(f"dataset_np/imagens_valid_{directory.split("\\")[1]}.npy", imagens_valid)
        np.save(f"dataset_np/labels_valid_{directory.split("\\")[1]}.npy", labels_valid)


if __name__ == "__main__":
    format_data("dataset_raw")