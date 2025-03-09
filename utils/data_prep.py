from include.imports import *
import os
import re
import shutil
from sklearn.model_selection import train_test_split

#Semente utilizada para criação dos datasets < USAR APENAS QUANDO FOR CRIAR UM NOVO DATASET
#-> EXISTE UMA DEFINIÇÃO DE SEMENTE NO ARQUIVO MAIN.PY >
"""
SEED = 36
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
"""

#Arrays Numpy
def load_data(angulo, folder = "np_dataset"):

    imagens_train = np.load(f"{folder}/imagens_train_{angulo}.npy")
    labels_train = np.load(f"{folder}/labels_train_{angulo}.npy")
    imagens_valid = np.load(f"{folder}/imagens_valid_{angulo}.npy")
    labels_valid = np.load(f"{folder}/labels_valid_{angulo}.npy")
    imagens_test = np.load(f"{folder}/imagens_test_{angulo}.npy")
    labels_test = np.load(f"{folder}/labels_test_{angulo}.npy")


    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test


def preprocess(image, max, min):
    image = (image - min) / (max - min)
    return image

def extract_id(filename):
    # Extrair o ID a partir do nome do arquivo
    return filename.split('_')[0]

"""
Função para converter os arquivos de texto em arrays numpy
"""
def to_array(directory, exclude = False, exclude_set = None):

    arquivos = os.listdir(directory)

    print(arquivos)

    healthy_path = os.path.join(directory, 'healthy')
    sick_path = os.path.join(directory, 'sick')

    healthy = os.listdir(healthy_path)
    sick = os.listdir(sick_path)

    imagens = []
    labels = []
    ids = []
    max_value = 0
    min_value = 1000

    for arquivo in healthy:

        if exclude and arquivo in exclude_set:
            print(f"Excluindo {arquivo}")
            continue

        path = os.path.join(healthy_path, arquivo)
        try:
          with open(path, 'r') as f:
            primeira_linha = f.readline()
            if ';' in primeira_linha:
              delimiter = ';'
            else:
              delimiter = ' '
          imagem = np.loadtxt(path, delimiter=delimiter)
          imagens.append(imagem)
          labels.append(0)
          ids.append(extract_id(arquivo))
        except ValueError as e:
          print(e)
          print(arquivo)
          continue

    for arquivo in sick:

        if exclude and arquivo in exclude_set:
            print(f"Excluindo {arquivo}")
            continue

        path = os.path.join(sick_path, arquivo)
        try:
          with open(path, 'r') as f:
            primeira_linha = f.readline()
            if ';' in primeira_linha:
              delimiter = ';'
            else:
              delimiter = ' '
            f.seek(0)
          imagem = np.loadtxt(path, delimiter=delimiter)
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

    train_percent = max(0.0,0.6 - mult_porcent)

    imagens_unicas = np.array(imagens_unicas)
    labels_unicos = np.array(labels_unicos)

    # Primeiro split: 60% treino, 40% restante
    imagens_train, imagens_rest, labels_train, labels_rest = train_test_split(imagens_unicas, labels_unicos, test_size= 1 - train_percent, shuffle = True)

    imagens_train = np.concatenate((imagens_train, mult_consultas_img), axis=0)
    labels_train = np.concatenate((labels_train, mult_consultas_label), axis=0)

    permutation = np.random.permutation(len(imagens_train))
    imagens_train = imagens_train[permutation]
    labels_train = labels_train[permutation]

    # Segundo split: 50% validação, 50% teste do restante (que é 20% cada do total original)

    imagens_valid, imagens_test, labels_valid, labels_test = train_test_split(imagens_rest, labels_rest, test_size=0.5, shuffle = True)


    for i in range(len(imagens_train)):
        max_value = max(max_value, np.max(imagens_train[i]))
        min_value = min(min_value, np.min(imagens_train[i]))


    for i in range(len(imagens_train)):
        imagens_train[i] = preprocess(imagens_train[i], max_value, min_value)

    for i in range(len(imagens_valid)):
        imagens_valid[i] = preprocess(imagens_valid[i], max_value, min_value)

    for i in range(len(imagens_test)):
        imagens_test[i] = preprocess(imagens_test[i], max_value, min_value)

    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test


def format_data(directory_raw, output_dir = "output_dir", exclude = False, exclude_path = ""):
    for angle in ["Frontal", "Right45", "Right90", "Left45", "Left90"]:

        if exclude:
            list = set()

            list = listar_imgs_nao_usadas(exclude_path, angle)
            
            imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = to_array(f"raw_dataset/{angle}", list)

        else:
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

        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}")

        np.save(f"{output_dir}/imagens_train_{angle}.npy", imagens_train)
        np.save(f"{output_dir}/labels_train_{angle}.npy", labels_train)
        np.save(f"{output_dir}/imagens_valid_{angle}.npy", imagens_valid)
        np.save(f"{output_dir}/labels_valid_{angle}.npy", labels_valid)
        np.save(f"{output_dir}/imagens_test_{angle}.npy", imagens_test)
        np.save(f"{output_dir}/labels_test_{angle}.npy", labels_test)    
    

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

    
# Função para salvar imagens e rótulos em arquivos NumPy
def save_numpy_data(images, labels, output_dir, position):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvar imagens e labels em arquivos separados
    np.save(os.path.join(output_dir, f"imagens_train_aug_{position}"), images)
    np.save(os.path.join(output_dir, f"labels_train_aug_{position}"), labels)
    print(f"Imagens e labels salvos em {output_dir}")


# Funções lambda para cada transformação separada
random_flip = lambda: keras.layers.RandomFlip("horizontal")
random_rotation = lambda: keras.layers.RandomRotation(0.05)
random_zoom = lambda: keras.layers.RandomZoom(0.3, 0.5)
random_brightness = lambda: keras.layers.RandomBrightness(factor=0.3, value_range=(0.0, 1.0))
random_contrast = lambda: keras.layers.RandomContrast(factor=0.3)

# Função para aplicar as transformações individualmente
def apply_transformation(image, transformation):
    return transformation()(image)

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
        i = 0
        for image, label in zip(train, labels):
            for transformation in transformations:  # Aplicar cada transformação separadamente
                augmented_image = apply_transformation(image, transformation)
                
                # Aleatoriamente decidir se vai aplicar random_flip
                if random.random() > 0.5:  # 50% de chance de aplicar o flip
                    augmented_image = apply_transformation(augmented_image, random_flip)
                

                all_images.append(augmented_image)
                all_labels.append(label)
                
                # Teste para verificar se estava aplicando a mesma transformação em todas as imagens e verficar
                # Reprodutiilidade da transformação com uma mesma sememente
                """
                if transformation == random_rotation and i < 10:
                    plt.subplot(1, 2, 1)
                    plt.imshow(image)
                    plt.axis("off")
                    plt.title("Original Image")
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(augmented_image)
                    plt.axis("off")
                    plt.title("Rotated Image")
                    plt.savefig(f"foto_aug3.{i}.png")

                    i += 1
                """
    
    #convertendo em np
    all_images = np.array([img for img in all_images])
    all_labels = np.array(all_labels)
    
    #se passado como parametro
    if resize:
        all_images = tf.image.resize_with_pad(all_images, target_size, target_size, method="bicubic")
    
    
    all_images = np.squeeze(all_images, axis=-1)
        
    print(all_images.shape)
    print(all_labels[all_labels == 1].shape)
    print(all_labels[all_labels == 0].shape)
    
    #teste
    #visualize_augmentation(all_images[:10], all_images[156:166], 10)
    
    
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


def create_aug_dataset(val_aug, output_dir="dataset_aug"):
    
    angles_list = ["Frontal", "Left45", "Left90", "Right45", "Right90"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for position in angles_list:
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(position)
        imagens_train, labels_train = apply_augmentation_and_expand(imagens_train, labels_train, val_aug, resize=False)
        
        #salvar imagens e labels em arquivos separados
        np.save(os.path.join(output_dir, f"imagens_train_{position}"), imagens_train)
        np.save(os.path.join(output_dir, f"labels_train_{position}"), labels_train)
                
        np.save(os.path.join(output_dir, f"imagens_valid_{position}"), imagens_valid)
        np.save(os.path.join(output_dir, f"labels_valid_{position}"), labels_valid)
        
        np.save(os.path.join(output_dir, f"imagens_test_{position}"), imagens_test)
        np.save(os.path.join(output_dir, f"labels_test_{position}"), labels_test)
        
        #verificação
        print(imagens_train.shape)
        print(labels_train.shape)






def filtrar_imgs_masks(angulo, img_path, mask_path):
    """
    Filtra imagens e máscaras baseado no ângulo especificado.
    Retorna listas de caminhos das imagens e máscaras.
    """
    angulos = {
        "Frontal": "1",
        "Left45": "4",
        "Right45": "2",
        "Left90": "5",
        "Right90": "3"
    }

    idx_angle = angulos.get(angulo)
    if not idx_angle:
        raise ValueError(f"Ângulo inválido: {angulo}")

    pattern = re.compile(f".*{idx_angle}\.S.*")

    imgs_files = sorted([img for img in os.listdir(img_path) if pattern.match(img)])
    masks_files = sorted([mask for mask in os.listdir(mask_path) if pattern.match(mask)])

    assert len(imgs_files) == len(masks_files), "Número de imagens e máscaras não corresponde!"

    data_imgs = [os.path.join(img_path, img) for img in imgs_files]
    data_masks = [os.path.join(mask_path, mask) for mask in masks_files]

    return data_imgs, data_masks


def load_imgs_masks(angulo, img_path, mask_path):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imgs_train, imgs_valid, masks_train, masks_valid.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]
    mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        imagens, mascaras, test_size=0.2, random_state=42
    )

    return np.array(imgs_train), np.array(imgs_valid), np.array(masks_train), np.array(masks_valid)

def load_imgs_masks_only(angulo, img_path, mask_path):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imagens e masks
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]
    mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    return np.array(imagens), np.array(mascaras)


def criar_pastas_yolo():
    """
    Cria a estrutura de diretórios para armazenar as imagens e máscaras do YOLO.
    """
    pastas = [
        "Yolo_dataset",
        "Yolo_dataset/images",
        "Yolo_dataset/images/train",
        "Yolo_dataset/images/val",
        "Yolo_dataset/masks",
        "Yolo_dataset/masks/train",
        "Yolo_dataset/masks/val",
    ]
    
    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)


def mover_arquivos_yolo(imgs_train, imgs_valid, masks_train, masks_valid):
    """
    Move imagens e máscaras para os diretórios adequados dentro de Yolo_dataset.
    """
    criar_pastas_yolo()

    for img in imgs_train:
        shutil.copy(img, "Yolo_dataset/images/train")
    for img in imgs_valid:
        shutil.copy(img, "Yolo_dataset/images/val")
    for mask in masks_train:
        shutil.copy(mask, "Yolo_dataset/masks/train")
    for mask in masks_valid:
        shutil.copy(mask, "Yolo_dataset/masks/val")


def YoLo_Data(angulo, img_path, mask_path):
    """
    Prepara os dados de imagens e máscaras e move para a estrutura YOLO.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        data_imgs, data_masks, test_size=0.2, random_state=42
    )

    mover_arquivos_yolo(imgs_train, imgs_valid, masks_train, masks_valid)
    
    input_dir = 'Yolo_dataset/masks/val'
    output_dir = 'Yolo_dataset/labels/val'

    masks_to_polygons(input_dir, output_dir)



def view_pred_mask(model, img):


    imagem = np.expand_dims(img, axis=0)
    imagem = np.expand_dims(img, axis=-1)

    pred = model.predict(imagem)

    pred = np.squeeze(pred, axis=0)

    mask = (pred > 0.5).astype(np.uint8)
    

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig("unet_pred_TESTE.png")
    plt.close()

import os
import cv2


def masks_to_polygons(input_dir, output_dir):
    
    for j in os.listdir(input_dir):
        image_path = os.path.join(input_dir, j)
        # load the binary mask and get its contours
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        H, W = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        # print the polygons
        with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))

            f.close()


#Listar os nomes das imagens que não quero usar para treinar o modelo de classificação

def listar_imgs_nao_usadas(directory, angulo):
    angulos = {
        "Frontal": "1",
        "Left45": "4",
        "Right45": "2",
        "Left90": "5",
        "Right90": "3"
    }

    idx_angle = angulos.get(angulo)
    if not idx_angle:
        raise ValueError(f"Ângulo inválido: {angulo}")
    
    pattern = re.compile(f".*{idx_angle}\.S.*")
    imgs_files = sorted([img for img in os.listdir(directory) if pattern.match(img)])
    nomesexcluidos = set()
    for file in imgs_files:
        # Extrair número do paciente e data do nome da segmentação (Ex: T0337.2.1.S.2019-11-13.00)
        match = re.search(r"T(\d+).*?(\d{4}-\d{2}-\d{2})", file)
        if match:
            paciente, data = match.groups()
            if angulo == "Frontal":
                nome_adequado = f"{int(paciente)}_img_Static-{angulo}_{data}.txt"  # Converte para o formato do raw_dataset
            else:
                nome_adequado = f"{int(paciente)}_img_Static-{angulo}°_{data}.txt"
            nomesexcluidos.add(nome_adequado)
    return nomesexcluidos





       



 




    