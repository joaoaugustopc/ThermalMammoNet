import cv2
from ultralytics import YOLO
from include.imports import *
from utils.data_prep import load_imgs_masks, YoLo_Data, masks_to_polygons,load_imgs_masks_only, copy_images_excluding_patients
from utils.files_manipulation import move_files_within_folder, create_folder
from src.models.yolo_seg import train_yolo_seg
from src.models.u_net import unet_model

# Use o tempo atual em segundos como semente
##VALUE_SEED = int(time.time() * 1000) % 15000
"""
VALUE_SEED = 7758
random.seed(VALUE_SEED)

seed = random.randint(0, 15000)

tf.random.set_seed(seed)

np.random.seed(seed)
"""



print("***SEMENTE USADA****:", VALUE_SEED)

# Salvar a semente em um arquivo de texto
with open("seed_global", "w") as seed_file:
    seed_file.write(f"VALUE_SEED: {VALUE_SEED}\n")
    seed_file.write(f"Random Seed: {seed}\n")


def train_models(models_objects, dataset: str, resize=False, target = 0, message=""):
    list = ["Frontal"]
    list = ["Frontal"]
    models = models_objects
                
    for angulo in list:
        

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo, dataset)
        
        if resize:
            # Para ajustar a dimensão das imagens para o modelo
            # Add uma dimensão para o canal de cor para o tf.image.resize_with_pad -> isso é por causa das dimensões do numpy
            imagens_train = np.expand_dims(imagens_train, axis=-1)
            imagens_valid = np.expand_dims(imagens_valid, axis=-1) 
            imagens_test = np.expand_dims(imagens_test, axis=-1)

            imagens_train = tf.image.resize_with_pad(imagens_train, target, target, method="bilinear")
            imagens_valid = tf.image.resize_with_pad(imagens_valid, target, target, method="bilinear")
            imagens_test = tf.image.resize_with_pad(imagens_test, target, target, method="bilinear")
            
            # Remover a dimensão do canal de cor
            imagens_train = np.squeeze(imagens_train, axis=-1)
            imagens_valid = np.squeeze(imagens_valid, axis=-1)       
            imagens_test = np.squeeze(imagens_test, axis=-1)
        
        #treinando cada modelo da lista
        for model_func in models_objects:

            #criando pasta com gráficos dos modelos
            os.makedirs(f"history/{model_func.__name__}", exist_ok=True)
            
            with open(f"modelos/random_seed.txt", "a") as f:
                f.write(f"Modelo:{model_func.__name__}\n")
                f.write(f"Angulo: {angulo}\n")

            for i in range(10):
                
                VALUE_SEED = int(time.time()*1000) % 15000
                random.seed(VALUE_SEED)
                
                seed = random.randint(0,15000)  
                
                tf.keras.utils.set_random_seed(seed)
                tf.config.experimental.enable_op_determinism()
                
                print(f"history/{model_func.__name__}/{model_func.__name__}_{angulo}_{i}_time.txt")
                
                start_time = time.time()
                
                checkpoint_path = f"modelos/{model_func.__name__}/{message}_{angulo}_{i}.h5"
                checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

                #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=15, min_lr=1e-5, min_delta=0.0001)

                #criando objeto e usando o modelo
                model = model_func()

                model.summary()
                

                # Salva a seed em um arquivo de texto
                with open("random_seed_vgg.txt", "a") as file:
                    file.write(str(seed))
                    file.write("\n")

                print("Seed gerada e salva em random_seed.txt:", seed)
                

                history = model.fit(imagens_train, labels_train, epochs = 500, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)
                
                end_time = time.time()
                
                # Avaliação do modelo com conjunto de teste
                if model_func.__name__ == "ResNet34":
                    with custom_object_scope({'ResidualUnit': ResidualUnit}):
                        best_model = keras.models.load_model(checkpoint_path)
                elif model_func.__name__ == "ResNet101":
                    with custom_object_scope({'BottleneckResidualUnit': BottleneckResidualUnit}):
                        best_model = keras.models.load_model(checkpoint_path)
                else:
                    best_model = keras.models.load_model(checkpoint_path)

                test_loss, test_accuracy = best_model.evaluate(imagens_test, labels_test, verbose=1)

                directory = f"history/{model_func.__name__}/{angulo}/treinamento/"
                os.makedirs(directory, exist_ok=True)

                with open(f"{directory}/{message}_{i}_time.txt", "w") as f:
                    f.write(f"Modelo: {model_func.__name__}\n")
                    f.write(f"Tempo de execução: {end_time - start_time}\n")
                    f.write(f"Loss: {history.history['loss']}\n")
                    f.write(f"Val_loss: {history.history['val_loss']}\n")
                    f.write(f"Accuracy: {history.history['accuracy']}\n")
                    f.write(f"Val_accuracy: {history.history['val_accuracy']}\n")
                    f.write(f"Test Loss: {test_loss}\n")
                    f.write(f"Test Accuracy: {test_accuracy}\n")
                    f.write(f"SEED{str(seed)}\n")
                    f.write("\n")
                    
                plot_convergence(history, model_func.__name__, angulo, i, "Vgg_16")

#TODO: calcular f1, IoU, recall, accuracy
#TODO: comparar unet com yolo

import cv2
from ultralytics import YOLO


# Função para calcular Pixel Accuracy
def pixel_accuracy(y_true, y_pred):
    # y_true e y_pred devem ter a mesma dimensão e serem binários
    correct = np.sum(y_true == y_pred)
    total = y_true.size
    return correct / total

import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1):
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    return dice

def txt_to_image(txt_file, output_image_path):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    # Substitui espaços por ponto e vírgula em cada linha
    lines = [line.replace(' ', ';') for line in lines]
    
    # Converte cada linha em uma lista de floats
    data = [list(map(float, line.strip().split(';'))) for line in lines]
    
    # Converte a lista de listas em uma matriz numpy
    image_array = np.array(data, dtype=np.float32)
    
    # Normaliza os valores para a faixa [0, 255]
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
    
    # Converte a matriz para uint8
    image_array = image_array.astype(np.uint8)
    
    # Salva a matriz como uma imagem
    image = Image.fromarray(image_array)
    image.save(output_image_path)
    print(f"Imagem salva em: {output_image_path}")
    

def transform_channels_normalize(angle, origin_folder, result_folder):
    """
    Converte imagens com 3 canais para escala de cinza, normaliza e salva em novos arquivos .npy.
    
    Parâmetros:
    - angle: string usada para identificar os arquivos (ex: "Frontal")
    - origin_folder: diretório onde estão os arquivos .npy originais
    - result_folder: diretório onde os novos arquivos serão salvos
    """
    imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angle, origin_folder)

    def processar_imagens(imagens):
        array = []
        for img in imagens:
            # Converte RGB para cinza
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (H, W)
            # Se já está em (H, W, 1) ou (H, W), deixamos como (H, W)
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(axis=-1)
            # Normaliza
            img = (img / 255.0).astype(np.float32)
            array.append(img)
        return np.array(array)

    # Processa os conjuntos
    imagens_train = processar_imagens(imagens_train)
    imagens_valid = processar_imagens(imagens_valid)
    imagens_test = processar_imagens(imagens_test)

    # Garante que o diretório de destino exista
    os.makedirs(result_folder, exist_ok=True)

    # Salva os arquivos
    np.save(f"{result_folder}/imagens_train_{angle}.npy", imagens_train)
    np.save(f"{result_folder}/labels_train_{angle}.npy", labels_train)
    np.save(f"{result_folder}/imagens_valid_{angle}.npy", imagens_valid)
    np.save(f"{result_folder}/labels_valid_{angle}.npy", labels_valid)
    np.save(f"{result_folder}/imagens_test_{angle}.npy", imagens_test)
    np.save(f"{result_folder}/labels_test_{angle}.npy", labels_test)

    print(f"Arquivos processados e salvos em '{result_folder}' para o ângulo '{angle}'.")

"""
    model_path: caminho do modelo da yolo, como 'runs/segment/train6/weights/best.pt'

"""
def segment_and_save_numpydataset(model_path, input_dir, output_dir, view):
    print(f"Iniciando segmentação para a visão: {view}")
    
    # Carregar o modelo
    model = YOLO(model_path)

    # Carregar dados (ajuste essa função conforme seu projeto)
    data_train, labels_train, data_valid, labels_valid, data_test, labels_test = load_data(view, input_dir)

    segmented_images = []

    for img in data_train:
        # Garantir formato adequado
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)

        img_resized = cv2.resize(img, (224, 224))

        # YOLO predict
        results = model.predict(img_resized, verbose=False)

        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            mask_tensor = results[0].masks.data[0]
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (224, 224))
            binary_mask = (mask > 0.5).astype(np.uint8)

            if binary_mask.ndim == 2:
                binary_mask = np.expand_dims(binary_mask, axis=-1)

            segmented_img = img_resized * binary_mask
        else:
            segmented_img = img_resized
            
        segmented_images.append(segmented_img)

    segmented_images = np.array(segmented_images)

    # Criar pasta de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Salvar arquivos segmentados e originais
    np.save(os.path.join(output_dir, f"imagens_train_{view}.npy"), segmented_images)
    np.save(os.path.join(output_dir, f"labels_train_{view}.npy"), labels_train)
    np.save(os.path.join(output_dir, f"imagens_valid_{view}.npy"), data_valid)
    np.save(os.path.join(output_dir, f"labels_valid_{view}.npy"), labels_valid)
    np.save(os.path.join(output_dir, f"imagens_test_{view}.npy"), data_test)
    np.save(os.path.join(output_dir, f"labels_test_{view}.npy"), labels_test)

    print(f"Segmentação concluída e dataset salvo para: {view}")


"""
    Segmenta imagens convertidas de arquivos .txt ou .png usando um modelo YOLO e retorna imagens png.

    Parâmetros:
    model_path : str
        Caminho para o arquivo de pesos YOLOv8 (.pt) treinado para segmentação.
    input_dir : str
        Caminho para a pasta contendo arquivos .txt representando imagens.
    output_dir : str
        Caminho onde as imagens segmentadas (.png) serão salvas.
    ext_txt : str
        Extensão dos arquivos de entrada (padrão: '.txt').
    ext_img : str
        Extensão das imagens convertidas (padrão: '.png').

    Retorno:
    None. As imagens segmentadas são salvas no diretório `output_dir`.
"""
def segment_and_save_pngdataset(model_path, input_dir, output_dir, ext_txt=".txt", ext_img=".png"):
  
    model = YOLO(model_path)
    create_folder(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(ext_txt):
            txt_path = os.path.join(input_dir, file)
            img_path = os.path.join(input_dir, f"{os.path.splitext(file)[0]}{ext_img}")
            
            txt_to_image(txt_path, img_path)

            img = cv2.imread(img_path)
            if img is None:
                print(f"[Erro] Não foi possível carregar a imagem: {img_path}")
                continue

            H, W, _ = img.shape
            results = model(img)

            for result in results:
                if result.masks is None:
                    continue
                for j, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy() * 255
                    mask_resized = cv2.resize(mask_np, (W, H))
                    segmented = cv2.bitwise_and(img, img, mask=mask_resized.astype(np.uint8))
                    out_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_seg_{j}{ext_img}")
                    cv2.imwrite(out_path, segmented)
                    print(f"[Salvo] {out_path}")


if __name__ == "__main__":
    ##train12 - Yolov8x-seg 
    #train_yolo_seg()

    #train_models([ResNet34], "Unet_dataset", resize=True, target=224, message="Unet_ResNet34_224_batch_8")
    #train_models([ResNet34], "Yolo_dataset", resize=True, target=224, message="ResNet34_224_batch_8")
    #train_models([ResNet34], "np_dataset_v2", resize=True, target=224, message="ResNet34V2_224_batch_8")

    move_files_to_folder(["Frontal_Unet.h5"],"modelos/unet")
    



    # Código utilizado para segmentar o dataset utilizado o YOLO-seg
    """
    model_path = "runs/segment/train12/weights/best.pt"

    model = YOLO(model_path)

    data_train, labels_train, data_valid, labels_valid, data_test, labels_test = load_data("Frontal","np_dataset_v2")
    
    segmented_images = []

    for img in data_train:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # Se a imagem for 2D (sem canal), converte para 3 canais
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Se a imagem tiver 1 canal (shape: (480,640,1)), replica o canal para formar 3 canais
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        # Redimensionar para 224x224
        img_resized = cv2.resize(img, (224, 224))
                
        # Obter as predições do YOLO-seg
        results = model.predict(img_resized)
        
        if results and len(results[0].masks) > 0:
            # Seleciona a primeira máscara (a de maior probabilidade)
            masks = results[0].masks
            mask_tensor = masks.data[0]
            mask = mask_tensor.cpu().numpy()

            # Redimensionar a máscara para 224x224
            if mask.shape[:2] != (224, 224):
                mask = cv2.resize(mask, (224, 224))

            # Converter a máscara para binária (threshold = 0.5)
            binary_mask = (mask > 0.5).astype(np.uint8)
            if binary_mask.ndim == 2:
                binary_mask = np.expand_dims(binary_mask, axis=-1)

            # Aplicar a máscara à imagem (multiplicação pixel a pixel)
            segmented_img = img_resized * binary_mask
        else:
            segmented_img = img_resized

        segmented_images.append(segmented_img)

    segmented_images = np.array(segmented_images)

    os.makedirs("Yolo_dataset", exist_ok=True)

    np.save(f"Yolo_dataset/imagens_train_Frontal.npy", segmented_images)
    np.save(f"Yolo_dataset/labels_train_Frontal.npy", labels_train)
    np.save(f"Yolo_dataset/imagens_valid_Frontal.npy", data_valid)
    np.save(f"Yolo_dataset/labels_valid_Frontal.npy", labels_valid)
    np.save(f"Yolo_dataset/imagens_test_Frontal.npy", data_test)
    np.save(f"Yolo_dataset/labels_test_Frontal.npy", labels_test)


    print("Segmentação concluída e dataset salvo!")
    """