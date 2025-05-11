import cv2
from ultralytics import YOLO
from include.imports import *
from utils.data_prep import load_imgs_masks, YoLo_Data, masks_to_polygons,load_imgs_masks_only, copy_images_excluding_patients, filter_dataset_by_id, load_raw_images,make_tvt_splits, augment_train_fold, normalize
from utils.files_manipulation import move_files_within_folder, create_folder
from src.models.yolo_seg import train_yolo_seg
from src.models.u_net import unet_model
from utils.stats import precision_score_, recall_score_, accuracy_score_, dice_coef_, iou_ 
import json

# Use o tempo atual em segundos como semente
##VALUE_SEED = int(time.time() * 1000) % 15000
"""
VALUE_SEED = 7758
random.seed(VALUE_SEED)

seed = random.randint(0, 15000)

tf.random.set_seed(seed)

np.random.seed(seed)
"""

def train_models_cv(models, raw_root , message, angle = "Frontal", k = 5, 
                    resize = False, resize_to = 224, n_aug = 1, batch = 8, seed = 42):
    
    X, y , patient_ids = load_raw_images(
        os.path.join(raw_root, angle),
        resize_to=resize_to, interp='bicubic')
    

    for fold, (tr_idx, va_idx, te_idx) in enumerate(
                        make_tvt_splits(X, y, patient_ids,
                                        k=k, val_size=0.2, seed=seed)):
        # salva índices para reuso posterior
        split_file = f"splits/{message}_{angle}_F{fold}.json"
        os.makedirs("splits", exist_ok=True)
        with open(split_file, "w") as f:
            json.dump({
                "train_idx": tr_idx.tolist(),
                "val_idx":   va_idx.tolist(),
                "test_idx":  te_idx.tolist()
            }, f)

        # ------ prepara dados -----------
        X_tr_orig, y_tr = X[tr_idx], y[tr_idx]
        X_val,    y_val = X[va_idx], y[va_idx]
        X_test,   y_test= X[te_idx], y[te_idx]

        # min/max APENAS dos originais
        mn, mx = X_tr_orig.min(), X_tr_orig.max()

        # augmenta & concatena
        X_tr_aug, y_tr_aug = augment_train_fold(X_tr_orig, y_tr,
                                                n_aug=n_aug, seed=fold)

        # normaliza
        X_tr = normalize(X_tr_aug, mn, mx)
        X_val= normalize(X_val,    mn, mx)
        X_test=normalize(X_test,   mn, mx)

        # ------ loop de modelos ---------
        for model_fn in models:
            model   = model_fn()
            ckpt    = f"modelos/{model_fn.__name__}/{message}_{angle}_F{fold}.h5"
            log_txt = f"history/{model_fn.__name__}/{message}_{angle}.txt"
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            os.makedirs(os.path.dirname(log_txt), exist_ok=True)

            model.fit(X_tr, y_tr_aug,
                      epochs=500,
                      validation_data=(X_val, y_val),
                      batch_size=batch,
                      callbacks=[
                          tf.keras.callbacks.EarlyStopping(
                              monitor='val_loss', patience=50,
                              min_delta=0.01, restore_best_weights=True),
                          tf.keras.callbacks.ModelCheckpoint(
                              ckpt, monitor='val_loss',
                              save_best_only=True)
                      ],
                      verbose=2, shuffle=True)

            # ---------- avaliação ----------
            y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()

            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                                   y_test, y_pred, average="binary",
                                   zero_division=0)

            # salva métrica fold‐a‐fold
            with open(log_txt, "a") as f:
                f.write(f"Fold {fold:02d}  "
                        f"Acc={acc:.4f}  "
                        f"Prec={prec:.4f}  "
                        f"Rec={rec:.4f}  "
                        f"F1={f1:.4f}\n")











def evaluate_segmentation(model_path, x_val, y_val):
    model = tf.keras.models.load_model(model_path)
    pred    = (model.predict(x_val) > 0.5).astype(np.uint8)
    true    = (y_val > 0.5).astype(np.uint8)

    metrics = {
        "precision": precision_score_(true, pred),
        "recall"   : recall_score_(true, pred),
        "accuracy" : accuracy_score_(true, pred),
        "dice"     : dice_coef_(true, pred),
        "iou"      : iou_(true, pred)
    }
    return metrics






#Função para transformar as imagens de .txt para .jpg
def raw_dataset_to_png(directory, exclude = False, exclude_set = None):

    arquivos = os.listdir(directory)

    print(arquivos)

    healthy_path = os.path.join(directory, 'healthy')
    sick_path = os.path.join(directory, 'sick')

    healthy = os.listdir(healthy_path)
    sick = os.listdir(sick_path)

    imagens = []
    labels = []
    ids = []

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

          os.makedirs("raw_png_dataset/Frontal/healthy", exist_ok=True)

          #Salvar a imagem como .png
          plt.imsave(os.path.join("raw_png_dataset/Frontal/healthy", arquivo.replace('.txt', '.png')), imagem, cmap='gray')
          
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

          os.makedirs("raw_png_dataset/Frontal/sick", exist_ok=True)
          
          plt.imsave(os.path.join("raw_png_dataset/Frontal/sick", arquivo.replace('.txt', '.png')), imagem, cmap='gray')

        except ValueError as e:
          print(e)
          print(arquivo)
          continue


def train_models(models_objects, dataset: str, resize=False, target = 0, message=""):
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

                #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=15, min_lr=1e-5, min_delta=0.0001)

                #criando objeto e usando o modelo
                model = model_func()

                model.summary()
                

                # Salva a seed em um arquivo de texto
                with open("modelos/random_seed.txt", "a") as file:
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
                    
                plot_convergence(history, model_func.__name__, angulo, i, message)

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


    """
    imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", True)
    YoLo_Data("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", "Yolo_dataset_aug_10_05", True)

    VALUE_SEED = int(time.time()*1000) % 15000
    random.seed(VALUE_SEED)

    print(f"Valor da semente: {VALUE_SEED}")
    with open("modelos/random_seed.txt", "a") as f:
        f.write(f"Valor da semente para treinar modelos com aumento de dados: {VALUE_SEED}\n")
                
    seed = random.randint(0,15000)  
                
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    model = unet_model()

    model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

    checkpoint = tf.keras.callbacks.ModelCheckpoint("modelos/unet/Frontal_Unet_AUG_V10_05.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')

    history = model.fit(imgs_train, masks_train, epochs = 500, validation_data= (imgs_valid, masks_valid), callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)

    # Gráfico de perda de treinamento
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'Training Loss Convergence for unet - Frontal')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_loss_convergence_Frontal_10_05_AUG.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Validation Loss Convergence for unet - Frontal')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_val_loss_convergence_Frontal10_05_AUG.png")
    plt.close()

    """


    VALUE_SEED = int(time.time()*1000) % 15000
    random.seed(VALUE_SEED)
    seed = random.randint(0,15000)

    semente = seed
    #train_yolo_seg("x", 500, "dataset.yaml", 224, seed=semente)
    #train_yolo_seg("n", 500, "dataset.yaml", 224, seed=semente)
    YoLo_Data("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", "Yolo_dataset_10_05", False)
    train_yolo_seg("x", 500, "dataset2.yaml", 224, seed=semente)
    train_yolo_seg("n", 500, "dataset2.yaml", 224, seed=semente)


    # -----------------------------------------------------------------------------------------
    imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", False)

    VALUE_SEED = int(time.time()*1000) % 15000
    random.seed(VALUE_SEED)

    with open("modelos/random_seed.txt", "a") as f:
        f.write(f"Valor da semente 2 para treinar modelos sem aumento de dados: {VALUE_SEED}\n")

    print(f"Valor da semente: {VALUE_SEED}")
                
    seed = random.randint(0,15000)  
                
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    model = unet_model()

    model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

    checkpoint = tf.keras.callbacks.ModelCheckpoint("modelos/unet/Frontal_Unet_V10_05.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')

    history = model.fit(imgs_train, masks_train, epochs = 500, validation_data= (imgs_valid, masks_valid), callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)

    # Gráfico de perda de treinamento
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'Training Loss Convergence for unet - Frontal')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_loss_convergence_Frontal_10_05.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Validation Loss Convergence for unet - Frontal')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_val_loss_convergence_Frontal_10_05.png")
    plt.close()


    semente = seed
    """
    """



    




    
    
    
    
    
    """
    original = "raw_dataset/Frontal"
    destino = "filtered_raw_dataset/Frontal"
    ids_para_remover = [
    10, 47, 94, 109, 114, 141, 156, 185, 197, 206,
    242, 258, 346, 363, 376, 400]
    
    filter_dataset_by_id(original, destino, ids_para_remover)
    """
    
    #raw_dataset_to_png("raw_dataset/Frontal")
     
    # 1. Treinamento do modelo UNET com o dataset AUMENTADO



    """
    """




    ### 2. Treinamento do modelo YoloX-seg com o dataset AUMENTADO



    #YoLo_Data("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", "Yolo_dataset_aug", True)
    #train_yolo_seg("x", 500, "dataset.yaml", 224, seed=7758)

    ### 3. Aumentando np_dataset_v2


    #create_aug_dataset(1, "np_dataset_v2","AUG_dataset_v2")



    # 4. Segmentando o AUG_dataset_v2 com a UNET



    """
    model = tf.keras.models.load_model("modelos/unet/Frontal_Unet_AUG.h5")

    data_train, labels_train, data_valid, labels_valid, data_test, labels_test = load_data("Frontal","AUG_dataset_v2")

    predictions = model.predict(data_train, batch_size = 4)

    masks = (predictions > 0.5).astype(np.uint8)

    masks = np.squeeze(masks, axis=-1)

    segmented_images = data_train * masks

    os.makedirs("Unet_dataset_AUG", exist_ok=True)

    np.save(f"Unet_dataset_AUG/imagens_train_Frontal.npy", segmented_images)
    np.save(f"Unet_dataset_AUG/labels_train_Frontal.npy", labels_train)
    np.save(f"Unet_dataset_AUG/imagens_valid_Frontal.npy", data_valid)
    np.save(f"Unet_dataset_AUG/labels_valid_Frontal.npy", labels_valid)
    np.save(f"Unet_dataset_AUG/imagens_test_Frontal.npy", data_test)
    np.save(f"Unet_dataset_AUG/labels_test_Frontal.npy", labels_test)
    """

    #######  5. Carregamento do modelo Yolo e Segmentação do dataset AUG_dataset_v2 #######

    """

    model_path = "runs/segment/train13/weights/best.pt"

    model = YOLO(model_path)

    data_train, labels_train, data_valid, labels_valid, data_test, labels_test = load_data("Frontal","AUG_dataset_v2")

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

    os.makedirs("Yolo_dataset_AUG", exist_ok=True)

    np.save(f"Yolo_dataset_AUG/imagens_train_Frontal.npy", segmented_images)
    np.save(f"Yolo_dataset_AUG/labels_train_Frontal.npy", labels_train)
    np.save(f"Yolo_dataset_AUG/imagens_valid_Frontal.npy", data_valid)
    np.save(f"Yolo_dataset_AUG/labels_valid_Frontal.npy", labels_valid)
    np.save(f"Yolo_dataset_AUG/imagens_test_Frontal.npy", data_test)
    np.save(f"Yolo_dataset_AUG/labels_test_Frontal.npy", labels_test)


    print("Segmentação concluída e dataset salvo!")
    """

    #### Código para remover 3 canais e "Renormalizar imagens" depois da segmentação do dataset utilizando a Yolo #####

    """
        a, b, c, d, e, f = load_data("Frontal", "Yolo_dataset_AUG")
        
        array = []

    # Exemplo hipotético de loop processando imagens
        for i, img in enumerate(a):
        # Se a imagem tiver 3 canais (H, W, 3), converta para cinza
        if img.ndim == 3 and img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (H, W)
            gray = np.expand_dims(gray, axis=-1)          # (H, W, 1)
            img = gray
            img = (img / 255.0).astype(np.float32)
        
        # Se ela já tiver shape (H, W, 1), não precisa fazer nada
        # Se estiver em 2D (H, W) e você quiser (H, W, 1), basta expandir a dimensão
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        
        # Agora 'img' tem shape (H, W, 1), pronto para o modelo de classificação
        # ...

        array.append(img)

        array = np.array(array).squeeze(axis=-1)

        print(array.shape)

        np.save("Yolo_dataset_AUG/imagens_train_Frontal.npy", array)

        """
    


    # 6. Treinamento do modelo ResNet34 com os datasets 
    #train_models([ResNet34], "Unet_dataset_AUG", resize=True, target=224, message="AUG_Unet_ResNet34_224_batch_8")
    #train_models([ResNet34], "Yolo_dataset_AUG", resize=True, target=224, message="AUG_ResNet34_224_batch_8")
    #train_models([ResNet34], "AUG_dataset_v2", resize=True, target=224, message="AUG_ResNet34V2_224_batch_8")


"""

    
    idx = 84
    
    img = imgs_train[idx]
    mask = masks_train[idx]

    

    mask = (mask > 0.5).astype(np.uint8)

    img_seg = img * mask

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title('Máscara')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_seg)
    plt.title('Imagem Segmentada')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("segmentada.png")

    """

    # Caminhos da imagem e máscara específicas


"""
    caminho_img = "TESTEAUG/images/train/aug_0142.png"
    caminho_mask = "TESTEAUG/masks/train/aug_0142.png"

    # 1. Carrega imagem e máscara como arrays normalizados [0, 1]
    img = np.array(Image.open(caminho_img).convert("L")) / 255.0
    mask = np.array(Image.open(caminho_mask).convert("L")) / 255.0

    # 2. Converte a máscara para binária
    mask = (mask > 0.5).astype(np.uint8)

    # 3. Aplica a máscara à imagem

    img_segmentada = img * mask

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_segmentada, cmap='gray')
    plt.title('Imagem Segmentada')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig("segmentada.png")
       


    """

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