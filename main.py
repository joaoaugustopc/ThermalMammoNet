import cv2
from ultralytics import YOLO
from include.imports import *
from utils.data_prep import load_imgs_masks, YoLo_Data, masks_to_polygons,load_imgs_masks_only, copy_images_excluding_patients
from utils.files_manipulation import move_files_within_folder, create_folder
from src.models.yolo_seg import train_yolo_seg
from src.models.u_net import unet_model
from src.models.Vgg_16 import Vgg_16

# Use o tempo atual em segundos como semente
VALUE_SEED = int(time.time() * 1000) % 15000

random.seed(VALUE_SEED)

seed = random.randint(0, 15000)

tf.random.set_seed(seed)

np.random.seed(seed)

print("***SEMENTE USADA****:", VALUE_SEED)

# Salvar a semente em um arquivo de texto
with open("seed_global", "w") as seed_file:
    seed_file.write(f"VALUE_SEED: {VALUE_SEED}\n")
    seed_file.write(f"Random Seed: {seed}\n")


def train_models(model, dataset: str, resize=False, target = 0, message="", learning_rate=0.00001):
    list = ["Frontal"]
                
    for angulo in list:

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo, dataset)
        
        if resize:
            # Para ajustar a dimensão das imagens para o modelo
            # Add uma dimensão para o canal de cor para o tf.image.resize_with_pad -> isso é por causa das dimensões do numpy
            imagens_train = np.expand_dims(imagens_train, axis=-1)
            imagens_valid = np.expand_dims(imagens_valid, axis=-1) 
            imagens_test = np.expand_dims(imagens_test, axis=-1)

            imagens_train = tf.image.resize_with_pad(imagens_train, target, target, method="bicubic")
            imagens_valid = tf.image.resize_with_pad(imagens_valid, target, target, method="bicubic")
            imagens_test = tf.image.resize_with_pad(imagens_test, target, target, method="bicubic")
            
            # Remover a dimensão do canal de cor
            imagens_train = np.squeeze(imagens_train, axis=-1)
            imagens_valid = np.squeeze(imagens_valid, axis=-1)       
            imagens_test = np.squeeze(imagens_test, axis=-1)
        
        #treinando cada modelo da lista

        #criando pasta com gráficos dos modelos
        os.makedirs(f"history/unet_vgg16/{model.__name__}", exist_ok=True)
        
        with open(f"modelos/random_seed.txt", "a") as f:
            f.write(f"Modelo:{model.__name__}\n")
            f.write(f"Angulo: {angulo}\n")

        for i in range(10):
            
            VALUE_SEED = int(time.time()*1000) % 15000
            random.seed(VALUE_SEED)
            
            seed = random.randint(0,15000)  
            
            tf.keras.utils.set_random_seed(seed)
            tf.config.experimental.enable_op_determinism()
            
            print(f"history/unet_vgg16/{model.__name__}/{model.__name__}_{angulo}_{i}_time.txt")
            
            start_time = time.time()
            
            checkpoint_path = f"modelos/unet_vgg16/{model.__name__}/{message}_{angulo}_{i}.h5"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, 
                                                        save_weights_only=False, mode='auto')
            
            earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

            #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=15, min_lr=1e-5, min_delta=0.0001)

            #criando objeto e usando o modelo
            modelo_object = model(learning_rate=learning_rate)
            modelo_train = modelo_object.model

            
            # Salva a seed em um arquivo de texto
            with open("random_seed_vgg.txt", "a") as file:
                file.write(str(seed))
                file.write("\n")

            print("Seed gerada e salva em random_seed.txt:", seed)
    


            history = modelo_train.fit(imagens_train, labels_train, epochs = 500, validation_data= (imagens_valid, labels_valid),
                                callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)
            
            end_time = time.time()
            
            # Avaliação do modelo com conjunto de teste
            if model.__name__ == "ResNet34":
                with custom_object_scope({'ResidualUnit': ResidualUnit}):
                    best_model = keras.models.load_model(checkpoint_path)
            elif model.__name__ == "ResNet101":
                with custom_object_scope({'BottleneckResidualUnit': BottleneckResidualUnit}):
                    best_model = keras.models.load_model(checkpoint_path)
            else:
                best_model = keras.models.load_model(checkpoint_path)

            test_loss, test_accuracy = best_model.evaluate(imagens_test, labels_test, verbose=1)

            directory = f"history/unet-vgg16/{model.__name__}/{angulo}/treinamento/"
            os.makedirs(directory, exist_ok=True)


            with open(f"{directory}/{message}_{i}_time.txt", "w") as f:
                f.write(f"Modelo: {model.__name__}\n")
                f.write(f"Tempo de execução: {end_time - start_time}\n")
                f.write(f"Loss: {history.history['loss']}\n")
                f.write(f"Val_loss: {history.history['val_loss']}\n")
                f.write(f"Accuracy: {history.history['accuracy']}\n")
                f.write(f"Val_accuracy: {history.history['val_accuracy']}\n")
                f.write(f"Test Loss: {test_loss}\n")
                f.write(f"Test Accuracy: {test_accuracy}\n")
                f.write("\n")
                    
            plot_convergence(history,f"{model.__name__}_unet_vgg16", angulo, i, "Vgg_16_unet")
            
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
    
    
def analysis_dataset(npy_data, dir):
    
    os.makedirs(dir, exist_ok=True)
    dataset = np.load(npy_data)
    
    print(dataset.shape)
    
    cont = 0
    for img in dataset:
        # Desnormaliza (de [0, 1] para [0, 255]) e converte para uint8
        img_uint8 = (img * 255).astype(np.uint8)

        filename = os.path.join(dir, f"imagem_{cont}.png")
        # Salva como imagem
        cv2.imwrite(filename, img_uint8)
        cont += 1

        


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

"""
    Verificar se o dataset está normalizado
"""
def test_normalize(dataset):
    data = np.load(dataset)
    
    if np.min(data) >= 0 and np.max(data) <= 1:
        print("Normalizado")
    else:
        print("Nao normalizado")

if __name__ == "__main__":

    

    # format_data("raw_dataset", "np_dataset_v2", exclude=True, exclude_path="Termografias_Dataset_Segmentação/images")



    #U-Net
    ##################################################################################################

    ######Código que possibilita testar a U-NEt em imagens especificas e verificar de forma visual######
    """
    img_test = np.load("np_dataset/imagens_test_Frontal.npy")

    print(img_test.shape)

    img = img_test[22]
    origin = img

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    print(img.shape)


    model = tf.keras.models.load_model("modelos/unet/unet.h5")

    pred = model.predict(img)

    pred = np.squeeze(pred, axis=0)

    if pred.shape[-1] == 1:
        pred = pred[:, :, 0]
        mask = (pred > 0.5).astype(np.uint8)
    else:
        mask = np.argmax(pred, axis=-1)

    plt.figure(figsize=(10, 5))
    plt.imshow(origin, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig("unet_pred.png")
    plt.close()
    """

    #train_yolo_seg()
    
    #pred_yolo_seg()
    """
    img = cv2.imread("imgTESTE()2.jpg", cv2.IMREAD_GRAYSCALE)

    mask = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)

    mask = (mask > 0).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig("YOLO_pred_TESTE.png")
    plt.close()
    """
    """
    # Transformar uma imagem em numpy para jpg em 224x224 e salvar
    img_test = np.load("np_dataset/imagens_test_Frontal.npy")

    img = img_test[16]
    origin = img

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    print(img.shape)

    img = tf.image.resize_with_pad(img, 224, 224, method="bicubic")

    img = np.squeeze(img, axis=-1)
    img = np.squeeze(img, axis=0)

    # Converter a imagem para o modo 'L' (luminância)
    img = Image.fromarray((img * 255).astype(np.uint8), mode='L')

    img.save("ImgTESTE()2.jpg")  

    """
    #pred_yolo_seg()

    #train_yolo_seg()

    #YoLo_Data("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks")
    """
    angles = ["Frontal", "Left45", "Right45", "Left90", "Right90"]

    for angle in angles:
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angle, "aug_dataset")

        # Mudança para encaixar na rede (se necessário)
        
        imagens_test = np.expand_dims(imagens_test, axis=-1)
        imagens_test = tf.image.resize_with_pad(imagens_test, 224, 224, method="bicubic")
        imagens_test = np.squeeze(imagens_test, axis=-1)

        # Lista para armazenar as matrizes de confusão

        for i in range(10):
            i = i + 1
            # Carregar o modelo
            
            with custom_object_scope({'ResidualUnit': ResidualUnit}):
                model = tf.keras.models.load_model(f"modelos/ResNet34/ResNet34_224x224_{angle}_{i}.h5")

            #Avaliação do modelo

            loss, accuracy = model.evaluate(imagens_test, labels_test, verbose=0)

            print(f"Modelo {angle}_{i}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}")
            print("\n")            
    """    

    #load_data("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks")

    """
    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(imgs, masks, test_size=0.2, random_state=42)



    loss, acc = model.evaluate(imgs_valid, masks_valid, verbose=1)

    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")
    """

    """
    model = tf.keras.models.load_model("modelos/unet/unet.h5")
    
    imagem = Image.open("output/ImgTESTE.jpg").convert('L')

    imagem = np.array(imagem)

    imagem = np.expand_dims(imagem, axis=0)
    imagem = np.expand_dims(imagem, axis=-1)

    imagem = imagem / 255.0

    pred = model.predict(imagem)

    pred = np.squeeze(pred, axis=0)

    
    mask = (pred > 0.5).astype(np.uint8)

    

    plt.figure(figsize=(10, 5))
    plt.imshow(imagem[0], cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig("unet_pred_TESTE.png")
    plt.close()
    """

    ######train U-Net model######
    """
    #imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks("Left45", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks")
    
    model = unet_model()

    model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1, mode='auto')

    checkpoint = tf.keras.callbacks.ModelCheckpoint("modelos/unet/L45unet.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')

    history = model.fit(imgs_train, masks_train, epochs = 200, validation_data= (imgs_valid, masks_valid), callbacks= [checkpoint, earlystop], batch_size = 4, verbose = 1, shuffle = True)
    
    # Gráfico de perda de treinamento
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'Training Loss Convergence for unet - Frontal')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_loss_convergence_L45.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Validation Loss Convergence for unet - Frontal')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_val_loss_convergence_L45.png")
    plt.close()
    """


    ######Código para avaliar o modelo U-Net######
    """
    model = keras.models.load_model("modelos/unet/L45unet.h5")

    img_test = np.load("np_dataset/imagens_test_Left45.npy")

    print(img_test.shape)

    loss, acc = model.evaluate(imgs_valid, masks_valid, verbose=1)

    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")
    """
    ########################################################################################################
    #U-NET



    #YOLO
    ########################################################################################################
    
    #Função para montar dataset para o modelo YOLO
    """
    YoLo_Data("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks")
    """

    #Função para treinar o modelo YOLO
    """
    train_yolo_seg()
    """

    #Caminho para carregar o modelo YOLO
    """
    model_path = 'runs/segment/train9/weights/best.pt'
    """

    # Transformar uma imagem em numpy para jpg em 224x224 e salvar
    """
    img_test = np.load("np_dataset/imagens_test_Frontal.npy")

    img = img_test[16]
    origin = img

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    print(img.shape)

    img = tf.image.resize_with_pad(img, 224, 224, method="bicubic")

    img = np.squeeze(img, axis=-1)
    img = np.squeeze(img, axis=0)

    # Converter a imagem para o modo 'L' (luminância)
    img = Image.fromarray((img * 255).astype(np.uint8), mode='L')

    img.save("ImgTESTE()2.jpg")  
    """

    ########################################################################################################
    #YOLO


    
    
    #AVALIANDO YOLO E U-NET
    ########################################################################################################
    """
    imgs, masks = load_imgs_masks_only("Frontal", "Yolo_dataset/images/val", "Yolo_dataset/masks/val")

    print(imgs.shape)
    print(masks.shape)

    model = tf.keras.models.load_model("modelos/unet/unet.h5")

    pred = model.predict(imgs)

    print(pred.shape)

    pred_masks = (pred > 0.5).astype(np.uint8)

    pred_masks = np.squeeze(pred_masks, axis=-1)

    acuracies = []



    for i in range(len(masks)):
        #pred_masks[i]= np.squeeze(pred_masks[i], axis=-1)

        acc = dice_coefficient(masks[i], pred_masks[i])
        print(f"Pixel Accuracy: {acc:.4f}")
        acuracies.append(acc)

    mean_accuracy = np.mean(acuracies)
    print(f"Pixel Accuracy Média: {mean_accuracy:.4f}")

    # Carregando o modelo YOLOv8-seg
    model = YOLO('runs/segment/train9/weights/best.pt')

    img_path = "Yolo_dataset/images/val"
    img_files = os.listdir(img_path)

    mask_path = "Yolo_dataset/masks/val"
    mask_files = os.listdir(mask_path)

    predicted_masks = []

    # Processa as imagens e gera as máscaras preditas
    for img_file in img_files:
        image_full_path = os.path.join(img_path, img_file)
    
        # Obtém as predições para a imagem
        results = model.predict(image_full_path, task='segment')
    
        # Cria uma máscara vazia com dimensões fixas (certifique-se de que são compatíveis com a ground truth)
        mask_union = np.zeros((192, 224), dtype=np.uint8)
    
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                mask_bin = (mask_np > 0.5).astype(np.uint8)
                mask_union = np.logical_or(mask_union, mask_bin).astype(np.uint8)
    
        predicted_masks.append(mask_union)

    # Carrega as máscaras ground truth (garantindo que sejam do mesmo tamanho e binárias)
    ground_truth_masks = []
    for mask_file in mask_files:
        mask_full_path = os.path.join(mask_path, mask_file)
        # Lê a imagem em escala de cinza
        mask_img = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    
        # Redimensiona, se necessário, para (192, 224)
        mask_img = cv2.resize(mask_img, (224, 192))
    
        # Binariza a máscara (ajuste o limiar conforme necessário)
        mask_bin = (mask_img > 127).astype(np.uint8)
        ground_truth_masks.append(mask_bin)

    # Calcula a Pixel Accuracy para cada par de máscara ground truth e predita
    pixel_accuracies = []
    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        acc = dice_coefficient(gt_mask, pred_mask)
        pixel_accuracies.append(acc)

    mean_accuracy = np.mean(pixel_accuracies)
    print(f"Pixel Accuracy Média: {mean_accuracy:.4f}")
    """

# criando mascaras

    # # Definir os caminhos
    # model_path = 'runs/segment/train6/weights/best.pt'
    # input_images_dir = 'train_pacients'
    # output_masks_dir = 'output_masks'

    # # Carregar o modelo YOLO com os pesos especificados
    # model = YOLO(model_path)

    # create_folder("output_masks")

    # # Processar cada imagem na pasta de entrada
    # for image_file in os.listdir(input_images_dir):
    #     if image_file.endswith('.txt'):
    #         txt_path = os.path.join(input_images_dir, image_file)
    #         output_image_path = os.path.join(input_images_dir, f"{os.path.splitext(image_file)[0]}.png")
            
    #         # Converte o arquivo txt para uma imagem
    #         txt_to_image(txt_path, output_image_path)
            
    #         # Carrega a imagem convertida
    #         img = cv2.imread(output_image_path)
    #         if img is None:
    #             print(f"Erro ao carregar a imagem: {output_image_path}")
    #             continue

    #         H, W, _ = img.shape

    #         # Aplicar o modelo YOLO na imagem
    #         results = model(img)

    #      # Processar os resultados e salvar as máscaras
    #         for result in results:
    #             for j, mask in enumerate(result.masks.data):
    #                 mask = mask.cpu().numpy() * 255  # Mover o tensor para a CPU antes de converter para numpy
    #                 mask = cv2.resize(mask, (W, H))
    #                 output_mask_path = os.path.join(output_masks_dir, f"{os.path.splitext(image_file)[0]}_mask_{j}.png")
    #                 cv2.imwrite(output_mask_path, mask)
    #                 print(f"Máscara salva em: {output_mask_path}")
    

    # # train_yolo_seg(type="n", epochs=200, dataset="dataset.yaml", imgsize=224, seed=VALUE_SEED)

    



    train_models(Vgg_16, "Unet", resize=True, target=224, message="vgg_unet", learning_rate=0.00001)
    # create_folder("history/Vgg_16_yolo_x")
    

