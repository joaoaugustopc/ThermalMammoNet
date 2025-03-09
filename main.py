import cv2
from ultralytics import YOLO
from include.imports import *
from utils.data_prep import load_imgs_masks, YoLo_Data, masks_to_polygons,load_imgs_masks_only, listar_imgs_nao_usadas, format_data
from src.models.yolo_seg import train_yolo_seg
from src.models.u_net import unet_model

# Use o tempo atual em segundos como semente

VALUE_SEED = int(time.time() * 1000) % 15000

random.seed(VALUE_SEED)

seed = random.randint(0, 15000)

tf.random.set_seed(seed)

np.random.seed(seed)


print("***SEMENTE USADA****:", VALUE_SEED)

def train_models(models_objects, dataset: str, resize=False, target = 0, message=""):
    list = ["Frontal","Left45", "Right45", "Left90", "Right90"]
    models = models_objects
                
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
        for model_func in models_objects:

            #criando pasta com gráficos dos modelos
            os.makedirs(f"history/{model_func.__name__}", exist_ok=True)
            
            with open(f"modelos/random_seed.txt", "a") as f:
                f.write(f"Modelo:{model_func.__name__}\n")
                f.write(f"Angulo: {angulo}\n")

            for i in range(10):
                
                print(f"history/{model_func.__name__}/{model_func.__name__}_{angulo}_{i}_time.txt")
                
                start_time = time.time()
                
                checkpoint_path = f"modelos/{model_func.__name__}/{message}_{angulo}_{i}.h5"
                checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')

                #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=15, min_lr=1e-5, min_delta=0.0001)

                #criando objeto e usando o modelo
                model = model_func().model

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
                    f.write("\n")
                    
                plot_convergence(history, model_func.__name__, angulo, i, message)


import re
if __name__ == "__main__":

    

    format_data("raw_dataset", "np_dataset_v2", exclude=True, exclude_path="Termografias_Dataset_Segmentação/images")



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
    
    ######Código para testar u-net no caso de uma paciente sem uma das mamas######
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
    ########################################################################################################
    #AVALIANDO YOLO E U-NET


    




