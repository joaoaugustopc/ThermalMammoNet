from include.imports import *
from utils.data_prep import load_imgs_masks

from src.models.u_net import unet_model

def main_func(models_list, mensagem = ""):
    
    list = ["Frontal","Left45", "Right45", "Left90", "Right90"]
    models = models_list
                
    for angulo in list:

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo, "aug_dataset")
        print(imagens_train.shape)
        
        print(labels_train[labels_train == 1].shape)
        print(labels_train[labels_train == 0].shape)

        
        # Para ajustar a dimensão das imagens para o modelo
        # Add uma dimensão para o canal de cor para o tf.image.resize_with_pad
        imagens_train = np.expand_dims(imagens_train, axis=-1)
        imagens_valid = np.expand_dims(imagens_valid, axis=-1) 
        imagens_test = np.expand_dims(imagens_test, axis=-1)

        imagens_train = tf.image.resize_with_pad(imagens_train, 224, 224, method="bicubic")
        imagens_valid = tf.image.resize_with_pad(imagens_valid, 224, 224, method="bicubic")
        imagens_test = tf.image.resize_with_pad(imagens_test, 224, 224, method="bicubic")
        
        # Remover a dimensão do canal de cor
        imagens_train = np.squeeze(imagens_train, axis=-1)
        imagens_valid = np.squeeze(imagens_valid, axis=-1)       
        imagens_test = np.squeeze(imagens_test, axis=-1)
        
        for model_func in models:

            model_name = model_func.__name__

            os.makedirs(f"history/{model_name}", exist_ok=True)

            for i in range(10):

                i = i + 1
                
                start_time = time.time()

                checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')

                #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=15, min_lr=1e-5, min_delta=0.0001)


                model = model_func()

                model.summary()

                history = model.fit(imagens_train, labels_train, epochs = 500, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint, earlystop], batch_size = 10, verbose = 1, shuffle = True)
                
                end_time = time.time()

                if model_name == "ResNet34":
                    with custom_object_scope({'ResidualUnit': ResidualUnit}):
                        best_model = keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5")
                elif model_name == "ResNet101":
                    with custom_object_scope({'BottleneckResidualUnit': BottleneckResidualUnit}):
                        best_model = keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5")
                else:
                    best_model = keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5")

                # Avaliação do modelo com conjunto de teste
                test_loss, test_accuracy = best_model.evaluate(imagens_test, labels_test, verbose=1)

                directory = f"history/{model_name}/{angulo}/treinamento/"
                os.makedirs(directory, exist_ok=True)

                with open(f"{directory}/{mensagem}_{angulo}_{i}_time.txt", "w") as f:
                    f.write(f"Modelo: {model_name}\n")
                    f.write(f"Tempo de execução: {end_time - start_time}\n")
                    f.write(f"Loss: {history.history['loss']}\n")
                    f.write(f"Val_loss: {history.history['val_loss']}\n")
                    f.write(f"Accuracy: {history.history['accuracy']}\n")
                    f.write(f"Val_accuracy: {history.history['val_accuracy']}\n")
                    f.write(f"Test Loss: {test_loss}\n")
                    f.write(f"Test Accuracy: {test_accuracy}\n")
                    f.write("\n")
                    
                plot_convergence(history, model_name, angulo, i, mensagem)


def train_models(models_objects, dataset, resize=False, target=0, message=""):
    list = ["Left90","Right90"]
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

            for i in range(5):

                i = i + 5
                
                print(f"history/{model_func.__name__}/{model_func.__name__}_{angulo}_{i}_time.txt")
                
                start_time = time.time()
                
                checkpoint_path = f"modelos/{model_func.__name__}/{model_func.__name__}{message}_{angulo}_{i}.h5"
                checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

                #criando objeto e usando o modelo
                model = model_func().model

                model.summary()
                
                # Use o tempo atual em segundos como semente
                value_seed = int(time.time() * 1000) % 15000
                random.seed(value_seed)

                # Agora gere o número aleatório
                seed = random.randint(0, 15000)

                # Salva a seed em um arquivo de texto
                with open("modelos/random_seed.txt", "a") as file:
                    file.write(str(seed))
                    file.write("\n")

                print("Seed gerada e salva em random_seed.txt:", seed)

                tf.random.set_seed(seed)
                

                history = model.fit(imagens_train, labels_train, epochs = 500, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)
                
                end_time = time.time()
                
                # Avaliação do modelo com conjunto de teste
                model = keras.models.load_model(checkpoint_path) #carregando o melhor modelo
                test_loss, test_accuracy = model.evaluate(imagens_test, labels_test, verbose=1)

                with open(f"history/{model_func.__name__}/{model_func.__name__}_{angulo}_{i}_time.txt", "w") as f:
                    f.write(f"Modelo: {model_func.__name__}\n")
                    f.write(f"Tempo de execução: {end_time - start_time}\n")
                    f.write(f"Loss: {history.history['loss']}\n")
                    f.write(f"Val_loss: {history.history['val_loss']}\n")
                    f.write(f"Accuracy: {history.history['accuracy']}\n")
                    f.write(f"Val_accuracy: {history.history['val_accuracy']}\n")
                    f.write(f"Test Loss: {test_loss}\n")
                    f.write(f"Test Accuracy: {test_accuracy}\n")
                    f.write("\n")
                    
                plot_convergence(history, model_func.__name__, angulo, i, "Vgg_16")





if __name__ == "__main__":


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
    
    imagem = Image.open("ImgTESTE.jpg").convert('L')

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

    
    #imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks("Left45", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks")
    
    """
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

    img_test = np.load("np_dataset/imagens_test_Frontal.npy")

    img = img_test[50]
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
    model = keras.models.load_model("modelos/unet/L45unet.h5")

    img_test = np.load("np_dataset/imagens_test_Left45.npy")

    loss, acc = model.evaluate(imgs_valid, masks_valid, verbose=1)

    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")
    """