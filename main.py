from include.imports import *
from utils.data_prep import load_imgs_masks

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
    list = ["Frontal", "Left45","Right45", "Left90", "Right90"]
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

            for i in range(10):

                
                start_time = time.time()
                
                checkpoint_path = f"modelos/{model_func.__name__}/{model_func.__name__}_{message}_{angulo}_{i}.h5"
                checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

                #criando objeto e usando o modelo
                model = model_func().model

                model.summary()

                history = model.fit(imagens_train, labels_train, epochs = 500, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)
                
                end_time = time.time()
                
                # Avaliação do modelo com conjunto de teste
                model = keras.models.load_model(checkpoint_path) #carregando o melhor modelo
                test_loss, test_accuracy = model.evaluate(imagens_test, labels_test, verbose=1)

                with open(f"history/{model_func.__name__}/{message}_{angulo}_{i}_time.txt", "w") as f:
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

    imgs, masks = load_imgs_masks("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks")

    print(imgs.shape)
    print(masks.shape)