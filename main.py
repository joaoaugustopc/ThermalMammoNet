from include.imports import *

def main_func(models_list, mensagem = ""):
    
    list = ["Frontal", "Left90", "Right90", "Left45", "Right45"]
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
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                              patience=10, min_lr=1e-5)


                model = model_func()

                model.summary()

                history = model.fit(imagens_train, labels_train, epochs = 500, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint, earlystop, reduce_lr], batch_size = 16, verbose = 1, shuffle = True)
                
                end_time = time.time()

                if model_name == "ResNet34":
                    with custom_object_scope({'ResidualUnit': ResidualUnit}):
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


if __name__ == "__main__":
        
    #main_func([ResNet34], "ResNet34_224x224_Bath16")

    #get_confusion_matrices("ResNet34", "ResNet34_224x224", "aug_dataset", resize=True, target = 224)

    get_auc_roc("ResNet34", "ResNet34_224x224", "aug_dataset", resize=True, target = 224)

    #files = [f"ResNet34_224x224_Bath16_Frontal_metrics.txt"]

    #move_files_to_folder(files, "history/ResNet34/Frontal/")

    
       
    
