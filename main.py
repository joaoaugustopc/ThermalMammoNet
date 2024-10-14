from include.imports import *

def main_func(models_list, mensagem = ""):
    
    list = ["Frontal","Left45","Right45", "Left90", "Right90"]
    models = models_list
                
    for angulo in list:

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo, "aug__dataset")
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

                start_time = time.time()

                checkpoint = tf.keras.callbacks.ModelCheckpoint(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')

                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                              patience=10, min_lr=1e-6)


                model = model_func()

                model.summary()

                history = model.fit(imagens_train, labels_train, epochs = 500, validation_data= (imagens_valid, labels_valid),
                                    callbacks= [checkpoint, earlystop, reduce_lr], batch_size = 8, verbose = 1, shuffle = True)
                
                end_time = time.time()

                best_model = keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5")

                # Avaliação do modelo com conjunto de teste
                test_loss, test_accuracy = best_model.evaluate(imagens_test, labels_test, verbose=1)

                with open(f"history/{model_name}/{mensagem}_{angulo}_{i}_time.txt", "w") as f:
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

    #main_func([ResNet34])
    #get_boxPlot("ResNet34")

    
    list = ["Frontal", "Left45", "Right45", "Left90", "Right90"]

    for angulo in list:

        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        imagens_train, labels_train = apply_augmentation_and_expand(imagens_train, labels_train, 2, resize=False)

        #Salvar dataset com o augmentation

        if not os.path.exists("aug_dataset"):
            os.makedirs("aug_dataset")

        np.save(f"aug_dataset/imagens_train_{angulo}.npy", imagens_train)
        np.save(f"aug_dataset/labels_train_{angulo}.npy", labels_train)

        np.save(f"aug_dataset/imagens_valid_{angulo}.npy", imagens_valid)
        np.save(f"aug_dataset/labels_valid_{angulo}.npy", labels_valid)

        np.save(f"aug_dataset/imagens_test_{angulo}.npy", imagens_test)
        np.save(f"aug_dataset/labels_test_{angulo}.npy", labels_test)
    

    



    





    
    main_func()        
    
