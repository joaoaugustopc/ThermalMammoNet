from include.imports import *

def test_model(model, imagens_test, labels_test):
    """
    imagens_test = np.expand_dims(imagens_test, axis = -1)
    imagens_test = np.repeat(imagens_test, 3, axis=-1)
    #imagens_test = tf.image.resize(imagens_test, (200, 200))
    """
   
    loss, acc = model.evaluate(imagens_test, labels_test)

    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")

    return loss, acc


def bloxPlot(acc_data, loss_data, title, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Boxplot para acurácia
    axs[0].boxplot(acc_data)
    axs[0].set_title('Acurácia dos modelos')
    axs[0].set_xlabel('Modelo')
    axs[0].set_ylabel('Acurácia')

    # Boxplot para loss
    axs[1].boxplot(loss_data)
    axs[1].set_title('Loss do modelo')
    axs[1].set_xlabel('Modelo')
    axs[1].set_ylabel('Loss')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#função alterada
def get_boxPlot(modelo):
    list = ["Frontal"] #alterei os angulos -> completar dps

    for angulo in list:
        acc = []
        loss = []
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo)
        
        #mudança para encaixar na rede
        #imagens_test = np.expand_dims(imagens_test, axis=-1)
        #imagens_test = tf.image.resize_with_pad(imagens_test, 224, 224, method="bicubic")
        #imagens_test = np.squeeze(imagens_test, axis=-1)
        #fim mudança        
        
        for i in range(10):
            with custom_object_scope({'ResidualUnit': ResidualUnit}):
              model = tf.keras.models.load_model(f"modelos/{modelo}_{angulo}_{i+1}.h5")
            
            #imagens_test = np.expand_dims(imagens_test, axis = -1)

            #imagens_test = np.repeat(imagens_test, 3, axis=-1)

            #imagens_test = tf.image.resize(imagens_test, (200, 200))

            loss_, acc_ = test_model(model, imagens_test, labels_test)

            acc.append(acc_)
            loss.append(loss_)
        
        #alterei
        bloxPlot(acc, loss, "ResNet34", f"ResNet34{angulo}.png")

        print(f"Acurácia média: {np.mean(acc)}")
        print(f"Loss médio: {np.mean(loss)}")
        print(f"Desvio padrão da acurácia: {np.std(acc)}")
        print(f"Desvio padrão do loss: {np.std(loss)}")
        print(f"Mediana da acurácia: {np.median(acc)}")
        print(f"Mediana do loss: {np.median(loss)}")

def data_distribution():
    for angle in ["Frontal", "Right45", "Right90", "Left45", "Left90"]:
        
        train = np.load(f"np_dataset/imagens_train_{angle}.npy")
        valid = np.load(f"np_dataset/imagens_valid_{angle}.npy")
        test = np.load(f"np_dataset/imagens_test_{angle}.npy")

        labels_train = np.load(f"np_dataset/labels_train_{angle}.npy")
        labels_valid = np.load(f"np_dataset/labels_valid_{angle}.npy")
        labels_test = np.load(f"np_dataset/labels_test_{angle}.npy")

        print("ANGLE:",angle)
        print("Train shape:",train.shape)
        print(labels_train.shape)
        print("valid shape:",valid.shape)
        print(labels_valid.shape)
        print("test shape:",test.shape)
        print(labels_test.shape)

        print("Train Healthy:",len(labels_train[labels_train == 0]))
        print("Train Sick:",len(labels_train[labels_train == 1]))

def plot_convergence(history, model_name, angulo, i, mensagem = ""):
        # Gráfico de perda de treinamento
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title(f'Training Loss Convergence for {model_name} - {angulo} - Run {i}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"history/{model_name}/{mensagem}_{angulo}_{i}_training_loss_convergence.png")
        plt.close()

        # Gráfico de perda de validação
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Validation Loss Convergence for {model_name} - {angulo} - Run {i}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"history/{model_name}/{mensagem}_{angulo}_{i}_validation_loss_convergence.png")
        plt.close()