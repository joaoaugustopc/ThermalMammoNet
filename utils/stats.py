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

"""
Params:
modelo: modelo para faer boxplot
dataset: nome do folder do datasetr utilizado
resize: se o dataset precisa ser redimensionado para entrar na rede
target: tamanho da foto redimensionada
"""
def get_boxPlot(model_name, dataset="np_dataset", resize=False, target=0):
    list = ["Frontal","Left45","Right45", "Left90", "Right90"] #alterei os angulos -> completar dps

    for angulo in list:
        acc = []
        loss = []
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angulo, dataset)
        
        # mudança para encaixar na rede
        if resize:
            imagens_test = np.expand_dims(imagens_test, axis=-1)
            imagens_test = tf.image.resize_with_pad(imagens_test, target, target, method="bicubic")
            imagens_test = np.squeeze(imagens_test, axis=-1)
        #fim mudança        
        
        for i in range(10):
            model = tf.keras.models.load_model(f"modelos/{model_name}/{model_name}_{angulo}_{i}.h5")

            loss_, acc_ = test_model(model, imagens_test, labels_test)

            acc.append(acc_)
            loss.append(loss_)
        
        bloxPlot(acc, loss, f"{model_name}", f"history/{model_name}/boxplot_{angulo}.png")

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

def get_confusion_matrices(model_name, dataset="np_dataset", resize=False, target=0):

    angles = ["Frontal", "Left45", "Right45", "Left90", "Right90"]

    # Dicionários para armazenar as métricas para cada ângulo
    metrics = {angle: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []} for angle in angles}

    for angle in angles:
        # Carregar os dados de teste
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angle, dataset)

        # Mudança para encaixar na rede (se necessário)
        if resize:
            imagens_test = np.expand_dims(imagens_test, axis=-1)
            imagens_test = tf.image.resize_with_pad(imagens_test, target, target, method="bicubic")
            imagens_test = np.squeeze(imagens_test, axis=-1)

        # Lista para armazenar as matrizes de confusão

        for i in range(10):
            # Carregar o modelo
            model = tf.keras.models.load_model(f"modelos/{model_name}/{model_name}_{angle}_{i}.h5")

            # Fazer previsões no conjunto de teste
            y_pred_prob = model.predict(imagens_test)
            y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

            # Obter os labels verdadeiros
            y_true = labels_test

            # Gerar a matriz de confusão
            cm = confusion_matrix(y_true, y_pred)

            # Calcular as métricas para este modelo
            TN, FP, FN, TP = cm.ravel()
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Printar as métricas
            print(f"Modelo {i} - {angle}")
            print(f"Acurácia: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1_score}")

            # Armazenar as métricas
            metrics[angle]['accuracy'].append(accuracy)
            metrics[angle]['precision'].append(precision)
            metrics[angle]['recall'].append(recall)
            metrics[angle]['f1_score'].append(f1_score)

            # Plotar e salvar a matriz de confusão para este modelo
            classes = ['Healthy', 'Sick']
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.title(f'Matriz de Confusão - {model_name} - {angle} - Modelo {i}')
            plt.ylabel('Classe Real')
            plt.xlabel('Classe Predita')
            plt.tight_layout()
            plt.savefig(f"history/{model_name}/confusion_matrix_{angle}_{i}.png")
            plt.close()

    generate_metrics_boxplots(metrics, model_name)

def generate_metrics_boxplots(metrics, model_name):

    os.makedirs(f"history/{model_name}/boxplots", exist_ok=True)

    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
        plt.figure(figsize=(10, 6))
        data = [metrics[angle][metric_name] for angle in metrics.keys()]
        plt.boxplot(data, labels=metrics.keys())
        plt.title(f'Boxplot da {metric_name.capitalize()} para cada ângulo - {model_name}')
        plt.ylabel(metric_name.capitalize())
        plt.xlabel('Ângulo')
        plt.grid(True)
        plt.savefig(f"history/{model_name}/boxplots/{metric_name}_boxplot.png")
        plt.close()

