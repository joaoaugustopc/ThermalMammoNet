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
def get_boxPlot(model_name, mensagem ="", dataset="np_dataset", resize=False, target=0):
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
            i = i + 1
            if model_name == "ResNet34":
                with custom_object_scope({'ResidualUnit': ResidualUnit}):
                    model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5")
            else:
                model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angulo}_{i}.h5")

            loss_, acc_ = test_model(model, imagens_test, labels_test)

            acc.append(acc_)
            loss.append(loss_)
        
        bloxPlot(acc, loss, f"{model_name}", f"{mensagem}_boxplot_{angulo}.png")

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
        plt.savefig(f"history/{model_name}/{angulo}/treinamento/{mensagem}_{angulo}_{i}_training_loss_convergence.png")
        plt.close()

        # Gráfico de perda de validação
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Validation Loss Convergence for {model_name} - {angulo} - Run {i}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"history/{model_name}/{angulo}/treinamento/{mensagem}_{angulo}_{i}_validation_loss_convergence.png")
        plt.close()

def get_confusion_matrices(model_name, mensagem="", dataset="np_dataset", resize=False, target=0):

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from src.models.resNet_101 import ResNet101, BottleneckResidualUnit

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
            i = i + 1
            # Carregar o modelo
            if model_name == "ResNet34":
                with custom_object_scope({'ResidualUnit': ResidualUnit}):
                    model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")
            elif model_name == "ResNet101":
                with custom_object_scope({'BottleneckResidualUnit': BottleneckResidualUnit}):
                    model = keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")
            else:
                model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")

            # Fazer previsões no conjunto de teste
            y_pred_prob = model.predict(imagens_test)
            y_pred = (y_pred_prob >= 0.50).astype(int).flatten()

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
            
            with open(f"history/{model_name}/{angle}/{mensagem}_{angle}_metrics.txt", "a") as f:
                f.write(f"Modelo: {model_name} - {i} - {angle}\n")
                f.write(f"Acurácia: {accuracy}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"F1 Score: {f1_score}\n\n\n")
            

            # Armazenar as métricas
            metrics[angle]['accuracy'].append(accuracy)
            metrics[angle]['precision'].append(precision)
            metrics[angle]['recall'].append(recall)
            metrics[angle]['f1_score'].append(f1_score)

            os.makedirs(f"history/{model_name}/{angle}/confusion_matrices/", exist_ok=True)

            # Plotar e salvar a matriz de confusão para este modelo
            classes = ['Healthy', 'Sick']
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.title(f'Matriz de Confusão - {model_name} - {angle} - Modelo {i}')
            plt.ylabel('Classe Real')
            plt.xlabel('Classe Predita')
            plt.tight_layout()
            plt.savefig(f"history/{model_name}/{angle}/confusion_matrices/{mensagem}_confusion_matrix_{angle}_{i}.png")
            plt.close()

    generate_metrics_boxplots(metrics, model_name, mensagem)

def generate_metrics_boxplots(metrics, model_name, mensagem=""):

    os.makedirs(f"history/{model_name}/boxplots", exist_ok=True)

    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
        os.makedirs(f"boxplots/{model_name}", exist_ok=True)
        plt.figure(figsize=(10, 6))
        data = [metrics[angle][metric_name] for angle in metrics.keys()]
        plt.boxplot(data, labels=metrics.keys())
        plt.title(f'Boxplot da {metric_name.capitalize()} para cada ângulo - {model_name}')
        plt.ylabel(metric_name.capitalize())
        plt.xlabel('Ângulo')
        plt.grid(True)
        plt.savefig(f"boxplots/{model_name}/{mensagem}_boxplots_{metric_name}_boxplot.png")
        plt.close()

def get_auc_roc(model_name,mensagem ="", dataset="np_dataset", resize=False, target=0):

    from sklearn.metrics import roc_auc_score, roc_curve

    angles = ["Frontal", "Left45", "Right45", "Left90", "Right90"]

    # Dicionário para armazenar as AUCs para cada ângulo
    auc_scores = {angle: [] for angle in angles}

    for angle in angles:
        # Carregar os dados de teste
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angle, dataset)

        # Pré-processamento das imagens de teste (se necessário)
        if resize:
            imagens_test = np.expand_dims(imagens_test, axis=-1)
            imagens_test = tf.image.resize_with_pad(imagens_test, target, target, method="bicubic")
            imagens_test = np.squeeze(imagens_test, axis=-1)

        for i in range(10):
            i = i + 1
            # Carregar o modelo
            if model_name == "ResNet34":
                with custom_object_scope({'ResidualUnit': ResidualUnit}):
                    model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")
            else:
                model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")

            # Fazer previsões no conjunto de teste
            y_pred_prob = model.predict(imagens_test).flatten()  # Probabilidades preditas

            # Obter os labels verdadeiros
            y_true = labels_test

            # Calcular a AUC
            auc = roc_auc_score(y_true, y_pred_prob)
            auc_scores[angle].append(auc)

            print(f"AUC para o modelo {i} no ângulo {angle}: {auc:.4f}")

            # Plotar a Curva ROC
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]

            os.makedirs(f"history/{model_name}/{angle}/roc_curves/", exist_ok=True)
            
            plt.figure()
            plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')  # Linha de referência (modelo aleatório)
            
            # Destacando o limiar ótimo (Youden)
            plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Limiar Ótimo: {best_threshold:.2f}', zorder=5)
            
            plt.xlabel('Taxa de Falsos Positivos')
            plt.ylabel('Taxa de Verdadeiros Positivos')
            plt.title(f'Curva ROC - {model_name} - {angle} - Modelo {i}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"history/{model_name}/{angle}/roc_curves/{mensagem}_roc_curve_{angle}_{i}.png")
            plt.close()
    
    plt.figure(figsize=(10, 6))
    data = [auc_scores[angle] for angle in angles]
    plt.boxplot(data, labels=angles)
    plt.title(f'Boxplot das AUCs para todos os ângulos - {model_name}')
    plt.ylabel('AUC')
    plt.xlabel('Ângulo')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"boxplots/{model_name}/{mensagem}_auc_boxplot_all_angles.png")
    plt.close()
    
def get_precision_recall_curves(model_name, mensagem="", dataset="np_dataset", resize=False, target=0):
    from sklearn.metrics import precision_recall_curve, auc, f1_score
    import seaborn as sns

    #angles = ["Frontal", "Left45", "Right45", "Left90", "Right90"]
    angles = ["Frontal"]

    # Dicionários para armazenar as métricas para cada ângulo
    metrics = {angle: {'best_thresholds': [], 'best_f1_scores': [], 'auc_pr': []} for angle in angles}

    for angle in angles:
        # Carregar os dados de teste
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angle, dataset)

        # Mudança para encaixar na rede (se necessário)
        if resize:
            imagens_test = np.expand_dims(imagens_test, axis=-1)
            imagens_test = tf.image.resize_with_pad(imagens_test, target, target, method="bicubic")
            imagens_test = np.squeeze(imagens_test, axis=-1)

        # Converter labels para formato adequado
        y_true = labels_test

        for i in range(10):
            i = i + 1
            # Carregar o modelo
            if model_name == "ResNet34":
                with custom_object_scope({'ResidualUnit': ResidualUnit}):
                    model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")
            else:
                model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")

            # Fazer previsões no conjunto de teste
            y_scores = model.predict(imagens_test).flatten()

            # Calcular a curva Precision-Recall
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

            # Calcular o F1-score para cada limiar
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

            # Encontrar o limiar com o melhor F1-score
            idx = np.argmax(f1_scores)
            best_threshold = thresholds[idx] if idx < len(thresholds) else 1.0
            best_f1 = f1_scores[idx]

            # Calcular a área sob a curva Precision-Recall
            auc_pr = auc(recall, precision)

            # Armazenar as métricas
            metrics[angle]['best_thresholds'].append(best_threshold)
            metrics[angle]['best_f1_scores'].append(best_f1)
            metrics[angle]['auc_pr'].append(auc_pr)

            # Plotar e salvar a curva Precision-Recall
            #os.makedirs(f"history/{model_name}/{angle}/precision_recall_curves/", exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'AUC PR = {auc_pr:.2f}')
            plt.scatter(recall[idx], precision[idx], marker='o', color='red',
                        label=f'Melhor F1 = {best_f1:.2f} (Limiar = {best_threshold:.2f})')
            plt.title(f'Curva Precision-Recall - {model_name} - {angle} - Modelo {i}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            #plt.savefig(f"history/{model_name}/{angle}/precision_recall_curves/{mensagem}_precision_recall_{angle}_{i}.png")
            plt.savefig(f"{mensagem}_precision_recall_{angle}_{i}.png")
            plt.close()

            # Salvar as métricas em um arquivo
            #with open(f"history/{model_name}/{angle}/{mensagem}_{angle}_precision_recall_metrics.txt", "a") as f:
            with open(f"{mensagem}_{angle}_precision_recall_metrics.txt", "a") as f:
                f.write(f"Modelo: {model_name} - {i} - {angle}\n")
                f.write(f"AUC PR: {auc_pr}\n")
                f.write(f"Melhor Limiar: {best_threshold}\n")
                f.write(f"Melhor F1 Score: {best_f1}\n\n\n")

    # Opcional: Gerar boxplots para as métricas coletadas
    generate_pr_metrics_boxplots(metrics, model_name, mensagem)


def generate_pr_metrics_boxplots(metrics, model_name, mensagem=""):
    os.makedirs(f"boxplots/{model_name}", exist_ok=True)

    for metric_name in ['best_f1_scores', 'best_thresholds', 'auc_pr']:
        plt.figure(figsize=(10, 6))
        data = [metrics[angle][metric_name] for angle in metrics.keys()]
        plt.boxplot(data, labels=metrics.keys())
        plt.title(f'Boxplot de {metric_name.replace("_", " ").capitalize()} para cada ângulo - {model_name}')
        plt.ylabel(metric_name.replace("_", " ").capitalize())
        plt.xlabel('Ângulo')
        plt.grid(True)
        #plt.savefig(f"boxplots/{model_name}/{mensagem}_boxplot_{metric_name}.png")
        plt.savefig(f"{mensagem}_boxplot_{metric_name}.png")
        plt.close()


def get_mean_metrics(model_name, mensagem = ""):
    angles = ["Frontal", "Left45", "Right45", "Left90", "Right90"]

    # Dicionários para armazenar as métricas para cada ângulo
    metrics = {angle: {'accuracy': [],'recall': [], 'AUC':[]} for angle in angles}

    for angle in angles:
        # Carregar os dados de teste
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(angle, "aug_dataset")

        # Mudança para encaixar na rede (se necessário)
        
        imagens_test = np.expand_dims(imagens_test, axis=-1)
        imagens_test = tf.image.resize_with_pad(imagens_test, 224, 224, method="bicubic")
        imagens_test = np.squeeze(imagens_test, axis=-1)

        # Lista para armazenar as matrizes de confusão

        for i in range(1,11):
            # Carregar o modelo
            
            if model_name == "ResNet34":
                with custom_object_scope({'ResidualUnit': ResidualUnit}):
                    model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")
            else:
                model = tf.keras.models.load_model(f"modelos/{model_name}/{mensagem}_{angle}_{i}.h5")
            
            # Fazer previsões no conjunto de teste
            y_pred_prob = model.predict(imagens_test)
            y_pred = (y_pred_prob >= 0.50).astype(int).flatten()

            # Obter os labels verdadeiros
            y_true = labels_test

            # Gerar a matriz de confusão
            cm = confusion_matrix(y_true, y_pred)

            # Calcular as métricas para este modelo
            TN, FP, FN, TP = cm.ravel()
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            auc = roc_auc_score(y_true, y_pred_prob)

        
            # Printar as métricas
            
            metrics[angle]['accuracy'].append(accuracy)
            metrics[angle]['recall'].append(recall)
            metrics[angle]['AUC'].append(auc)

        
        
        acc_mean = np.mean(metrics[angle]['accuracy'])
        rec_mean = np.mean(metrics[angle]['recall'])
        auc_mean = np.mean(metrics[angle]['AUC'])

        acc_std = np.std(metrics[angle]['accuracy'])
        rec_std = np.std(metrics[angle]['recall'])
        auc_std = np.std(metrics[angle]['AUC'])

        # Armazenar as métricas
        with open(f"{mensagem}_mean_metrics.txt", "a") as f:
            f.write(f"Ângulo: {angle}\n")
            f.write(f"Acurácia: {acc_mean} +/- {acc_std}\n")
            f.write(f"Recall: {rec_mean} +/- {rec_std}\n")
            f.write(f"AUC: {auc_mean} +/- {auc_std}\n")
            f.write("\n")



