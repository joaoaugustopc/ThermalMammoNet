import cv2
from ultralytics import YOLO
from include.imports import *
from utils.data_prep import load_imgs_masks, load_raw_images_ufpe, yolo_data, masks_to_polygons,load_imgs_masks_only, copy_images_excluding_patients, filter_dataset_by_id, load_raw_images,make_tvt_splits, augment_train_fold, normalize, tf_letterbox, listar_imgs_nao_usadas, load_imgs_masks_sem_padding,load_imgs_masks_recortado,tf_letterbox_Sem_padding, letterbox_center_crop, load_imgs_masks_Black_Padding, tf_letterbox_black,load_imgs_masks_distorcidas, apply_augmentation_and_expand_ufpe, yolo_data_2_classes
from utils.files_manipulation import move_files_within_folder, create_folder, move_folder
from src.models.yolo_seg import train_yolo_seg
from src.models.u_net import unet_model, unet_model_retangular
from utils.stats import precision_score_, recall_score_, accuracy_score_, dice_coef_, iou_ 
from eigemCAM import run_eigencam
import json
import glob
from sklearn.metrics import classification_report
from utils.files_manipulation import copy_file

import cv2
from ultralytics import YOLO
from tensorflow.keras import backend as K
import gc
import numpy as np
import os, json, time, shutil, random, gc
from pathlib import Path
from typing import List, Union

from pathlib import Path
import numpy as np
import cv2   # ou PIL.Image, se preferir
import csv
import os
import json as json_module  # Alias seguro
import json
import torch



from src.models.Vgg_16 import Vgg_16

# Use o tempo atual em segundos como semente
##VALUE_SEED = int(time.time() * 1000) % 15000
"""
VALUE_SEED = 7758
random.seed(VALUE_SEED)

seed = random.randint(0, 15000)

tf.random.set_seed(seed)

np.random.seed(seed)
"""

def clear_memory():
    """
    Limpa tudo que puder de RAM/VRAM para a próxima tentativa.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unet_segmenter(data_train, data_valid, data_test, path_model):
    model = tf.keras.models.load_model(path_model)
    train_predictions = model.predict(data_train, batch_size=4)
    valid_predictions = model.predict(data_valid, batch_size=4)
    test_predictions = model.predict(data_test, batch_size=4)

    masks_train = (train_predictions > 0.5).astype(np.uint8)
    masks_valid = (valid_predictions > 0.5).astype(np.uint8)
    masks_test = (test_predictions > 0.5).astype(np.uint8)

    masks_train = np.squeeze(masks_train, axis=-1)
    masks_valid = np.squeeze(masks_valid, axis=-1)
    masks_test = np.squeeze(masks_test, axis=-1)

    segmented_images_train = data_train * masks_train
    segmented_images_valid = data_valid * masks_valid
    segmented_images_test = data_test * masks_test

    return segmented_images_train, segmented_images_valid, segmented_images_test


def segment_with_yolo( X_train, X_valid, X_test, model_path):
    """
    Segmenta X_train, X_valid e X_test usando YOLO-Seg.
    Retorna as imagens segmentadas nas mesmas ordens.
    """
    
    def prepare_image(img):
        """Prepara imagem para o YOLO: uint8 RGB 3 canais, redimensionada"""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img
    
    PEITO_ID = 0       
    MARCADOR_ID = 1    

    def segment_batch(images, model):
        segmented = []
        for img in images:
            img_prepared = prepare_image(img)  # sua função
            H, W = img_prepared.shape[:2]

            # Predição
            results = model.predict(img_prepared, verbose=False)
            res = results[0]

            # Verifica se há máscaras
            has_masks = (
                res.masks is not None and
                res.masks.data is not None and
                len(res.masks.data) > 0
            )

            if has_masks:
                # (N_inst, h_m, w_m)
                masks_np = res.masks.data.cpu().numpy()
                # classes (N_inst,)
                classes = res.boxes.cls.cpu().numpy().astype(int)

                # Máscara final = união (OR) das instâncias de PEITO e MARCADOR
                union_mask = np.zeros((H, W), dtype=np.uint8)

                for m, c in zip(masks_np, classes):
                    if c in (PEITO_ID, MARCADOR_ID):
                        m_bin = (m > 0.5).astype(np.uint8)
                        # sempre redimensione máscaras com NEAREST
                        m_resized = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                        union_mask |= m_resized

                if union_mask.max() > 0:
                    # garante shape (H,W,1)
                    union_mask = union_mask[..., None]
                    segmented_img = img_prepared * union_mask  # aplica máscara (peito ∪ marcador)
                else:
                    print("Não encontrou instâncias das classes esperadas")
                    segmented_img = img_prepared
            else:
                print("Não encontrou mask")
                segmented_img = img_prepared

            segmented.append(segmented_img)

        return np.array(segmented)

    # Carrega modelo YOLO
    model = YOLO(model_path)

    # Segmenta os três conjuntos
    seg_train = segment_batch(X_train, model)
    seg_valid = segment_batch(X_valid, model)
    seg_test  = segment_batch(X_test, model)

    #a, b, c, d, e, f = load_data("Frontal", "Yolo_dataset")

    array = []

    # Exemplo hipotético de loop processando imagens
    for i, img in enumerate(seg_train):
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

    seg_train = np.array(array).squeeze(axis=-1)

    array = []

    for i, img in enumerate(seg_valid):
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

    seg_valid = np.array(array).squeeze(axis=-1)

    array = []

    for i, img in enumerate(seg_test):
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

    seg_test = np.array(array).squeeze(axis=-1)
    


    return seg_train, seg_valid, seg_test

def load_and_convert_temp(image_path):
    # Carregue seus dados de temperatura (ajuste conforme seu formato)
    temp_data = np.loadtxt(image_path)  # Ou outro método de carregamento
    
    # Aplicar os passos 1-3 acima
    temp_normalized = (temp_data - temp_data.min()) / (temp_data.ptp())
    temp_uint8 = (temp_normalized * 255).astype(np.uint8)
    temp_rgb = cv2.applyColorMap(temp_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(temp_rgb, cv2.COLOR_BGR2RGB)


def visualize_processed_images(images, labels, title, save_path=None):
    """
    Exibe ou salva uma amostra de imagens pré-processadas.
    Espera um array de imagens onde os valores não estão na faixa [0, 255].
    """
    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=16)
    num_to_display = min(len(images), 8)
    
    # Seleciona algumas imagens aleatórias para visualização
    indices = random.sample(range(len(images)), num_to_display)

    for i, idx in enumerate(indices):
        ax = plt.subplot(2, 4, i + 1)
        img = images[idx]
        
        # Normalização min-max para visualização
        img_range = img.max() - img.min()
        if img_range > 0:
            img_normalized = (img - img.min()) / img_range
        else:
            img_normalized = img

        # APLICAÇÃO DO MAPA DE CALOR:
        # 1. Converte a imagem normalizada (0.0-1.0) para 0-255
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        # 2. Aplica o mapa de cores (ex: INFERNO, como no seu exemplo)
        img_bgr = cv2.applyColorMap(img_uint8, cv2.COLORMAP_INFERNO)
        # 3. Converte de BGR para RGB para o Matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img_rgb)
        plt.title(f"Label: {labels[idx]}")
        plt.axis("off")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Imagem salva em: {save_path}")
    else:
        plt.show()


"""
FUNÇÃO PRINCIPAL PARA TREINAR OS MODELOS
"""
def train_model_cv(model, raw_root, message, angle="Frontal", k=5, 
                  resize=True, resize_method = "BlackPadding", resize_to=224, n_aug=0, batch=8, seed=42, 
                  segmenter="none", seg_model_path="",channel_method ="MapaCalor"):
    
    # DEBUG: vendo o nome do modeloo
    #print(model.__name__)
    
    
    exclude_set = listar_imgs_nao_usadas("Termografias_Dataset_Segmentação/images", angle)
    
    if "ufpe" in raw_root:
        X, y, patient_ids = load_raw_images_ufpe(
            os.path.join(raw_root, angle), exclude=False)
        print(f"Carregando imagens da UFPE: {X.shape}, {y.shape}, {len(patient_ids)} pacientes")
    else:
        X, y, patient_ids = load_raw_images(
            os.path.join(raw_root, angle), exclude=True, exclude_set=exclude_set)
        print(f"Carregando imagens: {X.shape}, {y.shape}, {len(patient_ids)} pacientes")

    print(f"Saudavéis: {np.sum(y==0)}, Doentes: {np.sum(y==1)}")

    
    with open("modelos/random_seed.txt", "a") as f:
        f.write(f"{message}\n"
                f"SEMENTE: {seed}\n")

    for fold, (tr_idx, va_idx, te_idx) in enumerate(
                    make_tvt_splits(X, y, patient_ids,
                                   k=k, val_size=0.25, seed=seed)):
        
        def run_fold():
        
            # salva índices para reuso posterior
            split_file = f"splits/{message}_{angle}_F{fold}.json"
            os.makedirs("splits", exist_ok=True)
            with open(split_file, "w") as f:
                json.dump({
                    "train_idx": tr_idx.tolist(),
                    "val_idx": va_idx.tolist(),
                    "test_idx": te_idx.tolist()
                }, f)

            # ------ prepara dados -----------
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val,    y_val = X[va_idx], y[va_idx]

            X_test,   y_test= X[te_idx], y[te_idx]

            print(f"Shape de treinamento fold {fold} antes do aumento de dados: {X_tr.shape}")
            print(f"Shape de validação fold {fold}: {X_val.shape}")
            print(f"Shape de teste fold {fold}: {X_test.shape}")

            

            # min/max APENAS dos originais
            mn, mx = X_tr.min(), X_tr.max()

            # normaliza
            X_tr = normalize(X_tr, mn, mx)
            X_val= normalize(X_val,    mn, mx)
            X_test=normalize(X_test,   mn, mx)

                

            # augmenta & concatena
            if n_aug > 0:
                if raw_root == "ufpe_temp":
                    X_tr, y_tr = augment_train_fold(X_tr, y_tr,         
                                                    n_aug=n_aug, seed=seed,dataset='ufpe')
                else:
                    X_tr, y_tr = augment_train_fold(X_tr, y_tr,         
                                                    n_aug=n_aug, seed=seed, dataset='uff')
                    
                with open("modelos/random_seed.txt", "a") as f:
                    f.write(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}\n")
            

            if resize:
                
                X_tr = np.expand_dims(X_tr, axis=-1)
                X_val= np.expand_dims(X_val, axis=-1)
                X_test= np.expand_dims(X_test, axis=-1)

                if resize_method == "GrayPadding":
                    X_tr = tf_letterbox(X_tr, resize_to)
                    X_val = tf_letterbox(X_val, resize_to)
                    X_test = tf_letterbox(X_test, resize_to)
                elif resize_method == "BlackPadding":
                    X_tr = tf_letterbox_black(X_tr, resize_to)
                    X_val = tf_letterbox_black(X_val, resize_to)
                    X_test = tf_letterbox_black(X_test, resize_to)
                elif resize_method == "Distorcido":
                    X_tr = tf.image.resize(X_tr, (224,224), method = "bilinear")
                    X_val = tf.image.resize(X_val, (224,224), method = "bilinear")
                    X_test = tf.image.resize(X_test, (224,224), method = "bilinear")

                X_tr = tf.clip_by_value(X_tr, 0, 1).numpy().squeeze(axis=-1)
                X_val = tf.clip_by_value(X_val, 0, 1).numpy().squeeze(axis=-1)
                X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(axis=-1)


            if segmenter != "none":
                if segmenter == "unet":
                    X_tr, X_val, X_test = unet_segmenter(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com UNet concluída.")   
                elif segmenter == "yolo":
                    X_tr, X_val, X_test = segment_with_yolo(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com YOLO concluída.")
                else:
                    raise ValueError("segmenter deve ser 'none', 'unet' ou 'yolo'")

                
            if isinstance(model, str):
                if model == "yolo":
                    print("Modelo YOLO selecionado.")
                
            else:
                
                if model.__name__ == "Vgg_16_pre_trained" or model.__name__ == "resnet50_pre_trained":
                    X_tr = (X_tr * 255).astype(np.uint8)
                    X_val = (X_val * 255).astype(np.uint8)
                    X_test = (X_test * 255).astype(np.uint8)
                    
                    # A VGG16 precisa do pré-processamento do ImageNet

                    if channel_method == "MapaCalor":

                        imgs_tr = []
                        imgs_val = []
                        imgs_test = []
                        
                        for img in X_tr:
                            img = img.astype(np.uint8)
                            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            imgs_tr.append(img)

                        for img in X_val:
                            img = img.astype(np.uint8)
                            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            imgs_val.append(img)

                        for img in X_test:
                            img = img.astype(np.uint8)
                            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            imgs_test.append(img)

                        X_tr = np.array(imgs_tr)
                        X_val = np.array(imgs_val)
                        X_test = np.array(imgs_test)

                    else:
                        print(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}")
                        X_tr = np.stack((X_tr,) * 3, axis=-1)
                        X_val = np.stack((X_val,) * 3, axis=-1)
                        X_test = np.stack((X_test,) * 3, axis=-1)
                        print(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}")

                    
                    if model.__name__ == "Vgg_16_pre_trained":
                        X_tr = vgg_preprocess_input(X_tr)
                        X_val = vgg_preprocess_input(X_val)
                        X_test = vgg_preprocess_input(X_test)
                    elif model.__name__ == "resnet50_pre_trained":
                        X_tr = resnet_preprocess_input(X_tr)
                        X_val = resnet_preprocess_input(X_val)
                        X_test = resnet_preprocess_input(X_test)

            # # ----------- VERIFICAÇÃO DA FAIXA DE VALORES -----------
            # print("\n--- Faixas de Valores após o Pré-processamento ---")
            # print(f"Conjunto de Treino: min={X_tr.min():.4f}, max={X_tr.max():.4f}")
            # print(f"Conjunto de Validação: min={X_val.min():.4f}, max={X_val.max():.4f}")
            # print(f"Conjunto de Teste: min={X_test.min():.4f}, max={X_test.max():.4f}")
            # print("---------------------------------------------------\n")
            

            if model == "yolo":
                save_split_to_png(X_tr, y_tr, "train", root=f"dataset_fold_{fold+1}")
                save_split_to_png(X_val, y_val, "val",   root=f"dataset_fold_{fold+1}")
                save_split_to_png(X_test, y_test, "test", root=f"dataset_fold_{fold+1}")

                print(f"Treinando YOLOv8 para o fold {fold+1} com seed {seed}...")

                # Treinamento YOLO
                model_f = YOLO('yolov8s-cls.pt')
                start_time = time.time() 

                model_f.train(
                    data=f"dataset_fold_{fold+1}",
                    epochs=100,
                    patience=50,
                    batch=16,
                    #lr0=0.0005,
                    optimizer='AdamW',
                    #weight_decay=0.0005,
                    #hsv_h=0.1,
                    #hsv_s=0.2,
                    #flipud=0.3,
                    #mosaic=0.1,
                    #mixup=0.1,
                    workers=0,
                    pretrained=False,
                    amp=False,
                    deterministic=True,
                    seed=seed,
                    project="runs/classify",
                    name=f"YOLOv8_cls_fold_{fold+1}_seed_{seed}"
                )
                
                end_time = time.time()

                # Validação
                metrics = model_f.val(
                    data=f"dataset_fold_{fold+1}",
                    project="runs/classify/val",
                    name=f"fold_{fold+1}_seed_{seed}",
                    save_json=True
                )

                # Dados para salvar
                results_to_save = {
                    'top1_accuracy': metrics.top1,
                    'top5_accuracy': metrics.top5,
                    'fitness': metrics.fitness,
                    'training_time_formatted': f"{end_time - start_time:.2f} s",  # Formatado como string
                    'all_metrics': metrics.results_dict,
                    'speed': metrics.speed
                }

                # Salvar em JSON
                with open(f'runs/classify/val/fold_{fold+1}_seed_{seed}/results_fold_{fold+1}_seed_{seed}.json', 'w') as f:
                    json_module.dump(results_to_save, f, indent=4)

                """
                # Extraindo métricas
                accuracy = metrics.results_dict['accuracy_top1']
                precision = metrics.results_dict['precision']
                recall = metrics.results_dict['recall']
                f1 = metrics.results_dict['f1']

                # Salvando no mesmo arquivo de log dos outros modelos
                with open(log_txt, "a") as f:
                    f.write(f"\nYOLO Validation - Fold {fold+1:02d}\n")
                    f.write(f"Accuracy: {accuracy:.4f}\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"F1-Score: {f1:.4f}\n")
                    f.write("-"*50 + "\n")  # Separador visual

                """

            else:

                if model == Vgg_16 or model.__name__ == 'Vgg_16_pre_trained':
                    obj = model()
                    model_f = obj.model
                    print("VGG")
                else:
                    model_f   = model()
                    print("ResNet")

                ckpt    = f"modelos/{model.__name__}/{message}_{angle}_F{fold}.h5"
                log_txt = f"history/{model.__name__}/{message}_{angle}.txt"
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                os.makedirs(os.path.dirname(log_txt), exist_ok=True)


                start_time = time.time()
                
                history = model_f.fit(X_tr, y_tr,
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
                
                end_time = time.time()

                # ---------- avaliação ----------
                y_pred = (model_f.predict(X_test) > 0.5).astype(int).ravel()

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
                            f"F1={f1:.4f}\n"
                            f"Tempo de treinamento={end_time - start_time:.2f} s\n")
                    
                plot_convergence(history, model.__name__, angle, fold, message)
            

            delete_folder("dataset_fold")
            clear_memory()
        
        max_retries = 2
        
        for attempt in range(1,max_retries + 1):
            try:
                run_fold()
                break
            except (tf.errors.ResourceExhaustedError, RuntimeError) as e:
                error_text = str(e).lower()
                if ("out of memory" not in error_text and
                    "oom" not in error_text and
                    "failed to allocate memory" not in error_text):
                    raise
                os.makedirs("logs", exist_ok=True)
                with open("logs/oom_errors.txt", "a") as f:
                    f.write(f"[Fold {fold+1}] OOM na tentativa {attempt}\n")

                if attempt == max_retries:
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/oom_errors.txt", "a") as f:
                        f.write(f"Máximo de tentativas atingido. Abortando …")
                        raise
                clear_memory()

# ------------------------------- inicio ufpe

# CARREGANDO IMAGENS DA UFPE       
def load_jpg_images(base_dir):
    """
    Carrega imagens da estrutura específica:
    base_dir/
        Frontal/
            healthy/
                imagens.jpg
            sick/
                imagens.jpg
    
    Returns:
        tuple: (images, labels) onde:
            - images: Array numpy das imagens
            - labels: Array numpy dos rótulos (0=healthy, 1=sick)
    """
    # Caminho completo para a pasta Frontal
    frontal_path = os.path.join(base_dir, 'Frontal')
    
    # Verifica se a estrutura está correta
    if not os.path.exists(frontal_path):
        raise ValueError(f"Diretório 'Frontal' não encontrado em {base_dir}")
    
    images = []
    labels = []
    
    # Processa cada classe
    for class_name in ['healthy', 'sick']:
        class_path = os.path.join(frontal_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"  Aviso: Pasta '{class_name}' não encontrada em {frontal_path}")
            continue
            
        label = 0 if class_name == 'healthy' else 1
        
        # Processa cada imagem
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(label)
                    else:
                        print(f"Falha ao carregar: {img_path}")
                except Exception as e:
                    print(f" Erro processando {img_path}: {str(e)}")
    
    if not images:
        available = os.listdir(frontal_path)
        raise ValueError(
            f"Nenhuma imagem válida encontrada.\n"
            f"Conteúdo de 'Frontal': {available}\n"
            f"Certifique-se de que existam:\n"
            f"Frontal/healthy/\n"
            f"Frontal/sick/\n"
            f"com imagens .jpg/.jpeg/.png dentro"
        )
    
    return np.array(images), np.array(labels)
             
def evaluate_segmentation(model_path, x_val, y_val):
    model = tf.keras.models.load_model(model_path)
    pred    = (model.predict(x_val) > 0.5).astype(np.uint8)
    true    = (y_val > 0.5).astype(np.uint8)

    pred    = np.squeeze(pred, axis=-1)

    metrics = {
        "precision": precision_score_(true, pred),
        "recall"   : recall_score_(true, pred),
        "accuracy" : accuracy_score_(true, pred),
        "dice"     : dice_coef_(true, pred),
        "iou"      : iou_(true, pred)
    }
    return metrics

def evaluate_yolo_on_folder(model_path, ds_root,
                            split="val", imgsz=(224,224), thr=0.5):
    """
    ds_root/
      ├ images/val/*.png
      ├ masks/val/*.png   (branco‑preto 0/255 ou 0/1)
    """
    model   = YOLO(model_path)
    img_dir = os.path.join(ds_root, "images", split)
    msk_dir = os.path.join(ds_root, "masks",  split)

    y_true, y_pred = [], []

    for img_file in glob.glob(os.path.join(img_dir, "*")):
        name   = os.path.splitext(os.path.basename(img_file))[0]
        msk_gt = cv2.imread(os.path.join(msk_dir, name + ".png"),
                            cv2.IMREAD_GRAYSCALE)
        # garante mesma resolução
        msk_gt = cv2.resize(msk_gt, imgsz, interpolation=cv2.INTER_NEAREST)
        msk_gt = (msk_gt > 127).astype(np.uint8)

        # --- predição ------------------------------------
        img = cv2.imread(img_file)
        img = cv2.resize(img, imgsz)
        res = model.predict(img, verbose=False)
        canvas = np.zeros(imgsz, np.uint8)
        if res and len(res[0].masks):
            for m in res[0].masks.data:
                m = cv2.resize(m.cpu().numpy(), imgsz)
                canvas |= (m > thr).astype(np.uint8)

        # ---- armazena vetores 1‑D p/ métricas ------------
        y_true.append(msk_gt.ravel())
        y_pred.append(canvas.ravel())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    metrics = {
        "precision": precision_score_(y_true, y_pred),
        "recall"   : recall_score_(   y_true, y_pred),
        "accuracy" : accuracy_score_( y_true, y_pred),
        "dice"     : dice_coef_(y_true, y_pred),
        "iou"      : iou_(y_true, y_pred)
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
            modelo_train = modelo_object

            #model.summary()
                

            # Salva a seed em um arquivo de texto
            with open("modelos/random_seed.txt", "a") as file:
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
                f.write(f"SEED{str(seed)}\n")
                f.write("\n")
                    
                plot_convergence(history, model_func.__name__, angulo, i, message)

#TODO: calcular f1, IoU, recall, accuracy
#TODO: comparar unet com yolo

# Função para calcular Pixel Accuracy
def pixel_accuracy(y_true, y_pred):
    # y_true e y_pred devem ter a mesma dimensão e serem binários
    correct = np.sum(y_true == y_pred)
    total = y_true.size
    return correct / total

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

#FABS
def save_split_to_png(images, labels, split_name, root="dataset_fold"):
    """
    Salva um split de imagens/labels em formato PNG no layout aceito pelo YOLOv8n-cls.

    └── dataset_fold/
        ├── train/
        │   ├── 0/
        │   │   ├── 00000.png
        │   │   └── ...
        │   └── 1/
        ├── val/
        └── test/
    Args
    ----
    images : np.ndarray  (N, H, W, C) ou (N, H, W)
    labels : Sequence[int]  rótulo inteiro por imagem
    split_name : str  "train" | "val" | "test"
    root : str ou Path  diretório-raiz do dataset
    """
    out_base = Path(root) / split_name
    out_base.mkdir(parents=True, exist_ok=True)

    for idx, (img, cls) in enumerate(zip(images, labels)):
        # Converte para uint8 [0-255] caso ainda esteja em float [0-1]
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Garante 3 canais
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        # YOLO-cls quer as pastas por classe
        class_dir = out_base / str(cls)
        class_dir.mkdir(parents=True, exist_ok=True)

        fname = class_dir / f"{idx:06d}.png"
        # cv2 espera BGR; converta se seu array já estiver RGB
        cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✓ {split_name} salvo em {out_base}")



def ppeprocessEigenCam(X, y, ids, splits_path, resize_method = "BlackPadding", segment = None, segmenter_path ="" ):
    
    
    with open (splits_path, "r") as f:
        splits = json.load(f)


    
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    X_test = X[test_idx]
    y_test = y[test_idx]
    ids_test = ids[test_idx]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]


    mn, mx = X_train.min(), X_train.max()

    # normaliza
    X_test=normalize(X_test,   mn, mx)
    X_tr=normalize(X_train, mn, mx)
    X_val=normalize(X_val,   mn, mx)

    

    X_tr = np.expand_dims(X_tr, axis=-1)
    X_val= np.expand_dims(X_val, axis=-1)
    X_test= np.expand_dims(X_test, axis=-1)

    # X_tr= tf.image.resize_with_pad(X_tr, 224, 224, method="bicubic")
    # X_val= tf.image.resize_with_pad(X_val, 224, 224, method="bicubic")
    # X_test= tf.image.resize_with_pad(X_test, 224, 224, method="bicubic")


    if resize_method == "GrayPadding":
        X_tr= tf_letterbox(X_tr, 224)
        X_val= tf_letterbox(X_val, 224)
        X_test= tf_letterbox(X_test, 224)
        print("Resize com GrayPadding concluído.")
    elif resize_method == "BlackPadding":
        X_tr= tf_letterbox_black(X_tr, 224)
        X_val= tf_letterbox_black(X_val, 224)
        X_test= tf_letterbox_black(X_test, 224)
        print("Resize com BlackPadding concluído.")
    elif resize_method == "Distorcido":
        X_tr = tf.image.resize(X_tr, (224,224), method = "bilinear")
        X_val = tf.image.resize(X_val, (224,224), method = "bilinear")
        X_test = tf.image.resize(X_test, (224,224), method = "bilinear")
        print("Resize Distorcido concluído.")

    X_tr = tf.clip_by_value(X_tr, 0, 1).numpy().squeeze(axis=-1)
    X_val = tf.clip_by_value(X_val, 0, 1).numpy().squeeze(axis=-1)
    X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(axis=-1)

    if segment != None:
        if segment == "unet":
            X_tr, X_val, X_test = unet_segmenter(X_tr, X_val, X_test, segmenter_path)
            print(f"Segmentação com UNet concluída.")   
        elif segment == "yolo":
            X_tr, X_val, X_test = segment_with_yolo(X_tr, X_val, X_test, segmenter_path)
            print(f"Segmentação com YOLO concluída.")
        elif segment == "none":
            print("Nenhum segmentador foi aplicado.")


    X_test = np.expand_dims(X_test, axis=-1)

    return X_test, y_test, ids_test




def prep_test_data(raw_root, angle, split_json, 
                    resize = True, resize_method = "GrayPadding", resize_to = 224,
                    segmenter = "none", seg_model_path="", rgb=False, channel_method = "MapaCalor"):
    
    """
    Função para preparar as imagens de teste para gerar as matrizes de confusão.
    Segue o mesmo procedimento de processamento do PipeLine de treinamento (train_models_cv)
    """

    exclude_set = listar_imgs_nao_usadas("Termografias_Dataset_Segmentação/images", angle)
    
    X, y, patient_ids = load_raw_images(
            os.path.join(raw_root, angle), exclude=True, exclude_set=exclude_set)
    with open(split_json, "r") as f:
        split = json.load(f)
    tr_idx, te_idx = np.array(split["train_idx"]), np.array(split["test_idx"])
    
    X_tr, X_test = X[tr_idx], X[te_idx]
    y_test       = y[te_idx]

    mn, mx = X_tr.min(), X_tr.max()

    X_test = normalize(X_test, mn, mx)

    if resize:
        X_test = np.expand_dims(X_test, -1)
        
        #X_test = tf.image.resize_with_pad(X_test, resize_to, resize_to, method="bicubic")
        if resize_method == "GrayPadding":
            X_test = tf_letterbox(X_test, resize_to)
        elif resize_method == "BlackPadding":
            X_test = tf_letterbox_black(X_test, resize_to)
        elif resize_method == "Distorcido":
            X_test = tf.image.resize(X_test, (resize_to, resize_to), method="bilinear")
        #X_test = tf_letterbox_Sem_padding(X_test, resize_to)
        #X_test = letterbox_center_crop(X_test, resize_to)
        X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(-1)

    if segmenter == "unet":
        _, _, X_test = unet_segmenter(X_test, X_test, X_test, seg_model_path)
        print(f"Segmentação com UNet concluída.")
    elif segmenter == "yolo":
        _, _, X_test = segment_with_yolo(X_test, X_test, X_test, seg_model_path)
        print(f"Segmentação com YOLO concluída.")

    if rgb:
        X_test = (X_test * 255).astype(np.uint8)

        if channel_method == "MapaCalor":
            imgs_test = []
            for img in X_test:
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_test.append(img)
            X_test = np.array(imgs_test, dtype=np.uint8)

        else:
            X_test = np.stack((X_test,) * 3, axis=-1)

    return X_test, y_test


def _plot_and_save_cm(cm, classes, title, out_png):

    """
    Recebe a matriz de confusão `cm` e os nomes das classes `classes`
    e plota a matriz de confusão com os rótulos das classes.
    O gráfico é salvo no caminho `out_png`.
    """


    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", square=True,
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predito"); plt.ylabel("Real"); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def evaluate_model_cm(model_path,          
                      output_path, 
                      split_json,      
                      raw_root,
                      message,
                      angle="Frontal",
                      resize=True,
                      resize_to=224,
                      resize_method="BlackPadding",
                      segmenter="none",
                      seg_model_path="",
                      classes=("Healthy", "Sick"), rgb=False, channel_method="MapaCalor"):
    """
    Avalia o modelo salvo no fold especificado e gera matriz de confusão.
    """
    os.makedirs(output_path, exist_ok=True)

    X_test, y_test = prep_test_data(raw_root, angle, split_json,
                                     resize,resize_method, resize_to,
                                     segmenter, seg_model_path,  rgb=rgb, channel_method=channel_method)

    
    
    with custom_object_scope({'ResidualUnit': ResidualUnit}):
        model = tf.keras.models.load_model(model_path, compile=False)



    if model.__class__.__name__ == "Vgg_16_pre_trained":
    
        X_test = vgg_preprocess_input(X_test)
    


    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    

    cm = confusion_matrix(y_test, y_pred)
    clf_rep = classification_report(y_test, y_pred, target_names=classes,
                                    output_dict=True, zero_division=0)


    out_png = os.path.join(output_path, f"cm_{message}_{angle}.png")
    
    _plot_and_save_cm(cm, classes,
                      f"Confusion Matrix – {message}",
                      out_png = out_png)
    

    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
                            y_test, y_pred, average="binary",
                            zero_division=0)
    
    out_txt = os.path.join(output_path, f"resultado_{message}_{angle}.txt")

    # salva métrica fold‐a‐fold
    with open(out_txt, "a") as f:
        f.write(f"Acc={acc:.4f}  "
                f"Prec={prec:.4f}  "
                f"Rec={rec:.4f}  "
                f"F1={f1:.4f}\n")

    
    K.clear_session(); gc.collect()

    return y_pred



# resize_with_tf_letterbox.py
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm  

def resize_imgs_masks_dataset(
    img_dir: str,
    mask_dir: str,
    output_base: str,
    target: int = 640,
    resize_method = "BlackPadding"
):
    """
    Redimensiona imagens (.jpg) e máscaras (.png) com tf_letterbox.

    Parâmetros
    ----------
    img_dir      Pasta com as .jpg originais
    mask_dir     Pasta com as .png originais (mesmo nome da imagem)
    output_base  Nova raiz que conterá:  output_base/images  e  output_base/masks
    target       Lado do quadrado de saída (ex. 640)
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    out_img = Path(output_base) / "images"
    out_mask = Path(output_base) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc="Redimensionando"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            print(f"[aviso] Máscara ausente para {stem} — pulando.")
            continue

        # ─── Leitura ----------------------------------------------------------------
        img  = tf.image.decode_jpeg(tf.io.read_file(str(img_path)), channels=3)
        mask = tf.image.decode_png(tf.io.read_file(str(mask_path)), channels=1)

        img  = tf.image.convert_image_dtype(img,  tf.float32)  # 0-1
        mask = tf.image.convert_image_dtype(mask, tf.float32)  # 0-1

        # ─── Letter-box (batch=1) ----------------------------------------------------
        if resize_method == "BlackPadding":
            img_lb  = tf_letterbox_black(tf.expand_dims(img,  0), target=target, mode='bilinear')
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode='nearest')
        elif resize_method == "Distorcido":
            img_lb  = tf.expand_dims(img,  0)
            img_lb = tf.image.resize(img_lb, (target, target), method='bilinear')
            mask_lb = tf.expand_dims(mask, 0)
            mask_lb = tf.image.resize(mask_lb, (target, target), method='nearest')
        elif resize_method == "GrayPadding":
            img_lb  = tf_letterbox(tf.expand_dims(img,  0), target=target, mode='bilinear')
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode='nearest')

        img_lb  = tf.squeeze(img_lb,  0)          # (H,W,3)
        mask_lb = tf.squeeze(mask_lb, 0)          # (H,W,1)

        # ─── Pós-processamento -------------------------------------------------------
        img_uint8  = tf.image.convert_image_dtype(img_lb, tf.uint8, saturate=True)
        mask_bin   = tf.cast(mask_lb > 0.5, tf.uint8) * 255  # 0 ou 255

        # ─── Grava ------------------------------------------------------------------
        tf.io.write_file(
            str(out_img / f"{stem}.jpg"),
            tf.io.encode_jpeg(img_uint8, quality=95)
        )
        tf.io.write_file(
            str(out_mask / f"{stem}.png"),
            tf.io.encode_png(mask_bin)
        )

    print(f"\nConcluído!  Novas pastas:\n  imagens → {out_img}\n  máscaras → {out_mask}")


def resize_imgs_masks_dataset_png(
    img_dir: str,
    mask_dir: str,
    output_base: str,
    target: int = 640,
    resize_method = "BlackPadding"
):
    """
    Redimensiona imagens (.jpg) e máscaras (.png) com tf_letterbox.

    Parâmetros
    ----------
    img_dir      Pasta com as .jpg originais
    mask_dir     Pasta com as .png originais (mesmo nome da imagem)
    output_base  Nova raiz que conterá:  output_base/images  e  output_base/masks
    target       Lado do quadrado de saída (ex. 640)
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    out_img = Path(output_base) / "images"
    out_mask = Path(output_base) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(img_dir.glob("*.png")), desc="Redimensionando"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            print(f"[aviso] Máscara ausente para {stem} — pulando.")
            continue

        # ─── Leitura ----------------------------------------------------------------
        img  = tf.image.decode_png(tf.io.read_file(str(img_path)), channels=1, dtype= tf.uint16)
        mask = tf.image.decode_png(tf.io.read_file(str(mask_path)), channels=1)

        img  = tf.image.convert_image_dtype(img,  tf.float32)  # 0-1
        mask = tf.image.convert_image_dtype(mask, tf.float32)  # 0-1

        # ─── Letter-box (batch=1) ----------------------------------------------------
        if resize_method == "BlackPadding":
            img_lb  = tf_letterbox_black(tf.expand_dims(img,  0), target=target, mode='bilinear')
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode='nearest')
        elif resize_method == "Distorcido":
            img_lb  = tf.expand_dims(img,  0)
            img_lb = tf.image.resize(img_lb, (target, target), method='bilinear')
            mask_lb = tf.expand_dims(mask, 0)
            mask_lb = tf.image.resize(mask_lb, (target, target), method='nearest')
        elif resize_method == "GrayPadding":
            img_lb  = tf_letterbox(tf.expand_dims(img,  0), target=target, mode='bilinear')
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode='nearest')

        img_lb  = tf.squeeze(img_lb,  0)          # (H,W,3)
        mask_lb = tf.squeeze(mask_lb, 0)          # (H,W,1)

        # ─── Pós-processamento -------------------------------------------------------
        img_uint16  = tf.image.convert_image_dtype(img_lb, tf.uint16, saturate=True)
        mask_bin   = tf.cast(mask_lb > 0.5, tf.uint8) * 255  # 0 ou 255

        # ─── Grava ------------------------------------------------------------------
        tf.io.write_file(
            str(out_img / f"{stem}.png"),
            tf.io.encode_png(img_uint16)
        )
        tf.io.write_file(
            str(out_mask / f"{stem}.png"),
            tf.io.encode_png(mask_bin)
        )

    print(f"\nConcluído!  Novas pastas:\n  imagens → {out_img}\n  máscaras → {out_mask}")



    from pathlib import Path
from tqdm import tqdm
import tensorflow as tf

def resize_imgs_two_masks_dataset(
    img_dir: str,
    mask_breast_dir: str,
    mask_marker_dir: str,
    output_base: str,
    target: int = 640,
    resize_method: str = "BlackPadding",
    min_val_mask: float = 0.5
):
    """
    Redimensiona imagens (.jpg) e DUAS máscaras (.png) preservando o alinhamento
    e grava em:

        output_base/
            images/
            masks_breast/
            masks_marker/

    Parâmetros
    ----------
    img_dir          Pasta com .jpg originais
    mask_breast_dir  Pasta com máscaras da classe Breast (mesmo nome da imagem)
    mask_marker_dir  Pasta com máscaras da classe Marker (mesmo nome da imagem)
    output_base      Raiz de saída
    target           Lado do quadrado (ex. 640)
    resize_method    ["BlackPadding", "GrayPadding", "Distorcido"]
    min_val_mask     limiar (float 0–1) p/ binarizar máscara
    """

    img_dir        = Path(img_dir)
    mask_breast_dir = Path(mask_breast_dir)
    mask_marker_dir = Path(mask_marker_dir)

    out_img   = Path(output_base) / "images"
    out_mb    = Path(output_base) / "masks_breast"
    out_mm    = Path(output_base) / "masks_marker"
    for p in (out_img, out_mb, out_mm):
        p.mkdir(parents=True, exist_ok=True)

    # ── loop principal ───────────────────────────────────────────────────────────
    for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc="Redimensionando"):
        stem = img_path.stem
        mb_path = mask_breast_dir / f"{stem}.png"
        mm_path = mask_marker_dir / f"{stem}.png"

        # Verifica a existência das duas máscaras
        if not (mb_path.exists() and mm_path.exists()):
            print(f"[aviso] Máscara faltando para {stem} — pulando.")
            continue

        # ── Leitura ──────────────────────────────────────────────────────────────
        img  = tf.image.decode_jpeg(tf.io.read_file(str(img_path)), channels=3)
        mb   = tf.image.decode_png (tf.io.read_file(str(mb_path)), channels=1)
        mm   = tf.image.decode_png (tf.io.read_file(str(mm_path)), channels=1)

        img = tf.image.convert_image_dtype(img, tf.float32)   # 0–1
        mb  = tf.image.convert_image_dtype(mb,  tf.float32)
        mm  = tf.image.convert_image_dtype(mm,  tf.float32)

        # ── Letter-box / resize ─────────────────────────────────────────────────
        if resize_method == "BlackPadding":
            img_lb  = tf_letterbox_black(tf.expand_dims(img, 0), target, mode='bilinear')
            mb_lb   = tf_letterbox_black(tf.expand_dims(mb,  0), target, mode='nearest')
            mm_lb   = tf_letterbox_black(tf.expand_dims(mm,  0), target, mode='nearest')

        elif resize_method == "GrayPadding":
            img_lb  = tf_letterbox(tf.expand_dims(img, 0), target, mode='bilinear')
            mb_lb   = tf_letterbox_black(tf.expand_dims(mb,  0), target, mode='nearest')
            mm_lb   = tf_letterbox_black(tf.expand_dims(mm,  0), target, mode='nearest')

        elif resize_method == "Distorcido":          # “stretch” p/ quadrado
            img_lb = tf.image.resize(tf.expand_dims(img, 0), (target, target),
                                     method='bilinear')
            mb_lb  = tf.image.resize(tf.expand_dims(mb,  0), (target, target),
                                     method='nearest')
            mm_lb  = tf.image.resize(tf.expand_dims(mm,  0), (target, target),
                                     method='nearest')
        else:
            raise ValueError(f"resize_method desconhecido: {resize_method}")

        img_lb = tf.squeeze(img_lb, 0)
        mb_lb  = tf.squeeze(mb_lb,  0)
        mm_lb  = tf.squeeze(mm_lb,  0)

        # ── Pós-processamento ───────────────────────────────────────────────────
        img_uint8 = tf.image.convert_image_dtype(img_lb, tf.uint8, saturate=True)
        mb_bin    = tf.cast(mb_lb > min_val_mask, tf.uint8) * 255
        mm_bin    = tf.cast(mm_lb > min_val_mask, tf.uint8) * 255

        # ── Grava ───────────────────────────────────────────────────────────────
        tf.io.write_file(str(out_img / f"{stem}.jpg"),
                         tf.io.encode_jpeg(img_uint8, quality=95))
        tf.io.write_file(str(out_mb  / f"{stem}.png"),
                         tf.io.encode_png(mb_bin))
        tf.io.write_file(str(out_mm  / f"{stem}.png"),
                         tf.io.encode_png(mm_bin))

    print(f"\nConcluído!\n  Imagens  → {out_img}\n  Breast   → {out_mb}\n  Marker   → {out_mm}")


    
import numpy as np
from PIL import Image


def carregar_e_salvar_imagens(pasta_origem, pasta_destino):    
    # Criar pasta de destino se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Listar todos os arquivos PNG
    arquivos = [f for f in os.listdir(pasta_origem) if f.endswith('.png')]
    
    print(f"Encontrados {len(arquivos)} arquivos PNG")
    
    for i, arquivo in enumerate(arquivos):
        caminho_origem = os.path.join(pasta_origem, arquivo)
        nome_base = os.path.splitext(arquivo)[0]
        caminho_destino = os.path.join(pasta_destino, f"{nome_base}.txt")
        
        # Carregar imagem
        img = plt.imread(caminho_origem)
        
        # Converter para escala de cinza se necessário
        img = img[:, :, 0]  # Pega apenas o primeiro canal
        
        # Salvar como arquivo de texto
        np.savetxt(caminho_destino, img, fmt="%.2f", delimiter=" ")
        
    
    print("Processamento concluído!")

def normalizar_imagens_texto(pasta_origem, pasta_destino):
    
    # Criar pasta de destino
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Listar arquivos de texto
    arquivos = [f for f in os.listdir(pasta_origem) if f.endswith('.txt')]
    print(f"Processando {len(arquivos)} arquivos de texto...")
    
    for arquivo in arquivos:
        # Carregar dados do arquivo texto
        caminho_origem = os.path.join(pasta_origem, arquivo)
        dados_imagem = np.loadtxt(caminho_origem)
        
        # Normalizar para [0, 1]
        min_val = np.min(dados_imagem)
        max_val = np.max(dados_imagem)
        
        if max_val > min_val:  # Evitar divisão por zero
            dados_normalizados = (dados_imagem - min_val) / (max_val - min_val)
        else:
            dados_normalizados = dados_imagem  # Se todos valores forem iguais
        
        # Salvar arquivo normalizado
        caminho_destino = os.path.join(pasta_destino, arquivo)
        np.savetxt(caminho_destino, dados_normalizados, fmt='%.2f', delimiter=' ')
    
    print("Normalização concluída!")



import numpy as np
import cv2
import os
import json

def transform_temp_img(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    limites = {}

    for fname in os.listdir(input_folder):
        if fname.endswith(".txt"):
            path = os.path.join(input_folder, fname)

            with open(path, 'r') as f:
                delim = ';' if ';' in f.readline() else ' '
                f.seek(0)
            temperatura = np.loadtxt(path, delimiter= delim)

            temp_min, temp_max = float(temperatura.min()), float(temperatura.max())
            limites[fname] = {"min": temp_min, "max": temp_max}

            norm = ((temperatura - temp_min) / (temp_max - temp_min) * 65535).astype(np.uint16)

            out_name = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(output_folder, out_name), norm)

    # salvar os limites em JSON
    with open(os.path.join(output_folder, "limites.json"), "w") as f:
        json.dump(limites, f, indent=4)


import os
import json
import cv2
import numpy as np

def recuperar_img(input_folder, output_folder):
    """
    Converte imagens PNG 16-bit de volta para matrizes de temperatura,
    usando os limites salvos em limites.json.
    O JSON deve ter as chaves no formato 'nome.txt'.
    """
    os.makedirs(output_folder, exist_ok=True)

    json_path = os.path.join(input_folder, "limites.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON de limites não encontrado em: {json_path}")

    with open(json_path, "r") as f:
        limites = json.load(f)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".png"):
            continue

        path = os.path.join(input_folder, fname)
        editada = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if editada is None:
            print(f"Aviso: não foi possível ler {fname}. Pulando.")
            continue

        if editada.ndim == 3:
            editada = editada[:, :, 0]

        # gerar a chave do JSON adicionando '.txt' ao nome base
        original_txt = os.path.splitext(fname)[0] + ".txt"

        if original_txt not in limites:
            print(f"AVISO: chave {original_txt} não encontrada. Chaves do JSON: {list(limites.keys())[:10]}")
            continue


        temp_min = limites[original_txt]["min"]
        temp_max = limites[original_txt]["max"]

        # converter de volta para escala de temperatura
        recuperada = editada.astype(np.float32) / 65535.0 * (temp_max - temp_min) + temp_min

        out_name = os.path.splitext(fname)[0] + ".txt"
        np.savetxt(os.path.join(output_folder, out_name), recuperada, fmt="%.6f")
        print(f"Arquivo recuperado: {out_name}")

import os
import json
import numpy as np

def gerar_limites_originais_txt(input_folder, output_json):
    """
    Gera limites (min e max) de cada arquivo TXT com matriz de temperatura
    e salva em JSON. Cada chave do JSON será 'nome.txt'.
    """
    limites = {}

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".txt"):
            path = os.path.join(input_folder, fname)
            
            # Tentar detectar delimitador
            with open(path, 'r') as f:
                first_line = f.readline()
            delimiter = None
            if ';' in first_line:
                delimiter = ';'
            elif ',' in first_line:
                delimiter = ','

            # Carregar o TXT
            if delimiter:
                temperatura = np.loadtxt(path, delimiter=delimiter)
            else:
                temperatura = np.loadtxt(path)

            temp_min, temp_max = float(temperatura.min()), float(temperatura.max())
            limites[fname] = {"min": temp_min, "max": temp_max}
            print(f"Adicionado: {fname} -> min: {temp_min}, max: {temp_max}")

    if not limites:
        print("AVISO: Nenhum arquivo TXT foi processado! Verifique o diretório.")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(limites, f, indent=4)

    print(f"Arquivo de limites salvo em: {output_json}")


def comparar_resultados_modelo_completo(exp1_base, exp1_modelo, exp2_base, exp2_modelo, mensagem="mensagem_comparacao:", output_dir="Comparacao"):
    """
    Compara dois modelos específicos de experimentos diferentes.
    Considera a estrutura exata:
    CAM_results/
        Acertos/<modelo>/Health
        Acertos/<modelo>/Sick
        Erros/<modelo>/Health
        Erros/<modelo>/Sick
    """

    def coletar_ids(exp_base, modelo):

        resultado = {"Acertos": {"Health": set(), "Sick": set()},
                    "Erros": {"Health": set(), "Sick": set()}}
        
        totais_arquivos = {"Acertos": {"Health": 0, "Sick": 0},
                       "Erros":   {"Health": 0, "Sick": 0}}
        
        padrao_chave = re.compile(r"^(id_\d+.*)\.png$", re.IGNORECASE)
                
        for tipo in ["Acertos", "Erros"]:
            for classe in ["Health", "Sick"]:
                class_path = os.path.join(exp_base, tipo, modelo, classe)
                if os.path.exists(class_path):
                    arquivos = [f for f in os.listdir(class_path) if f.lower().endswith(".png")]
                    totais_arquivos[tipo][classe] += len(arquivos)
                    for f in arquivos:
                        m = padrao_chave.match(f)
                        if m:
                            chave = m.group(1)  # p.ex. "id_365_overlay41"
                            resultado[tipo][classe].add(chave)

        return resultado, totais_arquivos


    exp1, exp1_totais = coletar_ids(exp1_base, exp1_modelo)
    exp2, exp2_totais = coletar_ids(exp2_base, exp2_modelo)

    #pasta de saída
    os.makedirs(output_dir, exist_ok=True)
    relatorio_path = os.path.join(output_dir, "relatorio_comparacao.txt")
    
    if mensagem == "mensagem_comparacao:":
        with open(relatorio_path, "w") as f:
            f.write(f"Comparando o modelo {exp1_modelo} com {exp2_modelo}\n")
    else:
        with open(relatorio_path, "w") as f:
            f.write(mensagem)

    with open(relatorio_path, "a") as report:            
        for classe in ["Health", "Sick"]:
            
            report.write(f"\n============================================= {classe} =============================================\n")

            exp1_ac = exp1["Acertos"][classe]
            exp1_er = exp1["Erros"][classe]
            exp2_ac = exp2["Acertos"][classe]
            exp2_er = exp2["Erros"][classe]
            
            report.write(f"\n---------------------------- Quantitativo total ----------------------------\n")
            report.write(f"Total de imagens saudáveis/acertos (modelo 1): {len(exp1_ac)}\n" if classe=="Health" else f"Total de imagens doentes/acertos (modelo 1): {len(exp1_ac)}\n")
            report.write(f"Total de imagens saudáveis/erros (modelo 1): {len(exp1_er)}\n" if classe=="Health" else f"Total de imagens doentes/erros (modelo 1): {len(exp1_er)}\n")
            report.write(f"Total de imagens saudáveis/acertos (modelo 2): {len(exp2_ac)}\n" if classe=="Health" else f"Total de imagens doentes/acertos (modelo 2): {len(exp2_ac)}\n")
            report.write(f"Total de imagens saudáveis/erros (modelo 2): {len(exp2_er)}\n" if classe=="Health" else f"Total de imagens doentes/erros (modelo 2): {len(exp2_er)}\n")
            report.write(f"\n------------------------------------------------------------------------------\n")

            
            # Categorias
            melhorou = exp1_er & exp2_ac      # Erro -> Acerto
            piorou = exp1_ac & exp2_er         # Acerto -> Erro
            manteve_ac = exp1_ac & exp2_ac
            manteve_er = exp1_er & exp2_er

            categorias = {
                "Imagens que o modelo 1 errava e agora o modelo 2 acerta:": melhorou,
                "Imagens que o modelo 1 acertava e agora o modelo 2 erra": piorou,
                "Manteve_Acerto": manteve_ac,
                "Manteve_Erro": manteve_er
            }

            #relatório
            report.write(f"\n---------------------------- Análise ----------------------------\n")
            for nome, ids_set in categorias.items():
                report.write(f"{nome}: {len(ids_set)} imagens\n")
            report.write("\n")
            for nome, ids_set in categorias.items():
                report.write(f"--- {nome} ---\n" + "\n".join(sorted(ids_set)) + "\n\n")

            #guarda as imagens
            for nome, ids_set in categorias.items():
                destino = os.path.join(output_dir, nome, classe)
                os.makedirs(destino, exist_ok=True)
                
                for img_id in ids_set:
                    # Procurar em todas as pastas possíveis
                    for tipo, base, modelo in [
                        ("Acertos", exp1_base, exp1_modelo),
                        ("Erros", exp1_base, exp1_modelo),
                        ("Acertos", exp2_base, exp2_modelo),
                        ("Erros", exp2_base, exp2_modelo)
                    ]:
                        class_path = os.path.join(base, tipo, modelo, classe)
                        if os.path.exists(class_path):
                            for f in os.listdir(class_path):
                                if f.lower().startswith(img_id.lower()) and f.lower().endswith(".png"):  
                                    shutil.copy(os.path.join(class_path, f), os.path.join(destino, f))

    print(f"Comparação concluída!")
    print(f"Relatório: {relatorio_path}")
    print(f"Imagens copiadas para: {output_dir}")

import os
import cv2

def unir_mascaras(pasta_breast, pasta_marker, pasta_saida):
    """
    Une máscaras de duas pastas diferentes (mesmo nome de arquivo)
    e salva a união em uma pasta final.

    Args:
        pasta_breast (str): Caminho da pasta com as máscaras dos seios.
        pasta_marker (str): Caminho da pasta com as máscaras dos marcadores.
        pasta_saida (str): Caminho da pasta onde serão salvas as máscaras unidas.
    """
    os.makedirs(pasta_saida, exist_ok=True)

    # Listar arquivos da pasta de seios
    for fname in os.listdir(pasta_breast):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue  # pular arquivos que não são imagens

        caminho_breast = os.path.join(pasta_breast, fname)
        caminho_marker = os.path.join(pasta_marker, fname)

        if not os.path.exists(caminho_marker):
            print(f"Aviso: não encontrei {fname} em {pasta_marker}, pulando...")
            continue

        # Carregar em escala de cinza
        mask_breast = cv2.imread(caminho_breast, 0)
        mask_marker = cv2.imread(caminho_marker, 0)

        if mask_breast is None or mask_marker is None:
            print(f"Aviso: não consegui carregar {fname}, pulando...")
            continue

        # Combinar (união)
        combined = cv2.bitwise_or(mask_breast, mask_marker)

        # Binarizar para garantir 0/255
        _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

        # Salvar na pasta de saída
        cv2.imwrite(os.path.join(pasta_saida, fname), combined)

    print(f"Máscaras unidas foram salvas em: {pasta_saida}")

# =========================================================================================
# --------------------------------COMPARACOES ESTATISTICAS--------------------------------
# =========================================================================================
# Para fazer ESSAS comparações, tem que ter gerados os relatórios acima de comparações em "comparar_resultados_modelo_completo" e gerar as pastas com os resultados

import os
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

def analisar_comparacoes_cruzadas(base_dir="comparacoes", output_dir="Analise_Global_Comparacoes"):
    """
    Analisa múltiplas estratégias de comparação através dos folds
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    estrategias = defaultdict(list)
    
    #   IDENTIFICAÇÃO DOS FOLDS
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if os.path.isdir(item_path):
            # Padrão flexível para identificar folds
            if '_F' in item and item.split('_F')[-1].isdigit():
                partes = item.split('_F')
                nome_estrategia = partes[0]
                fold_num = int(partes[1])
                
                estrategias[nome_estrategia].append((fold_num, item_path))
    
    # Ordenar por fold
    for estrategia in estrategias:
        estrategias[estrategia].sort(key=lambda x: x[0])
    
    print("  ESTRATÉGIAS E FOLDS IDENTIFICADOS:")
    for estrategia, folds in estrategias.items():
        print(f"   {estrategia}: {len(folds)} folds → {[f[0] for f in folds]}")
    
    if not estrategias:
        print(" Nenhuma pasta com padrão *_F* encontrada!")
        return None
    
    # Coletar métricas
    metricas_globais = coletar_metricas_globais(estrategias, output_dir)
    
    if metricas_globais is None:
        return None
    
    # Análises
    analisar_estrategias(metricas_globais, output_dir)
    analisar_consistencia_folds(metricas_globais, output_dir)
    
    return metricas_globais

def coletar_metricas_globais(estrategias, output_dir):
    """Coleta métricas de forma robusta baseada no formato conhecido"""
    
    metricas = {
        'estrategia': [],
        'fold': [],
        'classe': [],
        'categoria': [],
        'quantidade': []
    }
    
    total_metricas = 0
    
    for estrategia_nome, folds in estrategias.items():
        for fold, pasta in folds:
            relatorio_path = os.path.join(pasta, "relatorio_comparacao.txt")
            
            if os.path.exists(relatorio_path):
                print(f"📖 Processando: {estrategia_nome} - Fold {fold}")
                
                # Extrair métricas para Health e Sick
                for classe in ['Health', 'Sick']:
                    metricas_classe = extrair_metricas_por_classe(relatorio_path, classe)
                    
                    for categoria, quantidade in metricas_classe.items():
                        metricas['estrategia'].append(estrategia_nome)
                        metricas['fold'].append(fold)
                        metricas['classe'].append(classe)
                        metricas['categoria'].append(categoria)
                        metricas['quantidade'].append(quantidade)
                        total_metricas += 1
            else:
                print(f"   Relatório não encontrado: {relatorio_path}")
    
    print(f"  Total de métricas coletadas: {total_metricas}")
    
    if total_metricas == 0:
        print(" Nenhuma métrica válida foi coletada!")
        return None
    
    #   VERIFICAR INTEGRIDADE DOS DADOS
    comprimentos = {key: len(values) for key, values in metricas.items()}
    print(f"  Comprimentos das listas: {comprimentos}")
    
    if len(set(comprimentos.values())) != 1:
        print(" ERRO: As listas têm comprimentos diferentes!")
        return None
    
    try:
        df_metricas = pd.DataFrame(metricas)
        df_metricas.to_csv(os.path.join(output_dir, "metricas_brutas.csv"), index=False)
        print(f"  DataFrame criado com sucesso: {len(df_metricas)} registros")
        
        return {'df': df_metricas}
    
    except Exception as e:
        print(f" Erro ao criar DataFrame: {e}")
        return None

def extrair_metricas_por_classe(relatorio_path, classe_alvo):
    """
    Extrai métricas ESPECÍFICAS do formato exato do seu relatório
    """
    
    metricas = {
        'acertos_modelo1': 0,
        'erros_modelo1': 0, 
        'acertos_modelo2': 0,
        'erros_modelo2': 0,
        'melhorou': 0,
        'piorou': 0,
        'manteve_acerto': 0,
        'manteve_erro': 0
    }
    
    try:
        with open(relatorio_path, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
        
        dentro_classe = False
        
        for i, linha in enumerate(linhas):
            linha = linha.strip()
            
            # Identificar início da seção da classe
            if f"============================================= {classe_alvo} =============================================" in linha:
                dentro_classe = True
                print(f"     Encontrou seção: {classe_alvo}")
                continue
            elif dentro_classe and "=============================================" in linha and classe_alvo not in linha:
                dentro_classe = False
                continue
            
            if not dentro_classe:
                continue
            
            #   DEBUG: Mostrar linhas relevantes
            if "Total de imagens" in linha or "Imagens que o modelo" in linha or "Manteve_" in linha:
                print(f"     Linha {i}: {linha}")
            
            #   EXTRAÇÃO PRECISA - FORMATO EXATO DO SEU RELATÓRIO
            
            # 1. Quantitativos totais
            if "Total de imagens" in linha:
                # Saúde
                if classe_alvo == "Health":
                    if "acertos (modelo 1)" in linha and "saudáveis" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['acertos_modelo1'] = int(valor.group(1))
                            print(f"     acertos_modelo1 (Health): {metricas['acertos_modelo1']}")
                    
                    elif "erros (modelo 1)" in linha and "saudáveis" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['erros_modelo1'] = int(valor.group(1))
                            print(f"     erros_modelo1 (Health): {metricas['erros_modelo1']}")
                    
                    elif "acertos (modelo 2)" in linha and "saudáveis" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['acertos_modelo2'] = int(valor.group(1))
                            print(f"     acertos_modelo2 (Health): {metricas['acertos_modelo2']}")
                    
                    elif "erros (modelo 2)" in linha and "saudáveis" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['erros_modelo2'] = int(valor.group(1))
                            print(f"     erros_modelo2 (Health): {metricas['erros_modelo2']}")
                
                # Doença  
                else:  # Sick
                    if "acertos (modelo 1)" in linha and "doentes" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['acertos_modelo1'] = int(valor.group(1))
                            print(f"     acertos_modelo1 (Sick): {metricas['acertos_modelo1']}")
                    
                    elif "erros (modelo 1)" in linha and "doentes" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['erros_modelo1'] = int(valor.group(1))
                            print(f"     erros_modelo1 (Sick): {metricas['erros_modelo1']}")
                    
                    elif "acertos (modelo 2)" in linha and "doentes" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['acertos_modelo2'] = int(valor.group(1))
                            print(f"     acertos_modelo2 (Sick): {metricas['acertos_modelo2']}")
                    
                    elif "erros (modelo 2)" in linha and "doentes" in linha:
                        valor = re.search(r':\s*(\d+)', linha)
                        if valor:
                            metricas['erros_modelo2'] = int(valor.group(1))
                            print(f"     erros_modelo2 (Sick): {metricas['erros_modelo2']}")
            
            # 2. Categorias de comparação - FORMATO EXATO
            elif "Imagens que o modelo 1 errava e agora o modelo 2 acerta:" in linha:
                # Procura o número na MESMA linha
                valor = re.search(r':\s*(\d+)\s*imagens', linha)
                if valor:
                    metricas['melhorou'] = int(valor.group(1))
                    print(f"     melhorou: {metricas['melhorou']}")
                else:
                    # Tentativa alternativa
                    valor = re.search(r':\s*(\d+)', linha)
                    if valor:
                        metricas['melhorou'] = int(valor.group(1))
                        print(f"     melhorou (alt): {metricas['melhorou']}")
            
            elif "Imagens que o modelo 1 acertava e agora o modelo 2 erra" in linha:
                valor = re.search(r':\s*(\d+)\s*imagens', linha)
                if valor:
                    metricas['piorou'] = int(valor.group(1))
                    print(f"     piorou: {metricas['piorou']}")
                else:
                    valor = re.search(r':\s*(\d+)', linha)
                    if valor:
                        metricas['piorou'] = int(valor.group(1))
                        print(f"     piorou (alt): {metricas['piorou']}")
            
            elif "Manteve_Acerto:" in linha:
                valor = re.search(r':\s*(\d+)\s*imagens', linha)
                if valor:
                    metricas['manteve_acerto'] = int(valor.group(1))
                    print(f"     manteve_acerto: {metricas['manteve_acerto']}")
                else:
                    valor = re.search(r':\s*(\d+)', linha)
                    if valor:
                        metricas['manteve_acerto'] = int(valor.group(1))
                        print(f"     manteve_acerto (alt): {metricas['manteve_acerto']}")
            
            elif "Manteve_Erro:" in linha:
                valor = re.search(r':\s*(\d+)\s*imagens', linha)
                if valor:
                    metricas['manteve_erro'] = int(valor.group(1))
                    print(f"     manteve_erro: {metricas['manteve_erro']}")
                else:
                    valor = re.search(r':\s*(\d+)', linha)
                    if valor:
                        metricas['manteve_erro'] = int(valor.group(1))
                        print(f"     manteve_erro (alt): {metricas['manteve_erro']}")
    
    except Exception as e:
        print(f" Erro ao processar {relatorio_path}: {e}")
        import traceback
        traceback.print_exc()
    
    #   MOSTRAR RESULTADO FINAL
    print(f"     RESULTADOS EXTRAÍDOS para {classe_alvo}:")
    for key, value in metricas.items():
        print(f"      {key}: {value}")
    
    return metricas

def analisar_estrategias(metricas_globais, output_dir):
    """Analisa as diferenças entre as estratégias"""
    
    df = metricas_globais['df']
    
    with open(os.path.join(output_dir, "analise_estrategias.txt"), "w") as f:
        f.write("=== ANÁLISE COMPARATIVA ENTRE ESTRATÉGIAS ===\n\n")
        
        estrategias = df['estrategia'].unique()
        f.write(f"Estratégias identificadas: {', '.join(estrategias)}\n\n")
        
        # Configurar pandas para mostrar todas as colunas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        #   RESUMO GERAL COMPLETO
        f.write("  RESUMO GERAL COMPLETO:\n")
        resumo = df.groupby(['estrategia', 'categoria'])['quantidade'].mean().unstack()
        
        # Reordenar as colunas para uma visualização mais lógica
        colunas_ordenadas = [
            'acertos_modelo1', 'erros_modelo1', 
            'acertos_modelo2', 'erros_modelo2',
            'melhorou', 'piorou', 
            'manteve_acerto', 'manteve_erro'
        ]
        
        # Manter apenas as colunas que existem
        colunas_existentes = [col for col in colunas_ordenadas if col in resumo.columns]
        resumo = resumo[colunas_existentes]
        
        f.write(str(resumo.round(2)) + "\n\n")
        
        #  ANÁLISE DETALHADA DAS MELHORIAS
        f.write(" ANÁLISE DETALHADA DAS MELHORIAS:\n")
        
        for classe in ['Health', 'Sick']:
            f.write(f"\n🔬 CLASSE: {classe}\n")
            
            for estrategia in estrategias:
                # Melhorias
                melhorias = df[
                    (df['estrategia'] == estrategia) & 
                    (df['classe'] == classe) & 
                    (df['categoria'] == 'melhorou')
                ]['quantidade']
                
                # Pioras
                pioras = df[
                    (df['estrategia'] == estrategia) & 
                    (df['classe'] == classe) & 
                    (df['categoria'] == 'piorou')
                ]['quantidade']
                
                if len(melhorias) > 0 and len(pioras) > 0:
                    f.write(f"   {estrategia}:\n")
                    f.write(f"      Melhorias: {melhorias.mean():.1f} ± {melhorias.std():.1f}\n")
                    f.write(f"      Pioras: {pioras.mean():.1f} ± {pioras.std():.1f}\n")
                    
                    # Razão Melhoria/Piora
                    if pioras.mean() > 0:
                        razao = melhorias.mean() / pioras.mean()
                        f.write(f"      Razão M/P: {razao:.2f}\n")
            
            # Comparação estatística entre estratégias
            if len(estrategias) == 2:
                estrat1, estrat2 = estrategias
                
                melhorias1 = df[
                    (df['estrategia'] == estrat1) & 
                    (df['classe'] == classe) & 
                    (df['categoria'] == 'melhorou')
                ]['quantidade']
                
                melhorias2 = df[
                    (df['estrategia'] == estrat2) & 
                    (df['classe'] == classe) & 
                    (df['categoria'] == 'melhorou')
                ]['quantidade']
                
                if len(melhorias1) > 1 and len(melhorias2) > 1:
                    t_stat, p_value = stats.ttest_ind(melhorias1, melhorias2)
                    f.write(f"     Teste t {estrat1} vs {estrat2}: t={t_stat:.3f}, p={p_value:.3f}\n")
        
        # 📋 TABELA DETALHADA POR FOLD
        f.write(f"\n📋 DADOS DETALHADOS POR FOLD:\n")
        f.write("="*80 + "\n")
        
        for fold in sorted(df['fold'].unique()):
            f.write(f"\n  FOLD {fold}:\n")
            df_fold = df[df['fold'] == fold]
            
            for estrategia in df_fold['estrategia'].unique():
                f.write(f"   {estrategia}:\n")
                
                for classe in ['Health', 'Sick']:
                    f.write(f"      {classe}:\n")
                    
                    # Buscar todas as métricas para esta combinação
                    dados_classe = df_fold[
                        (df_fold['estrategia'] == estrategia) & 
                        (df_fold['classe'] == classe)
                    ]
                    
                    for _, row in dados_classe.iterrows():
                        f.write(f"         {row['categoria']}: {row['quantidade']}\n")

def analisar_consistencia_folds(metricas_globais, output_dir):
    """Analisa a consistência entre os folds com visualização completa"""
    
    df = metricas_globais['df']
    
    # Configurar o estilo dos gráficos
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # Gráfico 1: Todas as métricas por fold
    ax1 = plt.subplot(2, 3, 1)
    
    metricas_principais = ['melhorou', 'piorou', 'manteve_acerto', 'manteve_erro']
    cores = ['green', 'red', 'blue', 'orange']
    
    for i, metrica in enumerate(metricas_principais):
        if metrica in df['categoria'].unique():
            dados_metrica = df[df['categoria'] == metrica]
            
            for estrategia in dados_metrica['estrategia'].unique():
                subset = dados_metrica[dados_metrica['estrategia'] == estrategia]
                if not subset.empty:
                    plt.plot(subset['fold'], subset['quantidade'], 
                            marker='o', linewidth=2, markersize=6,
                            label=f'{estrategia} - {metrica}',
                            color=cores[i], alpha=0.7)
    
    plt.xlabel('Fold')
    plt.ylabel('Quantidade')
    plt.title('TODAS AS MÉTRICAS POR FOLD')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Heatmap das métricas
    ax2 = plt.subplot(2, 3, 2)
    
    # Preparar dados para heatmap
    heatmap_data = df.pivot_table(
        index=['estrategia', 'fold'], 
        columns='categoria', 
        values='quantidade',
        aggfunc='mean'
    ).fillna(0)
    
    # Reordenar colunas
    colunas_ordenadas = [col for col in [
        'acertos_modelo1', 'erros_modelo1', 'acertos_modelo2', 'erros_modelo2',
        'melhorou', 'piorou', 'manteve_acerto', 'manteve_erro'
    ] if col in heatmap_data.columns]
    
    heatmap_data = heatmap_data[colunas_ordenadas]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
    plt.title('HEATMAP - MÉTRICAS POR FOLD/ESTRATÉGIA')
    plt.tight_layout()
    
    # Gráfico 3: Comparação direta das melhorias
    ax3 = plt.subplot(2, 3, 3)
    
    dados_melhorias = df[df['categoria'] == 'melhorou']
    sns.boxplot(data=dados_melhorias, x='estrategia', y='quantidade', hue='classe', ax=ax3)
    plt.title('DISTRIBUIÇÃO DAS MELHORIAS')
    plt.ylabel('Número de Melhorias')
    
    # Gráfico 4: Razão Melhoria/Piora
    ax4 = plt.subplot(2, 3, 4)
    
    ratios = []
    for estrategia in df['estrategia'].unique():
        for fold in df['fold'].unique():
            for classe in ['Health', 'Sick']:
                try:
                    melhorou = df[
                        (df['estrategia'] == estrategia) & 
                        (df['fold'] == fold) & 
                        (df['classe'] == classe) & 
                        (df['categoria'] == 'melhorou')
                    ]['quantidade'].values[0]
                    
                    piorou = df[
                        (df['estrategia'] == estrategia) & 
                        (df['fold'] == fold) & 
                        (df['classe'] == classe) & 
                        (df['categoria'] == 'piorou')
                    ]['quantidade'].values[0]
                    
                    ratio = melhorou / piorou if piorou > 0 else melhorou
                    ratios.append({
                        'estrategia': estrategia,
                        'fold': fold,
                        'classe': classe,
                        'ratio': ratio
                    })
                except:
                    continue
    
    if ratios:
        df_ratios = pd.DataFrame(ratios)
        sns.boxplot(data=df_ratios, x='estrategia', y='ratio', hue='classe', ax=ax4)
        plt.title('RAZÃO MELHORIA/PIORA')
        plt.ylabel('Ratio (Melhorias/Pioras)')
    
    # Gráfico 5: Estabilidade entre folds
    ax5 = plt.subplot(2, 3, 5)
    
    # Calcular coeficiente de variação por estratégia
    cv_data = []
    for estrategia in df['estrategia'].unique():
        for categoria in df['categoria'].unique():
            valores = df[
                (df['estrategia'] == estrategia) & 
                (df['categoria'] == categoria)
            ]['quantidade']
            
            if len(valores) > 1:
                cv = valores.std() / valores.mean() if valores.mean() > 0 else 0
                cv_data.append({
                    'estrategia': estrategia,
                    'categoria': categoria,
                    'cv': cv
                })
    
    if cv_data:
        df_cv = pd.DataFrame(cv_data)
        cv_pivot = df_cv.pivot(index='categoria', columns='estrategia', values='cv')
        sns.heatmap(cv_pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax5)
        plt.title('COEFICIENTE DE VARIAÇÃO (Estabilidade)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analise_completa_folds.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Restaurar configurações padrão do pandas
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

# Adicionar também no final da análise_completa_comparacoes:
def analise_completa_comparacoes():
    print("  Iniciando análise completa...")
    
    metricas_globais = analisar_comparacoes_cruzadas()
    
    if metricas_globais is not None:
        print("  Análise completa concluída!")
        
        # Mostrar resumo no console também
        df = metricas_globais['df']
        print("\n" + "="*80)
        print("  RESUMO GERAL COMPLETO (CONSOLE):")
        print("="*80)
        
        resumo = df.groupby(['estrategia', 'categoria'])['quantidade'].mean().unstack()
        colunas_ordenadas = [
            'acertos_modelo1', 'erros_modelo1', 'acertos_modelo2', 'erros_modelo2',
            'melhorou', 'piorou', 'manteve_acerto', 'manteve_erro'
        ]
        colunas_existentes = [col for col in colunas_ordenadas if col in resumo.columns]
        resumo = resumo[colunas_existentes]
        
        print(resumo.round(2))
        print("\n Resultados salvos em: Analise_Global_Comparacoes/")
    
    return metricas_globais

import os, re, shutil
from collections import defaultdict

def comparar_modelos_por_id_com_consistencia(
    exp1_base, exp1_modelo,
    exp2_base, exp2_modelo,
    mensagem="mensagem_comparacao:",
    output_dir="Comparacao"
):
    """
    Lógica solicitada:
      1) Para cada classe (Health/Sick) e para cada experimento, agrupar arquivos por ID (id_###).
      2) Verificar se há >1 arquivo para o mesmo ID:
         - Se sim, checar se TODOS pertencem ao MESMO grupo (Acertos ou Erros) no experimento.
           * Se sim: marcar como CONSISTENTE e usar esse grupo (Acertos/Erros) na comparação.
           * Se não: anotar no relatório como 'revisão manual' (misto dentro do próprio experimento).
         - Se não (>1 arquivo): comparar normalmente.
      3) Só comparar IDs que são CONSISTENTES em AMBOS os experimentos.
         Categorias:
           - Erro -> Acerto (melhorou)
           - Acerto -> Erro (piorou)
           - Manteve_Acerto
           - Manteve_Erro
      4) IDs inconsistentes (mistos) em qualquer um dos experimentos vão para a seção
         'Revisão manual', com o detalhamento de quantos arquivos caíram em cada grupo.
    Estrutura assumida:
      CAM_results/
        Acertos/<modelo>/Health
        Acertos/<modelo>/Sick
        Erros/<modelo>/Health
        Erros/<modelo>/Sick
    """

    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def only_id(name: str) -> str:
        """Extrai 'id_###' do basename."""
        m = re.search(r"(id_\d+)", name, flags=re.IGNORECASE)
        return m.group(1) if m else name

    def listar_pngs(path):
        return [f for f in os.listdir(path) if f.lower().endswith(".png")]

    def coletar_por_experimento(exp_base: str, modelo: str):
        """
        Retorna, para cada classe:
          - por_id: dict[id] = {"Acertos": set(nomes_arquivos), "Erros": set(nomes_arquivos)}
          - totais_arquivos: contagem bruta de arquivos por grupo (Acertos/Erros)
        """
        por_id = { "Health": defaultdict(lambda: {"Acertos": set(), "Erros": set()}),
                   "Sick":   defaultdict(lambda: {"Acertos": set(), "Erros": set()}) }
        totais_arquivos = { "Health": {"Acertos": 0, "Erros": 0},
                            "Sick":   {"Acertos": 0, "Erros": 0} }

        for classe in ["Health", "Sick"]:
            for grupo in ["Acertos", "Erros"]:
                class_path = os.path.join(exp_base, grupo, modelo, classe)
                if not os.path.exists(class_path):
                    continue
                arquivos = listar_pngs(class_path)
                totais_arquivos[classe][grupo] += len(arquivos)

                for fname in arquivos:
                    _id = only_id(fname)
                    por_id[classe][_id][grupo].add(fname)

        return por_id, totais_arquivos

    # Coleta dos dois experimentos
    exp1_por_id, exp1_totais = coletar_por_experimento(exp1_base, exp1_modelo)
    exp2_por_id, exp2_totais = coletar_por_experimento(exp2_base, exp2_modelo)

    # Arquivo de relatório
    ensure_dir(output_dir)
    relatorio_path = os.path.join(output_dir, "relatorio_comparacao.txt")
    with open(relatorio_path, "w", encoding="utf-8") as f:
        if mensagem == "mensagem_comparacao:":
            f.write(f"Comparando o modelo {exp1_modelo} com {exp2_modelo}\n")
        else:
            f.write(mensagem.strip() + "\n")

    # Função para decidir consistência de um ID dentro de UM experimento
    def rotular_consistencia(entry: dict):
        """
        entry: {"Acertos": set(files), "Erros": set(files)}
        Retorna:
          - status: "CONSISTENTE" | "MISTO" | "AUSENTE"
          - grupo:  "Acertos" | "Erros" | None  (quando CONSISTENTE indica o grupo)
          - contagem: {"Acertos": n, "Erros": n}
          - total_arquivos: n_total
        Regras:
          - Se ambos vazios -> AUSENTE
          - Se ambos não vazios -> MISTO
          - Se apenas um não vazio -> CONSISTENTE, grupo = esse
        """
        ca = len(entry["Acertos"])
        ce = len(entry["Erros"])
        total = ca + ce
        if total == 0:
            return "AUSENTE", None, {"Acertos": ca, "Erros": ce}, total
        if ca > 0 and ce > 0:
            return "MISTO", None, {"Acertos": ca, "Erros": ce}, total
        if ca > 0:
            return "CONSISTENTE", "Acertos", {"Acertos": ca, "Erros": ce}, total
        return "CONSISTENTE", "Erros", {"Acertos": ca, "Erros": ce}, total

    # Loop por classe para montar o relatório
    with open(relatorio_path, "a", encoding="utf-8") as report:
        for classe in ["Health", "Sick"]:
            report.write(f"\n============================================= {classe} =============================================\n")

            # Totais brutos por ARQUIVO (só como referência inicial)
            if classe == "Health":
                report.write("\n---------------------------- Quantitativo total (por ARQUIVO) ----------------------------\n")
                report.write(f"Modelo 1 - Acertos: {exp1_totais['Health']['Acertos']} | Erros: {exp1_totais['Health']['Erros']}\n")
                report.write(f"Modelo 2 - Acertos: {exp2_totais['Health']['Acertos']} | Erros: {exp2_totais['Health']['Erros']}\n")
            else:
                report.write("\n---------------------------- Quantitativo total (por ARQUIVO) ----------------------------\n")
                report.write(f"Modelo 1 - Acertos: {exp1_totais['Sick']['Acertos']} | Erros: {exp1_totais['Sick']['Erros']}\n")
                report.write(f"Modelo 2 - Acertos: {exp2_totais['Sick']['Acertos']} | Erros: {exp2_totais['Sick']['Erros']}\n")

            # Conjuntos de IDs possíveis (união dos dois experimentos para a classe)
            ids_classe = set(exp1_por_id[classe].keys()) | set(exp2_por_id[classe].keys())

            # Buckets de comparação (apenas IDs consistentes em ambos os experimentos)
            melhorou = []      # Erros (exp1) -> Acertos (exp2)
            piorou = []        # Acertos (exp1) -> Erros (exp2)
            manteve_ac = []    # Acertos -> Acertos
            manteve_er = []    # Erros -> Erros

            # IDs para revisão manual (mistos em algum experimento)
            revisao_manual = []  # (id, detalhe_exp1, detalhe_exp2)

            # IDs ausentes em algum experimento (presentes só em um)
            ausentes = []  # (id, status_exp1, status_exp2)

            for _id in sorted(ids_classe, key=lambda x: (int(re.search(r"\d+", x).group()), x) if re.search(r"\d+", x) else (10**9, x)):
                e1_entry = exp1_por_id[classe].get(_id, {"Acertos": set(), "Erros": set()})
                e2_entry = exp2_por_id[classe].get(_id, {"Acertos": set(), "Erros": set()})

                s1, g1, c1, t1 = rotular_consistencia(e1_entry)
                s2, g2, c2, t2 = rotular_consistencia(e2_entry)

                # Se algum é AUSENTE, não dá para comparar -> registrar
                if s1 == "AUSENTE" or s2 == "AUSENTE":
                    ausentes.append((_id, s1, s2, c1, c2))
                    continue

                # Se algum é MISTO, mandar para revisão manual
                if s1 == "MISTO" or s2 == "MISTO":
                    detalhe_exp1 = f"exp1: Acertos={c1['Acertos']}, Erros={c1['Erros']}"
                    detalhe_exp2 = f"exp2: Acertos={c2['Acertos']}, Erros={c2['Erros']}"
                    revisao_manual.append((_id, detalhe_exp1, detalhe_exp2))
                    continue

                # Aqui ambos CONSISTENTES -> comparar grupos
                if g1 == "Erros" and g2 == "Acertos":
                    melhorou.append(_id)
                elif g1 == "Acertos" and g2 == "Erros":
                    piorou.append(_id)
                elif g1 == "Acertos" and g2 == "Acertos":
                    manteve_ac.append(_id)
                elif g1 == "Erros" and g2 == "Erros":
                    manteve_er.append(_id)

            # ---- Relatório: resultados de comparação (apenas IDs consistentes) ----
            report.write("\n---------------------------- Comparação (apenas IDs CONSISTENTES em ambos) ----------------------------\n")
            report.write(f"Erro -> Acerto (melhorou): {len(melhorou)} IDs\n")
            report.write(f"Acerto -> Erro (piorou):   {len(piorou)} IDs\n")
            report.write(f"Manteve_Acerto:            {len(manteve_ac)} IDs\n")
            report.write(f"Manteve_Erro:              {len(manteve_er)} IDs\n\n")

            def listar(titulo, items):
                report.write(f"--- {titulo} ---\n")
                for x in items:
                    report.write(f"{x}\n")
                report.write("\n")

            listar("Erro -> Acerto (melhorou)", melhorou)
            listar("Acerto -> Erro (piorou)", piorou)
            listar("Manteve_Acerto", manteve_ac)
            listar("Manteve_Erro", manteve_er)

            # ---- Relatório: revisão manual (IDs MISTOS em algum experimento) ----
            report.write("---------------------------- IDs para REVISÃO MANUAL (mistos em algum experimento) ----------------------------\n")
            report.write(f"Total: {len(revisao_manual)} IDs\n")
            for _id, d1, d2 in revisao_manual:
                report.write(f"{_id} | {d1} | {d2}\n")
            report.write("\n")

            # ---- Relatório: ausentes (sem contrapartida em algum experimento) ----
            report.write("---------------------------- IDs AUSENTES (presentes em apenas 1 experimento) ----------------------------\n")
            report.write(f"Total: {len(ausentes)} IDs\n")
            for _id, s1, s2, c1, c2 in ausentes:
                report.write(f"{_id} | exp1={s1} (A={c1['Acertos']},E={c1['Erros']}) | exp2={s2} (A={c2['Acertos']},E={c2['Erros']})\n")
            report.write("\n")

    print("Comparação concluída!")
    print(f"Relatório: {relatorio_path}")
    print(f"Pasta de saída: {output_dir}")

if __name__ == "__main__":
    
    SEMENTE = 13388

    for i in range(5):
        comparar_modelos_por_id_com_consistencia(
                exp1_base=f"Resultados_corrigidos_12_10_25/CAM_results",
                exp1_modelo=f"Vgg_AUG_CV_BlackPadding_13_09_25_F{i}",
                exp2_base=f"Resultados_corrigidos_12_10_25/CAM_results",
                exp2_modelo=f"VGG16_AUG_UFF_BlackPadding_NO_PAD_F{i}",
                output_dir=f"relatorio_normal_VS_normalNoPad/F{i}",
            )
        comparar_modelos_por_id_com_consistencia(
            exp1_base=f"Resultados_corrigidos_12_10_25/CAM_results",
            exp1_modelo=f"Vgg_AUG_CV_BlackPadding_13_09_25_F{i}",
            exp2_base=f"Resultados_corrigidos_12_10_25/CAM_results",
            exp2_modelo=f"Vgg_yolon_AUG_CV_BlackPadding_13_09_25_F{i}",
            output_dir=f"relatorio_normal_VS_Seg_normal/F{i}",
        )


    


    




    

    

    
    