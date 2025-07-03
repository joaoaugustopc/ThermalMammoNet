import cv2
from ultralytics import YOLO
from include.imports import *
from utils.data_prep import load_imgs_masks, yolo_data, masks_to_polygons,load_imgs_masks_only, copy_images_excluding_patients, filter_dataset_by_id, load_raw_images,make_tvt_splits, augment_train_fold, normalize, tf_letterbox, listar_imgs_nao_usadas, load_imgs_masks_sem_padding,load_imgs_masks_recortado,tf_letterbox_Sem_padding, letterbox_center_crop, load_imgs_masks_Black_Padding, tf_letterbox_black,load_imgs_masks_distorcidas
from utils.files_manipulation import move_files_within_folder, create_folder
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

    def segment_batch(images, model):
        segmented = []
        for img in images:
            img_prepared = prepare_image(img)

            results = model.predict(img_prepared, verbose=False)

            res = results[0]

            has_masks = (
            res.masks is not None and
            res.masks.data is not None and
            len(res.masks.data) > 0)

            if has_masks:
                mask_tensor = results[0].masks.data[0]
                mask = mask_tensor.cpu().numpy()

                if mask.shape[:2] != (224, 224):
                    mask = cv2.resize(mask, (224, 224))


                binary_mask = (mask > 0.5).astype(np.uint8)
                if binary_mask.ndim == 2:
                    binary_mask = np.expand_dims(binary_mask, axis=-1)

                segmented_img = img_prepared * binary_mask
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

def train_model_cv(model, raw_root , message, angle = "Frontal", k = 5, 
                    resize = True, resize_to = 224, n_aug = 0, batch = 8, seed = 42, segmenter = "none", seg_model_path = ""):
    
    exclude_set = listar_imgs_nao_usadas("Termografias_Dataset_Segmentação/images", angle)
    
    X, y , patient_ids = load_raw_images(
        os.path.join(raw_root, angle), exclude=True, exclude_set=exclude_set)
    

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

            # min/max APENAS dos originais
            mn, mx = X_tr.min(), X_tr.max()

            # normaliza
            X_tr = normalize(X_tr, mn, mx)
            X_val= normalize(X_val,    mn, mx)
            X_test=normalize(X_test,   mn, mx)

            if resize:

                X_tr = np.expand_dims(X_tr, axis=-1)
                X_val= np.expand_dims(X_val, axis=-1)
                X_test= np.expand_dims(X_test, axis=-1)

                # X_tr= tf.image.resize_with_pad(X_tr, resize_to, resize_to, method="bicubic")
                # X_val= tf.image.resize_with_pad(X_val, resize_to, resize_to, method="bicubic")
                # X_test= tf.image.resize_with_pad(X_test, resize_to, resize_to, method="bicubic")

                X_tr = tf_letterbox(X_tr, resize_to)
                X_val = tf_letterbox(X_val, resize_to)
                X_test = tf_letterbox(X_test, resize_to)

                X_tr = tf.clip_by_value(X_tr, 0, 1).numpy().squeeze(axis=-1)
                X_val = tf.clip_by_value(X_val, 0, 1).numpy().squeeze(axis=-1)
                X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(axis=-1)

            
            
            
            # augmenta & concatena
            if n_aug > 0:
                X_tr, y_tr = augment_train_fold(X_tr, y_tr,
                                                    n_aug=n_aug, seed=seed)
                
                with open("modelos/random_seed.txt", "a") as f:
                    f.write(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}\n")
            

            if segmenter != "none":
                if segmenter == "unet":
                    X_tr, X_val, X_test = unet_segmenter(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com UNet concluída.")   
                elif segmenter == "yolo":
                    X_tr, X_val, X_test = segment_with_yolo(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com YOLO concluída.")
                else:
                    raise ValueError("segmenter deve ser 'none', 'unet' ou 'yolo'")
            

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

                if model == Vgg_16:
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
                



def train_model_cv_BlackPadding(model, raw_root , message, angle = "Frontal", k = 5, 
                    resize = True, resize_to = 224, n_aug = 0, batch = 8, seed = 42, segmenter = "none", seg_model_path = ""):
    
    exclude_set = listar_imgs_nao_usadas("Termografias_Dataset_Segmentação/images", angle)
    
    X, y , patient_ids = load_raw_images(
        os.path.join(raw_root, angle), exclude=True, exclude_set=exclude_set)
    

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

            # min/max APENAS dos originais
            mn, mx = X_tr.min(), X_tr.max()

            # normaliza
            X_tr = normalize(X_tr, mn, mx)
            X_val= normalize(X_val,    mn, mx)
            X_test=normalize(X_test,   mn, mx)

            if resize:

                X_tr = np.expand_dims(X_tr, axis=-1)
                X_val= np.expand_dims(X_val, axis=-1)
                X_test= np.expand_dims(X_test, axis=-1)

                # X_tr= tf.image.resize_with_pad(X_tr, resize_to, resize_to, method="bicubic")
                # X_val= tf.image.resize_with_pad(X_val, resize_to, resize_to, method="bicubic")
                # X_test= tf.image.resize_with_pad(X_test, resize_to, resize_to, method="bicubic")

                X_tr = tf_letterbox_black(X_tr, resize_to)
                X_val = tf_letterbox_black(X_val, resize_to)
                X_test = tf_letterbox_black(X_test, resize_to)

                X_tr = tf.clip_by_value(X_tr, 0, 1).numpy().squeeze(axis=-1)
                X_val = tf.clip_by_value(X_val, 0, 1).numpy().squeeze(axis=-1)
                X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(axis=-1)

            
            
            
            # augmenta & concatena
            if n_aug > 0:
                X_tr, y_tr = augment_train_fold(X_tr, y_tr,
                                                    n_aug=n_aug, seed=seed)
                
                with open("modelos/random_seed.txt", "a") as f:
                    f.write(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}\n")
            

            if segmenter != "none":
                if segmenter == "unet":
                    X_tr, X_val, X_test = unet_segmenter(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com UNet concluída.")   
                elif segmenter == "yolo":
                    X_tr, X_val, X_test = segment_with_yolo(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com YOLO concluída.")
                else:
                    raise ValueError("segmenter deve ser 'none', 'unet' ou 'yolo'")
            

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

                if model == Vgg_16:
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



def train_model_cv_Distorcido(model, raw_root , message, angle = "Frontal", k = 5, 
                    resize = True, resize_to = 224, n_aug = 0, batch = 8, seed = 42, segmenter = "none", seg_model_path = ""):
    
    exclude_set = listar_imgs_nao_usadas("Termografias_Dataset_Segmentação/images", angle)
    
    X, y , patient_ids = load_raw_images(
        os.path.join(raw_root, angle), exclude=True, exclude_set=exclude_set)
    

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

            # min/max APENAS dos originais
            mn, mx = X_tr.min(), X_tr.max()

            # normaliza
            X_tr = normalize(X_tr, mn, mx)
            X_val= normalize(X_val,    mn, mx)
            X_test=normalize(X_test,   mn, mx)

            if resize:

                X_tr = np.expand_dims(X_tr, axis=-1)
                X_val= np.expand_dims(X_val, axis=-1)
                X_test= np.expand_dims(X_test, axis=-1)

                # X_tr= tf.image.resize_with_pad(X_tr, resize_to, resize_to, method="bicubic")
                # X_val= tf.image.resize_with_pad(X_val, resize_to, resize_to, method="bicubic")
                # X_test= tf.image.resize_with_pad(X_test, resize_to, resize_to, method="bicubic")


                X_tr = tf.image.resize(X_tr, (224,224), method = "bilinear")
                X_val = tf.image.resize(X_val, (224,224), method = "bilinear")
                X_test = tf.image.resize(X_test, (224,224), method = "bilinear")

                X_tr = tf.clip_by_value(X_tr, 0, 1).numpy().squeeze(axis=-1)
                X_val = tf.clip_by_value(X_val, 0, 1).numpy().squeeze(axis=-1)
                X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(axis=-1)

            
            
            
            # augmenta & concatena
            if n_aug > 0:
                X_tr, y_tr = augment_train_fold(X_tr, y_tr,
                                                    n_aug=n_aug, seed=seed)
                
                with open("modelos/random_seed.txt", "a") as f:
                    f.write(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}\n")
            

            if segmenter != "none":
                if segmenter == "unet":
                    X_tr, X_val, X_test = unet_segmenter(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com UNet concluída.")   
                elif segmenter == "yolo":
                    X_tr, X_val, X_test = segment_with_yolo(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com YOLO concluída.")
                else:
                    raise ValueError("segmenter deve ser 'none', 'unet' ou 'yolo'")
            

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

                if model == Vgg_16:
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



def train_model_cv_retangular(model, raw_root , message, angle = "Frontal", k = 5, 
                    resize = True, resize_to = 224, n_aug = 0, batch = 8, seed = 42, segmenter = "none", seg_model_path = ""):
    
    exclude_set = listar_imgs_nao_usadas("Termografias_Dataset_Segmentação/images", angle)
    
    X, y , patient_ids = load_raw_images(
        os.path.join(raw_root, angle), exclude=True, exclude_set=exclude_set)
    

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

            # min/max APENAS dos originais
            mn, mx = X_tr.min(), X_tr.max()

            # normaliza
            X_tr = normalize(X_tr, mn, mx)
            X_val= normalize(X_val,    mn, mx)
            X_test=normalize(X_test,   mn, mx)

            if resize:

                X_tr = np.expand_dims(X_tr, axis=-1)
                X_val= np.expand_dims(X_val, axis=-1)
                X_test= np.expand_dims(X_test, axis=-1)

                # X_tr= tf.image.resize_with_pad(X_tr, resize_to, resize_to, method="bicubic")
                # X_val= tf.image.resize_with_pad(X_val, resize_to, resize_to, method="bicubic")
                # X_test= tf.image.resize_with_pad(X_test, resize_to, resize_to, method="bicubic")

                X_tr = tf_letterbox_Sem_padding(X_tr, resize_to)
                X_val = tf_letterbox_Sem_padding(X_val, resize_to)
                X_test = tf_letterbox_Sem_padding(X_test, resize_to)

                X_tr = tf.clip_by_value(X_tr, 0, 1).numpy().squeeze(axis=-1)
                X_val = tf.clip_by_value(X_val, 0, 1).numpy().squeeze(axis=-1)
                X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(axis=-1)

            
            
            
            # augmenta & concatena
            if n_aug > 0:
                X_tr, y_tr = augment_train_fold(X_tr, y_tr,
                                                    n_aug=n_aug, seed=seed)
                
                with open("modelos/random_seed.txt", "a") as f:
                    f.write(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}\n")
            

            if segmenter != "none":
                if segmenter == "unet":
                    X_tr, X_val, X_test = unet_segmenter(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com UNet concluída.")   
                elif segmenter == "yolo":
                    X_tr, X_val, X_test = segment_with_yolo(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com YOLO concluída.")
                else:
                    raise ValueError("segmenter deve ser 'none', 'unet' ou 'yolo'")
            

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

                model_f   = model()
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


def train_model_cv_recortado(model, raw_root , message, angle = "Frontal", k = 5, 
                    resize = True, resize_to = 224, n_aug = 0, batch = 8, seed = 42, segmenter = "none", seg_model_path = ""):
    
    exclude_set = listar_imgs_nao_usadas("Termografias_Dataset_Segmentação/images", angle)
    
    X, y , patient_ids = load_raw_images(
        os.path.join(raw_root, angle), exclude=True, exclude_set=exclude_set)
    

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

            # min/max APENAS dos originais
            mn, mx = X_tr.min(), X_tr.max()

            # normaliza
            X_tr = normalize(X_tr, mn, mx)
            X_val= normalize(X_val,    mn, mx)
            X_test=normalize(X_test,   mn, mx)

            if resize:

                X_tr = np.expand_dims(X_tr, axis=-1)
                X_val= np.expand_dims(X_val, axis=-1)
                X_test= np.expand_dims(X_test, axis=-1)

                # X_tr= tf.image.resize_with_pad(X_tr, resize_to, resize_to, method="bicubic")
                # X_val= tf.image.resize_with_pad(X_val, resize_to, resize_to, method="bicubic")
                # X_test= tf.image.resize_with_pad(X_test, resize_to, resize_to, method="bicubic")

                X_tr = letterbox_center_crop(X_tr, resize_to)
                X_val = letterbox_center_crop(X_val, resize_to)
                X_test = letterbox_center_crop(X_test, resize_to)

                X_tr = tf.clip_by_value(X_tr, 0, 1).numpy().squeeze(axis=-1)
                X_val = tf.clip_by_value(X_val, 0, 1).numpy().squeeze(axis=-1)
                X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(axis=-1)

            
            
            
            # augmenta & concatena
            if n_aug > 0:
                X_tr, y_tr = augment_train_fold(X_tr, y_tr,
                                                    n_aug=n_aug, seed=seed)
                
                with open("modelos/random_seed.txt", "a") as f:
                    f.write(f"Shape de treinamento fold {fold} após o aumento de dados: {X_tr.shape}\n")
            

            if segmenter != "none":
                if segmenter == "unet":
                    X_tr, X_val, X_test = unet_segmenter(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com UNet concluída.")   
                elif segmenter == "yolo":
                    X_tr, X_val, X_test = segment_with_yolo(X_tr, X_val, X_test, seg_model_path)
                    print(f"Segmentação com YOLO concluída.")
                else:
                    raise ValueError("segmenter deve ser 'none', 'unet' ou 'yolo'")
            

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

                model_f   = model()
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
    models = models_objects
                
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
        for model_func in models_objects:

            #criando pasta com gráficos dos modelos
            os.makedirs(f"history/{model_func.__name__}", exist_ok=True)
            
            with open(f"modelos/random_seed.txt", "a") as f:
                f.write(f"Modelo:{model_func.__name__}\n")
                f.write(f"Angulo: {angulo}\n")

            for i in range(10):
                
                VALUE_SEED = int(time.time()*1000) % 15000
                random.seed(VALUE_SEED)
                
                seed = random.randint(0,15000)  
                
                tf.keras.utils.set_random_seed(seed)
                tf.config.experimental.enable_op_determinism()
                
                print(f"history/{model_func.__name__}/{model_func.__name__}_{angulo}_{i}_time.txt")
                
                start_time = time.time()
                
                checkpoint_path = f"modelos/{model_func.__name__}/{message}_{angulo}_{i}.h5"
                checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, 
                                                            save_weights_only=False, mode='auto')
                
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

                #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=15, min_lr=1e-5, min_delta=0.0001)

                #criando objeto e usando o modelo
                model = model_func()

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







def ppeprocessEigenCam(X, y, splits_path, segment = None, segmenter_path ="" ):
    
    
    with open (splits_path, "r") as f:
        splits = json.load(f)


    
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    X_test = X[test_idx]
    y_test = y[test_idx]

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

    X_tr= tf_letterbox(X_tr, 224)
    X_val= tf_letterbox(X_val, 224)
    X_test= tf_letterbox(X_test, 224)

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


    X_test = np.expand_dims(X_test, axis=-1)

    return X_test




def prep_test_data(raw_root, angle, split_json, 
                    resize = True, resize_to = 224,
                    segmenter = "none", seg_model_path=""):
    
    """
    Função para preparar as imagens de teste para gerar as matrizes de confusão.
    Segue o mesmo procedimento de processamento do PipeLine de treinamento (train_models_cv)
    """
    
    X, y, patient_ids = load_raw_images(os.path.join(raw_root, angle))
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
        X_test = tf_letterbox(X_test, resize_to)
        #X_test = tf_letterbox_Sem_padding(X_test, resize_to)
        #X_test = letterbox_center_crop(X_test, resize_to)
        X_test = tf.clip_by_value(X_test, 0, 1).numpy().squeeze(-1)

    if segmenter == "unet":
        _, _, X_test = unet_segmenter(X_test, X_test, X_test, seg_model_path)
        print(f"Segmentação com UNet concluída.")
    elif segmenter == "yolo":
        _, _, X_test = segment_with_yolo(X_test, X_test, X_test, seg_model_path)
        print(f"Segmentação com YOLO concluída.")

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
                      segmenter="none",
                      seg_model_path="",
                      classes=("Healthy", "Sick")):
    """
    Avalia o modelo salvo no fold especificado e gera matriz de confusão.
    """
    os.makedirs(output_path, exist_ok=True)

    X_test, y_test = prep_test_data(raw_root, angle, split_json,
                                     resize, resize_to,
                                     segmenter, seg_model_path)

    
    
    with custom_object_scope({'ResidualUnit': ResidualUnit}):
        model = tf.keras.models.load_model(model_path, compile=False)
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    

    cm = confusion_matrix(y_test, y_pred)
    clf_rep = classification_report(y_test, y_pred, target_names=classes,
                                    output_dict=True, zero_division=0)


    out_png = os.path.join(output_path, f"cm_{message}_{angle}.png")
    
    _plot_and_save_cm(cm, classes,
                      f"Confusion Matrix – {message}",
                      out_png = out_png)

    
    K.clear_session(); gc.collect()



# resize_with_tf_letterbox.py
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm   # barra de progresso opcional

# ──────────────────────────────────────────────────────────────────────────────
#  SE A SUA tf_letterbox NÃO PERMITIR MUDAR O PAD_COLOR,
#  NÃO HÁ PROBLEMA: A MÁSCARA É BINARIZADA DEPOIS.
# ──────────────────────────────────────────────────────────────────────────────
def resize_dataset_tf_Black(
    img_dir: str,
    mask_dir: str,
    output_base: str,
    target: int = 640,
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
        img_lb  = tf_letterbox_black(tf.expand_dims(img,  0), target=target, mode='bilinear')
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

def resize_dataset_tf_Distorcao(
    img_dir: str,
    mask_dir: str,
    output_base: str,
    target: int = 640,
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
        img_lb  = tf.expand_dims(img,  0)
        img_lb = tf.image.resize(img_lb, (target, target), method='bilinear')
        mask_lb = tf.expand_dims(mask, 0)
        mask_lb = tf.image.resize(mask_lb, (target, target), method='nearest')

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

def resize_dataset_tf_GrayPadding(
    img_dir: str,
    mask_dir: str,
    output_base: str,
    target: int = 640,
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

    


if __name__ == "__main__":


    
    SEMENTE = 13388
    
    tf.random.set_seed(SEMENTE)

    # tf.config.experimental.enable_op_determinism()

    gpus = tf.config.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    # 1. Treinar Yolo + Unet com BlackPadding 

    ##Unet
    # imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks_Black_Padding("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", True, True, 224)

    # model = unet_model()

    # model.summary()

    # earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

    # checkpoint = tf.keras.callbacks.ModelCheckpoint("modelos/unet/Frontal_Unet_AUG_BlackPadding.h5", monitor='val_loss', verbose=1, save_best_only=True, 
    #                                                         save_weights_only=False, mode='auto')

    # history = model.fit(imgs_train, masks_train, epochs = 500, validation_data= (imgs_valid, masks_valid), callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)

    # # Gráfico de perda de treinamento
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.title(f'Training Loss Convergence for unet - Frontal')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"unet_loss_convergence_Frontal_Unet_AUG_BlackPadding.png")
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title(f'Validation Loss Convergence for unet - Frontal')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"unet_val_loss_convergence_Frontal_Unet_AUG_BlackPadding.png")
    # plt.close()

    ## yolo

    # resize_dataset_tf_Black(
    #     img_dir="Termografias_Dataset_Segmentação/images",
    #     mask_dir="Termografias_Dataset_Segmentação/masks",
    #     output_base="Termografias_Dataset_Segmentação_224_BlackPadding",
    #     target=224          # mesmo tamanho definido no YAML da YOLO
    # )

    # yolo_data("Frontal", "Termografias_Dataset_Segmentação_224_BlackPadding/images", "Termografias_Dataset_Segmentação_224_BlackPadding/masks", "Yolo_dataset_BlackPadding", True)

    #train26
    #train_yolo_seg("n", 500, "dataset_black.yaml", seed=SEMENTE)






    # 2. Treinar Yolo + Unet com Distorção

    #Unet
    # imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks_distorcidas("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", True, True, 224)

    # model = unet_model()

    # model.summary()

    # earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

    # checkpoint = tf.keras.callbacks.ModelCheckpoint("modelos/unet/Frontal_Unet_AUG_Distorcao.h5", monitor='val_loss', verbose=1, save_best_only=True, 
    #                                                         save_weights_only=False, mode='auto')

    # history = model.fit(imgs_train, masks_train, epochs = 500, validation_data= (imgs_valid, masks_valid), callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)

    # # Gráfico de perda de treinamento
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.title(f'Training Loss Convergence for unet - Frontal')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"unet_loss_convergence_Frontal_Unet_AUG_Distorcao.png")
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title(f'Validation Loss Convergence for unet - Frontal')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"unet_val_loss_convergence_Frontal_Unet_AUG_Distorcao.png")
    # plt.close()

    ## yolo

    # resize_dataset_tf_Distorcao(
    #     img_dir="Termografias_Dataset_Segmentação/images",
    #     mask_dir="Termografias_Dataset_Segmentação/masks",
    #     output_base="Termografias_Dataset_Segmentação_224_Distorcao",
    #     target=224          # mesmo tamanho definido no YAML da YOLO
    # )

    # yolo_data("Frontal", "Termografias_Dataset_Segmentação_224_Distorcao/images", "Termografias_Dataset_Segmentação_224_Distorcao/masks", "Yolo_dataset_Distorcao", True)

    #train27
    # train_yolo_seg("n", 500, "dataset_distorcao.yaml", seed=SEMENTE)



    # 3. Treinar Yolo + Unet GrayPadding (Retangular não é possível)

    # imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks("Frontal", "Termografias_Dataset_Segmentação/images", "Termografias_Dataset_Segmentação/masks", True, True, 224)

    # VALUE_SEED = int(time.time()*1000) % 15000
    # random.seed(VALUE_SEED)

    # print(f"Valor da semente: {VALUE_SEED}")
    # with open("modelos/random_seed.txt", "a") as f:
    #     f.write(f"Valor da semente para treinar UNET com GRayPadding: {VALUE_SEED}\n")
                
    # seed = random.randint(0,15000)  
                
    # tf.keras.utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()

    # model = unet_model()

    # model.summary()

    # earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

    # checkpoint = tf.keras.callbacks.ModelCheckpoint("modelos/unet/Frontal_Unet_AUG_CV_GrayPadding.h5", monitor='val_loss', verbose=1, save_best_only=True, 
    #                                                         save_weights_only=False, mode='auto')

    # history = model.fit(imgs_train, masks_train, epochs = 500, validation_data= (imgs_valid, masks_valid), callbacks= [checkpoint, earlystop], batch_size = 8, verbose = 1, shuffle = True)

    # # Gráfico de perda de treinamento
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.title(f'Training Loss Convergence for unet - Frontal')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"unet_loss_convergence_Frontal_Unet_AUG_CV_GrayPadding.png")
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title(f'Validation Loss Convergence for unet - Frontal')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"unet_val_loss_convergence_Frontal_Unet_AUG_CV_GrayPadding.png")
    # plt.close()


    # resize_dataset_tf_GrayPadding(
    #     img_dir="Termografias_Dataset_Segmentação/images",
    #     mask_dir="Termografias_Dataset_Segmentação/masks",
    #     output_base="Termografias_Dataset_Segmentação_224_GrayPadding",
    #     target=224          # mesmo tamanho definido no YAML da YOLO
    # )

    # yolo_data("Frontal", "Termografias_Dataset_Segmentação_224_GrayPadding/images", "Termografias_Dataset_Segmentação_224_GrayPadding/masks", "Yolo_dataset_GrayPadding", True)

    #train28
    #train_yolo_seg("n", 500, "dataset_GrayPadding.yaml", seed=SEMENTE)



    # 4. treinar Vgg-16 : BlackPadding + Distocido + GrayPadding (Unet + Original + yolo)

    # train_model_cv_BlackPadding(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                segmenter="unet",
    #                message="Vgg_unet_AUG_CV_BlackPadding", seg_model_path="modelos/unet/Frontal_Unet_AUG_BlackPadding.h5")
    
    # train_model_cv_BlackPadding(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                segmenter= "yolo",
    #                message="Vgg_yolon_AUG_CV_BlackPadding", seg_model_path="runs/segment/train27/weights/best.pt")

    # train_model_cv_BlackPadding(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                message="Vgg_AUG_CV_BlackPadding")
    
    # train_model_cv_Distorcido(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                segmenter="unet",
    #                message="Vgg_unet_AUG_CV_Distorcido", seg_model_path="modelos/unet/Frontal_Unet_AUG_Distorcao.h5")
    
    # train_model_cv_Distorcido(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                segmenter= "yolo",
    #                message="Vgg_yolon_AUG_CV_Distorcido", seg_model_path="runs/segment/train28/weights/best.pt")
    
    # train_model_cv_Distorcido(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                message="Vgg_AUG_CV_Distorcido")
    
    # train_model_cv(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                segmenter="unet",
    #                message="Vgg_unet_AUG_CV_GrayPadding", seg_model_path="modelos/unet/Frontal_Unet_AUG_CV_GrayPadding.h5")
    
    # train_model_cv(Vgg_16,
    #                raw_root="filtered_raw_dataset",
    #                angle="Frontal",
    #                k=5,                 
    #                resize_to=224,
    #                n_aug=2,             
    #                batch=8,
    #                seed= SEMENTE,
    #                segmenter= "yolo",
    #                message="Vgg_yolon_AUG_CV_GrayPadding", seg_model_path="runs/segment/train29/weights/best.pt")
    

    train_model_cv(Vgg_16,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   message="Vgg_AUG_CV_GrayPadding")
    



    # 5. Treinar ResNet-34 : BlackPadding + Distorção + GrayPadding (Unet + Original + yolo)

    train_model_cv_BlackPadding(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   segmenter="unet",
                   message="ResNet34_unet_AUG_CV_BlackPadding", seg_model_path="modelos/unet/Frontal_Unet_AUG_BlackPadding.h5")
    
    train_model_cv_BlackPadding(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   segmenter= "yolo",
                   message="ResNet34_yolon_AUG_CV_BlackPadding", seg_model_path="runs/segment/train27/weights/best.pt")

    train_model_cv_BlackPadding(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   message="ResNet34_AUG_CV_BlackPadding")
    
    train_model_cv_Distorcido(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   segmenter="unet",
                   message="ResNet34_unet_AUG_CV_Distorcido", seg_model_path="modelos/unet/Frontal_Unet_AUG_Distorcao.h5")
    
    train_model_cv_Distorcido(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   segmenter= "yolo",
                   message="ResNet34_yolon_AUG_CV_Distorcido", seg_model_path="runs/segment/train28/weights/best.pt")
    
    train_model_cv_Distorcido(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   message="ResNet34_AUG_CV_Distorcido")
    
    train_model_cv(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   segmenter="unet",
                   message="ResNet34_unet_AUG_CV_GrayPadding", seg_model_path="modelos/unet/Frontal_Unet_AUG_CV_GrayPadding.h5")
    
    train_model_cv(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   segmenter= "yolo",
                   message="ResNet34_yolon_AUG_CV_GrayPadding", seg_model_path="runs/segment/train29/weights/best.pt")
    

    train_model_cv(ResNet34,
                   raw_root="filtered_raw_dataset",
                   angle="Frontal",
                   k=5,                 
                   resize_to=224,
                   n_aug=2,             
                   batch=8,
                   seed= SEMENTE,
                   message="ResNet34_AUG_CV_GrayPadding")





    



    



    

    


    