import os
import numpy as np
from ultralytics import YOLO
import cv2

"""
    Treina um modelo YOLOv8 para segmentação de imagens.
    Params:
    type : str
        Tipo do modelo YOLOv8 a ser utilizado. Pode ser:
        - 'n' : YOLOv8n-seg (nano)
        - 's' : YOLOv8s-seg (small)
        - 'm' : YOLOv8m-seg (medium)
        - 'l' : YOLOv8l-seg (large)
        - 'x' : YOLOv8x-seg (extra-large)
    epochs : int
        Número de épocas para o treinamento do modelo.
    dataset : str
        Caminho para o arquivo `.yaml` que define o conjunto de dados
        (com informações de path, classes e divisões train/val/test).
    imgsize : int
        Tamanho da imagem para redimensionamento (imgsz x imgsz).
    seed : int, opcional (padrão=-1)
        Valor da semente para reprodutibilidade. Se for -1, o treinamento
        não será determinístico. Caso contrário, o treinamento será
        reprodutível.

    Return:
    None
        Salva os pesos do modelo treinado na pasta padrão `runs/segment`.
    """
def train_yolo_seg(type, epochs, dataset, seed=-1):
    model = YOLO(f'yolov8{type}-seg.pt')  # load a pretrained model (recommended for training)

    deterministic = True if seed is not -1 else False
    model.train(data=dataset, epochs=epochs,seed=seed, imgsz=224, deterministic=deterministic)






    


