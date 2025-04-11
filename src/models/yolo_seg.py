import os
import numpy as np
from ultralytics import YOLO
import cv2



def train_yolo_seg():
    model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

    model.train(data='dataset.yaml', epochs=200, imgsz=224)






    


