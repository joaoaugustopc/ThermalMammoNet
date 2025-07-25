import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from src.models.resNet_34 import ResNet34, ResidualUnit, ResNet34_retangular
from src.models.Vgg_16 import Vgg_16
from src.models.googleLenet import googleLenet
from src.models.alexnet import AlexNet
#from src.models.vgg_16_trained import VGG16_trained
from src.models.resNet_152 import ResNet152_trained
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import time
import re
import shutil
from tensorflow.keras.utils import custom_object_scope
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil

from utils.data_prep import apply_augmentation_and_expand, load_data, create_aug_dataset, view_pred_mask, make_tvt_splits_without_ids, apply_augmentation_and_expand_jpg_ufpe
from utils.files_manipulation import delete_folder, delete_file, move_files_to_folder, rename_file, rename_folder
from utils.stats import plot_convergence, test_model, get_boxPlot, data_distribution, get_confusion_matrices, get_auc_roc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from src.models.resNet_101 import ResNet101, BottleneckResidualUnit

from src.models.vgg_16_trained import Vgg_16_pre_trained
from src.models.resnet50_pre_trained import resnet50_pre_trained
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
