import os
import sys
import cv2
import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
from glob import glob
from sklearn.metrics import (
    jaccard_score, accuracy_score, precision_score, recall_score
)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def BGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_pred_f = torch.flatten(y_pred)
    y_true_f = torch.flatten(y_true)

    intersection = (y_pred_f * y_true_f).sum()
    y_sum = y_pred_f.sum() + y_true_f.sum() 

    return (2. * intersection + smooth) / (y_sum + smooth)

def dice_coef_multiclass(y_true, y_pred, n_classes=80):
    dice = 0.
    print(torch.unique(y_true).shape)
    #for i in range(n_classes):

    #    weight = (torch.numel(y_true) / n_classes) / torch.numel(y_true[y_true==i])
    #    print(y_true.shape, torch.unique(y_true))
    #    print(f"{i}/{n_classes} - {torch.numel(y_true[y_true==i])}/{torch.numel(y_true)}")
    #    dice += weight * dice_coef(y_true[i,:,:], y_pred[i,:,:])
    #return dice / n_classes
    return 1

def pixelwise_accuracy(y, y_pred):
    return (y_pred == y).sum() / y.shape[1]






