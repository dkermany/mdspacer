import os
import sys
import cv2
from abc import ABC, abstractmethod
from glob import glob
from sklearn.metrics import (
    jaccard_score, accuracy_score, precision_score, recall_score
)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def BGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint: {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print(f"=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def dice_score(y, preds):
    assert preds.shape[1] == y.shape[1]
    smooth = 1.

    xflat = torch.flatten(preds)
    yflat = torch.flatten(y)
    intersection = (xflat * yflat).sum()

    return (2. * intersection + smooth) / (xflat.sum() + yflat.sum() + smooth)

def pixelwise_accuracy(y, preds):
    assert preds.shape[1] == y.shape[1]
    return (preds == y).sum() / y.shape[1]

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    total_y, total_preds = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            probs = torch.softmax(model(x), dim=1)
            preds = torch.argmax(probs, dim=1)
            total_y = torch.cat((total_y, y), dim=0)
            total_preds = torch.cat((total_preds, preds), dim=0)

    accuracy = accuracy_score(total_y, total_preds)
    precision = precision_score(total_y, total_preds)
    recall = recall_score(total_y, total_preds)
    IoU = jaccard_score(total_y, total_preds)
    dice = dice_score(total_y, total_preds)

    print(f"""
        Pixel-wise Accuracy: {accuracy}\n
        Precision: {precision}\n
        Recall: {recall}\n
        IoU: {IoU}\n
        Dice: {dice}
    """)

    model.train()





