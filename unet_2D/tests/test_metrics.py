import pytest
import torch
from torch import Tensor
from torchmetrics import JaccardIndex, Accuracy
from time import time

def test_iou():
    jaccard = JaccardIndex(num_classes=81, ignore_index=0).cuda()
    x = torch.randint(0, 81, (100, 256, 256), device="cuda")
    y = x[:]
    print(jaccard(x, y).item())
    

def test_acc():
    acc = Accuracy(num_classes=81, ignore_index=0).cuda()
    x = torch.randint(0, 81, (100, 256, 256), device="cuda")
    y = x[:]
    start = time()
    j = acc(x.flatten(), y.flatten()).item()
    print(time() - start)

test_iou()
