from torch import Tensor
from torch.utils.data import DataLoader

def _check_COCO_image(image):
    assert image.shape[0] == 3

def _check_COCO_mask(mask):
    assert mask.shape[0] == 3

def _check_CoNSeP_image(image):
    assert image.shape[0] == 3

def _check_inference_image(image):
    assert image.shape[0] == 3

def _check_save_labels_as_rgb_input(label):
    assert len(label.shape) == 3 and label.shape[1] == label.shape[2]
