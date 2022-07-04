import pytest
from torch.utils.data.dataloader import DataLoader
from pytest import MonkeyPatch
from unet.dataset import CoNSePDataset

def test_CoNSePDataset_invalid():
    with pytest.raises(TypeError):
        ds = CoNSePDataset()

def test_CoNSePDataset_load_invalid():
    with pytest.raises(ValueError):
        ds = CoNSePDataset("dgsyvr", "j29fj")


def test_CoNSePDataset_get_item():
    ds = CoNSePDataset(
        "/home/dkermany/data/CoNSeP/Test/Images",
        "/home/dkermany/data/CoNSeP/Test/Labels",
        image_ext="png",
        mask_ext="mat",
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    data, targets = next(iter(loader))
