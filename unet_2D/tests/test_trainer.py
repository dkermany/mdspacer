import pytest
from torch.utils.data.dataloader import DataLoader
from pytest import MonkeyPatch
from unet.train import UNetTrainer
from utils import load_checkpoint

def test_UNetTrainer_init():
    trainer = UNetTrainer(
        image_size=256,
        num_classes=1,
    )

def test_UNetTrainer_checkpoint_invalid():
    trainer = UNetTrainer(
        image_size=256,
        num_classes=1,
    )
    with pytest.raises(ValueError):
        trainer.set_checkpoint("sfdh")

def test_UNetTrainer_checkpoint_nofile():
    trainer = UNetTrainer(
        image_size=256,
        num_classes=1,
    )
    with pytest.raises(FileNotFoundError):
        load_checkpoint("sfdh.pth.tar", trainer.model, trainer.optimizer)

def test_UNetTrainer_checkpoint():
    trainer = UNetTrainer(
        image_size=256,
        num_classes=1,
    )
    load_checkpoint(
        "checkpoints/07012022/unet1.0-coco-epoch76.pth.tar",
        trainer.model,
        trainer.optimizer
    )
