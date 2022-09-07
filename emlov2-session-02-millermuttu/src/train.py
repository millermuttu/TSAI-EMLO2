import os
import torch
import hydra
import torch.nn as nn   
import torch.nn.functional as F
import logging
from pytorch_lightning import LightningModule, Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import timm
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, List, Optional
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path, instantiate

seed_everything(7)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

logger = logging.getLogger(__name__)


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str,
                train_data_dir: str,
                test_data_dir: str,
                train_dataloader_conf: Optional[DictConfig] = None,
                test_dataloader_conf: Optional[DictConfig] = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = train_data_dir
        self.test_dir = test_data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create()
        self.test_dataloader_conf = test_dataloader_conf or OmegaConf.create()
        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, **self.train_dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, **self.test_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, **self.test_dataloader_conf)

def create_model(channels,num_classes):
    model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class CustomModel(LightningModule):
    def __init__(self, channels, num_classes,lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model(channels,num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    dm = instantiate(cfg.data)
    channel,height,weight = dm.dims
    model = CustomModel(channel,dm.num_classes,lr=cfg.train.learning_rate)

    trainer = Trainer(
        max_epochs= cfg.train.max_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint("example.ckpt")
    if cfg.train.run_eval:
        trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()