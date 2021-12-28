"""A model for image classification task"""
# pylint: disable=R0901, R0913, R0914, W0613, W0221

import logging
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torch import nn
from torchvision.models import mobilenet_v2, vgg16, efficientnet_b7, densenet201, resnet152

from classification.data.datamodules import ClassificationDataModule
from classification.models.metrics import accuracy, cohen_kappa_score

logger = logging.getLogger(__name__)


class ClassificationModule(pl.LightningModule):
    """PyTorch-Lightning module for classification"""

    def __init__(
        self,
        learning_rate=2e-5,
        model_type="mobilenet_v2",
        augmentation_type="default",
    ):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.model_type == "mobilenet_v2":
            self.model = mobilenet_v2(pretrained=True)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=1, bias=True),
            )
        elif self.hparams.model_type == "vgg16":
            self.model = vgg16(pretrained=True)
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=4096, out_features=1024, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1024, out_features=1, bias=True),
            )
        elif self.hparams.model_type == "efficientnet_b7":
            self.model = efficientnet_b7(pretrained=True)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=2560, out_features=1, bias=True),
            )
        elif self.hparams.model_type == "densenet201":
            self.model = densenet201(pretrained=True)
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=1920, out_features=1, bias=True),
            )
        elif self.hparams.model_type == "resnet152":
            self.model = resnet152(pretrained=True)
            self.model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
        else:
            raise Exception(f"Model type '{model_type}' is not supported")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_images, y_true = batch
        y_pred = self(x_images)
        #loss = F.cross_entropy(y_pred, y_true)
        loss = F.mse_loss(y_pred, y_true.unsqueeze(1).float())
        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._valtest_step("val", batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self._valtest_epoch_end("val", outputs)

    def test_step(self, batch, batch_idx):
        return self._valtest_step("test", batch, batch_idx)

    def test_epoch_end(self, outputs):
        self._valtest_epoch_end("test", outputs)

    def _valtest_step(self, stage, batch, batch_idx):
        x_images, y_true = batch
        y_pred = self(x_images)
        #loss = F.cross_entropy(y_pred, y_true)
        loss = F.mse_loss(y_pred, y_true.unsqueeze(1).float())
        self.log(f"{stage}_loss", loss, prog_bar=True)

        #y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        #_, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            f"{stage}_loss": loss,
        }

    def _valtest_epoch_end(self, stage, outputs):
        y_true = torch.cat([o["y_true"] for o in outputs])
        y_pred = torch.cat([o["y_pred"] for o in outputs])
        format_preds(y_pred)
        y_pred = y_pred.long().squeeze(1)

        self.log(
            f"{stage}_accuracy",
            accuracy(y_pred, y_true),
        )
        self.log(
            f"{stage}_cohen_kappa_score",
            cohen_kappa_score(y_pred, y_true),
        )

        loss = torch.stack([o[f"{stage}_loss"] for o in outputs])
        self.log(f"{stage}_loss", torch.mean(loss))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def configure_trainer(
    save_dir: Path,
    trainer_args: Dict,
    early_stopping_args: Dict = None,
) -> Trainer:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    callbacks = []
    if early_stopping_args:
        callbacks.append(EarlyStopping(**early_stopping_args))

    return Trainer(
        **trainer_args,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir,
            name="",
            version="",
            default_hp_metric=False,
        ),
    )


def train(
    data_args: Dict,
    model_args: Dict,
    trainer_args: Dict,
    early_stopping_args: Dict,
    output_dir: Path,
    model_type: Optional[str] = None,
    augmentation_type: Optional[str] = None,
):
    model = ClassificationModule(
        model_type=model_type, augmentation_type=augmentation_type, **model_args
    )
    data = ClassificationDataModule(
        model_type=model_type,
        augmentation_type=augmentation_type,
        **data_args,
    )
    trainer = configure_trainer(
        output_dir, trainer_args, early_stopping_args
    )
    trainer.fit(model, datamodule=data)


def test(data_args: Dict, model_dir: Path, trainer_args: Dict):
    model = load_model(model_dir)
    data = ClassificationDataModule(
        **data_args,
        model_type=model.hparams.model_type,
        augmentation_type=model.hparams.augmentation_type,
    )
    trainer = configure_trainer(model_dir, trainer_args)
    trainer.test(model, datamodule=data)


def load_model(model_dir: Path, device: str = None):
    model_files = list(model_dir.glob('*.pth'))
    model_files.extend(list(model_dir.glob('*.ckpt')))
    model_path = model_files[0]
    if str(model_path).endswith("pth"):
        model = torch.load(model_path)
    else:
        model = ClassificationModule.load_from_checkpoint(model_path)
    if device:
        model = model.to(device)
    elif torch.cuda.is_available():
        model = model.to("cuda")
    return model


def format_preds(y_pred: torch.Tensor):
    y_pred[y_pred < 0.5] = 0
    y_pred[(y_pred >= 0.5) & (y_pred < 1.5)] = 1
    y_pred[(y_pred >= 1.5) & (y_pred < 2.5)] = 2
    y_pred[(y_pred >= 2.5) & (y_pred < 3.5)] = 3
    y_pred[(y_pred >= 3.5)] = 4