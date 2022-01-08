'''Script used for model training'''

from classification.config import read
from classification.models.models import train
from typing import Dict
from pytorch_lightning import seed_everything

def main(
    seed: int,
    model_type: str,
    augmentation_type: str,
    output_dir: str,
    data_args: Dict,
    model_args: Dict,
    trainer_args: Dict,
    early_stopping_args: Dict,
):
    """Runs training with passed argument dictionary"""
    seed_everything(seed)
    train(
        data_args,
        model_args,
        trainer_args,
        early_stopping_args,
        output_dir,
        model_type,
        augmentation_type
    )


if __name__ == "__main__":
    main(**read())
