"""A script used for model testing"""

from pathlib import Path
from typing import Dict

from pytorch_lightning import seed_everything

from classification.config import read
from classification.models.models import test


def main(
    seed: int,
    model_dir: str,
    data_args: Dict,
    trainer_args: Dict,
):
    """Runs tests with passed argument dictionary"""
    seed_everything(seed)
    model_dir = Path(model_dir)
    test(data_args, model_dir, trainer_args)


if __name__ == "__main__":
    main(**read())
