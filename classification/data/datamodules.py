from copy import copy

from classification.data.datasets import LabeledDataset
from classification.data.transforms import default_transform, augmented_transform_mobilenet_v2
from classification.data.samplers import ImbalancedDatasetSampler

from torch.utils.data import random_split, DataLoader, Subset
import pytorch_lightning as pl

class ClassificationDataModule(pl.LightningDataModule):
    """PyTorch-Lightning data module for classification"""

    def __init__(
        self,
        trainval_dataset_dir: str = None,
        trainval_split_size: float = 0.9,
        train_auto_balancing: bool = False,
        train_augmentation: bool = False,
        train_batch_size: int = 4,
        valtest_batch_size: int = 32,
        test_dataset_dir: str = None,
        num_workers: int = 12,
        model_type: str = "mobilenet_v2",
        augmentation_type: str = "default",
    ):
        """Data module used for basic pytorch lightning operations like
        applying augmentations, preparing the data loaders etc.

        Args:
            trainval_dataset_dir (str, optional): The dataset which
            will be randomly split into training and validation sets. Defaults to None.
            trainval_split_size (float, optional): Defines the relative size
            of training dataset. Defaults to 0.9.
            train_auto_balancing (bool, optional): If set to True then
            the class balancing will occur for training dataset. Defaults to False.
            train_augmentation (bool, optional): If set to True then
            the data augmentation will occur for training dataset. Defaults to False.
            train_batch_size (int, optional): How many samples are in training batch.
            Defaults to 4.
            valtest_batch_size (int, optional): How many samples are in validation
            and testing batches. Defaults to 32.
            test_dataset_dir (str, optional): The dataset on which the model
            will be tested. Defaults to None.
            num_workers (int, optional): How many workers are used for data loading.
            Defaults to 12.
            model_type (str): Type of the model. It impacts the augmentation used. Defaults to mobilenet_v2.
            augmentation_type (str): Type of the augmentation used in dataloaders.
        """
        super().__init__()
        self.trainval_dataset_dir = trainval_dataset_dir
        self.trainval_split_size = trainval_split_size
        self.train_auto_balancing = train_auto_balancing
        self.train_augmentation = train_augmentation
        self.train_batch_size = train_batch_size
        self.valtest_batch_size = valtest_batch_size
        self.test_dataset_dir = test_dataset_dir
        self.num_workers = num_workers

        if model_type == "mobilenet_v2":
            if augmentation_type == "default":
                self.transform = augmented_transform_mobilenet_v2
            else:
                raise Exception(
                    f"Data augmentation '{augmentation_type}' is not supported"
                )
            self.dataset_transform = default_transform()
        else:
            raise Exception(f"Model type '{model_type}' is not supported")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = self._load_dataset(
                self.trainval_dataset_dir,
                self.dataset_transform,
            )
            self.train_dataset, self.val_dataset = self._split_dataset(
                dataset, self.trainval_split_size
            )
            if self.train_augmentation:
                self._substitute_transforms(self.train_dataset, self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = self._load_dataset(
                self.test_dataset_dir,
                self.dataset_transform,
            )

    @staticmethod
    def _load_dataset(dataset_dir: str, transform):
        return LabeledDataset(root=dataset_dir, transform=transform)

    @staticmethod
    def _substitute_transforms(subset_of_dataset: Subset, transforms):
        '''Replaces the transform for subset's dataset. Copy is needed.'''
        dataset = copy(subset_of_dataset.dataset)
        dataset.transform = transforms()
        subset_of_dataset.dataset = dataset

    @staticmethod
    def _split_dataset(dataset: LabeledDataset, split_size: float):
        n_train = int(len(dataset) * split_size)
        n_val = len(dataset) - n_train
        return random_split(dataset, [n_train, n_val])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            sampler=ImbalancedDatasetSampler(
                self.train_dataset,
                callback_get_label=self._extract_label_from_dataset_subset,
            )
            if self.train_auto_balancing
            else None,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=not self.train_auto_balancing,
        )
    
    @staticmethod
    def _extract_label_from_dataset_subset(subset: Subset, idx: int):
        dataset = subset.dataset
        return ClassificationDataModule._extract_label_from_dataset(
            dataset, idx
        )

    @staticmethod
    def _extract_label_from_dataset(dataset: LabeledDataset, idx: int):
        return dataset.samples[idx][1]

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.valtest_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.valtest_batch_size,
            num_workers=self.num_workers,
        )
