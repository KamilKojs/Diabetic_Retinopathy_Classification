from typing import Optional, Callable, Tuple, Any, List
from pathlib import Path
import pandas as pd

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader

from classification.data.transforms import default_transform

class LabeledDataset(VisionDataset):
    """
    Labeled dataset used for model training
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = default_transform(),
    ) -> None:
        super().__init__(root, transform=transform)

        labels_path = f"{root}/labels.csv"
        samples = _load_samples(root, labels_path)
        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is ground truth of the target class.
        """
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def _load_samples(images_dir: str, labels_path: str):
    labels = pd.read_csv(labels_path)
    samples = [
        (
            sample_path,
            labels.loc[labels["image"] == sample_path.stem]["level"].values[0]
        ) 
        for sample_path 
        in Path(images_dir).rglob("*.jpeg")
    ]
    return samples
