from typing import Optional, Callable, Tuple, Any, List
from pathlib import Path

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

        covid_dir = f"{root}/COVID"
        normal_dir = f"{root}/Normal"
        pneumonia_dir = f"{root}/Viral_Pneumonia"
        lung_opacity_dir = f"{root}/Lung_Opacity"

        samples = _concat_datasets([covid_dir], [normal_dir, pneumonia_dir, lung_opacity_dir])
        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is ground truth of the target class.
        """
        path, illness, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, illness, target

    def __len__(self) -> int:
        return len(self.samples)


def _concat_datasets(positive_case_dirs: List[str], negative_case_dirs: List[str]):
    samples = []
    for positive_case_dir in positive_case_dirs:
        illness = Path(positive_case_dir).stem
        samples.extend([(sample_path, illness, 1) for sample_path in Path(positive_case_dir).rglob("*.png")])
    for negative_case_dir in negative_case_dirs:
        illness = Path(negative_case_dir).stem
        samples.extend([(sample_path, illness, 0) for sample_path in Path(negative_case_dir).rglob("*.png")])
    return samples