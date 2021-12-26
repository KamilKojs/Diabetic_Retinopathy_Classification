from collections import defaultdict

import torch
import torch.utils.data


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Args:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments
        - dataset and index
    """

    def __init__(self, dataset, callback_get_label, num_samples=None):

        # All elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        # If num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # Distribution of classes in the dataset
        label_to_count = defaultdict(int)
        for idx in self.indices:
            label = callback_get_label(dataset, idx)
            label_to_count[label] += 1

        # Weight for each sample
        weights = [
            1.0 / label_to_count[callback_get_label(dataset, idx)]
            for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples
