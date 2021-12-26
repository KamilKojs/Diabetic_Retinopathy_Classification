import torch
from torchvision import transforms

def default_transform():
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def augmented_transform_mobilenet_v2():
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.RandomCrop(256),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=15),
                ]
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
