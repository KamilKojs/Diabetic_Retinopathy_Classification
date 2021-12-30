from torchvision import transforms

def default_transform():
    return transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def augmented_transform():
    return transforms.Compose(
        [
            transforms.Resize(512),
            transforms.RandomCrop(512),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomPerspective(distortion_scale=0.1, p=0.5)
                ]
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
