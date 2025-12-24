"""
This file will create a subset of ImageNet dataset to be used for testing memorization.
"""

from collections import defaultdict
import os
import random
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from pathlib import Path
from torch.utils.data import Subset
from utils.data import normalize_01_into_pm1


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_imagenet_dataset(
    root: str = "/scratch/inf0/user/mparcham/ILSVRC2012",
    split: str = "val",                  # "train" or "val"
    transform: T.Compose | None = None,
    final_reso: int = 256,
    mid_reso: float = 1.125,
):
    """
    Build an ImageFolder dataset for ImageNet with VAR-style preprocessing.

    Transform (if not provided) follows the VAR setting:
      - Resize the shorter edge to round(mid_reso * final_reso) with LANCZOS
      - RandomCrop to (final_reso, final_reso)
      - ToTensor, then normalize from [0,1] to [-1,1] via normalize_01_into_pm1

    Note: We intentionally use RandomCrop for both train/val here to enable
    sampling multiple random crops per image downstream for robust activation
    estimation.
    """
    if transform is None:
        mid_px = round(mid_reso * final_reso)
        transform = T.Compose([
            T.Resize(mid_px, interpolation=InterpolationMode.LANCZOS),
            T.RandomCrop((final_reso, final_reso)),
            T.ToTensor(),
            normalize_01_into_pm1,
        ])

    dataset = ImageFolder(root=os.path.join(root, split), transform=transform)
    return dataset


def get_balanced_subset(
    dataset: ImageFolder,
    total_samples: int = 5000,
    shuffle: bool = True,
    seed: int = 0,
):
    """
    Create a balanced subset from a given ImageFolder dataset.
    Returns a torch.utils.data.Subset with `total_samples`, split evenly across classes.
    """

    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    # determine number of samples per class
    num_classes = len(class_to_indices)
    samples_per_class = total_samples // num_classes

    rng = random.Random(seed)
    selected_indices = []

    # select samples for each class
    for _, indices in class_to_indices.items():
        if shuffle:
            rng.shuffle(indices)
        selected_indices.extend(indices[:samples_per_class])

    # If total_samples not divisible by num_classes, fill with leftovers
    remaining = total_samples - len(selected_indices)
    if remaining > 0:
        leftovers = [
            idx for indices in class_to_indices.values()
            for idx in indices[samples_per_class:]
        ]
        if shuffle:
            rng.shuffle(leftovers)
        selected_indices.extend(leftovers[:remaining])

    return Subset(dataset, selected_indices)


def get_balanced_imagenet_dataset(
    root: str = "/scratch/inf0/user/mparcham/ILSVRC2012",
    split: str = "val",                  # "train" or "val"
    total_samples: int = 12800,           # 1% if imagenet training set
    shuffle: bool = True,
    seed: int = 0,
):
    """
    Convenience wrapper: builds the ImageNet dataset then returns a balanced subset.
    """
    dataset = build_imagenet_dataset(root=root, split=split)
    return get_balanced_subset(
        dataset=dataset,
        total_samples=total_samples,
        shuffle=shuffle,
        seed=seed,
    )
