"""CIFAR-10 on the GPU and the augmentation used by the original Data Diet code."""
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
DEFAULT_DATA_ROOT = os.environ.get("CIFAR10_ROOT", "/home/sofya/Downloads/data")
CHANNEL_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CHANNEL_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
NUM_CLASSES = 10


@dataclass
class Cifar10Data:
    """Images are normalized float32 tensors (N, 3, 32, 32); examples keep the original CIFAR-10 order."""
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    train_data_sha256: str  # fingerprint of the raw training images and labels, stored with every run

    @property
    def num_train_examples(self) -> int:
        return len(self.train_images)


def load_cifar10(device: str = "cuda", data_root: str = DEFAULT_DATA_ROOT) -> Cifar10Data:
    tensors, fingerprint = {}, hashlib.sha256()
    for split, is_train in (("train", True), ("test", False)):
        dataset = CIFAR10(data_root, train=is_train, download=False)
        if is_train:
            fingerprint.update(dataset.data.tobytes())
            fingerprint.update(np.asarray(dataset.targets, dtype=np.int64).tobytes())
        normalized = (dataset.data.astype(np.float32) / 255.0 - CHANNEL_MEAN) / CHANNEL_STD
        images = torch.from_numpy(normalized).permute(0, 3, 1, 2).contiguous().to(device)
        tensors[split] = (images, torch.tensor(dataset.targets, dtype=torch.long, device=device))
    return Cifar10Data(
        train_images=tensors["train"][0], train_labels=tensors["train"][1],
        test_images=tensors["test"][0], test_labels=tensors["test"][1],
        train_data_sha256=fingerprint.hexdigest(),
    )


def augment_like_official(images: torch.Tensor, crop_offset_rng: np.random.RandomState,
                          flip_generator: torch.Generator) -> torch.Tensor:
    """Official pipeline: REFLECT-pad by 4, ONE random 32x32 crop offset shared by the whole batch
    (tf.image.random_crop on a [B, H, W, C] tensor), independent horizontal flip for every image."""
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    offset_y, offset_x = crop_offset_rng.randint(0, 9, size=2)
    cropped = padded[:, :, offset_y:offset_y + 32, offset_x:offset_x + 32]
    flip_mask = torch.rand(images.shape[0], device=images.device, generator=flip_generator) < 0.5
    return torch.where(flip_mask[:, None, None, None], cropped.flip(3), cropped)
