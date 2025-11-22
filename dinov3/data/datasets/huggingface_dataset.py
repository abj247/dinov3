# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from datasets import load_dataset
from torchvision.datasets import VisionDataset

logger = logging.getLogger("dinov3")


class HuggingFaceDataset(VisionDataset):
    def __init__(
        self,
        *,
        path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(root=None, transforms=transforms, transform=transform, target_transform=target_transform)
        self.dataset = load_dataset(path, split=split, **kwargs)
        self.path = path
        self.split = split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        item = self.dataset[index]
        image = item["image"]
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Dummy target since we are doing self-supervised learning mostly, 
        # or if the dataset has a label, we could try to use it, but for DINOv3 pretraining
        # the target is usually not used or is just an index.
        # Let's see if there is a label column, otherwise return 0.
        target = item.get("label", 0)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.dataset)
