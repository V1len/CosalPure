from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .attacks import Attack
from .data import MaskDataset


class Solver(metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        attack: Attack,
        dataset: MaskDataset,
        device: Optional[torch.device] = None
    ):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.name = name
        self.model = model.eval().to(device)
        self.attack = attack
        self.dataset = dataset
        self.device = device

    def __call__(self):
        for sample in self.dataset:
            image = sample['image'][None]
            mask = sample['mask'][None]
            r = self.attack(image, mask)
            image = r[0][0].cpu()
            mask = r[1][0].cpu()
