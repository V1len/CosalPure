from collections import defaultdict
from typing import Optional, Any, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from PIL import Image


__all__ = ['MaskDataset']


class MaskDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_folder: str,
        mask_folder: Optional[str] = None,
        image_ext: str = 'jpg',
        mask_ext: str = 'png',
    ):

        super(MaskDataset, self).__init__()

        data_root = Path(data_root)
        image_folder = Path(data_root / image_folder)
        mask_folder = Path(data_root / mask_folder) if mask_folder is not None else None

        image_paths = sorted(list(image_folder.rglob(f'*.{image_ext}')))

        if mask_folder is not None:
            mask_paths = [mask_folder / image_path.relative_to(image_folder).with_suffix(f'.{mask_ext}') for image_path in image_paths]
        else:
            mask_paths = None
        group_to_indices = defaultdict(list)

        for i, p in enumerate(image_paths):
            group_to_indices[p.parent.name].append(i)

        self.data_root = data_root
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_ext = image_ext
        self.mask_ext = mask_ext

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.group_to_indices = group_to_indices

    def __getitem__(self, original_index: int) -> Dict[str, Any]:
        if original_index not in range(-len(self) - 1, len(self)):
            raise IndexError()

        index = original_index % len(self)

        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        mask_path = self.mask_paths and self.mask_paths[index]
        if mask_path is None or original_index < 0:
            mask = Image.new('1', image.size)
        else:
            mask = Image.open(self.mask_paths[index]).convert('1')

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }

    def __len__(self):
        return len(self.image_paths)
