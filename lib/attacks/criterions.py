from typing import Callable, TypeVar

import torch


Criterion = TypeVar(
    'Criterion',
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
)