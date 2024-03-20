from abc import ABCMeta, abstractmethod
from typing import Sequence, Any, Tuple

import torch

from .util import slic_tensor


class Perturbator(metaclass=ABCMeta):
    @abstractmethod
    def create_perturbation(self, tensor: torch.Tensor) -> Tuple[Sequence[torch.Tensor], Tuple[Any]]:
        pass

    @abstractmethod
    def apply_perturbation_(self, tensor: torch.Tensor, perturbation: Sequence[torch.Tensor], extra: Tuple[Any]):
        pass

    @abstractmethod
    def clamp_perturbation_(self, perturbation: Sequence[torch.Tensor], extra: Tuple[Any], epsilon: float):
        pass


class NoisePerturbator(Perturbator):
    def create_perturbation(self, tensor: torch.Tensor):
        return (torch.zeros_like(tensor).to(tensor),), (None,)

    def apply_perturbation_(self, tensor: torch.Tensor, perturbation: Sequence[torch.Tensor], extra: Tuple[Any]):
        assert len(perturbation) == 1
        tensor.add_(perturbation[0]).clamp_(0, 1)

    def clamp_perturbation_(self, perturbation: Sequence[torch.Tensor], extra: Tuple[Any], epsilon: float):
        assert len(perturbation) == 1
        perturbation[0].clamp_(-epsilon, epsilon)


class HistogramPerturbator(Perturbator):
    def __init__(self, bins: int = 256):
        super(HistogramPerturbator, self).__init__()
        self.bins = bins

    def create_perturbation(self, tensor: torch.Tensor):
        perturbation = torch.zeros(*tensor.shape[:2], self.bins).to(tensor)
        return (perturbation,), (None,)

    def apply_perturbation_(self, tensor: torch.Tensor, perturbation: Sequence[torch.Tensor], extra: Tuple[Any]):
        assert len(perturbation) == 1
        perturbation = perturbation[0]
        n, c, h, w = tensor.shape

        # p[batch][channel][spatial] = perturbation[batch][channel][q[batch][channel][spatial]]
        quantitized = (tensor.flatten(-2) * (self.bins - 1)).round().long()
        perturbation = torch.gather(perturbation, 2, quantitized)

        perturbation = perturbation.view_as(tensor)
        tensor.add_(perturbation).clamp_(0, 1)

    def clamp_perturbation_(self, perturbation: Sequence[torch.Tensor], extra: Tuple[Any], epsilon: float):
        assert len(perturbation) == 1
        perturbation[0].clamp_(-epsilon, epsilon)


class SuperPixelHistogramPerturbator(Perturbator):
    def __init__(self, bins: int = 256, segs: int = 100):
        super(SuperPixelHistogramPerturbator, self).__init__()
        self.bins = bins
        self.segs = segs

    def create_perturbation(self, tensor: torch.Tensor):
        perturbation = torch.zeros(*tensor.shape[:2], self.bins, self.segs).to(tensor)
        extra = slic_tensor(tensor, n_segments=self.segs)
        return (perturbation,), (extra,)

    def apply_perturbation_(self, tensor: torch.Tensor, perturbation: Sequence[torch.Tensor], extra: Tuple[torch.Tensor]):
        assert len(perturbation) == 1
        perturbation = perturbation[0]
        extra = extra[0]
        n, c, h, w = tensor.shape

        # perturbation[batch][channel][spatial][segment] = perturbation[batch][channel][quantitized[batch][channel][spatial][...segment]][segment]
        quantitized = (tensor.flatten(-2) * (self.bins - 1)).round().long().unsqueeze(-1).expand(-1, -1, -1, self.segs)
        perturbation = torch.gather(perturbation, 2, quantitized)

        # p[batch][channel][spatial][0] = perturbation[batch][channel][spatial][segment[segment[batch][...channel][spatial][0]]
        segment = extra.flatten(-2).unsqueeze(dim=-2).expand(-1, c, -1).unsqueeze(-1)
        perturbation = torch.gather(perturbation, 3, segment).squeeze(-1)

        perturbation = perturbation.view_as(tensor)
        tensor.add_(perturbation).clamp_(0, 1)

    def clamp_perturbation_(self, perturbation: Sequence[torch.Tensor], extra: Tuple[torch.Tensor], epsilon: float):
        assert len(perturbation) == 1
        perturbation[0].clamp_(-epsilon, epsilon)


class ComposedPerturbator(Perturbator):
    def __init__(self, composed: Sequence[Perturbator]):
        super(ComposedPerturbator, self).__init__()
        self.composed = composed

    def create_perturbation(self, tensor: torch.Tensor) -> Tuple[Sequence[torch.Tensor], Tuple[Tuple[Any]]]:
        perturbation = []
        extra = []
        for c in self.composed:
            p, e = c.create_perturbation(tensor)
            assert len(p) == 1, 'only support primitive perturbator for composing'
            perturbation.append(p[0])
            extra.append(e[0])
        return tuple(perturbation), tuple(extra)

    def apply_perturbation_(self, tensor: torch.Tensor, perturbation: Sequence[torch.Tensor], extra: Tuple[Any]):
        assert len(perturbation) == len(self.composed)
        assert len(extra) == len(self.composed)
        for c, p, e in zip(self.composed, perturbation, extra):
            c.apply_perturbation_(tensor, (p,), (e,))

    def clamp_perturbation_(self, perturbation: Sequence[torch.Tensor], extra: Tuple[Any], epsilon: float):
        assert len(perturbation) == len(self.composed)
        assert len(extra) == len(self.composed)
        for c, p, e in zip(self.composed, perturbation, extra):
            c.clamp_perturbation_((p,), (e,), epsilon)


class FirstPerturbator(Perturbator):
    def __init__(self, wrapped: Perturbator):
        super(FirstPerturbator, self).__init__()
        self.wrapped = wrapped

    def create_perturbation(self, tensor: torch.Tensor) -> Tuple[Sequence[torch.Tensor], Tuple[Any]]:
        return self.wrapped.create_perturbation(tensor[:1])

    def apply_perturbation_(self, tensor: torch.Tensor, perturbation: Sequence[torch.Tensor], extra: Tuple[Any]):
        self.wrapped.apply_perturbation_(tensor[:1], perturbation, extra)

    def clamp_perturbation_(self, perturbation: Sequence[torch.Tensor], extra: Tuple[Any], epsilon: float):
        self.wrapped.clamp_perturbation_(perturbation, extra, epsilon)

