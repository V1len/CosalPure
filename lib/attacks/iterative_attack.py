from math import inf
from typing import Optional

import torch
import torch.autograd as ag
import torch.nn as nn

from .attack import Attack
from .criterions import Criterion
from .perturbators import Perturbator


class IterativeAttack(Attack):
    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        perturbator: Perturbator,
        step: int,
        epsilon: float = inf,
        absolute_lr: Optional[float] = None,
        relative_lr: Optional[float] = None
    ):
        super(IterativeAttack, self).__init__(model)
        assert step >= 0, f'step should be non-negative integer, got {step}'
        assert epsilon > 0, f'epsilon should be positive float, got {epsilon}'
        self.criterion = criterion
        self.perturbator = perturbator
        self.step = step
        self.epsilon = epsilon
        self.absolute_lr = absolute_lr
        self.relative_lr = relative_lr

    def __call__(
        self,
        tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        returns_iter: bool = False
    ):
        it = self._make_iter(tensor, target)
        if returns_iter:
            return it

        for last in it:
            pass
        return last

    def _make_iter(
        self,
        tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ):
        perturbation, extra = self.perturbator.create_perturbation(tensor)
        for p in perturbation: p.requires_grad_()
        perturbated = tensor.clone()
        self.perturbator.apply_perturbation_(perturbated, perturbation, extra)
        predictions = self.model(perturbated)
        yield perturbated, predictions, perturbation, extra

        for _ in range(self.step):
            loss = self.criterion(predictions) if target is None else self.criterion(predictions, target)
            with torch.no_grad():
                grad = ag.grad(loss, perturbation)
                for g in grad: g.sign_()
                perturbation = [p.sub_(g * self.lr) for p, g in zip(perturbation, grad)]
                if self.epsilon is not None:
                    self.perturbator.clamp_perturbation_(perturbation, extra, self.epsilon)
            perturbated = tensor.clone()
            self.perturbator.apply_perturbation_(perturbated, perturbation, extra)
            predictions = self.model(perturbated)
            yield perturbated, predictions, perturbation, extra

    @property
    def lr(self) -> float:
        if self.absolute_lr is not None:
            assert self.relative_lr is None, 'relative_lr is mutually exclusive to absolute_lr'
            return self.absolute_lr
        assert self.epsilon != inf, 'epsilon should be not inf to determine lr when absolute_lr is None'
        if self.relative_lr is not None:
            return self.epsilon * self.relative_lr
        # both absolute_lr and relative_lr are None
        return self.epsilon / self.step
