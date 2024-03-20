import torch
import torch.autograd as ag
import torch.nn as nn

import kornia

from .attack import Attack
from .criterions import Criterion


class TestAttack(Attack):
    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        step: int,
        lr: float,
        alpha: float = 1.
    ):
        super(TestAttack, self).__init__(model)
        assert step >= 0, f'step should be non-negative integer, got {step}'
        assert lr >= 0, f'lr should be non-negative float, got {lr}'
        assert alpha >= 0, f'alpha should be non-negative float, got {alpha}'
        self.criterion = criterion
        self.step = step
        self.lr = lr
        self.alpha = alpha

    def __call__(self, tensor: torch.Tensor):
        # identity bias field
        n, c, h, w = tensor.size()
        noise = torch.zeros_like(tensor).requires_grad_()
        params = [noise]

        losses = []
        for n_iter in range(self.step + 1):
            pert = tensor.clone()
            pert = kornia.rgb_to_luv(pert)
            pert = pert.add_(noise * 100)
            pert = kornia.luv_to_rgb(pert)
            pert = pert.clamp_(0, 1)
            pred = self.model(pert)

            # calculate loss and sparsity constraint terms
            crit = self.criterion(pred)
            # diff_h = (noise[:, :, :, 1:] - noise[:, :, :, :-1])
            # diff_v = (noise[:, :, 1:, :] - noise[:, :, :-1, :])
            # diff = torch.cat((diff_h.flatten(1), diff_v.flatten(1)), dim=1)
            # sparsity = diff.pow_(2).sum()

            sparsity = noise.pow(2).sum()

            loss = torch.zeros(1).to(tensor)
            loss.add_(crit)
            loss.add_(sparsity, alpha=self.alpha)

            losses.append(crit.item())

            # grad and update bias field
            if n_iter < self.step:
                with torch.no_grad():
                    grads = ag.grad(loss, params)
                    for param, grad in zip(params, grads):
                        param.sub_(grad.sign_().mul_(self.lr))

        print(f'%delta: {(losses[0] - losses[-1]) / losses[0]:.2%}')

        diff = pert - tensor
        print(f'min/max/absmean: {diff.min().item():.2f}/{diff.max().item():.2f}/{diff.abs().mean().item():.2f}')

        return pert, pred, noise
