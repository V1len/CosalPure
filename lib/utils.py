import torch


def measure(prediction: torch.Tensor, target: torch.Tensor, b: float = 0.3, eps: float = 1e-5) -> torch.Tensor:
    prediction = prediction >= .5
    target = target >= .5
    total = prediction.numel()
    tp = (target & prediction).sum().item()
    tn = (~target & ~prediction).sum().item()
    fp = (~target & prediction).sum().item()
    fn = (target & ~prediction).sum().item()
    
    p = (tp + eps) / (tp + fp + eps)
    r = (tp + eps) / (tp + fn + eps)
    fs = ((b + 1) * p * r + eps) / (b * p + r + eps)
    mae = (fp + fn) / total
    return {'precision': p, 'recall': r, 'fmeasure': fs, 'mae': mae}