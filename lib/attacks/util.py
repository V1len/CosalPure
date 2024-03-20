import torch
from skimage.segmentation import slic


def slic_tensor(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    device = tensor.device
    segs = []
    for im in tensor.unbind(dim=0):
        im = im.permute(1, 2, 0).double().cpu().numpy()
        seg = slic(im, *args, **kwargs)
        seg = torch.tensor(seg, device=device).long()
        segs.append(seg)
    return torch.stack(segs, dim=0)