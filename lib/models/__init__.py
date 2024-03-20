from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchcv.model_provider import get_model as ptcv_get_model


def build_model(model_cfg: Dict[str, Any]):
    name = model_cfg.get('name')
    if name == 'PoolNet':
        model = build_poolnet(model_cfg)
    elif name == 'EGNet':
        model = build_egnet(model_cfg)
    elif name == 'BASNet':
        model = build_basnet(model_cfg)
    elif name == 'U2Net':
        model = build_u2net(model_cfg)
    elif name == 'GCAGC':
        model = build_gcagc(model_cfg)
    elif name == 'GICD':
        model = build_gicd(model_cfg)
    elif name == 'Cls':
        model = build_cls(model_cfg)
    elif name == 'HGD':
        model = build_hgd(model_cfg)
    elif name == 'ZeroDCE':
        model = build_zerodce(model_cfg)
    elif name == 'GCoNet':
        model = build_gconet(model_cfg)
    else:
        raise RuntimeError(f'Unknown model name {name}')

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def build_poolnet(model_cfg):
    # build model
    backbone = model_cfg.get('backbone', 'resnet')
    joint = model_cfg.get('joint', False)
    mode = model_cfg.get('mode', 'sal')
    model = PoolNetWrapper(backbone, joint, mode)

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.load_state_dict(torch.load(weights_path))

    return model


def build_egnet(model_cfg):
    # build model
    backbone = model_cfg.get('backbone', 'resnet')
    mode = model_cfg.get('mode', 'sal')
    model = EGNetWrapper(backbone, mode)

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.load_state_dict(torch.load(weights_path))

    return model


def build_basnet(model_cfg):
    # build model
    model = BASNetWrapper()

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.load_state_dict(torch.load(weights_path))

    return model


def build_u2net(model_cfg):
    # build model
    model = U2NetWrapper()

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.load_state_dict(torch.load(weights_path))

    return model


def build_gcagc(model_cfg):
    # build model
    backbone = model_cfg.get('backbone', 'hrnet')
    mode = model_cfg.get('mode', 'cosal')
    model = GCAGCWrapper(backbone, mode)

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.load_state_dict(torch.load(weights_path))

    return model


def build_gicd(model_cfg):
    # build model
    mode = model_cfg.get('mode', 'cosal')
    detach_cls = model_cfg.get('detach_cls', False)
    model = GICDWrapper(mode, detach_cls)

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.ginet.load_state_dict(torch.load(weights_path))

    return model


def build_cls(model_cfg):
    # build model
    cls_name = model_cfg.get('cls_name', 'resneta152b')
    stages = model_cfg.get('stages')
    model = ClsWrapper(cls_name, stages)
    return model


def build_hgd(model_cfg):
    # build model
    model = HGDWrapper()
    return model


def build_zerodce(model_cfg):
    # build model
    model = ZeroDCEWrapper()

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.load_state_dict(torch.load(weights_path))

    return model


def build_gconet(model_cfg):
    # build model
    model = GCoNetWrapper()

    # load weights
    weights_path = model_cfg.get('weights_path')
    model.wrapped.ginet.load_state_dict(torch.load(weights_path))

    return model


class PoolNetWrapper(nn.Module):
    def __init__(self, backbone, joint, mode):
        super(PoolNetWrapper, self).__init__()

        if joint:
            from .PoolNet.networks.joint_poolnet import build_model
        else:
            from .PoolNet.networks.poolnet import build_model

        assert mode in ('sal', 'feat')
            
        self.backbone = backbone
        self.joint = joint
        self.mode = mode
        self.wrapped = build_model(backbone)
        # See L132@EGNet/dataset.py
        self.register_buffer('image_mean', torch.tensor((122.67892, 104.00699, 116.66877)))

    def forward(self, x):
        # centerization
        x = (x * 255.) - self.image_mean[:, None, None]

        # resize if too small
        h, w = x.shape[-2:]
        new_h, new_w = min(max(h, 112), 960), min(max(w, 112), 960)
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        x = x[:, [2, 1, 0]]
        
        if self.mode == 'sal':
            r = self.wrapped(x, 1) if self.joint else self.wrapped(x)
            r = r.sigmoid()
            r = F.interpolate(r, size=(h, w), mode='bilinear', align_corners=True)
            return r
        elif self.mode == 'feat':
            conv2merge, infos = self.wrapped.base(x)
            if self.wrapped.base_model_cfg == 'resnet':
                conv2merge = self.wrapped.convert(conv2merge)
            return conv2merge


class EGNetWrapper(nn.Module):
    def __init__(self, backbone, mode):
        super(EGNetWrapper, self).__init__()

        from .EGNet.model import build_model

        assert mode in ('sal', 'feat')

        self.backbone = backbone
        self.mode = mode
        self.wrapped = build_model(backbone)
        # See L74@PoolNet/dataset/dataset.py
        self.register_buffer('image_mean', torch.tensor((122.67892, 104.00699, 116.66877)))

    def forward(self, x):
        # centerization
        x = (x * 255.) - self.image_mean[:, None, None]

        # resize if too small
        h, w = x.shape[-2:]
        new_h, new_w = min(max(h, 112), 960), min(max(w, 112), 960)
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        x = x[:, [2, 1, 0]]

        if self.mode == 'sal':
            _, _, r = self.wrapped(x)
            r = r[-1]
            r = r.sigmoid()
            r = F.interpolate(r, size=(h, w), mode='bilinear', align_corners=True)
            return r
        elif self.mode == 'feat':
            conv2merge, infos = self.wrapped.base(x)
            if self.wrapped.base_model_cfg == 'resnet':
                conv2merge = self.wrapped.convert(conv2merge)
            return conv2merge


class BASNetWrapper(nn.Module):
    def __init__(self):
        super(BASNetWrapper, self).__init__()

        from .BASNet.model import BASNet

        self.wrapped = BASNet(3, 1)
        # See L141@BASNet/data_loader.py
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(self.image_mean[:, None, None]).div_(self.image_std[:, None, None])

        # resize if too small
        h, w = x.shape[-2:]
        new_h, new_w = min(max(h, 112), 960), min(max(w, 112), 960)
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)

        r = self.wrapped(x)
        r = r[0]
        r = r.sub_(r.flatten(1).min(1)[0][:, None, None, None])
        r = r.div_(r.flatten(1).max(1)[0][:, None, None, None])
        r = F.interpolate(r, size=(h, w), mode='bilinear', align_corners=True)
        return r


class U2NetWrapper(nn.Module):
    def __init__(self):
        super(U2NetWrapper, self).__init__()

        from .U2Net.model import U2NET as U2Net

        self.wrapped = U2Net(3, 1)
        # See L141@BASNet/data_loader.py
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(self.image_mean[:, None, None]).div_(self.image_std[:, None, None])

        # resize if too small
        h, w = x.shape[-2:]
        new_h, new_w = 320, 320
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)

        r = self.wrapped(x)
        r = r[0]
        r = r.sub_(r.flatten(1).min(1)[0][:, None, None, None])
        r = r.div_(r.flatten(1).max(1)[0][:, None, None, None])
        r = F.interpolate(r, size=(h, w), mode='bilinear', align_corners=True)
        return r


class GCAGCWrapper(nn.Module):
    def __init__(self, backbone, mode):
        super(GCAGCWrapper, self).__init__()

        if backbone == 'hrnet':
            from .GCAGC.model3.model2_graph4_hrnet_sal import Model2 as build_model
        else:
            raise RuntimeError(f'Unknown backbone {backbone}')

        assert mode in ('sal', 'cosal', 'feat')
        self.backbone = backbone
        self.mode = mode

        self.wrapped = build_model()

        # See L21@GCAGC/test.py
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(self.image_mean[:, None, None]).div_(self.image_std[:, None, None])

        
        if self.mode == 'sal':
            sal = self.wrapped(x)[0]
            sal.sigmoid_()
            return sal
        elif self.mode == 'cosal':
            cosal = self.wrapped.cosalnet(x)[0]
            cosal.unsqueeze_(dim=1)
            cosal.sub_(cosal.min()).div_(cosal.max() + 1e-8)
            return cosal
        elif self.mode == 'feat':
            feat = self.wrapped.cosalnet.prnet(x)[1:4]
            return feat
        else:
            raise RuntimeError()


class GICDWrapper(nn.Module):
    def __init__(self, mode, detach_cls):
        super(GICDWrapper, self).__init__()
        assert mode in ('cosal',)
        self.mode = mode
        self.detach_cls = detach_cls

        from .GICD.models import GICD
        self.wrapped = GICD(detach_cls)

        # See L22@GICD/dataset.py
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(self.image_mean[:, None, None]).div_(self.image_std[:, None, None])

        return self.wrapped(x)


class ClsWrapper(nn.Module):
    def __init__(self, cls_name, stages):
        super(ClsWrapper, self).__init__()
        if isinstance(stages, int):
            stages = [stages]
        self.cls_name = cls_name
        self.stages = stages
        self.max_stage = max(stages)
        self.features = ptcv_get_model(cls_name, pretrained=True).features

        # See https://github.com/osmr/imgclsmob/blob/a5c5bf8d2f3777d16d0898e2cd6572e32de17f2c/pytorch/datasets/imagenet1k_cls_dataset.py#L66
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(self.image_mean[:, None, None]).div_(self.image_std[:, None, None])

        feat = []
        last = x
        for stage, layer in enumerate(self.features):
            last = layer(last)
            if stage in self.stages:
                feat.append(last)
            if stage >= self.max_stage:
                break

        return feat


class HGDWrapper(nn.Module):
    def __init__(self):
        super(HGDWrapper, self).__init__()
        from .HGD.Exps.sample.model import get_model
        _, net = get_model()
        self.wrapped = net.net.denoise


    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(0.5).div_(0.5)

        # resize if too small
        h, w = x.shape[-2:]
        
        # See L9@HDG/Exps/sample/model.py
        new_h, new_w = 299, 299
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        r = self.wrapped(x)
        r = F.interpolate(r, size=(h, w), mode='bilinear', align_corners=True)
        r = r.mul_(0.5).add_(0.5).clamp_(0, 1)
        return r


class ZeroDCEWrapper(nn.Module):
    def __init__(self):
        super(ZeroDCEWrapper, self).__init__()
        from .ZeroDCE.ZeroDCE_code.model import enhance_net_nopool
        self.wrapped = enhance_net_nopool()


    def forward(self, x):
        _, r, _ = self.wrapped(x)
        return r


class GCoNetWrapper(nn.Module):
    def __init__(self):
        super(GCoNetWrapper, self).__init__()

        from .GCoNet.models import GCoNet

        self.wrapped = GCoNet()
        self.wrapped.set_mode('test')

        # See L238@GCoNet/dataset.py
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(self.image_mean[:, None, None]).div_(self.image_std[:, None, None])

        return self.wrapped(x)[-1]
