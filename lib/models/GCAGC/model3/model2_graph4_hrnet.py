import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
#from resnext import ResNeXt101
from torch.autograd import Variable
from default import _C as config
from cls_hrnet import get_cls_net
model5 = get_cls_net(config)
model5.cuda()
###############
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.prnet = model5
        self.gc4_1 = GraphConvolution(384,  192)
        self.gc4_2 = GraphConvolution(192,  192)
        self.gc3_1 = GraphConvolution(192, 96)
        self.gc3_2 = GraphConvolution(96,  96)
        self.gc2_1 = GraphConvolution(96,  48)
        self.gc2_2 = GraphConvolution(48,  48)
        # 
        #self.pro3=nn.Conv2d(192, 192, 1)
        # decoder
        de_in_channels=int(48+96+192)
        de_layers = make_decoder_layers(decoder_archs['d16'], de_in_channels, batch_norm=True)
        self.decoder = DOCSDecoderNet(de_layers)
#        self.decoder=nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.ConvTranspose2d(64, 1, 64, 32, 16))
#        self.initialize_weights()
    def forward(self, img):
        with torch.no_grad():
            fea = self.prnet(img)       #### layer4=(B*N, C, H, W)  
        out4 = unsqz_fea(fea[3])        #### fea5=(B,N,392,H,W)
        out3 = unsqz_fea(fea[2])        #### fea5=(B,N,192,H,W)
        out2 = unsqz_fea(fea[1])
        for a in range(out4.shape[0]):
            ######################## layer4
            feat4 = out4[a].permute(0, 2, 3, 1).reshape(-1, out4.shape[2]) # (N*H*W,C)           
            adj4 = torch.mm(feat4, torch.t(feat4)) # (N*H*W, N*H*W)
            adj4 = row_normalize(adj4)
            gc4 = F.relu(self.gc4_1(feat4, adj4))
            gc4 = F.relu(self.gc4_2(gc4, adj4))    # (N*H*W,192)
            gc4 = gc4.reshape(out4[a].shape[0], out4[a].shape[2], out4[a].shape[3], 192)
            gc4 = gc4.permute(0, 3, 1, 2)       # (N,192,H,W)
            gc4 = F.upsample(gc4, scale_factor=4, mode='bilinear')
            ######################### layer3
            feat3=out3[a].permute(0,2,3,1).reshape(-1, out3.shape[2])
            adj3=torch.mm(feat3, torch.t(feat3))
            adj3=row_normalize(adj3)
            gc3=F.relu(self.gc3_1(feat3, adj3))
            gc3=F.relu(self.gc3_2(gc3, adj3))   # (5*14*14,96)
            gc3 = gc3.reshape(out3[a].shape[0], out3[a].shape[2], out3[a].shape[3], 96)
            gc3 = gc3.permute(0, 3, 1, 2)       # (N,96,H,W) 
            gc3 = F.upsample(gc3, scale_factor=2, mode='bilinear')
            ######################### layer2
            feat2=out2[a].permute(0,2,3,1).reshape(-1, out2.shape[2])
            adj2=torch.mm(feat2, torch.t(feat2))            
            adj2=row_normalize(adj2)
            gc2=F.relu(self.gc2_1(feat2, adj2))
            gc2=F.relu(self.gc2_2(gc2, adj2))
            gc2 = gc2.reshape(out2[a].shape[0], out2[a].shape[2], out2[a].shape[3], 48)
            gc2 = gc2.permute(0, 3, 1, 2)
            #########################
            gc_fuse = torch.cat((gc4, gc3, gc2), dim=1)
            if a == 0:
                gcx_out = gc_fuse
            else:
                gcx_out = torch.cat((gcx_out, gc_fuse), dim=0)
        ######################
        out_final=self.decoder(gcx_out)
        out_final=F.upsample(out_final, size=img.size()[2:], mode='bilinear')
        out_final=out_final.sigmoid().squeeze()
        return out_final, fea[3], fea[2], fea[1]
        
#    def initialize_weights(self):
#        pretrained_dict=torch.load('/home/litengpeng/CODE/semantic-segmentation/CPD-HR2/hrnetv2_w48_imagenet_pretrained.pth')
#        net_dict=self.prnet.state_dict()
#        for key,value in pretrained_dict.items():
#          if key in net_dict.keys():
#             net_dict[key]=value
#        net_dict.update(net_dict)
#        self.prnet.load_state_dict(net_dict)    
def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, dim=1)
    r_inv = 1 / (rowsum + 1e-10)
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx
def unsqz_fea(dim4_data):
    split_data = torch.split(dim4_data, 5, dim=0)
    for i in range(len(split_data)):
        if i == 0:
            dim5_data = split_data[i].unsqueeze(dim=0)
        else:
            dim5_data = torch.cat((dim5_data, split_data[i].unsqueeze(dim=0)), dim=0)
    return dim5_data

def sqz_fea(dim5_data):
    if dim5_data.size(1) == 1:
        return dim5_data.squeeze()
    else:
        b = dim5_data.size(0)
        for i in range(b):
            if i == 0:
                new_dim4_data = dim5_data[i, :, :, :, :]
            else:
                new_dim4_data = torch.cat((new_dim4_data, dim5_data[i, :, :, :, :]), dim=0)
    return new_dim4_data    
################################
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
#################
decoder_archs = {
    'd16': [336, 'd128', 128, 128, 'd64', 64, 64, 'c1']
}

def make_decoder_layers(cfg, in_channels, batch_norm=True):
    layers = []
    for v in cfg:
        if type(v) is str:
            if v[0] == 'd':
                v = int(v[1:])
                convtrans2d = nn.ConvTranspose2d(in_channels, v, kernel_size=4, stride=2, padding=1)
                if batch_norm:
                    layers += [convtrans2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [convtrans2d, nn.ReLU()]
                in_channels = v
            elif v[0] == 'c':
                v = int(v[1:])
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)
class DOCSDecoderNet(nn.Module):
    def __init__(self, features):
        super(DOCSDecoderNet, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)
#####################
# def pca_group(fmap, group_size):
#     fmap_split = torch.split(fmap, group_size, dim=0)
#     for i in range(len(fmap_split)):
#         cur_fmap = fmap_split[i]
#         p4pca = cur_fmap.permute(0, 2, 3, 1).contiguous().view(-1, cur_fmap.size()[1])
#         p4pca_mean = torch.mean(p4pca, dim=0)
#         p4pca = p4pca - p4pca_mean.expand_as(p4pca)
#         U, S, V = torch.svd(torch.t(p4pca))
#         C = torch.mm(p4pca, U[:, 0].view(-1, 1))
#         a4pca = C.view(cur_fmap.size()[0], cur_fmap.size()[2], cur_fmap.size()[3], 1)
#         cur_mask = a4pca.permute(0, 3, 1, 2)
#         # ddd = cur_mask.detach().cpu().numpy()
#         if i == 0:
#             pca_mask = cur_mask
#         else:
#             pca_mask = torch.cat(([pca_mask, cur_mask]), dim=0)
#     # ddd = pca_mask.detach().cpu().numpy()
#     return pca_mask