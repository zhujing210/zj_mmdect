from pickle import FALSE
import torch.nn as nn
import torch.nn.functional as F

import logging
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init, normal_init
from ..builder import BACKBONES


###hourglass+mobilenet  之前的 9py
# class convbnrelu(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=1, stride=1, pad=0, groups=1,
#                                         use_bias=False, use_bn=True, use_relu=True):
#         super(convbnrelu, self).__init__()
#         self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
#                      padding=pad, groups=groups, bias=use_bias)
#         self.bn = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.out_channels = planes
#         self.use_bn = use_bn
#         self.use_relu = use_relu
#     def forward(self,x):
#         out = self.conv(x)
#         if self.use_bn:
#             out = self.bn(out)
#         if self.use_relu:
#             out = self.relu(out)
#         return out

# ####hourglass+mobilenet   与 pretrain 保持一致
class convbnrelu(nn.Module):
    def __init__(self,inplanes,planes,kernel_size=1,stride=1,pad=0,groups=1,use_bn=True,use_relu=True, use_bias=True):
        super(convbnrelu,self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=pad,groups=groups, bias=use_bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = planes
        self.use_bn = use_bn
        self.use_relu = use_relu
    def forward(self,x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_relu:
            out = self.relu(out)
        return out


class bottleneck_v2(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, hg=2):
        super(bottleneck_v2, self).__init__()

        self.reduce = convbnrelu(inplanes, inplanes//hg, kernel_size=1, stride=1, pad=0)
        self.conv2 = convbnrelu(inplanes//hg, planes, kernel_size=3, stride=1, pad=1, use_relu=False)
        #self.pointwise = convbnrelu(planes*2, planes, kernel_size=1, stride=1, pad=0, use_bn=False)

        # one_split = torch.ones((planes, 1, 1)).float()
        # idx = torch.arange(0, planes).long()
        # filters = torch.zeros((planes, planes, 1, 1)).float()
        # filters[idx, idx] = one_split
        # self.filters = filters.repeat(1,2,1,1)
        
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out =  self.conv2(self.reduce(x))
        out += residual

        # out = torch.cat([residual, out], 1)
        # out = F.conv2d(out, Variable(self.filters, requires_grad=False).cuda(), padding=0)
        #TODO need bn???  order .w.r.t relu
        out = self.bn(out)
        out = self.relu(out)

        return out 


@BACKBONES.register_module()
class DgMobilenet(nn.Module):
    """
    DgMobilenet backbone for single-shot-detection.

     Args:
        out_feature_indices (Sequence[int]): Output from which layer.
    """
    def __init__(self, out_feature_indices=(7, 10, 13, 16)):
        super().__init__()
        self.out_feature_indices = out_feature_indices
        self.features = self._make_layers(3)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def _make_layers(self, in_planes): 
        layers = []
        scale = 1
        block = bottleneck_v2 #bottleneck_v2 #inverted_residual_bottleneck #

        layers+=[convbnrelu(in_planes,int(scale*16),kernel_size=3,stride=2,pad=1)]
        layers+=[block(int(scale*16),int(scale*16),hg=2)]
        #layers+=[block(int(scale*16),int(scale*16),hg=2)]

        layers+=[convbnrelu(int(scale*16),int(scale*32),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*24),int(scale*24))]
        layers+=[block(int(scale*32),int(scale*32))]
        layers+=[block(int(scale*32),int(scale*32))] #4, 1/4,

        layers+=[convbnrelu(int(scale*32),int(scale*64),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*32),int(scale*32))]
        layers+=[block(int(scale*64),int(scale*64))]
        layers+=[block(int(scale*64),int(scale*64))] #7, 1/8

        layers+=[convbnrelu(int(scale*64),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*40),int(scale*40))]
        layers+=[block(int(scale*128),int(scale*128))]
        layers+=[block(int(scale*128),int(scale*128))] #10, 1/16

        layers+=[convbnrelu(int(scale*128),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*48),int(scale*48))]
        layers+=[block(int(scale*128),int(scale*128))]
        layers+=[block(int(scale*128),int(scale*128))] #13, 1/32

        layers+=[convbnrelu(int(scale*128),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*256),int(scale*256))]
        layers+=[block(int(scale*128),int(scale*128))]
        layers+=[block(int(scale*128),int(scale*128))] #16, 1/64

        # layers+=[convbnrelu(int(scale*256),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*128),int(scale*128))]
        # layers+=[block(int(scale*128),int(scale*128))] #19, 1/128

        # layers+=[convbnrelu(int(scale*128),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*128),int(scale*128))]
        # layers+=[block(int(scale*128),int(scale*128))] #22, 1/256

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""

        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class DgMobilenetFCOS(DgMobilenet):
    def _make_layers(self, in_planes):
        layers = []
        scale = 1
        block = bottleneck_v2 #bottleneck_v2 #inverted_residual_bottleneck #

        layers+=[convbnrelu(in_planes,int(scale*16),kernel_size=3,stride=2,pad=1)]
        layers+=[block(int(scale*16),int(scale*16),hg=2)]

        layers+=[convbnrelu(int(scale*16),int(scale*32),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*24),int(scale*24))]
        layers+=[block(int(scale*32),int(scale*32))]
        layers+=[block(int(scale*32),int(scale*32))] #4, 1/4,

        layers+=[convbnrelu(int(scale*32),int(scale*64),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*32),int(scale*32))]
        layers+=[block(int(scale*64),int(scale*64))]
        layers+=[block(int(scale*64),int(scale*64))] #7, 1/8

        layers+=[convbnrelu(int(scale*64),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*40),int(scale*40))]
        layers+=[block(int(scale*128),int(scale*128))]
        layers+=[block(int(scale*128),int(scale*128))] #10, 1/16

        layers+=[convbnrelu(int(scale*128),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[block(int(scale*48),int(scale*48))]
        layers+=[block(int(scale*128),int(scale*128))]
        layers+=[block(int(scale*128),int(scale*128))] #13, 1/32

        return nn.Sequential(*layers)


@BACKBONES.register_module()
class DgMobilenet_80Channels(DgMobilenet):
    def _make_layers(self, i): 
        layers = []
        scale = 1
        layers+=[convbnrelu(i,int(scale*24),kernel_size=3,stride=2,pad=1,use_bias=False)] # 0, 1/2, 2, 3, for test pretrain
        layers+=[bottleneck_v2(int(scale*24),int(scale*24))]  # 1, 1/2, 2, 7
        layers+=[bottleneck_v2(int(scale*24),int(scale*24))]  # 2, 1/2, 2, 11

        layers+=[convbnrelu(int(scale*24),int(scale*32),kernel_size=3,stride=2,pad=1,use_bias=False)] # 3, 1/4, 4, 15, for test pretrain
        layers+=[bottleneck_v2(int(scale*32),int(scale*32))]  # 4, 1/4, 4, 23 
        layers+=[bottleneck_v2(int(scale*32),int(scale*32))] #5, setp: 1/4, j:4, rf : 31

        layers+=[convbnrelu(int(scale*32),int(scale*64),kernel_size=3,stride=2,pad=1,use_bias=False)]  # 6, 1/8, 8, 39, for test pretrain
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))]  # 7, 1/8, 8, 55
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))]  # 8, 1/8, 8, 71
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))]  # 9, 1/8, 8, 87
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))] #10, 1/8, 8, 103

        layers+=[convbnrelu(int(scale*64),int(scale*80),kernel_size=3,stride=2,pad=1,use_bias=False)]  # 11, 1/16, 16, 119, for test pretrain
        layers+=[bottleneck_v2(int(scale*80),int(scale*80))] # 12, 1/16, 16, 151
        layers+=[bottleneck_v2(int(scale*80),int(scale*80))] # 13, 1/16, 16, 183
        # layers+=[bottleneck_v2(int(scale*80),int(scale*80))] # 14, 1/16, 16, 215
        # layers+=[bottleneck_v2(int(scale*80),int(scale*80))] #15 ,1/16 , 16, 247
        
        return nn.Sequential(*layers)

@BACKBONES.register_module()
class DgMobilenet_128Channels(DgMobilenet):
    def _make_layers(self, i): 
        layers = []
        scale = 1
        layers+=[convbnrelu(i,int(scale*24),kernel_size=3,stride=2,pad=1)] # 0, 1/2, 2, 3
        layers+=[bottleneck_v2(int(scale*24),int(scale*24))]  # 1, 1/2, 2, 7
        layers+=[bottleneck_v2(int(scale*24),int(scale*24))]  # 2, 1/2, 2, 11

        layers+=[convbnrelu(int(scale*24),int(scale*32),kernel_size=3,stride=2,pad=1)] # 3, 1/4, 4, 15
        layers+=[bottleneck_v2(int(scale*32),int(scale*32))]  # 4, 1/4, 4, 23 
        layers+=[bottleneck_v2(int(scale*32),int(scale*32))] #5, setp: 1/4, j:4, rf : 31

        layers+=[convbnrelu(int(scale*32),int(scale*64),kernel_size=3,stride=2,pad=1)]  # 6, 1/8, 8, 39
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))]  # 7, 1/8, 8, 55
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))]  # 8, 1/8, 8, 71
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))]  # 9, 1/8, 8, 87
        layers+=[bottleneck_v2(int(scale*64),int(scale*64))] #10, 1/8, 8, 103

        layers+=[convbnrelu(int(scale*64),int(scale*128),kernel_size=3,stride=2,pad=1)]  # 11, 1/16, 16, 119
        layers+=[bottleneck_v2(int(scale*128),int(scale*128))] # 12, 1/16, 16, 151
        layers+=[bottleneck_v2(int(scale*128),int(scale*128))] # 13, 1/16, 16, 183
        layers+=[bottleneck_v2(int(scale*128),int(scale*128))] # 14, 1/16, 16, 215
        layers+=[bottleneck_v2(int(scale*128),int(scale*128))] #15 ,1/16 , 16, 247

        # layers+=[convbnrelu(int(scale*128),int(scale*128),kernel_size=3,stride=2,pad=1)]  # 16, 1/32, 32, 279
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))] # 17, 1/32, 32, 343
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))]   # 18, 1/32, 32, 407
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))]   # 19, 1/32, 32, 471
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))]    #20, 1/32, 32, 535

        # layers+=[convbnrelu(int(scale*128),int(scale*128),kernel_size=3,stride=2,pad=1)]
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))]
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))] 
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))]
        # layers+=[bottleneck_v2(int(scale*128),int(scale*128))] #25, 1/64, 64, 127
        
        return nn.Sequential(*layers)