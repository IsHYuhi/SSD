import torch.nn as nn
import torch
from itertools import product
from math import sqrt
import pandas as pd
import torch.nn.init as init

def make_vgg():
    layers = []
    in_channels = 3

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            #default: floor_mode
            #change ceil_mode
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(cfg[-1], 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def make_extras():
    layers = []
    in_channels = 1024

    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)

def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

    #conv4_3（source1)
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]

    # last layer (source2)
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]

    # extra (source3)
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]

    # extra (source4)
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    # extra (source5)
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    # extra (source6)
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()#inplement super class __init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale #initial value of weight
        self.reset_parameters() #initialize parameters
        self.eps = 1e-10

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)# all weights => 20

    def forward(self, x):
        #norm: torch.Size([batch_num, 1, 38, 38])
        #x: torch.Size([batch_num, 512, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x/norm
        x = torch.div(x, norm)

        #weight: torch.Size([512]) => torch.Size([batch_num, 512, 38, 38]) #expand_as(x)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        self.image_size = cfg['input_size'] #300

        self.feature_maps = cfg['feature_maps']#'feature_maps': [38, 19, 10, 5, 3, 1]
        self.num_priors = len(cfg["feature_maps"])#source 6
        self.steps = cfg['steps'] #DBox pixel size, 'steps':[8, 16, 32, 64, 100, 300]
        self.min_sizes = cfg['min_sizes']# smaller Dbox size [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes']# larger Dbox size [45, 99, 153, 207, 261, 315]

        self.aspect_ratios = cfg['aspect_ratios'] # aspect ratios

    def make_dbox_list(self):
        '''generate DBox'''
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2): # fP2

                # 300/ 'steps':[8, 16, 32, 64, 100, 300]
                f_k = self.image_size / self.steps[k]

                #center axis (cx, cy), where 0~1
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 'max_sizes': [45, 99, 153, 207, 261, 315]
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # defBox[cx, cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # Dbox => torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        #DBox -> 0~1
        output.clamp_(max=1, min=0)

        return output

class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase =phase # train or inference
        self.num_classes = cfg["num_classes"] #21

        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        if phase == 'inference':
            self.detect = Detect()



vgg_test = make_vgg()
print(vgg_test)

extras_test = make_extras()
print(extras_test)

ssd_cfg = {
    'num_classes': 21,
    'input_size': 300,
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
    'feature_maps': [38, 19, 10, 5, 3, 1], # output size from each source
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

dbox = DBox(ssd_cfg)
dbox_list = dbox.make_dbox_list()

print(pd.DataFrame(dbox_list.numpy()))

ssd_test = SSD(phase="train", cfg=ssd_cfg)
print(ssd_test)