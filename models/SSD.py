import torch.nn as nn
import torch
from itertools import product
from math import sqrt
import pandas as pd
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function

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


def decode(loc, dbox_list):
    """
    DBox2BBox
    [cx, cy, width, height] to [xmin, ymin, xmax, ymax]
    Parameters
    ----------
    loc: [8732, 4]
        [Δcx, Δcy, Δwidth, Δheight]

    dbox_list: [8732, 4]
        [cx, cy, width, height]

    Returns
    -------
    boxes: [xmin, ymin, xmax, ymax]
    """

    #boxes: torch.Size([8732, 4])
    boxes = torch.cat((
        dbox_list[:, :2] + loc [:, :2] * 0.1 * dbox_list[:, 2:], #to (cx + Δcx*0.1*width, cy + Δcy*0.1*height): [8732, 2]
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)) #to (width + Δcx*0.1*cx, width + Δcy*0.1*cy): [8732, 2]
        , dim=1)# to adjusted (cx, cy, width, height): [8732, 4]

    #[cx, cy, width, height] to [xmin, ymin, xmax, ymax]
    boxes[:, :2] -= boxes[:, 2:] / 2 #(cx, cy) to (xmin, ymin)
    boxes[:, 2:] += boxes[:, :2] #(width, height) to (xmax, ymax)

    return boxes

def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppression

    Parameters
    ----------
    boxes: [the number of object over threshold, 4]

    scores: [the number of object over threshold]

    Returns
    -------
    keep: list
        index of conf in the descending order
    count: int
        the number of object over threshold
    """

    # placeholder for return
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()# all elements are zero

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2-x1, y2-y1)

    #copy for IoU
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # sort scores in ascending order
    v, idx = scores.sort(0)

    #extract top_k index of BBox
    idx = idx[-top_k:]

    # loop while elements of idx is not 0
    while idx.numel() > 0:
        i = idx[-1] # index of maximum conf to i

        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        # substract 1 because last index is stored in keep
        idx = idx[:-1]

        #eliminate BBox covered much with the BBox in the keep
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_selsct(y1, 0, idx, out=tmp_y1)
        torch.index_selsct(x2, 0, idx, out=tmp_x2)
        torch.index_selsct(y2, 0, idx, out=tmp_y2)

        #clamp current BBox=index conflict with i for all BBox
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_x1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x1, max=x2[i])
        tmp_y2 = torch.clamp(tmp_x1, max=y2[i])

        # w, h
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        #area
        inter = tmp_w*tmp_h

        #intersect in IoU
        rem_areas = torch.index_select(area, 0, idx)# each BBox original area
        union = (rem_areas - inter) + area[i]# and
        IoU = inter/union

        # leave idx IoU less than or equal to overlap
        idx = idx[IoU.le(overlap)]# le = Less than or Equal to

    return keep, count

class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)# normalize conf with softmax

        self.conf_thresh = conf_thresh

        self.top_k = top_k

        self.nms_thresh = nms_thresh # consider BBox for the same object where IoU >= thresh in nm_supression

    def forward(self, loc_data, conf_data, dbox_list):
        """
        forward

        Parameters
        ----------
        loc_data: [batch_num, 8732, 4]
            offset information
        conf_data: [batch_num, 8732, num_classes]
            conf
        dbox_list:[8732, 4]
            DBox

        Returns
        -------
        output: torch.Size([batch_num, num_classes, top_k, 5])
            [batch_num, classes, top_k, BBox], BBox[xmin, ymin, xmax, ymax, class]
        """

        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_data = self.softmax(conf_data)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        #conf_data: [batch_num, 8732, num_classes] => [batch_num, num_classes, 8732]
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):

            decoded_boxes = decode(loc_data[i], dbox_list)

            #copy
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):# from 1 because of backgroud:0

                c_mask = conf_scores[cl].gt(self.conf_thresh)# gt means greater than

                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0: # sum of element
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)

                boxes = decoded_boxes[l_mask].view(-1, 4)#decoded_boxes[lmask]:[n*4]=>[n,4]

                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output

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

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        #source3~6
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):

            loc.append(l(x).permute(0, 2, 3, 1).contiguous())# it shows variables located in memory when using view(), so contiguous sorted on memory.
            conf.append(c(x).permute(0,2, 3, 1).contiguous())

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":
            return self.detect(output[0], output[1], output[2])

        else:
            return output



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