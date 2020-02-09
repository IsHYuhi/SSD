import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
from utils.data_augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

def make_datapath_list(rootpath):
    """
    make datapath list

    Params
    -----
    rortpath: str

    Returns
    -------
    ret: train_img_list, train_anno_list, val_img_list, val_anno_list
    """

    imgpath_template = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = os.path.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = os.path.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = os.path.join(rootpath + 'ImageSets/Main/val.txt')

    train_img_list = []
    train_anno_list = []

    for line in open(train_id_names):
        file_id = line.strip() #remove space and indention
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = []
    val_anno_list = []

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


class Anno_xml2list(object):
    """
    xml to list after standardization

    Attributes
    ----------
    classes: list
        store name of class
    """

    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
        xml to list after standardization

        Params
        ------
        xml_path: str

        width: int

        height: int

        Returns
        -------
        ret: [[xmin, ymin, xmax, ymax, label_ind], ...]
        """

        ret = []

        xml = ET.parse(xml_path).getroot()

        #loop with the number of object in image
        for obj in xml.iter('object'):

            #Exclude detection set to difficult
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            bndbox = []

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                #Reset origin(1,1) to (0,0)
                cur_pixel = int(bbox.find(pt).text) - 1

                # standlization with width and height
                if pt == 'xmin' or pt =='xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            label_idx = self.classes.index(name)
            #[xmin, ymin, xmax, ymax, label_ind]
            bndbox.append(label_idx)

            ret += [bndbox]

        return np.array(ret)

class DataTransform():
    """
    pre-process
    resize 300x300
    data augment when training

    Attributes
    ----------
    input_size: int

    color mean: (B, G, R)
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        """
        Params
        ------
        phase: 'train' or 'val'
        """
        return self.data_transform[phase](img, boxes, labels)

class VOCDataset(data.Dataset):
    """
    Inherit PyTorch's Dataset class

    Attributes
    ----------
    img_list: list

    anno_list: list

    phase: 'train' or 'test'

    transform: object

    transform_anno: object
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self,index):

        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape

        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4])

        #[H][W][C] => [C][H][W]
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        #gt means ground truth
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))#boxes[n][4],label[n] -> boxes[n][4] + label[n][1] = gt[n][5]

        return img, gt, height, width
