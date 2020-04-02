import torch
from torch.utils.data import DataLoader, Dataset
import typing
from typing import Tuple, List
import os
import math
import cv2
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)
import numpy as np
import random


class Transform(object):
    '''
    rescale the image to 3*299*299
    '''

    def __call__(self, image: np.array):
        '''

        :param image: H*W*3
        :return:      299*299*3
        '''
        assert image.shape[2] == 3
        H = image.shape[0]
        W = image.shape[1]
        scale = max([math.ceil(299 / H), math.ceil(299 / W)])
        New_H, New_W = H * scale, W * scale
        transform = Compose([transforms.Resize(height=New_H, width=New_W),
                             transforms.RandomCrop(height=299, width=299),
                             ToTensorV2()], p=1.0)
        output = transform(image=image)['image']
        return output


class mydataset(Dataset):
    def __init__(self, dataset_dir: str, subdir: str, phase: str, input_shape: Tuple[int, int]):
        '''

        :param dataset_dir: where the images are stored.
                        Contains folders with the classnames
        :param phase: 'train' or 'val'
        :param subdir: 'white' or 'nbi'
        :param train_portion:  The proportion of train images in all images
        :param image_paths: list of image paths
        '''
        super(mydataset, self).__init__()
        self.input_shape = input_shape
        self.phase = phase
        self.data_dir = dataset_dir
        self.subdir = subdir
        if phase == 'train':
            names = os.listdir(os.path.join(dataset_dir, 'Images', subdir))
            sorted(names)
            num_train = int(0.7 * len(names))
            name_list = {'train': names[:num_train], 'val': names[num_train:]}
            self.name_list = name_list[phase]
            np.save(os.path.join(dataset_dir, subdir + '_split.npy'), name_list)
        elif phase == 'val':
            self.name_list = np.load(os.path.join(dataset_dir, subdir + '_split.npy'), allow_pickle=True).item()[phase]
        else:
            raise ValueError('Wrong phase')

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        '''

        :param index:

        :return: images,labels
        read image and crop it to square, check if it's
        too small ! ! !    The standard is 3*299*299
        Labels should be 1
        *********do some augmentation if necessary

        '''
        img_path = os.path.join(self.data_dir, 'Images', self.subdir, self.name_list[index])
        annot_path = os.path.join(self.data_dir, 'Annotations', str.split(self.name_list[index], '.')[0] + '.npy')
        image = cv2.imread(img_path)
        image = self.aug(image)
        annot = np.load(annot_path, allow_pickle=True).item()
        # label = int(annot['class'] == 1)
        label = torch.tensor(int(annot['class'] == 1))
        return image, label, self.name_list[index]

    def __len__(self):
        return len(self.name_list)

    def aug(self, image):
        '''
        center crop, resize, affine, hue, saturation,
        '''
        if self.phase == 'train':
            imgaug = Compose([
                transforms.Resize(height=self.input_shape[0], width=self.input_shape[1]),
                RandomRotate90(),
                Flip(),
                Transpose(),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=.2),
                    MedianBlur(blur_limit=3, p=.1),
                    Blur(blur_limit=3, p=.1),
                ], p=0.2),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
                ToTensorV2(),
            ], p=1)
        else:
            imgaug = Compose([transforms.Resize(height=self.input_shape[0], width=self.input_shape[1]),
                              ToTensorV2()])
        image = imgaug(image=image)['image']
        return image


def mydataloader(dataset_dir: str, subdir: str, phase: str, batch_size: int, input_shape: Tuple[int, int]):
    dataset = mydataset(dataset_dir, subdir=subdir, phase=phase, input_shape=input_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    images, labels, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels, paths
