from __future__ import print_function, division
import os.path as osp
import math
import warnings
import sys
import os
import csv
import random
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from .extra_aug import ExtraAugmentation
from .registry import DATASETS
from torch.utils.data.sampler import Sampler
from torchvision.transforms import functional as F
from .transforms import (BboxTransform, ImageTransform, MaskTransform,
                         Numpy2Tensor, SegMapTransform)
from .utils import random_scale, to_tensor
import scipy.io as sio
import cv2


@DATASETS.register_module
class RGBD(Dataset):
    CLASSES = ('head',)
    def __init__(self, ann_file, img_prefix, class_file, mode, img_scale, flip_ratio, img_norm_cfg,
                 skip_img_without_anno=True, size_divisor=None, multiscale_mode='value', resize_keep_ratio=True,
                 debug=False, **kwargs):
        print('init dataset...')
        self.train_file = ann_file
        self.img_prefix = img_prefix
        self.class_list = class_file
        self.mode = mode
        self.img_scales = img_scale if isinstance(img_scale, list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        self.size_divisor = size_divisor  # size_divisor (used for FPN)
        self.img_norm_cfg = img_norm_cfg
        self.depth_norm_cfg = dict(
            mean=[0], std=[1], to_rgb=False, rgb_format=False)
        self.density_norm_cfg = dict(
            mean=[0], std=[1], to_rgb=False, rgb_format=False)

        self.resize_keep_ratio = resize_keep_ratio
        self.skip_img_without_anno = skip_img_without_anno
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.depth_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.depth_norm_cfg)
        self.density_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.density_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()
        self.proposals = None

        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())
        self.img_ids = list(self.image_data.keys())

        if self.mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode:
            while True:
                data = self.prepare_train_img(idx)
                if data is None:
                    idx = self.rand_another()
                    continue
                return data
        else:
            return self.prepare_test_img(idx)

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def prepare_train_img(self, idx):
        img = mmcv.imread(self.image_names[idx])

        # load annotation
        # annot is a numpy array (N, 5).
        # N means there are N objects in the image, 5 means {x1,y1,x2,y2,class_id}
        annot = self.load_annotations(idx)
        gt_bboxes = annot[:, 0:4]
        gt_labels = annot[:, 4]

        depth = self.load_depth(idx)

        loc = self.load_point(idx)
        dmap = create_dmap(img, loc, depth, 0.6, downscale=4.0)

        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('skp the image %s that has no valid gt bbox' % self.image_names[idx])
            return None

        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        depth, _img_shape, _pad_shape, _scale_factor = self.depth_transform(
            depth, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        dmap, _dmap_shape, _d_pad_shape, _d_scale_factor = self.density_transform(
            dmap, dmap.shape, flip, keep_ratio=self.resize_keep_ratio)
        depth = to_tensor(depth)
        depth = depth.unsqueeze(0)
        dmap = to_tensor(dmap)
        dmap = dmap.unsqueeze(0)
        img = to_tensor(img)
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor, flip)
        ori_shape = (img.shape[1], img.shape[2])
        img_meta = dict(
            idx=idx,
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip
        )
        data = dict(
            img=DC(img, stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            depth=DC(depth, stack=True),
            dmap=DC(dmap, stack=True)
        )
        data['gt_labels'] = DC(to_tensor(gt_labels))
        return data

    '''test data'''
    def prepare_test_img(self, idx):
        img = mmcv.imread(self.image_names[idx])
        img_name = self.image_names[idx]
        name_index = img_name.rfind('_')
        img_name = img_name[name_index + 1:]

        annot = self.load_annotations(idx)

        depth = self.load_depth(idx)

        use_gt_dmap = True
        if use_gt_dmap:
            loc = self.load_point(idx)
            dmap = create_dmap(img, loc, depth, 0.6, downscale=4.0)
        else:
            dmap_path = "/root/vis/1.0.init/84_36/"
            dmap_path = dmap_path + img_name
            dmap = cv2.imread(dmap_path, cv2.IMREAD_GRAYSCALE)
            dmap = dmap.astype('float64')
            dmap = dmap / 255

        def prepare_single(img, scale, flip, proposal=None):
            ori_shape = (img.shape[0], img.shape[1], img.shape[2])
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                idx=idx,
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
        depth, _depth_shape, _depth_pad_shape, _scale_factor = self.depth_transform(
            depth, scale, False, keep_ratio=self.resize_keep_ratio)
        dmap, _dmap_shape, _densi_pad_shape, _densi_scale_factor = self.density_transform(
            dmap, dmap.shape, False, keep_ratio=self.resize_keep_ratio)

        depth = to_tensor(depth)
        depth = depth.unsqueeze(0)
        depth = DC(to_tensor(depth), stack=True)
        dmap = to_tensor(dmap)
        dmap = dmap.unsqueeze(0)
        dmap = DC(dmap, stack=True)
        data = dict(img=imgs, img_meta=img_metas, anno=annot, depth=depth, dmap=dmap)
        return data

    def load_depth(self, image_index):
        img_name = self.image_names[image_index]
        mat_name = img_name.replace('img', 'depth').replace('IMG', 'GT').replace('.png', '.mat')
        depth = sio.loadmat(mat_name)
        depth = depth['depth']
        depth[depth > 20000] = 0
        depth[depth < 0] = 0
        depth = depth / 20000
        return depth

    def load_point(self, image_index):
        img_name = self.image_names[image_index]
        mat_name = img_name.replace('img', 'gt').replace('IMG', 'DEPTH').replace('.png', '.mat')
        loc = sio.loadmat(mat_name)
        loc = loc['image_info'][0][0][0][0][0].astype(np.float32)
        return loc

    def load_annotations(self, image_index):
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        if len(annotation_list) == 0:
            return annotations

        for idx, a in enumerate(annotation_list):
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2
            annotation[0, 4] = 1
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def load_classes(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1
            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))  # 0

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def _rand_another(self):
        pool = len(self.image_name)-1
        return np.random.choice(pool)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img = mmcv.imread(self.image_names[i])
            if img.shape[1] / img.shape[0] > 1:
                self.flag[i] = 1

def create_dmap(img, gtLocation, depth, sigma, downscale=4.0):
    height, width, cns = img.shape
    raw_width, raw_height = width, height
    width = math.floor(width / downscale)
    height = math.floor(height / downscale)
    raw_loc = gtLocation
    gtLocation = gtLocation / downscale
    gaussRange = 25
    pad = int((gaussRange - 1) / 2)
    densityMap = np.zeros((int(height + gaussRange - 1), int(width + gaussRange - 1)))
    for gtidx in range(gtLocation.shape[0]):
        if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
            xloc = int(math.floor(gtLocation[gtidx, 0]) + pad)
            yloc = int(math.floor(gtLocation[gtidx, 1]) + pad)
            x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
            x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)
            y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
            y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
            depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
            if depth_mean != 0:
                kernel = GaussianKernel((25, 25), sigma=sigma / depth_mean)
            else:
                kernel = GaussianKernel((25, 25), sigma=sigma)
            densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
    densityMap = densityMap[pad:pad + height, pad:pad + width]
    return densityMap

def GaussianKernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian kernel which is equal to MATLAB's fspecial('gaussian',[shape],[sigma])
    """
    radius_x, radius_y = [(radius-1.)/2. for radius in shape]
    y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
    h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))  # (25,25),max()=1~h[12][12]
    h[h < (np.finfo(h.dtype).eps*h.max())] = 0
    return h
