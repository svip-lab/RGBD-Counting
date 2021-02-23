import os.path as osp
import warnings

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .extra_aug import ExtraAugmentation
from .registry import DATASETS
from .transforms import (BboxTransform, ImageTransform, MaskTransform,
                         Numpy2Tensor, SegMapTransform)
from .utils import random_scale, to_tensor


@DATASETS.register_module
class CustomDataset(Dataset):
    """
    [Custom] dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,  # 32
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 skip_img_without_anno=True,
                 test_mode=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        '''self.img_infos'''
        # here we get self.img_infos - 存储了全部图像的基本信息
        self.img_infos = self.load_annotations(ann_file)
        # print(type(self.img_infos))  # list of 118287(all)
        # print(self.img_infos[0]) e.g.
        # {'license': 3, 'file_name': '000000391895.jpg', 'coco_url': 'http://images.cocodataset.org/train2017/000000391895.jpg',
        # 'height': 360, 'width': 640, 'date_captured': '2013-11-14 11:18:45', 'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
        # 'id': 391895, 'filename': '000000391895.jpg'}

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale, list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transform
        # resize + normalize + flip + pad + transpose
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        # rescale + flip + pad the first dimension to 'max_num_gts'
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio
        self.skip_img_without_anno = skip_img_without_anno

    '''__len__'''
    def __len__(self):
        '''self.img_infos'''
        return len(self.img_infos)

    '''__getitem__'''
    def __getitem__(self, idx):
        # test data
        if self.test_mode:
            return self.prepare_test_img(idx)
        # train data
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            # print(data)
            return data

    '''train_data'''
    def prepare_train_img(self, idx):
        '''
        load data
        '''
        img_info = self.img_infos[idx]
        # load image'
        '''img'''
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # print(img)  # 这里的img数据是[0-255]原大小
        # print(img.shape)  # [x, x, 3]不同大小的图像
        # load proposals if necessary
        '''proposals'''
        if self.proposals is not None:
            # ×
            proposals = self.proposals[idx][:self.num_max_proposals]
            # print(proposals.shape)
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        #print(gt_bboxes)  # [n, 4] - 每个object对应的坐标
        gt_labels = ann['labels']
        ######## print(gt_labels)  # [n,] - 每个object对应的label
        if self.with_crowd:
            # ×
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        '''
        transform
        '''
        # extra augmentation
        if self.extra_aug is not None:
            # ×
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False  # flip_ratio=0.5
        # randomly sample a scale(self.multiscale_mode='value') - multiscale_mode
        img_scale = random_scale(self.img_scales, self.multiscale_mode)  # (1333, 800)
        '''img_transform'''
        # img_shape:(800,1293,3)/(1067,800,3)/..., # transform后的shape
        # pad_shape:(1088,800,3)/(1216,800,3)/..., # pad后的shape
        # scale_factor(1.666)/1.877/2.011...       # shape改变的scale_factor
        # 这里进行了归一化
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        ##################
        if self.with_seg:
            # ×
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix,img_info['file_name'].replace('jpg', 'png')),flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            # ×
            proposals = self.bbox_transform(proposals, img_shape, scale_factor, flip)
            proposals = np.hstack([proposals, scores]) if scores is not None else proposals
        ##################
        '''bbox_transform'''
        # print(gt_bboxes)  # 变换前的相应bboxes
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor, flip)
        # print(gt_bboxes)  # 一系列4坐标的bboxes
        ##################
        if self.with_crowd:
            # ×
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            # ×
            gt_masks = self.mask_transform(ann['masks'], pad_shape, scale_factor, flip)
        ##################

        '''
        concat data
        '''
        ori_shape = (img_info['height'], img_info['width'], 3)  # 最初的img shape
        '''img_meta'''
        img_meta = dict(  # 元img数据
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        # print(img)  # 这里的img已经通过norm被压缩到了(-1,1)
        '''data'''
        data = dict(  # all data
            img=DC(to_tensor(img), stack=True),  # img
            img_meta=DC(img_meta, cpu_only=True),  # img_meta
            gt_bboxes=DC(to_tensor(gt_bboxes)))  # gt_bboxes
        if self.proposals is not None:
            # ×
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))  # gt_labels
        if self.with_crowd:
            # ×
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            # ×
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            # ×
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)

        return data

    '''test data'''
    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        '''load'''
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            # ×
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        '''transform'''
        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                # ×
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack([_proposal, score
                                       ]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:  # 单尺度here
            # prepare_single
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)  # None
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

    '''__init__'''
    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

