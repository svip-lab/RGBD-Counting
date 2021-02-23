import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector

affine_par = True

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(self, backbone, neck=None, bbox_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.bbox_head = builder.build_head(bbox_head)

        self.conv11 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv21 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.conv31 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(64)
        self.conv32 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(64)
        self.conv42 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(256)
        self.conv43 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn43 = nn.BatchNorm2d(256)
        self.conv51 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(64)
        self.conv52 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(256)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn53 = nn.BatchNorm2d(256)

        self.init_weights(pretrained=pretrained)
        self.up = nn.Upsample(size=[136, 240], mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward_train(self, img, img_metas, depth, dmap, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        x = self.extract_feat(img)

        densi_fea = self.density_attention(depth, dmap)

        outs = self.bbox_head(x, densi_fea)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def density_attention(self, depth, dmap):
        depth_pixel = depth
        depth1 = depth_pixel[..., ::4, ::4]
        depth2 = depth_pixel[..., ::8, ::8]
        depth3 = depth_pixel[..., ::16, ::16]
        depth4 = depth_pixel[..., ::32, ::32]
        depth5 = depth_pixel[..., ::64, ::64]

        mask1 = (depth1 > 0.45) * (depth1 <= 1)
        mask2 = (depth2 >= 0.38) * (depth2 < 0.9)
        mask3 = (depth3 >= 0.3) * (depth3 < 0.48)
        mask4 = (depth4 >= 0.15) * (depth4 < 0.35)
        mask5 = (depth5 >= 0.01) * (depth5 < 0.3)
        mask1 = mask1.cuda().float()
        mask2 = mask2.cuda().float()
        mask3 = mask3.cuda().float()
        mask4 = mask4.cuda().float()
        mask5 = mask5.cuda().float()

        pre_dmap = dmap
        pre_dmap = self.up(pre_dmap)
        pre_dmap1 = pre_dmap * mask1
        pre_dmap2 = pre_dmap[..., ::2, ::2] * mask2
        pre_dmap3 = pre_dmap[..., ::4, ::4] * mask3
        pre_dmap4 = pre_dmap[..., ::8, ::8] * mask4
        pre_dmap5 = pre_dmap[..., ::16, ::16] * mask5

        # stage 1
        pre_dmap1 = self.conv11(pre_dmap1)
        pre_dmap1 = self.bn11(pre_dmap1)
        pre_dmap1 = self.relu(pre_dmap1)
        pre_dmap1 = self.conv12(pre_dmap1)
        pre_dmap1 = self.bn12(pre_dmap1)
        pre_dmap1 = self.relu(pre_dmap1)
        pre_dmap1 = self.conv13(pre_dmap1)
        pre_dmap1 = self.bn13(pre_dmap1)
        pre_dmap1 = self.sigmoid(pre_dmap1)
        # stage 2
        pre_dmap2 = self.conv21(pre_dmap2)
        pre_dmap2 = self.bn21(pre_dmap2)
        pre_dmap2 = self.relu(pre_dmap2)
        pre_dmap2 = self.conv22(pre_dmap2)
        pre_dmap2 = self.bn22(pre_dmap2)
        pre_dmap2 = self.relu(pre_dmap2)
        pre_dmap2 = self.conv23(pre_dmap2)
        pre_dmap2 = self.bn23(pre_dmap2)
        pre_dmap2 = self.sigmoid(pre_dmap2)
        # stage 3
        pre_dmap3 = self.conv31(pre_dmap3)
        pre_dmap3 = self.bn31(pre_dmap3)
        pre_dmap3 = self.relu(pre_dmap3)
        pre_dmap3 = self.conv32(pre_dmap3)
        pre_dmap3 = self.bn32(pre_dmap3)
        pre_dmap3 = self.relu(pre_dmap3)
        pre_dmap3 = self.conv33(pre_dmap3)
        pre_dmap3 = self.bn33(pre_dmap3)
        pre_dmap3 = self.sigmoid(pre_dmap3)
        # stage 4
        pre_dmap4 = self.conv41(pre_dmap4)
        pre_dmap4 = self.bn41(pre_dmap4)
        pre_dmap4 = self.relu(pre_dmap4)
        pre_dmap4 = self.conv42(pre_dmap4)
        pre_dmap4 = self.bn42(pre_dmap4)
        pre_dmap4 = self.relu(pre_dmap4)
        pre_dmap4 = self.conv43(pre_dmap4)
        pre_dmap4 = self.bn43(pre_dmap4)
        pre_dmap4 = self.sigmoid(pre_dmap4)
        # stage 5
        pre_dmap5 = self.conv51(pre_dmap5)
        pre_dmap5 = self.bn51(pre_dmap5)
        pre_dmap5 = self.relu(pre_dmap5)
        pre_dmap5 = self.conv52(pre_dmap5)
        pre_dmap5 = self.bn52(pre_dmap5)
        pre_dmap5 = self.relu(pre_dmap5)
        pre_dmap5 = self.conv53(pre_dmap5)
        pre_dmap5 = self.bn53(pre_dmap5)
        pre_dmap5 = self.sigmoid(pre_dmap5)
        # iterable
        densi_fea = []
        densi_fea.append(pre_dmap1)
        densi_fea.append(pre_dmap2)
        densi_fea.append(pre_dmap3)
        densi_fea.append(pre_dmap4)
        densi_fea.append(pre_dmap5)
        return densi_fea

    def simple_test(self, img, img_meta, anno, depth, dmap, rescale=False):
        x = self.extract_feat(img)

        densi_fea = self.density_attention(depth, dmap)

        outs = self.bbox_head(x, densi_fea)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
