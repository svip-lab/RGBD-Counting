from ..registry import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module
class RetinaNet(SingleStageDetector):
    # e.g.retinanet_x101_64x4d_fpn_1x.py:
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
