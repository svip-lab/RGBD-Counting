from torch import nn

from mmdet.utils import build_from_cfg

# 此处不会在执行registry而是直接进行sys.modules查询得到
from .registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                       ROI_EXTRACTORS, SHARED_HEADS)


# model = build_detector(
#     cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
def build_detector(cfg, train_cfg=None, test_cfg=None):
    # print(DETECTORS)  # 1次
    # Registry(name=detector, items=['CascadeRCNN', 'TwoStageDetector', 'DoubleHeadRCNN',
    # 'FastRCNN', 'FasterRCNN', 'SingleStageDetector', 'FCOS', 'GridRCNN',
    # 'HybridTaskCascade', 'MaskRCNN', 'MaskScoringRCNN', 'RetinaNet', 'RPN'])
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
    # 这里返回的已经是搭建好的model了 -
    # 传入的cfg确实是model的所有信息，但registry只传入了DETECTORS为什么搭建出了整个模型？
    # 也即build的registry里什么时候传入的backbone、neck等信息？


'''上述7个大类的build函数均调用该build函数，通过registry标志进行选择(BACKBONES等)
其中DETECTORS特有训练和检测的配置信息，从原始cfg中分离出来
这里传入的cfg实际上是cfg.model，也就是参数配置cfg中的模型配置'''
# return build(cfg.model, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
def build(cfg, registry, default_args=None):
    # 多次进入build，每次cfg给一部分进行搭建(即cfg和registry的print是交替进行的，而非cfg一次输出完)
    # print(cfg)  # cfg.model -> cfg.model.backbone(ResNeXt) -> .neck(FPN) -> .ReinaHead(head) -> ...
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        # print('1')
        return nn.Sequential(*modules)
    else:
        '''CORE'''
        # print('2')  # √
        # print(registry)  # detector -> backbone -> neck -> head -> loss -> loss
        return build_from_cfg(cfg, registry, default_args)


'''
7个基本大类的build按照Registry的不同占位层进行不同配置
主要原因是每个大类对应不同功能，落实在_module_dict包含不同的层(class)
因此分开进行build
e.g.build_backbone(cfg),build_neck(cfg),build_roi_extractor(cfg)...  
'''
# 何时调用？-> RetinaNet继承SingleStageDetector时__init__
# self.backbone = builder.build_backbone(backbone)
def build_backbone(cfg):
    return build(cfg, BACKBONES)


#         if neck is not None:
#             self.neck = builder.build_neck(neck)
def build_neck(cfg):
    return build(cfg, NECKS)


# self.bbox_head = builder.build_head(bbox_head)
def build_head(cfg):
    return build(cfg, HEADS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)

