from mmdet.utils import Registry
# registry的真正实现部分在mmdet.utils的registry.py
# 这里的registry只是注册关键字，utils/registry存放的才是真正的注册实现函数
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
# build_from_cfg(cfg.model, registry=DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
DETECTORS = Registry('detector')
