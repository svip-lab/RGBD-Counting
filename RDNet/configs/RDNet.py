# model settings
model = dict(
    type='RetinaNet',
    pretrained='modelzoo://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=2,
        scales_per_octave=4,
        anchor_ratios=[1.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0/9.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.35,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=230)
# dataset settingscounting
dataset_type = 'RGBD'
data_root = '/p300/Dataset/SIST_RGBD/RGBDmerge_540P/Part_A/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.csv',
        img_prefix=data_root + 'train_img/',
        class_file=data_root + 'class.csv',
        mode=True,
        img_scale=(960, 540),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test.csv',
        img_prefix=data_root + 'test_img/',
        class_file=data_root + 'class.csv',
        mode=False,
        img_scale=(960, 540),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.csv',
        img_prefix=data_root + 'test_img/',
        class_file=data_root + 'class.csv',
        mode=False,
        img_scale=(960, 540),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='Adam', lr=0.00005, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    target_lr=0.00001)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 200
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/p300/work_dirs/9.73.imagenet'
load_from = None
resume_from = None
workflow = [('train', 1)]
