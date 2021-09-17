# -*- coding: utf-8 -*- 
# model settings in default
# model settings
input_size = [300,300]
model = dict(
    type='SingleStageDetector',
    #pretrained='open-mmlab://vgg16_caffe',
    pretrained=None,
    backbone=dict(
        type = 'DgMobilenet',
        out_feature_indices=(7, 10, 13, 16)),
    # backbone=dict(
    #     type='SSDVGG',
    #     input_size=input_size,
    #     depth=16,
    #     with_last_pool=False,
    #     ceil_mode=True,
    #     out_indices=(3, 4),
    #     out_feature_indices=(22, 34),
    #     l2_norm_scale=None), #20
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        # in_channels=(512, 1024, 512, 256, 256, 256),
        in_channels=(64, 128, 128, 128),
        # num_classes=80,
        num_classes=3,
        anchor_generator=dict(
            type='DgSSDAnchorGenerator',
            #MIN_SIZES = [21, 45, 99, 153, 207, 261],
            MIN_SIZES = [[10,16], [32,42,56], [70,90,120], [160,210,300]],
            #MAX_SIZES = [45, 99, 153, 207, 261, 315],
            MAX_SIZES = [],
            ASPECT_RATIOS = [[1/2, 2], [1/2, 2, 1/3, 3], [1/2, 2, 1/3, 3], [1/2, 2, 1/3, 3]],
            #_steps = [8, 16, 32, 64, 100, 300],
            _steps = [8, 16, 32, 64],
            flip = False,
            _clip = False,
            input_size = input_size),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True


# dataset settings
dataset_type = 'DGVOCDataset'
data_root = None #nouse
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    # samples_per_gpu=12,
    # workers_per_gpu=3,
    samples_per_gpu=12,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=5, 
        dataset=dict(
            type=dataset_type,
            ann_file=(('coco2017', 'train2017_voc.txt'),
            ('DmsPersonCatDog', 'train_v2.txt'),
            ),
            img_prefix='/mnt/data/jingzhudata/',
            pipeline=train_pipeline)),
        # type=dataset_type,
        # ann_file=(('coco2017', 'train2017_voc.txt'),
        # ('DmsPersonCatDog', 'train_v2.txt'),),
        # img_prefix='/mnt/data/jingzhudata/',
        # pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=(
            #('coco2017', 'val2017_voc.txt'),
            ('DmsPersonCatDog', 'test_v2.txt'),
        ),
        img_prefix='/mnt/data/jingzhudata/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=(
            #('coco2017', 'val2017_voc.txt'),
            ('DmsPersonCatDog', 'test_v2.txt'),
        ),
        img_prefix='/mnt/data/jingzhudata/',
        pipeline=test_pipeline)
)

evaluation = dict(interval=1, metric='mAP')
# optimizer
#voc
# optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 20])

#coco
#optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()


# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[6, 15, 22])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)

# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/home/jingzhu/mmdet/work_dirs/dgmobilenet_voc_baseline/latest.pth'
workflow = [('train', 1)]
work_dir = './work_dirs/dgmobilenet_voc_baseline'