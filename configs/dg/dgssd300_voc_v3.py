# -*- coding: utf-8 -*- 
# model settings in default
# model settings
input_size = (384,640)
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(
        type = 'DgMobilenet',
        out_feature_indices=(7, 10, 13, 16)),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        in_channels=(64, 128, 128, 128),
        num_classes=3,
        anchor_generator=dict(
            type='DgSSDAnchorGeneratorv2',
            #MIN_SIZES = [[10,16], [32,42,56], [70,90,120], [160,210,300]],
            #MIN_SIZES = [[10, 16], [32, 51, 81], [128,158,195], [256,322,406]],
            MIN_SIZES = [[10, 16], [23, 34, 51], [76, 113], [168, 250, 360]], # (1/2~2/3)RF: 41, 89, 185, 360
            MAX_SIZES = [],
            ASPECT_RATIOS = [[0.5, 2], [0.5, 2, 0.3,3], [0.5, 2, 0.3, 3], [0.5, 2, 0.3, 3]],
            strides = [8, 16, 32, 64],
            flip = False,
            ),
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
    dict(type='DgScale', scale_factor=(0.5, 1)),
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
        #ratio_range=(1, 4)
        ratio_range=(1, 2),
        ),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        #min_crop_size=0.3
        min_crop_size=0.2,
        ),
    #dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Expand2Canvas', size=input_size, mean=[123.675, 116.28, 103.53]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=[0.4, 0.2], direction=['horizontal', 'diagonal']),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        flip=False,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            dict(type='Expand2Canvas', size=input_size, mean=[123.675, 116.28, 103.53]),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    # samples_per_gpu=12,
    # workers_per_gpu=3,
    samples_per_gpu=16,
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
    step=[6, 16, 22])

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
resume_from = None
workflow = [('train', 1)]
work_dir = '/mnt/data/jingzhudata/work_dirs/dgmobilenet_voc_v3'