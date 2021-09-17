# -*- coding: utf-8 -*- 
# model settings in default
# model settings
"""
wallet && phone detection.
44epoches, original dataaugmentation.
"""
input_size = (576, 960)
classes = ('phone', 'money')
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(
        type = 'DgMobilenet_128Channels',
        out_feature_indices=(10, 15)),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        in_channels=(64, 128),
        num_classes=len(classes),
        anchor_generator=dict(
            type='DgSSDAnchorGeneratorv2',
            MIN_SIZES = [[11, 15, 22], [32, 43, 61]], # iou_0.5  # w/h=1/r
            MAX_SIZES = [],
            ASPECT_RATIOS = [[0.33, 1/2, 2, 3], [0.33, 1/2, 2, 3]],
            strides = [8, 16],
            flip = False),
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
dataset_type = 'DGVOCDatasetTrainPhoneWallet'
test_dataset_type = 'DGVOCDatasetTrainPhoneWallet'
data_root = None #nouse
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # # dict(
    # #     type='CopyPaste',
    # #     classes = classes,
    # #     least_category = 'poi_phone',
    # #     p = 0.1,
    # # ),
    # dict(
    #     type='PhotoMetricDistortion',
    #     gray = False,
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    # dict(
    #     type='Expand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     ratio_range=(1, 2),
    #     ),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #     #min_crop_size=0.3
    #     min_crop_size=0.2,
    #     ),
    dict(type='Expand2Canvas', size=input_size, mean=img_norm_cfg['mean'], use_base=False, gray_img=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
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
            dict(type='Expand2Canvas', size=input_size, mean=img_norm_cfg['mean'], use_base=True, gray_img=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,    
    train=dict(
                type=dataset_type,
                classes = classes,
                ann_file=(
                    ('', 'has_people_phone_rand3w_train2.lst'),
                    ('', 'jianhang_0412_RMB_rand3w_train2.lst'), 
                    ('', 'money_phone_20210710_rand2w_2w_train.lst'),
                    ('','colloect_phone_money_20210708_train.lst'),
                    ),
                img_prefix='/mnt/datadisk0/jingzhudata/phone_money/',
                pipeline=train_pipeline
        ),
    val=dict(
        type=test_dataset_type,
        classes = classes,
        ann_file=(
            #('coco2017', 'val2017_voc.txt'),
           ('', 'all_test.lst'),
        ),
        img_prefix='/mnt/datadisk0/jingzhudata/phone_money',
        pipeline=test_pipeline),
    test=dict(
        type=test_dataset_type,
        classes = classes,
        ann_file=(
            #('coco2017', 'val2017_voc.txt'),
           ('', 'all_test.lst'),
        ),
        img_prefix='/mnt/datadisk0/jingzhudata/phone_money',
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
    warmup_iters=1400,
    warmup_ratio=0.001,
    step=[12, 22, 32, 42])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=44)

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
load_from =  None
resume_from = None
workflow = [('train', 1)]
work_dir = '/mnt/datadisk0/jingzhudata/work_dirs/dg_mb_ssd_wallet_phonepy_original_aug_test'