# model settings
input_size = (576, 960)
classes = ('phone', 'money')
model = dict(
    type='RetinaNet',
    # pretrained='torchvision://resnet50',
    pretrained=None,
    backbone=dict(
        type = 'DgMobilenet_128Channels',
        out_feature_indices=(5, 10, 15)),
    neck=dict(
        type='FPN',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[64, 64, 128],
        #out_channels=256,
        out_channels=128,
        start_level=1,
        # add_extra_convs='on_input',
        # num_outs=5
        num_outs=2,
        ),
    bbox_head=dict(
        type='RetinaHead',
        #num_classes=80,
        num_classes=len(classes),
        #in_channels=256,
        in_channels=128,
        stacked_convs=4,
        # feat_channels=256,
        feat_channels=128,
        # anchor_generator=dict(
        #     type='AnchorGenerator',
        #     octave_base_scale=4,
        #     scales_per_octave=3,
        #     ratios=[0.5, 1.0, 2.0],
        #     strides=[8, 16, 32, 64, 128]),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.33, 0.5, 1.0, 2.0, 3.0],
            strides=[8, 16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
cudnn_benchmark = True

dataset_type = 'DGVOCDatasetTrainPhoneWallet'
test_dataset_type = 'DGVOCDataset'
data_root = None #nouse
img_norm_cfg = dict(mean=[128, 128, 128], std=[1, 1, 1], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='PhotoMetricDistortion',
    #     gray = True,
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    
    # dict(
    #     type='Expand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     #ratio_range=(1, 4)
    #     ratio_range=(1, 2),
    #     ),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #     #min_crop_size=0.3
    #     min_crop_size=0.2,
    #     ),
    # dict(type='Expand2Canvas', size=input_size, mean=img_norm_cfg['mean'], use_base=False),
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
            dict(type='Expand2Canvas', size=input_size, mean=img_norm_cfg['mean'], use_base=True),
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
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=5e-4)
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
resume_from = '/mnt/datadisk0/jingzhudata/work_dirs/dgretinanet_mb_fpn_wallet_phonepy_original_aug/latest.pth'
workflow = [('train', 1)]
work_dir = '/mnt/datadisk0/jingzhudata/work_dirs/dgretinanet_mb_fpn_wallet_phonepy_original_aug'
