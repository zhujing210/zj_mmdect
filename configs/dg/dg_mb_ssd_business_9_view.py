# -*- coding: utf-8 -*- 
# model settings in default
# model settings

input_size = (192, 320)

# dataset settings
dataset_type = 'DGVOCDatasetViewCarout'
# TODO ignore fjs
test_dataset_type = 'DGVOCDatasetViewCarout'
data_root = None #nouse
img_norm_cfg = dict(mean=[128, 128, 128], std=[1, 1, 1], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        gray = True,
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
    dict(type='Expand2Canvas', size=input_size, mean=img_norm_cfg['mean'], use_base=False),
    # dict(type = 'Albu', 
    #      transforms=albu_train_transform,
    #         bbox_params=dict(
    #             type='BboxParams',
    #             format='pascal_voc',  # x,y,x,y
    #             label_fields=['gt_labels'],
    #             min_visibility=0.0),
    #         keymap={
    #             'img': 'image',
    #             'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=False),
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

classes = ('human', 'vehicle')
data = dict(
    # samples_per_gpu=12,
    # workers_per_gpu=3,
    # 128
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes = classes,
        ann_file=(
        ('', 'carout_train.list'),
        ),
        img_prefix='/mnt/datadisk0/jingzhudata/',
        pipeline=train_pipeline),
    val=dict(
        type=test_dataset_type,
        classes = classes,
        ann_file=(
            #('coco2017', 'val2017_voc.txt'),
            ('', 'carout_train.list'),
        ),
        img_prefix='/mnt/datadisk0/jingzhudata/',
        pipeline=test_pipeline),
    test=dict(
        type=test_dataset_type,
        classes = classes,
        ann_file=(
            #('coco2017', 'val2017_voc.txt'),
            ('', 'carout_train.list'),
        ),
        img_prefix='/mnt/datadisk0/jingzhudata/',
        pipeline=test_pipeline)
)

