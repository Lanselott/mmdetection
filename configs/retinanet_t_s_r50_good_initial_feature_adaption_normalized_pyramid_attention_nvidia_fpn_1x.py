# model settings
BLOCK_ALIGN = False
PYRAMID_ALIGN = True
PRI_PYRAMID_ALIGN = False
PYRAMID_CORRELATION = False
HEAD_ALIGN = False
FREEZE_TEACHER = False
RATIO = 2
DOWNSAMPLE_RATIO = 1
COPY_TEACHER_FPN = False
GOOD_INITIAL = True
BN_TOPK_SELECTION = False
ROUSE_STUDENT_POINT = 7330 * 13
USE_INTERMEDIATE_LEARNER = False
# inference parameters
SWITCH_TO_INTER_LEARNER = False
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResTSNet',
        depth=50,
        s_depth=50,
        t_s_ratio=RATIO,
        spatial_ratio=DOWNSAMPLE_RATIO,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        freeze_teacher=FREEZE_TEACHER,
        good_initial=GOOD_INITIAL,
        bn_topk_selection=BN_TOPK_SELECTION,
        rouse_student_point=ROUSE_STUDENT_POINT,
        apply_block_wise_alignment=BLOCK_ALIGN,
        feature_adaption=True,
        constant_term=False,
        conv_downsample=True,
    ),
    neck=dict(
        type='FPNTS',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        s_in_channels=[128, 256, 512, 1024],
        s_out_channels=128,
        start_level=1,
        t_s_ratio=RATIO,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
        apply_block_wise_alignment=BLOCK_ALIGN,
        copy_teacher_fpn=COPY_TEACHER_FPN,
        freeze_teacher=FREEZE_TEACHER,
        rouse_student_point=ROUSE_STUDENT_POINT),
    bbox_head=dict(
        type='RetinaTSHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        s_feat_channels=128,
        dynamic_weight=True,
        norm_pyramid=True,
        pyramid_wise_attention=True,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        apply_block_wise_alignment=BLOCK_ALIGN,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
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
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/2017/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
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
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
