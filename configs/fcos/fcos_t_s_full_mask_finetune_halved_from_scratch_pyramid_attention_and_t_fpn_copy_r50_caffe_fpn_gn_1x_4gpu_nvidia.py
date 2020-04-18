# model settings
BLOCK_ALIGN = False
PYRAMID_ALIGN = True
HEAD_ALIGN = False
FREEZE_TEACHER = False
RATIO = 2
model = dict(
    type='FCOSTS',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResTSNet',
        depth=50,
        t_s_ratio=RATIO,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe',
        pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
        apply_block_wise_alignment=BLOCK_ALIGN,
        freeze_teacher=FREEZE_TEACHER),
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
        copy_teacher_fpn=True,
        apply_block_wise_alignment=BLOCK_ALIGN,
        freeze_teacher=FREEZE_TEACHER),
    bbox_head=dict(
        type='FCOSTSFullMaskHead',
        num_classes=81,
        in_channels=256,
        s_in_channels=128,
        stacked_convs=4,
        feat_channels=256,
        s_feat_channels=128,
        t_s_ratio=RATIO,
        training=True,
        eval_student=False,
        learn_when_train=True,
        finetune_student=True,
        apply_iou_similarity=False,
        apply_soft_cls_distill=False,
        apply_feature_alignment=False,  # regression
        temperature=1,
        align_level=0,
        apply_block_wise_alignment=BLOCK_ALIGN,
        apply_pyramid_wise_alignment=PYRAMID_ALIGN,
        copy_teacher_fpn=True,
        pyramid_wise_attention=True,
        pyramid_full_attention=False,
        pyramid_attention_only=True,
        pyramid_attention_factor=1,
        apply_data_free_mode=False,
        apply_head_wise_alignment=HEAD_ALIGN,
        freeze_teacher=FREEZE_TEACHER,
        block_teacher_attention=False,
        attention_threshold=0.5,  # duplicate
        # student distillation params
        beta=1.5,
        gamma=2,
        adap_distill_loss_weight=0.3,
        strides=[8, 16, 32, 64, 128],
        pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
        reg_head_hint_loss=dict(type='MSELoss', loss_weight=1),
        cls_head_hint_loss=dict(type='MSELoss', loss_weight=1),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        # loss_s_t_cls=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_s_t_reg=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_s_t_cls=dict(type='MSELoss', loss_weight=5),
        loss_s_t_reg=dict(type='MSELoss', loss_weight=5),
        t_s_distance=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=1.0),
        # loss_iou_similiarity=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_iou_similiarity=dict(type='MSELoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
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
data_root = '/coco/data/2017/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
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
    imgs_per_gpu=4,
    workers_per_gpu=4,
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
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
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
work_dir = './work_dirs/fcos_r50_caffe_fpn_gn_1x_4gpu'
load_from = None  #'./fcos_t_s_finetune_halved_student_from_scratch_epoch_12.pth'
resume_from = './fcos_t_s_t_fpn_copy_epoch_7.pth'
workflow = [('train', 1)]
