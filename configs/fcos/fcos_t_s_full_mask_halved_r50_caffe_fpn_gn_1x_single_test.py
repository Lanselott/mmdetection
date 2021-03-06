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
# Asynmetric backbone settings
S_IN_CHANNELS = [128, 256, 512, 1024]

model = dict(
    type='FCOSTS',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResTSNet',
        depth=50,
        s_depth=50,
        t_s_ratio=RATIO,
        spatial_ratio=DOWNSAMPLE_RATIO,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe',
        pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
        apply_block_wise_alignment=BLOCK_ALIGN,
        freeze_teacher=FREEZE_TEACHER,
        good_initial=GOOD_INITIAL,
        feature_adaption=False,
        train_mode=False,
        conv_downsample=False,
        constant_term=False,
        pure_student_term=True,
        kernel_adaption=True,
        bn_topk_selection=BN_TOPK_SELECTION,
        rouse_student_point=ROUSE_STUDENT_POINT),
    neck=dict(
        type='FPNTS',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # s_in_channels=[128, 256, 512, 1024],
        s_in_channels=S_IN_CHANNELS,
        s_out_channels=128,
        start_level=1,
        t_s_ratio=RATIO,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
        pure_student_term=True,
        apply_block_wise_alignment=BLOCK_ALIGN,
        copy_teacher_fpn=COPY_TEACHER_FPN,
        freeze_teacher=FREEZE_TEACHER,
        rouse_student_point=ROUSE_STUDENT_POINT,
        kernel_meta_learner=False),
    bbox_head=dict(
        type='FCOSTSFullMaskHead',
        num_classes=81,
        in_channels=256,
        s_in_channels=128,
        stacked_convs=4,
        feat_channels=256,
        s_feat_channels=128,
        t_s_ratio=RATIO,
        spatial_ratio=DOWNSAMPLE_RATIO,
        training=True,
        eval_student=True,
        eval_teacher_backbone=False,
        pyramid_merging=False,
        learn_when_train=True,
        finetune_student=True,
        apply_autoencoder=False,
        apply_sharing_alignment=False,
        cls_aware_attention=False,
        train_teacher=False,
        apply_iou_similarity=False,
        sorting_match=False,
        apply_posprocessing_similarity=False,
        apply_soft_regression_distill=False,
        apply_selective_regression_distill=False,
        apply_soft_cls_distill=False,
        pyramid_decoupling=False,
        temperature=1,
        align_level=0,
        apply_block_wise_alignment=BLOCK_ALIGN,
        apply_pyramid_wise_alignment=PYRAMID_ALIGN,
        apply_discriminator=False,
        siamese_distill=False,
        copy_teacher_fpn=COPY_TEACHER_FPN,
        multi_levels=1,  # 1 or 5
        multi_branches=False,
        naive_conv=True,  # NOTE: Check it before evaluation!
        apply_pri_pyramid_wise_alignment=PRI_PYRAMID_ALIGN,
        simple_pyramid_alignment=False,
        learn_from_missing_annotation=False,
        block_wise_attention=False,
        pyramid_wise_attention=True,
        logistic_train_first=False,
        pyramid_train_first=False,
        learn_from_teacher_backbone=False,
        ignore_low_ious=False,
        pyramid_full_attention=False,
        pyramid_factor=1,
        pyramid_correlation=PYRAMID_CORRELATION,
        pyramid_learn_high_quality=False,
        pyramid_attention_only=False,
        interactive_learning=False,
        se_attention=False,
        direct_downsample=False,
        use_intermediate_learner=USE_INTERMEDIATE_LEARNER,
        norm_pyramid=False,
        apply_sharing_auxiliary_fpn=False,
        use_student_backbone=False,
        switch_to_inter_learner=SWITCH_TO_INTER_LEARNER,
        corr_out_channels=64,
        head_attention_factor=1,
        pyramid_cls_reg_consistent=False,
        pyramid_nms_aware=False,
        dynamic_weight=True,
        head_wise_attention=False,
        apply_head_wise_alignment=HEAD_ALIGN,
        head_align_levels=[0, 1],
        align_to_teacher_logits=False,
        block_teacher_attention=False,
        head_teacher_reg_attention=False,
        consider_cls_reg_distribution=False,
        freeze_teacher=FREEZE_TEACHER,
        rouse_student_point=ROUSE_STUDENT_POINT,  #6 * 7330,
        # student distillation params
        beta=1.5,
        gamma=2,
        adap_distill_loss_weight=0.1,
        strides=[8, 16, 32, 64, 128],
        pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
        # pyramid_hint_loss=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     # reduction='none',
        #     loss_weight=1.0),
        reg_head_hint_loss=dict(type='MSELoss', loss_weight=1),
        cls_head_hint_loss=dict(type='MSELoss', loss_weight=1),
        cls_reg_distribution_hint_loss=dict(type='MSELoss', loss_weight=1),
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
        loss_regression_distill=dict(type='IoULoss', loss_weight=1.0),
        reg_distill_threshold=0.5,
        inner_opt=False,
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
data_root = 'data/2017/'
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
    imgs_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer = dict(type='Adam', lr=0.0002, betas=(0.5, 0.999), eps=1e-08)
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
load_from = None  # 'work/dirs/fcos_t_s_scratch_model/fcos_t_s_finetune_halved_student_from_scratch_epoch_12.pth'
resume_from = None
workflow = [('train', 1)]
