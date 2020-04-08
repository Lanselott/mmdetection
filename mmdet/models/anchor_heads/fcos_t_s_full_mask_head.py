import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, bbox_overlaps
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
from IPython import embed
INF = 1e8


@HEADS.register_module
class FCOSTSFullMaskHead(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 s_in_channels,
                 feat_channels=256,
                 s_feat_channels=64,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 t_s_ratio=1,
                 spatial_ratio=1,
                 eval_student=True,
                 training=True,
                 learn_when_train=False,
                 finetune_student=False,
                 train_teacher=False,
                 apply_iou_similarity=False,
                 apply_posprocessing_similarity=False,
                 apply_soft_regression_distill=False,
                 choose_better_iou=False,
                 apply_soft_cls_distill=False,
                 apply_soft_centerness_distill=False,
                 temperature=1,
                 apply_feature_alignment=False,
                 fix_student_train_teacher=False,
                 train_student_only=False,
                 align_level=1,
                 apply_block_wise_alignment=False,
                 apply_pyramid_wise_alignment=False,
                 apply_head_wise_alignment=False,
                 simple_pyramid_alignment=False,
                 head_align_levels=[0],
                 apply_data_free_mode=False,
                 learn_from_missing_annotation=False,
                 block_wise_attention=False,
                 pyramid_wise_attention=False,
                 pyramid_full_attention=False,
                 corr_out_channels=32,
                 pyramid_correlation=False,
                 pyramid_cls_reg_consistent=False,
                 pyramid_nms_aware=False,
                 pyramid_attention_factor=1,
                 head_attention_factor=1,
                 dynamic_weight=False,
                 head_wise_attention=False,
                 align_to_teacher_logits=False,
                 block_teacher_attention=False,
                 head_teacher_reg_attention=False,
                 consider_cls_reg_distribution=False,
                 teacher_iou_attention=False,
                 attention_threshold=0.5,
                 freeze_teacher=False,
                 beta=1,
                 gamma=1,
                 adap_distill_loss_weight=0.5,
                 pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
                 reg_head_hint_loss=dict(type='MSELoss', loss_weight=1),
                 cls_head_hint_loss=dict(type='MSELoss', loss_weight=1),
                 cls_reg_distribution_hint_loss=dict(
                     type='MSELoss', loss_weight=1),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_t_logits_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_t_logits_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_s_t_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_s_t_reg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 t_s_distance=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_regression_distill=dict(type='IoULoss', loss_weight=1),
                 reg_distill_threshold=0.5,
                 loss_iou_similiarity=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(FCOSTSFullMaskHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.s_in_channels = s_in_channels
        self.feat_channels = feat_channels
        self.s_feat_channels = s_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.t_s_ratio = t_s_ratio
        self.spatial_ratio = spatial_ratio
        self.align_level = align_level
        self.apply_block_wise_alignment = apply_block_wise_alignment
        self.apply_pyramid_wise_alignment = apply_pyramid_wise_alignment
        self.simple_pyramid_alignment = simple_pyramid_alignment
        self.block_wise_attention = block_wise_attention
        self.pyramid_wise_attention = pyramid_wise_attention
        self.pyramid_full_attention = pyramid_full_attention
        self.pyramid_correlation = pyramid_correlation
        self.corr_out_channels = corr_out_channels
        self.pyramid_cls_reg_consistent = pyramid_cls_reg_consistent
        self.pyramid_nms_aware = pyramid_nms_aware
        self.pyramid_attention_factor = pyramid_attention_factor
        self.head_attention_factor = head_attention_factor
        self.dynamic_weight = dynamic_weight
        self.head_wise_attention = head_wise_attention
        self.apply_head_wise_alignment = apply_head_wise_alignment
        self.head_align_levels = head_align_levels
        self.align_to_teacher_logits = align_to_teacher_logits
        self.block_teacher_attention = block_teacher_attention
        self.head_teacher_reg_attention = head_teacher_reg_attention
        self.consider_cls_reg_distribution = consider_cls_reg_distribution
        self.teacher_iou_attention = teacher_iou_attention
        self.attention_threshold = attention_threshold
        self.freeze_teacher = freeze_teacher
        self.beta = beta
        self.gamma = gamma
        self.adap_distill_loss_weight = adap_distill_loss_weight
        self.training = training
        self.eval_student = eval_student
        self.learn_when_train = learn_when_train
        self.finetune_student = finetune_student
        self.train_teacher = train_teacher
        self.apply_iou_similarity = apply_iou_similarity
        self.apply_posprocessing_similarity = apply_posprocessing_similarity
        self.apply_soft_regression_distill = apply_soft_regression_distill
        self.choose_better_iou = choose_better_iou
        self.apply_soft_cls_distill = apply_soft_cls_distill
        self.apply_soft_centerness_distill = apply_soft_centerness_distill
        self.temperature = temperature
        self.apply_feature_alignment = apply_feature_alignment
        self.apply_data_free_mode = apply_data_free_mode
        self.learn_from_missing_annotation = learn_from_missing_annotation
        self.fix_student_train_teacher = fix_student_train_teacher
        self.train_student_only = train_student_only
        self.pyramid_hint_loss = build_loss(pyramid_hint_loss)
        self.reg_head_hint_loss = build_loss(reg_head_hint_loss)
        self.cls_head_hint_loss = build_loss(cls_head_hint_loss)
        self.cls_reg_distribution_hint_loss = build_loss(
            cls_reg_distribution_hint_loss)
        self.loss_cls = build_loss(loss_cls)
        self.loss_t_logits_cls = build_loss(loss_t_logits_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_t_logits_bbox = build_loss(loss_t_logits_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_s_t_cls = build_loss(loss_s_t_cls)
        self.loss_s_t_reg = build_loss(loss_s_t_reg)
        self.t_s_distance = build_loss(t_s_distance)
        self.loss_regression_distill = build_loss(loss_regression_distill)
        self.reg_distill_threshold = reg_distill_threshold
        self.loss_iou_similiarity = nn.BCELoss(
        )  # build_loss(loss_iou_similiarity)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self._init_teacher_layers()
        self._init_student_layers()

    def _init_teacher_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_student_layers(self):
        self.s_cls_convs = nn.ModuleList()
        self.s_reg_convs = nn.ModuleList()

        self.s_t_reg_head_align = nn.ModuleList()
        self.s_t_cls_head_align = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.s_in_channels if i == 0 else self.s_feat_channels
            self.s_cls_convs.append(
                ConvModule(
                    chn,
                    self.s_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.s_reg_convs.append(
                ConvModule(
                    chn,
                    self.s_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        '''
        # Align student feature to teacher
        '''
        self.t_s_correlation_conv = nn.ModuleList()
        if self.apply_pyramid_wise_alignment:
            self.t_s_pyramid_align = nn.Conv2d(
                self.s_feat_channels, self.feat_channels, 3, padding=1)

        if self.apply_head_wise_alignment:
            # NOTE: head wise + learn from logits
            for i in range(self.stacked_convs + 1):
                # s->t
                self.s_t_reg_head_align.append(
                    ConvModule(
                        self.s_feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                self.s_t_cls_head_align.append(
                    ConvModule(
                        self.s_feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))

        self.fcos_s_cls = nn.Conv2d(
            self.s_feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_s_reg = nn.Conv2d(self.s_feat_channels, 4, 3, padding=1)
        self.fcos_s_centerness = nn.Conv2d(
            self.s_feat_channels, 1, 3, padding=1)

        # teacher and student scales should be different
        self.s_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)
        # student model
        for m in self.s_cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.s_reg_convs:
            normal_init(m.conv, std=0.01)
        bias_s_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_s_cls, std=0.01, bias=bias_s_cls)
        normal_init(self.fcos_s_reg, std=0.01)
        normal_init(self.fcos_s_centerness, std=0.01)
        if self.apply_pyramid_wise_alignment:
            normal_init(self.t_s_pyramid_align, std=0.01)

        if self.apply_head_wise_alignment:
            for m in self.s_t_cls_head_align:
                normal_init(m.conv, std=0.01)
            for m in self.s_t_reg_head_align:
                normal_init(m.conv, std=0.01)

        if self.freeze_teacher:
            self.freeze_teacher_layers()

    def freeze_teacher_layers(self):
        for m in [self.cls_convs, self.reg_convs]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        self.fcos_cls.eval()
        self.fcos_reg.eval()
        self.fcos_centerness.eval()
        for m in [self.fcos_cls, self.fcos_reg, self.fcos_centerness]:
            for param in m.parameters():
                param.requires_grad = False

        for scale in self.scales:
            scale.eval()
            for m in scale.parameters():
                m.requires_grad = False

    def forward(self, feats):
        t_feats = feats[0]
        s_feats = feats[1]

        # hint_losses_list = []
        if self.apply_block_wise_alignment:
            hint_pairs = feats[2]
            hint_pairs += tuple('N')
            return multi_apply(self.forward_single, t_feats, s_feats,
                               self.scales, self.s_scales, hint_pairs)
        else:
            return multi_apply(self.forward_single, t_feats, s_feats,
                               self.scales, self.s_scales)

    def forward_single(self, t_x, s_x, scale, s_scale, hint_pairs=None):
        t_cls_feat = t_x
        t_reg_feat = t_x
        # student head model
        s_cls_feat = s_x
        s_reg_feat = s_x
        cls_hint_pairs = []
        reg_hint_pairs = []

        if self.apply_pyramid_wise_alignment:
            pyramid_hint_pairs = []
            pyramid_hint_pairs.append(s_x)
            pyramid_hint_pairs.append(t_x)

        corr_pairs = []

        for i in range(len(self.cls_convs)):
            cls_layer = self.cls_convs[i]
            cls_s_layer = self.s_cls_convs[i]
            t_cls_feat = cls_layer(t_cls_feat)
            s_cls_feat = cls_s_layer(s_cls_feat)

            if self.apply_head_wise_alignment:
                cls_hint_pairs.append([s_cls_feat, t_cls_feat])

        cls_score = self.fcos_cls(t_cls_feat)
        s_cls_score = self.fcos_s_cls(s_cls_feat)
        centerness = self.fcos_centerness(t_cls_feat)
        s_centerness = self.fcos_s_centerness(s_cls_feat)

        for j in range(len(self.reg_convs)):
            reg_layer = self.reg_convs[j]
            s_reg_layer = self.s_reg_convs[j]
            t_reg_feat = reg_layer(t_reg_feat)
            s_reg_feat = s_reg_layer(s_reg_feat)

            if self.apply_head_wise_alignment:
                reg_hint_pairs.append([s_reg_feat, t_reg_feat])

        # wrap reg/cls head features
        if self.apply_head_wise_alignment:
            head_hint_pairs = []
            head_hint_pairs.append(cls_hint_pairs)
            head_hint_pairs.append(reg_hint_pairs)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(t_reg_feat)).float().exp()
        s_bbox_pred = s_scale(self.fcos_s_reg(s_reg_feat)).float().exp()

        if self.training:
            if self.apply_pyramid_wise_alignment and not self.apply_head_wise_alignment:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, hint_pairs, pyramid_hint_pairs, None, corr_pairs
            elif self.apply_head_wise_alignment and not self.apply_pyramid_wise_alignment:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, hint_pairs, None, head_hint_pairs, None
            elif self.apply_pyramid_wise_alignment and self.apply_head_wise_alignment:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, hint_pairs, pyramid_hint_pairs, head_hint_pairs, corr_pairs
            else:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, hint_pairs, None, None, corr_pairs
        else:
            if self.eval_student:
                return s_cls_score, s_bbox_pred, s_centerness
            else:
                return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             s_cls_scores,
             s_bbox_preds,
             s_centernesses,
             hint_pairs,
             pyramid_hint_pairs,
             head_hint_pairs,
             corr_pairs,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        loss_cls, loss_bbox, loss_centerness, _, t_flatten_cls_scores, flatten_labels, t_iou_maps, t_pos_inds, t_pred_bboxes, t_gt_bboxes, block_distill_masks, _, t_pred_centerness, t_all_pred_bboxes, t_pos_points = self.loss_single(
            cls_scores,
            bbox_preds,
            centernesses,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore=None)
        s_loss_cls, s_loss_bbox, s_loss_centerness, cls_avg_factor, s_flatten_cls_scores, _, s_iou_maps, _, s_pred_bboxes, s_gt_bboxes, _, pos_centerness_targets, s_pred_centerness, s_all_pred_bboxes, _ = self.loss_single(
            s_cls_scores,
            s_bbox_preds,
            s_centernesses,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore=None,
            spatial_ratio=self.spatial_ratio)

        loss_dict = {}
        assert self.fix_student_train_teacher != self.learn_when_train

        if self.learn_when_train:
            if self.apply_block_wise_alignment:
                hint_pairs = hint_pairs[:4]  # Remove placeholder
                for j, hint_feature in enumerate(hint_pairs):
                    s_block_feature = hint_feature[0]
                    t_block_feature = hint_feature[1].detach()
                    if self.block_teacher_attention:
                        # Apply method to partially update hint losses
                        block_distill_masks = torch.nn.functional.upsample(
                            block_distill_masks,
                            size=t_block_feature.shape[2:])
                        attention_weight = block_distill_masks.expand(
                            -1, t_block_feature.shape[1], -1, -1)
                        hint_loss = self.pyramid_hint_loss(
                            s_block_feature,
                            t_block_feature,
                            weight=attention_weight)
                    else:
                        hint_loss = self.pyramid_hint_loss(
                            s_block_feature, t_block_feature)
                    loss_dict.update(
                        {'hint_loss_block_{}'.format(j): hint_loss})
            # NOTE: pyramid wise alignment
            if self.apply_pyramid_wise_alignment:
                t_pyramid_feature_list = []
                s_pyramid_feature_list = []
                for j, pyramid_hint_pair in enumerate(pyramid_hint_pairs):
                    if self.spatial_ratio > 1:
                        s_pyramid_feature = self.t_s_pyramid_align(
                            F.interpolate(
                                pyramid_hint_pair[0],
                                size=pyramid_hint_pair[1].shape[2:],
                                mode='nearest'))
                    else:
                        s_pyramid_feature = self.t_s_pyramid_align(
                            pyramid_hint_pair[0])
                    t_pyramid_feature = pyramid_hint_pair[1].detach()
                    t_pyramid_feature_list.append(
                        t_pyramid_feature.permute(0, 2, 3, 1).reshape(
                            -1, self.feat_channels))
                    s_pyramid_feature_list.append(
                        s_pyramid_feature.permute(0, 2, 3, 1).reshape(
                            -1, self.feat_channels))

                t_pyramid_feature_list = torch.cat(t_pyramid_feature_list)
                s_pyramid_feature_list = torch.cat(s_pyramid_feature_list)

                if self.pyramid_wise_attention:
                    t_pred_cls = t_flatten_cls_scores.max(1)[1]
                    s_pred_cls = s_flatten_cls_scores.max(1)[1]
                    cls_attention_weight = (t_pred_cls == s_pred_cls).float()
                    # cls_attention_weight *= self.pyramid_attention_factor
                    s_pred_bbox_quality = bbox_overlaps(
                        s_pred_bboxes, t_gt_bboxes,
                        is_aligned=True).detach().mean()
                    if self.pyramid_full_attention:
                        iou_attention_weight = bbox_overlaps(
                            s_all_pred_bboxes, t_all_pred_bboxes,
                            is_aligned=True).detach()
                        attention_iou_pyramid_hint_loss = self.pyramid_attention_factor * self.pyramid_hint_loss(
                            s_pyramid_feature_list,
                            t_pyramid_feature_list,
                            weight=iou_attention_weight,
                            avg_factor=iou_attention_weight.sum())
                    else:
                        iou_attention_weight = bbox_overlaps(
                            s_pred_bboxes, t_pred_bboxes,
                            is_aligned=True).detach()
                        attention_iou_pyramid_hint_loss = self.pyramid_attention_factor * self.pyramid_hint_loss(
                            s_pyramid_feature_list[t_pos_inds],
                            t_pyramid_feature_list[t_pos_inds],
                            weight=iou_attention_weight,
                            avg_factor=iou_attention_weight.sum())
                    # attention_cls_pyramid_hint_loss = self.pyramid_attention_factor * self.pyramid_hint_loss(
                    #     s_pyramid_feature_list,
                    #     t_pyramid_feature_list,
                    #     weight=cls_attention_weight)
                    loss_dict.update({
                        # 'attention_cls_pyramid_hint_loss':
                        # attention_cls_pyramid_hint_loss,
                        'attention_iou_pyramid_hint_loss':
                        attention_iou_pyramid_hint_loss
                    })
                pyramid_hint_loss = self.pyramid_hint_loss(
                    s_pyramid_feature_list, t_pyramid_feature_list)
                loss_dict.update({'pyramid_hint_loss': pyramid_hint_loss})
            # NOTE: apply pyramid correlation
            if self.pyramid_correlation:
                '''
                affinity_size = t_pos_inds.shape[0]
                t_affinity_matrix = torch.zeros(
                    [affinity_size, affinity_size, self.feat_channels], dtype=t_pyramid_feature_list.dtype, device=t_pyramid_feature_list.device)
                s_affinity_matrix = torch.zeros(
                    [affinity_size, affinity_size, self.feat_channels], dtype=s_pyramid_feature_list.dtype, device=s_pyramid_feature_list.device)

                for i, t_pyramid_feat_i in enumerate(t_pyramid_feature_list[t_pos_inds]):
                    for j, t_pyramid_feat_j in enumerate(t_pyramid_feature_list[t_pos_inds]):
                        # get distance for pyramid pixel-wise features
                        # of inner/intra instances, a affinity matrix
                        t_affinity_matrix[i][j] = t_pyramid_feat_i - t_pyramid_feat_j

                t_affinity_matrix = t_affinity_matrix / \
                    (t_affinity_matrix.max(2)[0].unsqueeze(2) + 1e-6)
                t_affinity_matrix = torch.norm(t_affinity_matrix, dim=2)

                for i, s_pyramid_feat_i in enumerate(s_pyramid_feature_list[t_pos_inds]):
                    for j, s_pyramid_feat_j in enumerate(s_pyramid_feature_list[t_pos_inds]):
                        # get distance for pyramid pixel-wise features
                        # of inner/intra instances, a affinity matrix
                        s_affinity_matrix[i][j] = s_pyramid_feat_i - s_pyramid_feat_j

                s_affinity_matrix = s_affinity_matrix / \
                    (s_affinity_matrix.max(2)[0].unsqueeze(2) + 1e-6)
                s_affinity_matrix = torch.norm(s_affinity_matrix, dim=2)

                corr_affinity_loss = self.pyramid_hint_loss(
                    s_affinity_matrix, t_affinity_matrix)
                loss_dict.update({'corr_affinity_loss': corr_affinity_loss})
                '''
                t_corr_map = torch.mm(
                    t_pyramid_feature_list[t_pos_inds], t_pyramid_feature_list[t_pos_inds].permute(1, 0))
                t_corr_map_range = t_corr_map.max() - t_corr_map.min()
                t_corr_map = (t_corr_map - t_corr_map.min()) / t_corr_map_range
                s_corr_map = torch.mm(
                    s_pyramid_feature_list[t_pos_inds], s_pyramid_feature_list[t_pos_inds].permute(1, 0))
                s_corr_map_range = s_corr_map.max() - s_corr_map.min()
                s_corr_map = (s_corr_map - s_corr_map.min()) / s_corr_map_range

                corr_loss = 10 * self.pyramid_hint_loss(s_corr_map, t_corr_map)
                loss_dict.update({'corr_loss': corr_loss})

            # NOTE: head wise alignment
            if self.apply_head_wise_alignment:
                t_cls_heads_feature_list = []
                s_cls_heads_feature_list = []
                t_reg_heads_feature_list = []
                s_reg_heads_feature_list = []
                # pyramids
                for j, head_hint_pair in enumerate(head_hint_pairs):
                    cls_head_pair = head_hint_pair[0]
                    reg_head_pair = head_hint_pair[1]
                    reg_head_stacked_loss_list = []
                    cls_head_stacked_loss_list = []
                    # store temp tensors
                    t_cls_head_feature_list = []
                    s_cls_head_feature_list = []
                    t_reg_head_feature_list = []
                    s_reg_head_feature_list = []
                    # towers
                    for k in self.head_align_levels:
                        s_cls_head_feature = self.s_t_cls_head_align[k](
                            cls_head_pair[k][0])
                        s_reg_head_feature = self.s_t_reg_head_align[k](
                            reg_head_pair[k][0])
                        t_cls_head_feature = cls_head_pair[k][1].detach()
                        t_reg_head_feature = reg_head_pair[k][1].detach()
                        # learn from intermediate layers
                        t_cls_head_feature_list.append(t_cls_head_feature)
                        s_cls_head_feature_list.append(s_cls_head_feature)
                        t_reg_head_feature_list.append(t_reg_head_feature)
                        s_reg_head_feature_list.append(s_reg_head_feature)

                    t_cls_heads_feature_list.append(
                        torch.cat(t_cls_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      len(self.head_align_levels) *
                                      self.feat_channels))
                    s_cls_heads_feature_list.append(
                        torch.cat(s_cls_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      len(self.head_align_levels) *
                                      self.feat_channels))
                    t_reg_heads_feature_list.append(
                        torch.cat(t_reg_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      len(self.head_align_levels) *
                                      self.feat_channels))
                    s_reg_heads_feature_list.append(
                        torch.cat(s_reg_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      len(self.head_align_levels) *
                                      self.feat_channels))
                t_cls_heads_feature_list = torch.cat(t_cls_heads_feature_list,
                                                     0)
                s_cls_heads_feature_list = torch.cat(s_cls_heads_feature_list,
                                                     0)
                t_reg_heads_feature_list = torch.cat(t_reg_heads_feature_list,
                                                     0)
                s_reg_heads_feature_list = torch.cat(s_reg_heads_feature_list,
                                                     0)

                if self.head_wise_attention:
                    pos_t_reg_heads_feature = t_reg_heads_feature_list[
                        t_pos_inds]
                    pos_s_reg_heads_feature = s_reg_heads_feature_list[
                        t_pos_inds]
                    iou_attention_weight = bbox_overlaps(
                        s_pred_bboxes, t_pred_bboxes,
                        is_aligned=True).detach()
                    t_pred_cls = t_flatten_cls_scores.max(1)[1]
                    s_pred_cls = s_flatten_cls_scores.max(1)[1]
                    cls_attention_inds = (
                        t_pred_cls == s_pred_cls).nonzero().reshape(-1)
                    # head attention loss
                    cls_head_attention_hint_loss = self.head_attention_factor * self.cls_head_hint_loss(
                        s_cls_heads_feature_list[cls_attention_inds],
                        t_cls_heads_feature_list[cls_attention_inds])
                    reg_head_attention_hint_loss = self.head_attention_factor * self.reg_head_hint_loss(
                        pos_s_reg_heads_feature,
                        pos_t_reg_heads_feature,
                        weight=iou_attention_weight,
                        avg_factor=iou_attention_weight.sum())
                    loss_dict.update({
                        'reg_head_attention_hint_loss':
                        reg_head_attention_hint_loss
                    })
                    loss_dict.update({
                        'cls_head_attention_hint_loss':
                        cls_head_attention_hint_loss
                    })

                cls_head_hint_loss = self.head_attention_factor * self.cls_head_hint_loss(
                    s_cls_heads_feature_list, t_cls_heads_feature_list)
                reg_head_hint_loss = self.head_attention_factor * self.reg_head_hint_loss(
                    s_reg_heads_feature_list, t_reg_heads_feature_list)
                loss_dict.update({'reg_head_hint_loss': reg_head_hint_loss})
                loss_dict.update({'cls_head_hint_loss': cls_head_hint_loss})

            # NOTE: learn from logits
            if self.align_to_teacher_logits:
                t_cls_logits_list = []
                t_bbox_preds_list = []
                branch_level = 0
                # pyramids
                for j, pyramid_hint_pair in enumerate(pyramid_hint_pairs):
                    s_cls_head_feature = pyramid_hint_pair[0]
                    s_reg_head_feature = pyramid_hint_pair[0]

                    # align student/teacher tensor sizes
                    s_cls_head_feature = self.s_t_cls_head_align[-1](
                        s_cls_head_feature)
                    s_reg_head_feature = self.s_t_reg_head_align[-1](
                        s_reg_head_feature)

                    for cls_conv, reg_conv in zip(self.cls_convs,
                                                  self.reg_convs):
                        s_cls_head_feature = cls_conv(s_cls_head_feature)
                        s_reg_head_feature = reg_conv(s_reg_head_feature)

                    t_cls_logits_list.append(self.fcos_cls(s_cls_head_feature))
                    t_bbox_preds_list.append(self.scales[j](
                        self.fcos_reg(s_reg_head_feature)).float().exp())
                flatten_t_cls_scores = [
                    t_cls_logits.permute(0, 2, 3,
                                         1).reshape(-1, self.cls_out_channels)
                    for t_cls_logits in t_cls_logits_list
                ]
                flatten_t_bbox_preds = [
                    t_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)
                    for t_bbox_preds in t_bbox_preds_list
                ]
                flatten_t_cls_scores = torch.cat(flatten_t_cls_scores)
                flatten_t_bbox_preds = torch.cat(flatten_t_bbox_preds)
                t_pos_logits_bboxes = flatten_t_bbox_preds[t_pos_inds]
                t_pos_logits_bboxes = distance2bbox(t_pos_points,
                                                    t_pos_logits_bboxes)

                t_logits_cls = self.loss_cls(
                    flatten_t_cls_scores,
                    flatten_labels,
                    avg_factor=cls_avg_factor)
                t_logits_reg = self.loss_bbox(
                    t_pos_logits_bboxes,
                    t_gt_bboxes,
                    weight=pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())
                loss_dict.update(t_logits_cls=t_logits_cls)
                loss_dict.update(t_logits_reg=t_logits_reg)

            if self.finetune_student:
                if not self.apply_data_free_mode:
                    loss_dict.update(
                        s_loss_bbox=s_loss_bbox,
                        s_loss_centerness=s_loss_centerness,
                        s_loss_cls=s_loss_cls)
                if self.apply_iou_similarity:
                    assert self.spatial_ratio == 1
                    if self.apply_posprocessing_similarity:
                        # TODO: not done yet
                        pass
                    else:
                        loss_iou_similiarity = self.loss_iou_similiarity(
                            s_iou_maps, t_iou_maps.detach())
                        loss_dict.update(
                            loss_iou_similiarity=loss_iou_similiarity)

                if self.apply_soft_regression_distill:
                    # calcuate iou of student boxes with teacher boxes and ground truths
                    # choose the better iou as guidance
                    t_s_ious = bbox_overlaps(
                        s_pred_bboxes, t_pred_bboxes, is_aligned=True)
                    gt_s_ious = bbox_overlaps(
                        s_pred_bboxes, s_gt_bboxes, is_aligned=True)
                    t_iou_inds = (t_s_ious > gt_s_ious).nonzero().reshape(-1)
                    gt_iou_inds = (t_s_ious <= gt_s_ious).nonzero().reshape(-1)

                    if len(t_iou_inds) > 0:
                        # learn from teacher
                        s_distill_loss_bbox = self.loss_bbox(
                            s_pred_bboxes[t_iou_inds],
                            t_pred_bboxes[t_iou_inds],
                            weight=pos_centerness_targets[t_iou_inds],
                            avg_factor=pos_centerness_targets[t_iou_inds].sum(
                            ))
                    else:
                        s_distill_loss_bbox = t_pred_bboxes[t_iou_inds].sum()

                    if self.choose_better_iou:
                        if len(gt_iou_inds) > 0:
                            # learn from ground truth
                            s_loss_bbox = self.loss_bbox(
                                s_pred_bboxes[gt_iou_inds],
                                s_gt_bboxes[gt_iou_inds],
                                weight=pos_centerness_targets[gt_iou_inds],
                                avg_factor=pos_centerness_targets[gt_iou_inds].
                                sum())
                        else:
                            s_loss_bbox = s_gt_bboxes[gt_iou_inds].sum()
                    else:
                        s_loss_bbox = self.loss_bbox(
                            s_pred_bboxes,
                            s_gt_bboxes,
                            weight=pos_centerness_targets,
                            avg_factor=pos_centerness_targets.sum())
                    loss_dict.update(s_distill_loss_bbox=s_distill_loss_bbox)
                    loss_dict.update(s_loss_bbox=s_loss_bbox)

                if self.apply_soft_cls_distill:
                    if self.spatial_ratio > 1:
                        # upsample student to match the size
                        # TODO: currently not use
                        assert True
                    s_tempered_cls_scores = s_flatten_cls_scores / self.temperature
                    s_gt_labels = (t_flatten_cls_scores.detach() /
                                   self.temperature).sigmoid()
                    # CE(KL distance) between teacher and student
                    t_s_distribution_distance = self.t_s_distance(
                        s_tempered_cls_scores, s_gt_labels)
                    t_entropy = -(
                        s_gt_labels * s_gt_labels.log() + (1 - s_gt_labels) *
                        (1 - s_gt_labels).log())
                    adaptive_distillation_weight = (
                        1 - (-t_s_distribution_distance -
                             self.beta * t_entropy).exp())**self.gamma
                    adaptive_distillation_loss = self.adap_distill_loss_weight * (
                        adaptive_distillation_weight *
                        t_s_distribution_distance).sum() / (
                            len(t_pos_inds) + cls_scores[0].size(0))
                    loss_dict.update(
                        adaptive_distillation_loss=adaptive_distillation_loss)
                if self.apply_soft_centerness_distill:
                    t_pred_centerness = t_pred_centerness.detach().sigmoid()
                    # compare teacher and gt
                    self.loss_centerness.reduction = 'none'
                    s_centerness_diff = self.loss_centerness(
                        s_pred_centerness, pos_centerness_targets)
                    s_distill_diff = self.loss_centerness(
                        s_pred_centerness, t_pred_centerness)
                    t_center_inds = (s_distill_diff <
                                     s_centerness_diff).nonzero().reshape(-1)
                    gt_center_inds = (s_distill_diff >=
                                      s_centerness_diff).nonzero().reshape(-1)
                    # get loss
                    self.loss_centerness.reduction = 'mean'
                    if len(gt_center_inds) > 0:
                        s_loss_centerness = self.loss_centerness(
                            s_pred_centerness[gt_center_inds],
                            pos_centerness_targets[gt_center_inds])
                    else:
                        s_loss_centerness = pos_centerness_targets[
                            gt_center_inds].sum()
                    if len(t_center_inds) > 0:
                        s_distill_loss_centerness = self.loss_centerness(
                            s_pred_centerness[t_center_inds],
                            t_pred_centerness[t_center_inds])
                    else:
                        s_distill_loss_centerness = pos_centerness_targets[
                            t_center_inds].sum()
                    loss_dict.update(
                        s_loss_centerness=s_loss_centerness,
                        s_distill_loss_centerness=s_distill_loss_centerness)
                if self.consider_cls_reg_distribution:
                    t_cls_reg_distance = t_flatten_cls_scores[
                        t_pos_inds].sigmoid().max(
                            1)[0] * t_pred_centerness.sigmoid() - t_iou_maps
                    s_cls_reg_distance = s_flatten_cls_scores[
                        t_pos_inds].sigmoid().max(
                            1)[0] * s_pred_centerness.sigmoid() - s_iou_maps
                    cls_reg_dist_loss = self.cls_reg_distribution_hint_loss(
                        s_cls_reg_distance, t_cls_reg_distance)
                    loss_dict.update(cls_reg_dist_loss=cls_reg_dist_loss)
                if self.apply_data_free_mode:
                    df_t_pred_centerness = t_pred_centerness.sigmoid()
                    df_t_confidence, df_t_labels = t_flatten_cls_scores.sigmoid(
                    ).max(1)
                    # select high confidence
                    df_t_labels[(df_t_confidence < 0.3).nonzero()] = 0
                    df_avg_factor = df_t_labels.nonzero().shape[0]
                    df_loss_bbox = self.loss_bbox(
                        s_pred_bboxes,
                        t_pred_bboxes,
                        weight=df_t_pred_centerness,
                        avg_factor=df_t_pred_centerness.sum())
                    df_loss_cls = self.loss_cls(
                        s_flatten_cls_scores,
                        df_t_labels,
                        avg_factor=df_avg_factor)
                    df_loss_centerness = self.loss_centerness(
                        s_pred_centerness, df_t_pred_centerness)
                    loss_dict.update(
                        df_loss_cls=df_loss_cls,
                        df_loss_bbox=df_loss_bbox,
                        df_loss_centerness=df_loss_centerness)
                if self.learn_from_missing_annotation:
                    # learn from high confidence predictions
                    # from teacher network without annotations
                    t_cls_logits, t_learned_labels = t_flatten_cls_scores.sigmoid(
                    ).max(1)
                    t_learned_labels[t_pos_inds] = 0  # mask annotations
                    t_missing_cls_anno_inds = t_learned_labels.nonzero(
                    ).reshape(-1)
                    recovered_anno_mask = t_missing_cls_anno_inds[(
                        t_cls_logits[t_missing_cls_anno_inds] >=
                        0.75).nonzero().reshape(-1)]
                    recovered_pos_avg_factor = len(recovered_anno_mask)
                    if recovered_pos_avg_factor == 0:
                        recovered_loss_bboxes = t_all_pred_bboxes[
                            recovered_anno_mask].sum()
                        recovered_loss_cls = s_flatten_cls_scores[
                            recovered_anno_mask].sum()
                    else:
                        recovered_loss_weight = 1
                        recovered_loss_bboxes = recovered_loss_weight * self.loss_bbox(
                            s_all_pred_bboxes[recovered_anno_mask],
                            t_all_pred_bboxes[recovered_anno_mask])
                        recovered_loss_cls = recovered_loss_weight * self.loss_cls(
                            s_flatten_cls_scores[recovered_anno_mask],
                            t_learned_labels[recovered_anno_mask],
                            avg_factor=recovered_pos_avg_factor)
                    loss_dict.update(
                        recovered_loss_cls=recovered_loss_cls,
                        recovered_loss_bboxes=recovered_loss_bboxes)
                if not self.freeze_teacher:
                    loss_dict.update(
                        loss_cls=loss_cls,
                        loss_bbox=loss_bbox,
                        loss_centerness=loss_centerness)

        elif self.fix_student_train_teacher:
            loss_dict.update(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness)
        else:
            loss_dict.update(
                loss_cls=loss_cls,
                s_loss_cls=s_loss_cls,
                adaptive_distillation_loss=adaptive_distillation_loss,
                loss_bbox=loss_bbox,
                s_loss_bbox=s_loss_bbox,
                loss_centerness=loss_centerness,
                s_loss_centerness=s_loss_centerness,
                loss_s_t_reg=loss_s_t_reg)

        return loss_dict

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    gt_bboxes,
                    gt_labels,
                    img_metas,
                    cfg,
                    gt_bboxes_ignore=None,
                    spatial_ratio=1):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        # Handle down sampled spatial size of student model
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device, spatial_ratio)
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        block_distill_masks = []
        if self.block_teacher_attention:
            for i, label in enumerate(labels):
                distill_masks = (label.reshape(
                    num_imgs, 1, featmap_sizes[i][0], featmap_sizes[i][1]) >
                    0).float()
                block_distill_masks.append(
                    torch.nn.functional.upsample(
                        distill_masks, size=featmap_sizes[0]))
            block_distill_masks = torch.cat(block_distill_masks,
                                            1).sum(1).unsqueeze(1)
            block_distill_masks = (block_distill_masks > 0).float()

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        cls_avg_factor = num_pos + num_imgs
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=cls_avg_factor)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        decoded_bbox_preds = distance2bbox(flatten_points, flatten_bbox_preds)

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            '''
            Generate IoU map of Teacher and Student model
            '''
            if self.apply_posprocessing_similarity:
                is_aligned = True
            else:
                is_aligned = False
            pos_iou_maps = bbox_overlaps(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                is_aligned=is_aligned)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return loss_cls, loss_bbox, loss_centerness, cls_avg_factor, flatten_cls_scores, flatten_labels, pos_iou_maps, pos_inds, pos_decoded_bbox_preds, pos_decoded_target_preds, block_distill_masks, pos_centerness_targets, pos_centerness, decoded_bbox_preds, pos_points

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None,
                   require_grad=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        if self.eval_student:
            mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                          bbox_preds[0].device,
                                          self.spatial_ratio)
        else:
            mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                          bbox_preds[0].device)
        result_list = []
        if require_grad:
            for img_id in range(len(img_metas)):
                cls_score_list = [
                    cls_scores[i][img_id] for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id] for i in range(num_levels)
                ]
                centerness_pred_list = [
                    centernesses[i][img_id] for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor']
                det_bboxes = self.get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    centerness_pred_list,
                                                    mlvl_points, img_shape,
                                                    scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        else:
            for img_id in range(len(img_metas)):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                centerness_pred_list = [
                    centernesses[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor']
                det_bboxes = self.get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    centerness_pred_list,
                                                    mlvl_points, img_shape,
                                                    scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device, spatio_ratio=1):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i],
                                       self.strides[i] * spatio_ratio, dtype,
                                       device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
