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
                 apply_soft_regression_distill=False,
                 apply_adaptive_distillation=False,
                 temperature=1,
                 apply_feature_alignment=False,
                 fix_student_train_teacher=False,
                 train_student_only=False,
                 align_level=1,
                 apply_block_wise_alignment=False,
                 apply_pyramid_wise_alignment=False,
                 apply_head_wise_alignment=False,
                 cosine_similarity=False,
                 block_teacher_attention=False,
                 head_teacher_reg_attention=False,
                 teacher_iou_attention=False,
                 attention_threshold=0.5,
                 freeze_teacher=False,
                 beta=1,
                 gamma=1,
                 adap_distill_loss_weight=0.5,
                 t_hint_loss=dict(type='MSELoss', loss_weight=1),
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
        self.apply_head_wise_alignment = apply_head_wise_alignment
        self.cosine_similarity = cosine_similarity
        self.block_teacher_attention = block_teacher_attention
        self.head_teacher_reg_attention = head_teacher_reg_attention
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
        self.apply_soft_regression_distill = apply_soft_regression_distill
        self.apply_adaptive_distillation = apply_adaptive_distillation
        self.temperature = temperature
        self.apply_feature_alignment = apply_feature_alignment
        self.fix_student_train_teacher = fix_student_train_teacher
        self.train_student_only = train_student_only
        self.t_hint_loss = build_loss(t_hint_loss)
        self.pyramid_hint_loss = build_loss(pyramid_hint_loss)
        self.reg_head_hint_loss = build_loss(reg_head_hint_loss)
        self.cls_head_hint_loss = build_loss(cls_head_hint_loss)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_s_t_cls = build_loss(loss_s_t_cls)
        self.loss_s_t_reg = build_loss(loss_s_t_reg)
        self.t_s_distance = build_loss(t_s_distance)
        self.loss_regression_distill = build_loss(loss_regression_distill)
        self.reg_distill_threshold = reg_distill_threshold
        self.loss_iou_similiarity = nn.BCELoss(
        )  #build_loss(loss_iou_similiarity)
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
        # TODO: Align each level individually?
        '''
        self.t_s_cls_align = nn.Conv2d(
            self.s_feat_channels, self.feat_channels, 3, padding=1)
        self.t_s_reg_align = nn.Conv2d(
            self.s_feat_channels, self.feat_channels, 3, padding=1)

        if self.apply_pyramid_wise_alignment:
            self.t_s_pyramid_align = nn.Conv2d(
                self.s_feat_channels, self.feat_channels, 3, padding=1)

        if self.apply_head_wise_alignment:
            for i in range(self.stacked_convs):
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
        normal_init(self.t_s_cls_align, std=0.01)
        normal_init(self.t_s_reg_align, std=0.01)
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
        self.fcos_cls.requires_grad = False
        self.fcos_reg.eval()
        self.fcos_reg.requires_grad = False
        self.fcos_centerness.eval()
        self.fcos_centerness.requires_grad = False

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

        for i in range(len(self.cls_convs)):
            cls_layer = self.cls_convs[i]
            cls_s_layer = self.s_cls_convs[i]
            t_cls_feat = cls_layer(t_cls_feat)
            s_cls_feat = cls_s_layer(s_cls_feat)

            if self.apply_head_wise_alignment:
                cls_hint_pairs.append([s_cls_feat, t_cls_feat])

            if i == self.align_level:
                s_align_cls_feat = s_cls_feat
                t_aligned_cls_feat = t_cls_feat

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

            if j == self.align_level:
                s_align_reg_feat = s_reg_feat
                t_aligned_reg_feat = t_reg_feat

        # wrap reg/cls head features
        if self.apply_head_wise_alignment:
            head_hint_pairs = []
            head_hint_pairs.append(cls_hint_pairs)
            head_hint_pairs.append(reg_hint_pairs)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(t_reg_feat)).float().exp()
        s_bbox_pred = s_scale(self.fcos_s_reg(s_reg_feat)).float().exp()
        '''
        # feature align to teacher
        '''
        if self.spatial_ratio == 1:
            s_align_reg_feat = self.t_s_reg_align(s_align_reg_feat)
            s_align_cls_feat = self.t_s_cls_align(s_align_cls_feat)
        else:
            # upsample + conv
            s_align_reg_feat = self.t_s_reg_align(
                F.interpolate(
                    s_align_reg_feat,
                    size=t_aligned_reg_feat.shape[2:],
                    mode='nearest'))
            s_align_cls_feat = self.t_s_cls_align(
                F.interpolate(
                    s_align_cls_feat,
                    size=t_aligned_reg_feat.shape[2:],
                    mode='nearest'))
        if self.training:
            if self.apply_pyramid_wise_alignment and not self.apply_head_wise_alignment:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, t_aligned_cls_feat, s_align_cls_feat, t_aligned_reg_feat, s_align_reg_feat, hint_pairs, pyramid_hint_pairs, None
            elif self.apply_head_wise_alignment and not self.apply_pyramid_wise_alignment:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, t_aligned_cls_feat, s_align_cls_feat, t_aligned_reg_feat, s_align_reg_feat, hint_pairs, None, head_hint_pairs
            elif self.apply_pyramid_wise_alignment and self.apply_head_wise_alignment:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, t_aligned_cls_feat, s_align_cls_feat, t_aligned_reg_feat, s_align_reg_feat, hint_pairs, pyramid_hint_pairs, head_hint_pairs
            else:
                return cls_score, bbox_pred, centerness, s_cls_score, s_bbox_pred, s_centerness, t_aligned_cls_feat, s_align_cls_feat, t_aligned_reg_feat, s_align_reg_feat, hint_pairs, None
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
             t_cls_feats,
             s_cls_feats,
             reg_feats,
             s_reg_feats,
             hint_pairs,
             pyramid_hint_pairs,
             head_hint_pairs,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        loss_cls, loss_bbox, loss_centerness, _, t_flatten_cls_scores, t_iou_maps, t_pos_inds, t_pred_bboxes, t_gt_bboxes, block_distill_masks, _ = self.loss_single(
            cls_scores,
            bbox_preds,
            centernesses,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore=None)
        s_hard_loss_cls, s_loss_bbox, s_loss_centerness, cls_avg_factor, s_cls_scores, s_iou_maps, _, s_pred_bboxes, s_gt_bboxes, _, pos_centerness_targets = self.loss_single(
            s_cls_scores,
            s_bbox_preds,
            s_centernesses,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore=None,
            spatial_ratio=self.spatial_ratio)

        flatten_s_cls_feat = [
            s_cls_feat.permute(0, 2, 3, 1).reshape(-1, self.s_feat_channels)
            for s_cls_feat in s_cls_feats
        ]
        flatten_t_cls_feat = [
            t_cls_feat.permute(0, 2, 3, 1).reshape(-1, self.s_feat_channels)
            for t_cls_feat in t_cls_feats
        ]
        flatten_s_reg_feat = [
            s_reg_feat.permute(0, 2, 3, 1).reshape(-1, self.s_feat_channels)
            for s_reg_feat in s_reg_feats
        ]
        flatten_t_reg_feat = [
            t_reg_feat.permute(0, 2, 3, 1).reshape(-1, self.s_feat_channels)
            for t_reg_feat in reg_feats
        ]
        flatten_s_cls_feat = torch.cat(flatten_s_cls_feat)
        flatten_t_cls_feat = torch.cat(flatten_t_cls_feat)
        flatten_s_reg_feat = torch.cat(flatten_s_reg_feat)
        flatten_t_reg_feat = torch.cat(flatten_t_reg_feat)

        loss_dict = {}
        assert self.fix_student_train_teacher != self.learn_when_train

        if self.learn_when_train:
            loss_dict.update(s_hard_loss_cls=s_hard_loss_cls)
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
                        hint_loss = self.t_hint_loss(
                            s_block_feature,
                            t_block_feature,
                            weight=attention_weight)
                    else:
                        hint_loss = self.t_hint_loss(s_block_feature,
                                                     t_block_feature)
                    loss_dict.update(
                        {'hint_loss_block_{}'.format(j): hint_loss})
            if self.apply_pyramid_wise_alignment:
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

                    pyramid_hint_loss = self.pyramid_hint_loss(
                        s_pyramid_feature, t_pyramid_feature)
                    loss_dict.update({
                        'pyramid_hint_loss_block_{}'.format(j):
                        pyramid_hint_loss
                    })
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
                    for k in range(self.stacked_convs):
                        s_cls_head_feature = self.s_t_cls_head_align[k](
                            cls_head_pair[k][0])
                        s_reg_head_feature = self.s_t_reg_head_align[k](
                            reg_head_pair[k][0])
                        t_cls_head_feature = cls_head_pair[k][1].detach()
                        t_reg_head_feature = reg_head_pair[k][1].detach()

                        t_cls_head_feature_list.append(t_cls_head_feature)
                        s_cls_head_feature_list.append(s_cls_head_feature)
                        t_reg_head_feature_list.append(t_reg_head_feature)
                        s_reg_head_feature_list.append(s_reg_head_feature)

                    t_cls_heads_feature_list.append(
                        torch.cat(t_cls_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      self.stacked_convs * self.feat_channels))
                    s_cls_heads_feature_list.append(
                        torch.cat(s_cls_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      self.stacked_convs * self.feat_channels))
                    t_reg_heads_feature_list.append(
                        torch.cat(t_reg_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      self.stacked_convs * self.feat_channels))
                    s_reg_heads_feature_list.append(
                        torch.cat(s_reg_head_feature_list,
                                  1).permute(0, 2, 3, 1).reshape(
                                      -1,
                                      self.stacked_convs * self.feat_channels))
                t_cls_heads_feature_list = torch.cat(t_cls_heads_feature_list,
                                                     0)
                s_cls_heads_feature_list = torch.cat(s_cls_heads_feature_list,
                                                     0)
                t_reg_heads_feature_list = torch.cat(t_reg_heads_feature_list,
                                                     0)
                s_reg_heads_feature_list = torch.cat(s_reg_heads_feature_list,
                                                     0)
                if not self.head_teacher_reg_attention:
                    reg_head_hint_loss = self.reg_head_hint_loss(
                        s_reg_heads_feature_list, t_reg_heads_feature_list)
                else:
                    pos_t_reg_heads_feature = t_reg_heads_feature_list[
                        t_pos_inds]
                    pos_s_reg_heads_feature = s_reg_heads_feature_list[
                        t_pos_inds]
                    select_update_reg_inds = (t_iou_maps >
                                              s_iou_maps).nonzero().reshape(-1)
                    if self.cosine_similarity is True:
                        self.reg_head_hint_loss = torch.nn.CosineEmbeddingLoss(
                            margin=0.2, reduction='mean')
                        reg_head_hint_loss = self.reg_head_hint_loss(
                            pos_s_reg_heads_feature, pos_t_reg_heads_feature,
                            torch.ones_like(pos_t_reg_heads_feature[:, 0]))
                    else:
                        if len(select_update_reg_inds) > 0:
                            reg_head_hint_loss = self.reg_head_hint_loss(
                                pos_s_reg_heads_feature[
                                    select_update_reg_inds],
                                pos_t_reg_heads_feature[select_update_reg_inds]
                            )
                        else:
                            reg_head_hint_loss = pos_t_reg_heads_feature[
                                select_update_reg_inds].sum()
                # cls_head_hint_loss = self.cls_head_hint_loss(
                #     s_cls_heads_feature_list, t_cls_heads_feature_list)
                loss_dict.update({'reg_head_hint_loss': reg_head_hint_loss})
                # loss_dict.update({'cls_head_hint_loss': cls_head_hint_loss})

            if self.apply_feature_alignment:
                if str(self.loss_s_t_cls) == 'MSELoss()':
                    loss_s_t_reg = self.loss_s_t_reg(
                        flatten_s_reg_feat[t_pos_inds],
                        flatten_t_reg_feat[t_pos_inds].detach())
                elif str(self.loss_s_t_cls) == 'CrossEntropyLoss()':
                    loss_s_t_reg = self.loss_s_t_reg(
                        flatten_s_reg_feat[t_pos_inds],
                        flatten_t_reg_feat[t_pos_inds].detach().sigmoid())
                loss_dict.update(loss_s_t_reg=loss_s_t_reg)
            if self.finetune_student:
                loss_dict.update(
                    s_loss_bbox=s_loss_bbox,
                    s_loss_centerness=s_loss_centerness,
                    s_hard_loss_cls=s_hard_loss_cls)
                if self.apply_iou_similarity:
                    assert self.spatial_ratio == 1
                    loss_iou_similiarity = self.loss_iou_similiarity(
                        s_iou_maps, t_iou_maps.detach())
                    loss_dict.update(loss_iou_similiarity=loss_iou_similiarity)
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
                    if len(gt_iou_inds) > 0:
                        # learn from ground truth
                        s_loss_bbox = self.loss_bbox(
                            s_pred_bboxes[gt_iou_inds],
                            t_pred_bboxes[gt_iou_inds],
                            weight=pos_centerness_targets[gt_iou_inds],
                            avg_factor=pos_centerness_targets[gt_iou_inds].sum(
                            ))
                    else:
                        s_loss_bbox = t_pred_bboxes[gt_iou_inds].sum()

                    loss_dict.update(s_distill_loss_bbox=s_distill_loss_bbox)
                    loss_dict.update(s_loss_bbox=s_loss_bbox)
                if self.apply_adaptive_distillation:
                    if self.spatial_ratio > 1:
                        # upsample student to match the size
                        # TODO: currently not use
                        assert True
                    s_tempered_cls_scores = s_cls_scores / self.temperature
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
                if self.train_teacher:
                    # currently duplicate
                    assert self.train_teacher != self.freeze_teacher
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
                s_hard_loss_cls=s_hard_loss_cls,
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
            pos_iou_maps = bbox_overlaps(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                is_aligned=True)
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

        return loss_cls, loss_bbox, loss_centerness, cls_avg_factor, flatten_cls_scores, pos_iou_maps, pos_inds, pos_decoded_bbox_preds, pos_decoded_target_preds, block_distill_masks, pos_centerness_targets

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
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
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
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
