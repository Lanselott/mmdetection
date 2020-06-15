import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead
from IPython import embed


@HEADS.register_module
class RetinaTSHead(AnchorHead):
    """
    An anchor-based head used in [1]_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 eval_student=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.eval_student = eval_student
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaTSHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.s_cls_convs = nn.ModuleList()
        self.s_reg_convs = nn.ModuleList()
        self.s_t_align_convs = nn.ModuleList()

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
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        for i in range(self.stacked_convs):
            # chn = self.in_channels if i == 0 else self.s_feat_channels
            self.s_cls_convs.append(
                ConvModule(
                    self.s_feat_channels,
                    self.s_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.s_reg_convs.append(
                ConvModule(
                    self.s_feat_channels,
                    self.s_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        self.s_retina_cls = nn.Conv2d(
            self.s_feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.s_retina_reg = nn.Conv2d(
            self.s_feat_channels, self.num_anchors * 4, 3, padding=1)
        self.s_t_align_convs.append(
            nn.Conv2d(self.s_feat_channels, self.feat_channels, 3, padding=1))

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.s_cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.s_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.s_t_align_convs:
            normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
        normal_init(self.s_retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.s_retina_reg, std=0.01)

    def forward_single(self, x, s_x, aligned_fpn=None):
        cls_feat = x
        reg_feat = x
        s_cls_feat = s_x
        s_reg_feat = s_x
        
        # align to teacher
        for s_t_align_conv in self.s_t_align_convs:
            s_x = s_t_align_conv(s_x)

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        for s_cls_conv in self.s_cls_convs:
            s_cls_feat = s_cls_conv(s_cls_feat)
        for s_reg_conv in self.s_reg_convs:
            s_reg_feat = s_reg_conv(s_reg_feat)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        s_cls_score = self.s_retina_cls(s_cls_feat)
        s_bbox_pred = self.s_retina_reg(s_reg_feat)
        if self.eval_student and not self.training:
            return s_cls_score, s_bbox_pred
        elif not self.eval_student and not self.training:
            return cls_score, bbox_pred
        elif self.training:
            return tuple([cls_score, s_cls_score, x,
                          s_x]), tuple([bbox_pred, s_bbox_pred])
