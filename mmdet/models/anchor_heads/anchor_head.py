from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from ..builder import build_loss
from ..registry import HEADS

import math
from IPython import embed


@HEADS.register_module
class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories including the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 s_feat_channels=128,
                 dynamic_weight=False,
                 pyramid_wise_attention=False,
                 feature_adaption=False,
                 norm_pyramid=False,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
                 apply_block_wise_alignment=False,
                 multi_stage_train=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.s_feat_channels = s_feat_channels
        self.pyramid_wise_attention = pyramid_wise_attention
        self.feature_adaption = feature_adaption
        self.norm_pyramid = norm_pyramid
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.train_step = 0
        self.dynamic_weight = dynamic_weight

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))

        self.loss_cls = build_loss(loss_cls)
        self.pyramid_hint_loss = build_loss(pyramid_hint_loss)
        self.loss_bbox = build_loss(loss_bbox)
        self.apply_block_wise_alignment = apply_block_wise_alignment
        self.multi_stage_train = multi_stage_train
        self.fp16_enabled = False

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x, s_x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        if type(feats) is tuple:
            # branch for distillation
            if self.pure_student_term and not self.eval_student:
                return multi_apply(self.forward_single, feats[0], feats[1],
                                   feats[4])
            else:
                return multi_apply(self.forward_single, feats[0], feats[1])
        else:
            return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # NOTE: if cls scores/ bbox preds are tuples, it's distillation mode
        # cls_score tuple: [cls_score, s_cls_score, x, s_x],
        # bbox_pred tuple: [bbox_pred, s_bbox_pred]
        # classification loss

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        if self.dynamic_weight:
            if self.norm_pyramid:
                attention_lambda = 1000 + 1200 * (self.train_step // 7330)
                # attention_lambda = 500 + 20 * (self.train_step // 7330)
                pyramid_lambda = 100
            else:
                attention_lambda = 15.0 / (
                    1.0 + math.exp(-2 * (self.train_step // 7330 - 1)))
                # pyramid_lambda = 8 / (1 + 0.33 * self.train_step // 7330)
                pyramid_lambda = 8.0
        else:
            if self.norm_pyramid:
                attention_lambda = 2000
                pyramid_lambda = 100
            else:
                attention_lambda = 15.0
                pyramid_lambda = 1.0

        if type(cls_score) is tuple:
            t_cls_score = cls_score[0].permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels)
            s_cls_score = cls_score[1].permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels)
            if self.adapt_on_channel:
                adapt_channels = (
                    self.feat_channels - self.s_feat_channels) // 2 + self.s_feat_channels
                x_feats = cls_score[2].permute(0, 2, 3,
                                               1).reshape(-1, adapt_channels)
                s_x_feats = cls_score[3].permute(0, 2, 3, 1).reshape(
                    -1, adapt_channels)
            else:
                x_feats = cls_score[2].permute(0, 2, 3,
                                               1).reshape(-1, self.feat_channels)
                s_x_feats = cls_score[3].permute(0, 2, 3, 1).reshape(
                    -1, self.feat_channels)

                if self.feature_adaption:
                    # adaption_factor = 0.5
                    adaption_factor = self.train_step / (7330 * 12)
                    s_x_feats = adaption_factor * s_x_feats + \
                        (1 - adaption_factor) * x_feats

            if self.pure_student_term:
                s_pure_cls_score = cls_score[4].permute(0, 2, 3, 1).reshape(
                    -1, self.cls_out_channels)
        else:
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)
        if type(cls_score) is tuple:
            t_loss_cls = self.loss_cls(
                t_cls_score,
                labels,
                label_weights,
                avg_factor=num_total_samples)
            s_loss_cls = self.loss_cls(
                s_cls_score,
                labels,
                label_weights,
                avg_factor=num_total_samples)
            if self.pure_student_term:
                s_pure_loss_cls = self.loss_cls(
                    s_pure_cls_score,
                    labels,
                    label_weights,
                    avg_factor=num_total_samples)
        else:
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)

        if type(cls_score) is tuple:
            t_bbox_pred = bbox_pred[0].permute(0, 2, 3, 1).reshape(-1, 4)
            s_bbox_pred = bbox_pred[1].permute(0, 2, 3, 1).reshape(-1, 4)
            if self.pure_student_term:
                s_pure_bbox_pred = bbox_pred[2].permute(0, 2, 3,
                                                        1).reshape(-1, 4)

            if self.norm_pyramid:
                s_x_feats = F.normalize(s_x_feats, p=2, dim=1)
                x_feats = F.normalize(x_feats, p=2, dim=1)

            pyramid_hint_loss = pyramid_lambda * self.pyramid_hint_loss(
                s_x_feats, x_feats.detach())

            if self.pyramid_wise_attention:
                bbox_num_pos = bbox_weights.reshape(self.num_anchors, -1,
                                                    4).sum(0).sum(1)
                pos_bbox_inds = bbox_num_pos.nonzero().reshape(-1)
                t_pos_bbox_pred = t_bbox_pred[pos_bbox_inds]
                s_pos_bbox_pred = s_bbox_pred[pos_bbox_inds]
                anchors_weights = bbox_num_pos[pos_bbox_inds] / 4

                if len(t_pos_bbox_pred) > 0:
                    bbox_distance = torch.abs(t_pos_bbox_pred -
                                              s_pos_bbox_pred).sum(1)
                    attention_weight = (
                        1 - bbox_distance / bbox_distance.max()).detach()
                    # attention_weight *= anchors_weights

                    pos_attention_pyramid_hint_loss = attention_lambda * self.pyramid_hint_loss(
                        s_x_feats[pos_bbox_inds],
                        x_feats[pos_bbox_inds].detach(),
                        weight=attention_weight)

                else:
                    pos_attention_pyramid_hint_loss = t_pos_bbox_pred.sum()

            t_loss_bbox = self.loss_bbox(
                t_bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)

            t_g_bbox_dists = torch.abs(bbox_targets - t_bbox_pred).mean(1)
            high_qual_t_bbox_mask = (
                t_g_bbox_dists <= t_g_bbox_dists.mean()).float().view(-1, 1)
            if self.t_low_bbox_mask:
                s_loss_bbox = self.loss_bbox(
                    s_bbox_pred,
                    bbox_targets,
                    bbox_weights * high_qual_t_bbox_mask,
                    avg_factor=num_total_samples)
            else:
                s_loss_bbox = self.loss_bbox(
                    s_bbox_pred,
                    bbox_targets,
                    bbox_weights,
                    avg_factor=num_total_samples)

            if self.pure_student_term:
                s_pure_loss_bbox = self.loss_bbox(
                    s_pure_bbox_pred,
                    bbox_targets,
                    bbox_weights,
                    avg_factor=num_total_samples)

        else:
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)

        if type(cls_score) is tuple:
            if self.pyramid_wise_attention:
                if self.finetune_student:
                    return s_loss_cls, s_loss_bbox, pyramid_hint_loss, pos_attention_pyramid_hint_loss
                else:
                    if self.pure_student_term:
                        return t_loss_cls, s_loss_cls, t_loss_bbox, s_loss_bbox, pyramid_hint_loss, pos_attention_pyramid_hint_loss, s_pure_loss_bbox
                    else:
                        # if self.train_step >= 7330 * 3:
                        return t_loss_cls, s_loss_cls, t_loss_bbox, s_loss_bbox, pyramid_hint_loss, pos_attention_pyramid_hint_loss
                        # else:
                        #     return t_loss_cls, t_loss_bbox,  # s_loss_cls, s_loss_bbox, pyramid_hint_loss, pos_attention_pyramid_hint_loss
            else:
                if self.finetune_student:
                    return s_loss_cls, s_loss_bbox, pyramid_hint_loss
                else:
                    return t_loss_cls, s_loss_cls, t_loss_bbox, s_loss_bbox, pyramid_hint_loss
        else:
            return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        self.train_step += 1
        if type(cls_scores[0]) is tuple:
            # TODO: add settings and anchor targets for
            # different feature sizes between teacher and student
            featmap_sizes = [featmap[0].size()[-2:] for featmap in cls_scores]
            device = cls_scores[0][0].device
        else:
            featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
            device = cls_scores[0].device

        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        loss_dict = {}
        if len(cls_scores[0]) > 2:
            # NOTE: well. distillation mode
            if self.pyramid_wise_attention:
                if self.finetune_student:
                    s_loss_cls, s_loss_bbox, pyramid_hint_loss, pos_attention_pyramid_hint_loss = multi_apply(
                        self.loss_single,
                        cls_scores,
                        bbox_preds,
                        labels_list,
                        label_weights_list,
                        bbox_targets_list,
                        bbox_weights_list,
                        num_total_samples=num_total_samples,
                        cfg=cfg)
                    if self.multi_stage_train and self.train_step < 7330 * 12: # 7330 * 6:
                        loss_dict.update({
                            'pyramid_hint_loss':
                            pyramid_hint_loss,
                            'pos_attention_pyramid_hint_loss':
                            pos_attention_pyramid_hint_loss
                        })
                    else:
                        loss_dict.update({
                            's_loss_cls':
                            s_loss_cls,
                            's_loss_bbox':
                            s_loss_bbox,
                            'pyramid_hint_loss':
                            pyramid_hint_loss,
                            'pos_attention_pyramid_hint_loss':
                            pos_attention_pyramid_hint_loss
                        })
                    return loss_dict
                else:
                    if self.pure_student_term:
                        t_loss_cls, s_loss_cls, t_loss_bbox, s_loss_bbox, pyramid_hint_loss, pos_attention_pyramid_hint_loss, s_pure_loss_bbox = multi_apply(
                            self.loss_single,
                            cls_scores,
                            bbox_preds,
                            labels_list,
                            label_weights_list,
                            bbox_targets_list,
                            bbox_weights_list,
                            num_total_samples=num_total_samples,
                            cfg=cfg)
                        loss_dict.update({
                            't_loss_cls': t_loss_cls,
                            't_loss_bbox': t_loss_bbox,
                            's_loss_cls': s_loss_cls,
                            's_loss_bbox': s_loss_bbox,
                            'pyramid_hint_loss': pyramid_hint_loss,
                            'pos_attention_pyramid_hint_loss':
                            pos_attention_pyramid_hint_loss,
                            's_pure_loss_bbox': s_pure_loss_bbox,
                        })
                        return loss_dict
                    else:
                        # if self.train_step >= 7330 * 3:
                        t_loss_cls, s_loss_cls, t_loss_bbox, s_loss_bbox, pyramid_hint_loss, pos_attention_pyramid_hint_loss = multi_apply(
                            self.loss_single,
                            cls_scores,
                            bbox_preds,
                            labels_list,
                            label_weights_list,
                            bbox_targets_list,
                            bbox_weights_list,
                            num_total_samples=num_total_samples,
                            cfg=cfg)
                        if self.multi_stage_train and self.train_step < 7330 * 6:
                            loss_dict.update({
                                't_loss_cls':
                                t_loss_cls,
                                't_loss_bbox':
                                t_loss_bbox,
                                'pyramid_hint_loss':
                                pyramid_hint_loss,
                                'pos_attention_pyramid_hint_loss':
                                pos_attention_pyramid_hint_loss
                            })
                        else:
                            loss_dict.update({
                                't_loss_cls':
                                t_loss_cls,
                                't_loss_bbox':
                                t_loss_bbox,
                                's_loss_cls':
                                s_loss_cls,
                                's_loss_bbox':
                                s_loss_bbox,
                                'pyramid_hint_loss':
                                pyramid_hint_loss,
                                'pos_attention_pyramid_hint_loss':
                                pos_attention_pyramid_hint_loss
                            })
                        '''
                        else:
                            t_loss_cls, t_loss_bbox = multi_apply(
                                self.loss_single,
                                cls_scores,
                                bbox_preds,
                                labels_list,
                                label_weights_list,
                                bbox_targets_list,
                                bbox_weights_list,
                                num_total_samples=num_total_samples,
                                cfg=cfg)
                            loss_dict.update({
                                't_loss_cls': t_loss_cls,
                                't_loss_bbox': t_loss_bbox
                            })
                        '''
                        return loss_dict
            else:
                if self.finetune_student:
                    s_loss_cls, s_loss_bbox, pyramid_hint_loss = multi_apply(
                        self.loss_single,
                        cls_scores,
                        bbox_preds,
                        labels_list,
                        label_weights_list,
                        bbox_targets_list,
                        bbox_weights_list,
                        num_total_samples=num_total_samples,
                        cfg=cfg)
                    loss_dict.update({
                        's_loss_cls': s_loss_cls,
                        's_loss_bbox': s_loss_bbox,
                        'pyramid_hint_loss': pyramid_hint_loss
                    })
                    return loss_dict

                else:
                    t_loss_cls, s_loss_cls, t_loss_bbox, s_loss_bbox, pyramid_hint_loss = multi_apply(
                        self.loss_single,
                        cls_scores,
                        bbox_preds,
                        labels_list,
                        label_weights_list,
                        bbox_targets_list,
                        bbox_weights_list,
                        num_total_samples=num_total_samples,
                        cfg=cfg)
                    loss_dict.update({
                        't_loss_cls': t_loss_cls,
                        's_loss_cls': s_loss_cls,
                        't_loss_bbox': t_loss_bbox,
                        's_loss_bbox': s_loss_bbox,
                        'pyramid_hint_loss': pyramid_hint_loss
                    })
                    return loss_dict

        else:
            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples,
                cfg=cfg)
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        """
        Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(num_classes=9, in_channels=1)
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
