import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F

from mmdet.core import multi_apply, multiclass_nms, multiclass_nms_sorting, distance2bbox, bbox2delta, bbox_overlaps
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule
from mmdet.ops import DeformConv, MaskedConv2d

from IPython import embed

INF = 1e8
MIN = 1e-8


@HEADS.register_module
class DDBV3Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 apply_conditional_consistency_on_regression=False,
                 use_deformable=False,
                 mask_origin_bbox_loss=False,
                 iou_delta=0.0,
                 apply_iou_cache=False,
                 mask_sort=False,
                 weighted_mask=False,
                 weight_balance=False,
                 weight_balance_threshold=0.5,
                 consistency_weight=False,
                 apply_consistency_on_cls=True,
                 normalize_centerness=False,
                 apply_cls_ignore_area=True,
                 cls_aware=False,
                 sorted_warmup=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_sorted_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_dist_scores=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 bd_threshold=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(DDBV3Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.apply_conditional_consistency_on_regression = apply_conditional_consistency_on_regression
        self.use_deformable = use_deformable
        self.mask_origin_bbox_loss = mask_origin_bbox_loss
        self.iou_delta = iou_delta
        self.apply_iou_cache = apply_iou_cache
        self.mask_sort = mask_sort
        self.weighted_mask = weighted_mask
        self.weight_balance = weight_balance
        self.weight_balance_threshold = weight_balance_threshold
        self.consistency_weight = consistency_weight
        self.apply_consistency_on_cls = apply_consistency_on_cls
        self.normalize_centerness = normalize_centerness
        self.apply_cls_ignore_area = apply_cls_ignore_area
        self.cls_aware = cls_aware
        self.sorted_warmup = sorted_warmup
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_sorted_bbox = build_loss(loss_sorted_bbox)
        self.loss_dist_scores = build_loss(loss_dist_scores)
        self.bd_threshold = bd_threshold
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.sc_image_counter = 0
        self.sc_avg_ratio = 0

        self._init_layers()

    def _init_layers(self):
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
        if not self.use_deformable:
            self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
            self.fcos_centerness = nn.Conv2d(
                self.feat_channels, 1, 3, padding=1)
        else:
            kernel_size = 3
            deformable_groups = 1
            offset_channels = kernel_size * kernel_size * 2

            self.reg_offset = nn.Conv2d(
                self.feat_channels,
                deformable_groups * offset_channels,
                1,
                bias=False)
            self.reg_adaption = DeformConv(
                self.feat_channels,
                4,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                deformable_groups=deformable_groups)

            self.centerness_offset = nn.Conv2d(
                self.feat_channels,
                deformable_groups * offset_channels,
                1,
                bias=False)
            self.centerness_adaption = DeformConv(
                self.feat_channels,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)

        if self.use_deformable:
            normal_init(self.reg_offset, std=0.1)
            normal_init(self.reg_adaption, std=0.01)
            normal_init(self.centerness_offset, std=0.1)
            normal_init(self.centerness_adaption, std=0.01)
        else:
            normal_init(self.fcos_reg, std=0.01)
            normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        cls_score = self.fcos_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # trick: centerness to reg branch
        if not self.use_deformable:
            centerness = self.fcos_centerness(reg_feat)
        else:
            centerness_offset = self.centerness_offset(reg_feat.detach())
            centerness = self.centerness_adaption(reg_feat, centerness_offset)
        '''
        # scale the bbox_pred of different level
        bbox_pred = scale(self.fcos_reg(reg_feat)).exp()
        '''
        '''
        # tricks in https://github.com/tianzhi0549/FCOS/blob/4697f876b5aeead5f60423e0d04ea7e3e1282790/fcos_core/modeling/rpn/fcos/fcos.py
        '''
        if not self.use_deformable:
            bbox_pred = scale(self.fcos_reg(reg_feat))
        else:
            reg_offset = self.reg_offset(reg_feat.detach())
            bbox_pred = scale(self.reg_adaption(reg_feat, reg_offset))

        bbox_pred = self.relu(bbox_pred)

        return cls_score, bbox_pred, centerness

    # def regression_hook()

    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        dist_conf_mask_list = []

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        '''
        normalized bbox: bbox_targets
        origin bbox: bbox_origins
        '''
        labels, bbox_targets, bbox_strided_targets, bbox_strides = self.fcos_target(
            all_level_points, gt_bboxes, gt_labels)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds
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
        flatten_bbox_strided_targets = torch.cat(bbox_strided_targets)
        flatten_bbox_strides = torch.cat(bbox_strides)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_bbox_strided_targets = flatten_bbox_strided_targets[pos_inds]
        pos_bbox_strides = flatten_bbox_strides[pos_inds]

        pos_centerness = flatten_centerness[pos_inds]
        pos_labels = flatten_labels[pos_inds]
        '''
        # BUG: do not use
        for id in range(1, len(pos_labels)):
            if pos_labels[id] == pos_labels[id-1]:
                instance_counter[id] = start_id
            else:
                start_id += 1
                instance_counter[id] = start_id
        instance_counter = instance_counter.int()
        '''
        if num_pos > 0:
            pos_points = flatten_points[pos_inds]
            '''
            NOTE: 
            Strided box and Origin box
            '''
            pos_decoded_bbox_preds = distance2bbox(
                pos_points, pos_bbox_preds * pos_bbox_strides)
            pos_decoded_bbox_strided_preds = distance2bbox(
                pos_points, pos_bbox_preds)
            pos_decoded_strided_targets = distance2bbox(
                pos_points, pos_bbox_strided_targets)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # create centerness targets, iou based, NOTE: strided as FCOS implementation
            '''
            pos_centerness_targets = bbox_overlaps(pos_decoded_bbox_strided_preds.clone().detach(), pos_decoded_strided_targets, is_aligned=True)
            '''
            pos_centerness_targets = bbox_overlaps(
                pos_decoded_bbox_preds.clone().detach(),
                pos_decoded_target_preds,
                is_aligned=True)
            '''
            sorting bbox branch
            '''
            '''
            instance levels mask
            '''
            # generate instance levels index
            instance_counter = torch.zeros(num_pos, device=pos_labels.device)
            remove = torch.zeros(num_pos, device=pos_labels.device)
            obj_id = 0

            # NOTE: get mask for each obj
            for i in range(len(pos_decoded_target_preds)):
                if remove[i] == 0:
                    current_bbox = pos_decoded_target_preds[i]
                    mask = ((pos_decoded_target_preds == current_bbox
                             ).sum(1) == 4).nonzero()
                    instance_counter[mask] = obj_id
                    remove[mask] = 1
                    obj_id += 1

            instance_counter = instance_counter.int()
            obj_ids = torch.bincount(instance_counter).nonzero().int()

            for obj_id in obj_ids:
                dist_conf_mask_list.append(
                    (instance_counter == obj_id).float())

            masks_for_all = torch.ones_like(instance_counter).float()
            '''
            # consistency reduction: keep consistency between classification and regression
            # if pixel i  regression < mean & classification < mean, label as zero and do not regression on these pixels.
            '''
            pos_scores = flatten_cls_scores[pos_inds]
            if not self.cls_aware:
                pos_scores, _ = pos_scores.sigmoid().max(1)
            else:
                cls_pos_inds = torch.arange(
                    0, len(pos_scores), out=torch.LongTensor())
                pos_scores = pos_scores[cls_pos_inds,
                                        pos_labels[cls_pos_inds] - 1]
                # pos_scores_list = []
                # for i in range(len(pos_scores)):
                #     pos_scores_list.append(pos_scores[i, pos_labels[i] -
                #                                       1].reshape(-1, 1))
                # pos_scores = torch.cat(pos_scores_list).reshape(-1)

            for dist_conf_mask in dist_conf_mask_list:
                obj_mask_inds = dist_conf_mask.nonzero().reshape(-1)
                pos_centerness_obj = pos_centerness_targets[obj_mask_inds]
                pos_scores_obj = pos_scores[obj_mask_inds]
                # pos_scores_obj = pos_scores
                # mean IoU of an object
                regression_reduced_mean = pos_centerness_obj.mean()
                classification_reduced_mean = pos_scores_obj.mean()

                regression_mask = pos_centerness_obj < regression_reduced_mean
                classification_mask = pos_scores_obj < classification_reduced_mean

                # consistency:
                consistency_mask = (regression_mask + classification_mask) == 2
                masks_for_all[obj_mask_inds[consistency_mask]] = 0
            # cls branch
            reduced_mask = (masks_for_all == 0).nonzero()

            # NOTE: for sc statistic
            test = self.sc_statistic(flatten_labels, labels, featmap_sizes, gt_masks)
            
            if self.apply_consistency_on_cls:
                if self.apply_cls_ignore_area:
                    flatten_labels[pos_inds[reduced_mask]] = -1
                    saved_inds = (flatten_labels != -1).nonzero().reshape(-1)
                    flatten_labels = flatten_labels[saved_inds]
                    flatten_cls_scores = flatten_cls_scores[saved_inds]
                else:
                    # the pixels where IoU from sorted branch lower than 0.5 are labeled as negative (background) zero
                    flatten_labels[pos_inds[reduced_mask]] = 0

            saved_target_mask = masks_for_all.nonzero().reshape(-1)
            pos_centerness = pos_centerness[saved_target_mask].reshape(-1)
            '''
            consistency between regression and classification
            '''
            # bbox branch
            pos_decoded_target_preds = pos_decoded_target_preds[
                saved_target_mask].reshape(-1, 4)
            pos_decoded_bbox_preds = pos_decoded_bbox_preds[
                saved_target_mask].reshape(-1, 4)
            pos_bbox_strides = pos_bbox_strides[saved_target_mask].reshape(
                -1, 4)

            pos_centerness_targets = pos_centerness_targets[
                saved_target_mask].reshape(-1)

            if self.normalize_centerness:
                pos_centerness_targets = torch.pow(pos_centerness_targets, 2)
                pos_centerness_targets = pos_centerness_targets / pos_centerness_targets.max(
                )

            pos_inds = flatten_labels.nonzero().reshape(-1)
            num_pos = len(pos_inds)
            '''
            self.draw_sc_masks(flatten_cls_scores, pos_decoded_bbox_preds,
                          pos_decoded_target_preds, pos_inds, featmap_sizes)
            '''

            # NOTE: clone, avoid inplace operations
            pos_decoded_sort_bbox_preds = pos_decoded_bbox_preds.clone()
            none_sort_pos_decoded_sort_bbox_preds = pos_decoded_bbox_preds.clone(
            )
            '''
            # NOTE: update sorting rules:
            # 1. first order bboxes (four boundries 'outside' the gt) 
            # always better than second order boxes (partial inside gt with same sorting ranks)
            # 2. for second order boxes, it is not trival to get best bboxes 

            delta_x1, delta_y1, delta_x2, delta_y2 = gt - pred
            delta_x1 >= 0, delta_y1 >= 0, delta_x2 <= 0, delta_y2 <= 0
            '''
            pos_dist_scores = torch.abs(pos_decoded_target_preds -
                                        pos_decoded_bbox_preds)
            pos_dist_scores_sorted = torch.abs(
                pos_decoded_target_preds -
                pos_decoded_sort_bbox_preds).detach()

            pos_dist_scores = pos_dist_scores.permute(
                1, 0).contiguous()  # [pos_inds * 4] -> [4 * pos_inds]

            pos_gradient_update_mapping = torch.zeros_like(
                pos_dist_scores_sorted, dtype=torch.int64)

            pos_gradient_update_anti_mapping = torch.zeros_like(
                pos_dist_scores_sorted, dtype=torch.int64)

            inds_shift = 0
            for dist_conf_mask in dist_conf_mask_list:
                obj_mask_inds = (
                    (dist_conf_mask[saved_target_mask] +
                     masks_for_all[saved_target_mask]) == 2).nonzero()

                # global merging
                _, sorted_inds = torch.sort(
                    pos_dist_scores[:, obj_mask_inds], dim=1, descending=True)
                pos_decoded_sort_bbox_preds[
                    obj_mask_inds, 0] = pos_decoded_sort_bbox_preds[
                        obj_mask_inds, 0][sorted_inds[0]].reshape(-1, 1)
                pos_decoded_sort_bbox_preds[
                    obj_mask_inds, 1] = pos_decoded_sort_bbox_preds[
                        obj_mask_inds, 1][sorted_inds[1]].reshape(-1, 1)
                pos_decoded_sort_bbox_preds[
                    obj_mask_inds, 2] = pos_decoded_sort_bbox_preds[
                        obj_mask_inds, 2][sorted_inds[2]].reshape(-1, 1)
                pos_decoded_sort_bbox_preds[
                    obj_mask_inds, 3] = pos_decoded_sort_bbox_preds[
                        obj_mask_inds, 3][sorted_inds[3]].reshape(-1, 1)

                pos_gradient_update_mapping[obj_mask_inds,
                                            0] = inds_shift + sorted_inds[0]
                pos_gradient_update_mapping[obj_mask_inds,
                                            1] = inds_shift + sorted_inds[1]
                pos_gradient_update_mapping[obj_mask_inds,
                                            2] = inds_shift + sorted_inds[2]
                pos_gradient_update_mapping[obj_mask_inds,
                                            3] = inds_shift + sorted_inds[3]

                pos_gradient_update_anti_mapping[inds_shift + sorted_inds[0],
                                                 0] = obj_mask_inds
                pos_gradient_update_anti_mapping[inds_shift + sorted_inds[1],
                                                 1] = obj_mask_inds
                pos_gradient_update_anti_mapping[inds_shift + sorted_inds[2],
                                                 2] = obj_mask_inds
                pos_gradient_update_anti_mapping[inds_shift + sorted_inds[3],
                                                 3] = obj_mask_inds

                inds_shift += sorted_inds.shape[1]

            sorted_ious_weights = bbox_overlaps(
                pos_decoded_sort_bbox_preds,
                pos_decoded_target_preds,
                is_aligned=True).detach()
            ious_weights = bbox_overlaps(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                is_aligned=True).detach()

            _bd_sort_iou = torch.cat([
                sorted_ious_weights[pos_gradient_update_anti_mapping[..., 0]].
                reshape(-1, 1), sorted_ious_weights[
                    pos_gradient_update_anti_mapping[..., 1]].reshape(-1, 1),
                sorted_ious_weights[pos_gradient_update_anti_mapping[..., 2]].
                reshape(-1, 1), sorted_ious_weights[
                    pos_gradient_update_anti_mapping[..., 3]].reshape(-1, 1)
            ], 1)
            _bd_iou = ious_weights.reshape(-1, 1).repeat(1, 4)
            '''
            # NOTE: the grad of sorted branch is in sort order, diff from origin
            '''
            sort_gradient_mask = (_bd_sort_iou >
                                  (_bd_iou - self.iou_delta)).float()

            sort_gradient_mask[..., 0] = sort_gradient_mask[
                pos_gradient_update_mapping[..., 0], 0]
            sort_gradient_mask[..., 1] = sort_gradient_mask[
                pos_gradient_update_mapping[..., 1], 1]
            sort_gradient_mask[..., 2] = sort_gradient_mask[
                pos_gradient_update_mapping[..., 2], 2]
            sort_gradient_mask[..., 3] = sort_gradient_mask[
                pos_gradient_update_mapping[..., 3], 3]

            origin_gradient_mask = ((_bd_sort_iou - self.iou_delta) <=
                                    _bd_iou).float()
            sort_mask = torch.cat([
                sort_gradient_mask[pos_gradient_update_mapping[..., 0], 0].
                reshape(-1, 1),
                sort_gradient_mask[pos_gradient_update_mapping[..., 1], 1].
                reshape(-1, 1),
                sort_gradient_mask[pos_gradient_update_mapping[..., 2], 2].
                reshape(-1, 1),
                sort_gradient_mask[pos_gradient_update_mapping[..., 3], 3].
                reshape(-1, 1)
            ], 1)
            pos_decoded_sort_bbox_preds.register_hook(
                lambda grad: grad * sort_mask)
            pos_decoded_bbox_preds.register_hook(
                lambda grad: grad * origin_gradient_mask)

            loss_sorted_bbox = self.loss_sorted_bbox(
                pos_decoded_sort_bbox_preds,
                pos_decoded_target_preds,
                weight=sorted_ious_weights,
                avg_factor=sorted_ious_weights.sum())
            # origin boxes
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=ious_weights,
                avg_factor=ious_weights.sum())
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                avg_factor=num_pos + num_imgs)  # avoid pruned_num_pos is 0
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_sorted_bbox = pos_decoded_bbox_preds.sum()
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        if self.sorted_warmup > 0:
            self.sorted_warmup -= 1
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                # loss_sorted_bbox=loss_sorted_bbox,
                loss_centerness=loss_centerness)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_sorted_bbox=loss_sorted_bbox,
                loss_centerness=loss_centerness)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None,
                   nms=True):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
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
                                                scale_factor, cfg, rescale,
                                                nms)
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
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, points, centerness, stride in zip(
                cls_scores, bbox_preds, mlvl_points, centernesses,
                self.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = bbox_pred * stride

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = scores.max(dim=1)
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

        if nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness)

            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_centerness

    def get_points(self, featmap_sizes, dtype, device):
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
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
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
        concat_lvl_bbox_strided_targets = []
        concat_lvl_strides = []
        for i in range(num_levels):
            stride = torch.ones(1, device=bbox_targets_list[0][0].device)
            stride[0] = self.strides[i]

            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_bbox_strided_targets.append(
                torch.cat([
                    bbox_targets[i] / self.strides[i]
                    for bbox_targets in bbox_targets_list
                ]))
            concat_lvl_strides.append(
                torch.cat([
                    stride.expand_as(bbox_targets[i])
                    for bbox_targets in bbox_targets_list
                ]))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_bbox_strided_targets, concat_lvl_strides

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

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

        # cut = 3
        # w_stride = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) / cut
        # h_stride = (gt_bboxes[..., 3] - gt_bboxes[..., 1]) / cut

        # center_left = xs - (gt_bboxes[..., 0] + w_stride)
        # center_right = gt_bboxes[..., 2] - w_stride - xs
        # center_top = ys - (gt_bboxes[..., 1] + h_stride)
        # center_bottom = gt_bboxes[..., 3] - h_stride - ys

        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # bbox_center_targets = torch.stack((center_left, center_top, center_right, center_bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # inside_gt_bbox_center_mask = bbox_center_targets.min(-1)[0] > 0
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        # areas[inside_gt_bbox_center_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets


    def draw_sc_masks(self, flatten_cls_scores, pos_decoded_bbox_preds,
                    pos_decoded_target_preds, pos_inds, featmap_sizes):

        stand_size = featmap_sizes[0]
        heatmap = torch.zeros(
            stand_size,
            dtype=pos_decoded_bbox_preds.dtype,
            device=pos_decoded_bbox_preds.device)
        reg_iou_heatmap = torch.zeros(
            flatten_cls_scores.shape[0],
            dtype=pos_decoded_bbox_preds.dtype,
            device=pos_decoded_bbox_preds.device)
        pos_ious_scores = bbox_overlaps(
            pos_decoded_bbox_preds.detach(),
            pos_decoded_target_preds,
            is_aligned=True)

        cls_scores_heatmap = flatten_cls_scores.sigmoid().max(1)[0]
        reg_iou_heatmap[pos_inds] = pos_ious_scores

        hook = 0
        for i, featmap_size in enumerate(featmap_sizes):
            w, h = featmap_size

            sub_heatmap = reg_iou_heatmap[hook:hook + w * h]
            sub_heatmap = sub_heatmap.reshape(w, h)
            # heatmap += nn.functional.interpolate(
            #     sub_heatmap, size=stand_size,
            #     mode='nearest').view(stand_size[0], stand_size[1])

            # heatmap = sub_heatmap.view(w, h)

            if len(sub_heatmap.nonzero()) > 0:
                inds = sub_heatmap.nonzero()
                # upsample with stride
                stride = int(stand_size[0] / w)
                print(stride)
                for i, ind in enumerate(inds):
                    w_ind = ind[0]
                    h_ind = ind[1]
                    heatmap[w_ind * stride, h_ind *
                            stride] = sub_heatmap[w_ind, h_ind]

            hook += w * h

        heatmap = heatmap.detach().cpu().numpy()
        from matplotlib import pyplot as plt
        plt.imshow(heatmap)
        # Saving image
        plt.savefig('temp_vis/heatmap-451090.png')
        # embed()

        # import seaborn as sns
        # ax = sns.heatmap(heatmap)
        # misc.imsave("sc_heatmaps_3249.png", heatmap.detach().cpu().numpy())


    def sc_statistic(self, flatten_labels, labels, featmap_sizes, gt_masks):
        crop_point = 0
        area = labels[0].shape[0]
        _w, _h = featmap_sizes[0]
        n = area // (_w * _h)
        sum_masks = torch.zeros([n, _w, _h], device=labels[0].device).float()
        for featmap_size, label in zip(featmap_sizes, labels):
            w, h = featmap_size
            mask = flatten_labels[crop_point:crop_point + n * w * h].reshape(
                n, w, h).float()
            sum_masks += F.interpolate(
                mask.unsqueeze(0), size=featmap_sizes[0],
                mode='nearest').reshape(n, _w, _h)
            crop_point += n * w * h
        from scipy.misc import imsave
        for sum_mask, gt_mask in zip(sum_masks, gt_masks):
            self.sc_image_counter += 1
            gt_mask = torch.tensor(gt_mask, device=sum_masks.device).float().sum(0)
            gt_mask = F.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0),
                size=featmap_sizes[0],
                mode='nearest').reshape(_w, _h)

            gt_mask = gt_mask.reshape(-1)
            sum_mask = sum_mask.reshape(-1)
            gt_mask[gt_mask != 0] = 1
            sum_mask[sum_mask != 0] = 1

            gt_area = gt_mask.sum()
            sc_area = ((sum_mask + gt_mask) == 2).sum()
            inside_ratio = sc_area / gt_area
            effective_ratio = sc_area / sum_mask.sum()
            # print("effective_ratio:", effective_ratio)
            # print("sc_ratio:", sc_ratio)
            self.sc_avg_ratio += effective_ratio
            # imsave("./visualize/sum_masks_{}.png".format(self.sc_image_counter), sum_mask.cpu().numpy())
            if self.sc_image_counter == 1000:
                print("avg_sc_ratio:", self.sc_avg_ratio / self.sc_image_counter)
                embed()
        return None
