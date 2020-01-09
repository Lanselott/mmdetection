import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, bbox_overlaps
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
from IPython import embed
INF = 1e8


@HEADS.register_module
class FCOSFCHead(nn.Module):
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
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 topk_select_num=9):
        super(FCOSFCHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.topk_select_num = topk_select_num
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
        
        # self.fcos_topk_reg = nn.Linear(self.feat_channels, 4)
        # self.topk_obj_reg = nn.Linear(self.topk_select_num, 1)
        self.topk_obj_reg = nn.Conv2d(256, 4, 3) # (in, out, kernel_size)

        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        
        # normal_init(self.fcos_topk_reg, std=0.01)
        normal_init(self.topk_obj_reg, std=0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
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
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness, reg_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             reg_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(reg_feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
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
        flatten_reg_feats = [
            reg_feat.permute(0, 2, 3, 1).reshape(-1, 256)
            for reg_feat in reg_feats
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_reg_feats = torch.cat(flatten_reg_feats)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_reg_feats = flatten_reg_feats[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            pos_iou_scores = bbox_overlaps(pos_decoded_bbox_preds, pos_decoded_target_preds, is_aligned=True).clamp(min=1e-6)
            # Instance level op
            dist_conf_mask_list = []
            # generate instance levels index
            instance_counter = torch.zeros(num_pos, device=pos_points.device)
            remove = torch.zeros(num_pos, device=pos_points.device)
            obj_id = 0
            
            # NOTE: get mask for each obj
            topk_select_num = self.topk_select_num
            for i in range(len(pos_decoded_target_preds)):
                if remove[i] == 0:
                    current_bbox = pos_decoded_target_preds[i]                
                    mask = ((pos_decoded_target_preds == current_bbox).sum(1)==4).nonzero()
                    instance_counter[mask] = obj_id
                    remove[mask] = 1
                    obj_id += 1

            instance_counter = instance_counter.int()
            obj_ids = torch.bincount(instance_counter).nonzero().int()
            opt_threshold = 0.9
            for obj_id in obj_ids:
                dist_conf_mask_list.append((instance_counter==obj_id).float())
            
            obj_topk_pos_bbox_preds = []
            obj_topk_pos_decoded_bbox_targets = []
            obj_pos_reg_feat_list = []
            mean_obj_topk_inds_list = []

            for dist_conf_mask in dist_conf_mask_list:
                obj_mask_inds = dist_conf_mask.nonzero().reshape(-1)
                obj_pos_iou_scores = pos_iou_scores[obj_mask_inds]
                obj_pos_reg_feat = pos_reg_feats[obj_mask_inds]
                obj_sample_nums = obj_mask_inds.size()[0]

                if obj_sample_nums > topk_select_num:
                    obj_scores, obj_topk_inds = torch.topk(obj_pos_iou_scores, topk_select_num)
                    if (obj_topk_inds == 0).sum() > 1 or obj_scores.min() < opt_threshold:
                        continue
                else:
                    # obj_scores, obj_topk_inds = obj_pos_iou_scores.topk(obj_sample_nums)
                    continue

                mean_obj_topk_inds_list.append(pos_points[obj_mask_inds[obj_topk_inds]].sum(0).reshape(1, 2) / self.topk_select_num)
                '''
                # UPDATE
                '''
                obj_pos_reg_feat_list.append(obj_pos_reg_feat[obj_topk_inds].reshape(1, -1, 256))
                obj_topk_pos_decoded_bbox_targets.append(pos_decoded_target_preds[obj_mask_inds[obj_topk_inds]][0].reshape(1, 4))
            
            if len(obj_pos_reg_feat_list) != 0:
                obj_pos_reg_feats = torch.cat(obj_pos_reg_feat_list)
                mean_obj_topk_inds = torch.cat(mean_obj_topk_inds_list)
                obj_topk_pos_bbox_preds = distance2bbox(
                                            mean_obj_topk_inds, 
                                            self.topk_obj_reg(obj_pos_reg_feats.permute(0, 2, 1).contiguous().reshape(-1, 256, 3 ,3)).float().exp().reshape(-1, 4))

                obj_topk_pos_decoded_bbox_targets = torch.cat(obj_topk_pos_decoded_bbox_targets)
                # print("length of topks", len(obj_topk_pos_bbox_preds))
                loss_topk_bbox = self.loss_bbox(
                    obj_topk_pos_bbox_preds,
                    obj_topk_pos_decoded_bbox_targets)
                    # weight=pos_centerness_targets,
                    # avg_factor=len(obj_topk_pos_bbox_preds))
            else:
                loss_topk_bbox = torch.zeros(0, device=pos_bbox_preds.device).sum()
            # print("loss_topk_bbox:", loss_topk_bbox)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds)
                # weight=pos_centerness_targets,
                # avg_factor=pos_centerness_targets.sum())
            # loss_centerness = self.loss_centerness(pos_centerness,
            #                                        pos_centerness_targets)
        else:
            loss_topk_bbox = pos_bbox_preds.sum()
            loss_bbox = pos_bbox_preds.sum()
            # loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_topk_bbox=loss_topk_bbox,
            loss_bbox=loss_bbox)
            # loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   reg_feats,
                   img_metas,
                   cfg,
                   rescale=None):
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
            reg_feat_list = [
                reg_feats[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                reg_feat_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          reg_feats,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_reg_feat = []
        _mlvl_points = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, reg_feat, points in zip(
                cls_scores, bbox_preds, centernesses, reg_feats, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            
            reg_feat = reg_feat.permute(1, 2, 0).reshape(-1, 256)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                reg_feat = reg_feat[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_reg_feat.append(reg_feat)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            _mlvl_points.append(points)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_reg_feat = torch.cat(mlvl_reg_feat)
        _mlvl_points = torch.cat(_mlvl_points)
        # topk_mlvl_bboxes_ious, topk_inds = bbox_overlaps(mlvl_bboxes, mlvl_bboxes, is_aligned=False).clamp(min=1e-6).topk(self.topk_select_num)
        # topk_encoded_boxes_list = []
        # mean_points_list = []
        # for selected_inds in topk_inds:
        #     mean_points_list.append(_mlvl_points[selected_inds].mean(0).reshape(1, 2))
        #     topk_encoded_boxes_list.append(mlvl_reg_feat[selected_inds].reshape(1, self.topk_select_num, -1))
        #     # topk_boxes_list.append(distance2bbox(mean_points, self.topk_obj_reg(mlvl_reg_feat[selected_inds].permute(1, 0).contiguous().reshape(1, 256, 3, 3)).float().exp().reshape(1 ,4)))
        # topk_encoded_boxes = torch.cat(topk_encoded_boxes_list)
        # mean_points = torch.cat(mean_points_list)
        # topk_boxes = distance2bbox(mean_points, self.topk_obj_reg(topk_encoded_boxes.permute(0, 2, 1).contiguous().reshape(-1, 256, 3, 3)).float().exp().reshape(-1 ,4))

        det_bboxes, det_labels, det_inds = multiclass_nms(
            mlvl_bboxes,
            # topk_boxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            with_inds=True)
            # score_factors=mlvl_centerness)

        nms_selected_boxes_ious, nms_selected_boxes_inds = bbox_overlaps(det_bboxes[:, :4], mlvl_bboxes, is_aligned=False).topk(self.topk_select_num)

        infer_threshold = 0.95
        saved_bboxes_inds = nms_selected_boxes_ious[:, -1] < infer_threshold
        saved_det_bboxes = det_bboxes[saved_bboxes_inds]
        removed_det_bboxes = det_bboxes[(saved_bboxes_inds!=1).nonzero()].reshape(-1, 5)
        nms_selected_boxes_inds = nms_selected_boxes_inds[(saved_bboxes_inds!=1).nonzero()].reshape(-1, 9)
        topk_encoded_boxes_list = []
        mean_points_list = []
        for selected_inds in nms_selected_boxes_inds:
            mean_points_list.append(_mlvl_points[selected_inds].sum(0).reshape(1, 2) / self.topk_select_num)
            topk_encoded_boxes_list.append(mlvl_reg_feat[selected_inds].reshape(1, self.topk_select_num, -1))

        topk_encoded_boxes = torch.cat(topk_encoded_boxes_list)
        mean_points = torch.cat(mean_points_list)
        topk_boxes = distance2bbox(mean_points, self.topk_obj_reg(topk_encoded_boxes.permute(0, 2, 1).contiguous().reshape(-1, 256, 3, 3)).float().exp().reshape(-1 ,4))
        topk_boxes = torch.cat([topk_boxes, removed_det_bboxes[:, 4:]], 1)
        final_det_bboxes = torch.cat([saved_det_bboxes, topk_boxes])

        return final_det_bboxes, det_labels

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
