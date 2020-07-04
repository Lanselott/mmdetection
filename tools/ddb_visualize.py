from mmdet.apis import init_detector, inference_detector, show_result, show_result_with_gt, train_detector
from mmdet.datasets.coco import CocoDataset
from mmdet.core import bbox_overlaps

from mmcv import Config

import torch
import numpy as np
from IPython import embed

config_file = './configs/fcos/ddb_v3_r50_caffe_fpn_gn_1x_single_test.py'
checkpoint_file = './work/dirs/ddb_v3_single/epoch_1.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
gt_iou_threshold = 0.1

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
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
data_root = 'data/2017/'
ann_file = 'annotations/instances_train2017.json'
img_prefix = 'train2017/'
img_scale = (1333, 800)
train_cfg = Config.fromfile(config_file)

coco_dataset = CocoDataset(
    ann_file,
    pipeline=test_pipeline,
    img_prefix=img_prefix,
    data_root=data_root)

img_name_list = [coco_dataset.img_infos[i]['file_name'] for i in range(len(coco_dataset.img_infos))]

for j in range(1):
    rand = np.random.randint(10000)
    # img_infos = coco_dataset.img_infos[rand]
    # img_id = img_infos['id']
    # img_file_name = img_infos['file_name']
    img_file_name = '000000009898.jpg'
    # imgs = './data/2017/train2017/' + img_file_name
    imgs = './temp_img/' + img_file_name
    img_id = [i for i, x in enumerate(img_name_list) if x == img_file_name]
    img_annotation_boxes = coco_dataset.get_ann_info(img_id[0])['bboxes']
    img_annotation_labels = coco_dataset.get_ann_info(img_id[0])['labels']

    annotations = np.ones(
        [img_annotation_boxes.shape[0], img_annotation_boxes.shape[1] + 1])
    annotations[:, :4] = img_annotation_boxes
    if len(annotations) > 2:
        j = j - 1
        continue
    result = inference_detector(model, imgs)
    # result = train_detector(model, imgs, train_cfg)
    _result = result.copy()

    _dr_source_result = [np.zeros((0, 5), dtype=float)] * 80  # result.copy()

    for i in range(len(img_annotation_labels)):
        inds = img_annotation_labels[i]
        result[inds - 1] = np.concatenate(
            (result[inds - 1], annotations[i].reshape(-1, 5)), 0)

    origin_bd_weight = []
    dr_bd_weight = []
    selected_result = []
    selected_dr_result = []
    # sort
    for k in range(len(img_annotation_labels)):
        inds = img_annotation_labels[k]

        _pred_boxes = _result[inds - 1][:, :4]
        if _pred_boxes.shape[0] == 0:
            continue
        ious = bbox_overlaps(
            torch.from_numpy(_pred_boxes).float(),
            torch.from_numpy(annotations[k][:4]).reshape(-1, 4).float().expand(
                _pred_boxes.shape[0], 4),
            is_aligned=True)

        _pred_boxes = _pred_boxes[(
            ious >= gt_iou_threshold).nonzero()[:, 0]].reshape(-1, 4)
        ious = ious[(ious >= gt_iou_threshold).nonzero()[:, 0]]
        if _pred_boxes.shape[0] == 0:
            continue

        delta = np.abs(_pred_boxes - annotations[k][:4])
        sort_max_inds = np.argsort(delta, axis=0).reshape(-1, 4)
        sort_max_inds_map_back = np.zeros_like(sort_max_inds)

        for l in range(len(sort_max_inds_map_back)):
            sort_max_inds_map_back[sort_max_inds[l, 0], 0] = l
            sort_max_inds_map_back[sort_max_inds[l, 1], 1] = l
            sort_max_inds_map_back[sort_max_inds[l, 2], 2] = l
            sort_max_inds_map_back[sort_max_inds[l, 3], 3] = l

        l = _pred_boxes[sort_max_inds[:, 0], 0].reshape(-1, 1)
        t = _pred_boxes[sort_max_inds[:, 1], 1].reshape(-1, 1)
        r = _pred_boxes[sort_max_inds[:, 2], 2].reshape(-1, 1)
        b = _pred_boxes[sort_max_inds[:, 3], 3].reshape(-1, 1)
        scores = np.ones((_pred_boxes.shape[0], )).reshape(-1, 1)
        dr_box = np.concatenate([l, t, r, b], axis=1)
        dr_box = np.concatenate([dr_box, scores], axis=1)
        dr_ious = bbox_overlaps(
            torch.from_numpy(dr_box[:, :4]).float(),
            torch.from_numpy(annotations[k][:4]).reshape(-1, 4).float().expand(
                dr_box.shape[0], 4),
            is_aligned=True)

        dr_ious = dr_ious.reshape(-1, 1).expand(-1, 4)
        ious = ious.reshape(-1, 1).expand(-1, 4)

        ious_map = torch.zeros_like(dr_ious)

        ious_map[:, 0] = ious[sort_max_inds[:, 0], 0]
        ious_map[:, 1] = ious[sort_max_inds[:, 1], 1]
        ious_map[:, 2] = ious[sort_max_inds[:, 2], 2]
        ious_map[:, 3] = ious[sort_max_inds[:, 3], 3]
        dr_boundary_weight = dr_ious
        # dr_boundary_weight = torch.where(dr_ious >= ious_map, dr_ious,
        #                                  ious_map)

        # dr_boundary_weight should be in sorted order
        origin_boundary_weight = ious
        origin_bd_weight.append(origin_boundary_weight)
        dr_bd_weight.append(dr_boundary_weight)
        _pred_boxes = np.concatenate([_pred_boxes, scores], axis=1)
        result[inds - 1] = _pred_boxes
        _result[inds - 1] = dr_box
        selected_result.append(_pred_boxes)
        selected_dr_result.append(dr_box)

    dr_bd_weight = torch.cat(dr_bd_weight)
    origin_bd_weight = torch.cat(origin_bd_weight)
    selected_result = np.concatenate(selected_result)
    selected_dr_result = np.concatenate(selected_dr_result)

    # before dr
    show_result_with_gt(
        imgs,
        selected_result,  #result,
        None,  # model.CLASSES,
        img_annotation_boxes,
        origin_bd_weight,
        score_thr=0.00,
        show=False,
        thickness=2,
        out_file='{}_result_before_dr.png'.format(img_file_name[:-4]))
    # TODO: visualize D&R module
    show_result_with_gt(
        imgs,
        selected_dr_result,  #_result,
        None,  # model.CLASSES,
        img_annotation_boxes,
        dr_bd_weight,
        score_thr=0.00,
        show=False,
        thickness=2,
        out_file='{}_result_after_dr.png'.format(img_file_name[:-4]))

    #
    '''
    show_result_without_names(imgs, _dr_source_result, model.CLASSES, score_thr=0.99, out_file='dr_source_{}'.format(img_file_name))
    '''
    # origin
    # show_result(imgs, _dr_source_result, model.CLASSES, score_thr=1, out_file='origin_{}'.format(img_file_name))

# show_result(img, result, model.CLASSES)
# test a list of images and write the results to image files
# imgs = ['test1.jpg', 'test2.jpg']
'''
for i, result in enumerate(inference_detector(model, imgs)):
    # show_result_with_center(imgs[i], result, model.CLASSES, score_thr=0.5, out_file='result_{}.jpg'.format(i))
    show_result(imgs, result, model.CLASSES, score_thr=0.1, out_file='result_{}.jpg'.format(i))
'''
