import copy
from os.path import dirname, exists, join

from mmdet.apis import init_detector, inference_detector, show_result, show_result_with_gt, train_detector
from mmdet.datasets.coco import CocoDataset
from mmdet.core import bbox_overlaps
from mmdet.datasets.pipelines import Compose

import mmcv
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

import torch
import numpy as np

from IPython import embed


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['bbox_fields'] = []
        # results['ann_info'] = anns
        return results


def _get_config_directory():
    """ Find the predefined detector config directory """
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')

    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """
    Load a configuration as a python module
    """
    from xdoctest.utils import import_module_from_path
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = import_module_from_path(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """
    Grab configs necessary to create a detector. These are deep copied to allow
    for safe modification of parameters without influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))
    return model, train_cfg, test_cfg


def draw_heatmap():
    config_file = 'fcos/ddb_v3_r50_caffe_fpn_gn_1x_single_test.py'
    checkpoint_file = './work/dirs/ddb_v3_single/epoch_12.pth'
    model, train_cfg, test_cfg = _get_detector_cfg(config_file)

    # build the model from a config file and a checkpoint file
    from mmdet.models import build_detector
    detector = build_detector(model, train_cfg=train_cfg, test_cfg=test_cfg)
    checkpoint = load_checkpoint(detector, checkpoint_file)

    if 'CLASSES' in checkpoint['meta']:
        detector.CLASSES = checkpoint['meta']['CLASSES']
    else:
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use COCO classes by default.')
        detector.CLASSES = get_classes('coco')

    train_cfg = Config.fromfile('configs/' + config_file)
    detector.cfg = train_cfg  # save the config in the model for convenience
    detector.to('cuda:0')
    detector.eval()
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
    ann_file = 'annotations/instances_val2017.json'
    img_prefix = 'train2017/'
    img_scale = (1333, 800)

    coco_dataset = CocoDataset(
        ann_file,
        pipeline=test_pipeline,
        img_prefix=img_prefix,
        data_root=data_root)

    img_name_list = [
        coco_dataset.img_infos[i]['file_name']
        for i in range(len(coco_dataset.img_infos))
    ]

    for j in range(1):
        rand = np.random.randint(10000)
        # 2017 train
        # img_file_name = '000000009898.jpg'
        # img_file_name = '000000003209.jpg'
        # img_file_name = '000000001238.jpg'
        # img_file_name = '000000003249.jpg'
        # img_file_name = '000000000382.jpg'
        # img_file_name = '000000000706.jpg'
        # img_file_name = '000000003217.jpg'
        # img_file_name = '000000002445.jpg'
        # img_file_name = '000000012443.jpg'
        # 2017 val
        # img_file_name = '000000085665.jpg' 
        # img_file_name = '000000084477.jpg' 
        img_file_name = '000000451090.jpg' 
        # imgs = './data/2017/train2017/' + img_file_name
        imgs = './temp_img/' + img_file_name
        img_id = [i for i, x in enumerate(img_name_list) if x == img_file_name]
        img_metas = coco_dataset.get_ann_info(img_id[0])
        # img_metas.pop('bboxes_ignore', None)
        img_annotation_boxes = img_metas['bboxes']
        img_annotation_labels = img_metas['labels']

        device = next(detector.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + detector.cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        # data = dict(img=imgs, ann_info=img_metas)
        data = dict(img=imgs)
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        input_tensor = data['img'][0]
        gt_bboxes = []
        gt_labels = []

        gt_bboxes.append(torch.cuda.FloatTensor(img_annotation_boxes))
        gt_labels.append(torch.cuda.LongTensor(img_annotation_labels))
        '''
        mmcv.imshow_det_bboxes(
            mmcv.imread(imgs),
            img_annotation_boxes,
            img_annotation_labels,
            class_names=detector.CLASSES,
            score_thr=0,
            show=False,
            out_file='input_with_gt.png')
        '''
        losses = detector.forward(
            input_tensor,
            img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            return_loss=True)


if __name__ == '__main__':
    draw_heatmap()
