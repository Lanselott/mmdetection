from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.datasets.coco import CocoDataset
import numpy as np
from IPython import embed

config_file = './configs/fcos/consistency_deconv_r50_caffe_fpn_gn_1x_single_test.py'
checkpoint_file = 'work/dirs/upsample_deconv/upsample_deconv_epoch_12.pth'

# build the model from a config file and a checkpoint file

# test a single image and show the results
imgs = ['./data/2014/trainval/COCO_train2014_000000289515.jpg','./data/2014/trainval/COCO_train2014_000000577762.jpg','./data/2014/trainval/COCO_val2014_000000581886.jpg','./data/2014/trainval/COCO_train2014_000000574957.jpg','./data/2014/trainval/COCO_val2014_000000576085.jpg'] # or img = mmcv.imread(img), which will only load it once
# imgs = ['./data/2014/trainval/COCO_train2014_000000288421.jpg','./data/2014/trainval/COCO_train2014_000000576598.jpg','./data/2014/trainval/COCO_val2014_000000579294.jpg','./data/2014/trainval/COCO_train2014_000000573252.jpg','./data/2014/trainval/COCO_val2014_000000573113.jpg'] # or img = mmcv.imread(img), which will only load it once

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
i = 0
for img in imgs:
    result = inference_detector(model, img)
    # or save the visualization results to image files
    # embed()
    show_result(img, result, model.CLASSES, score_thr=0.3, show=False, out_file='result_{}.jpg'.format(i))
    i += 1
