#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import re
from random import randint
from PIL import Image

data_base = '/home/ubuntu/data/UECFOOD256/'
data_dir = os.path.join(data_base, 'data')

def get_classes():
    cat_file = os.path.join(data_base, 'category.txt')
    with open(cat_file, 'r') as f:
        data = f.read()
    class_names = re.findall('(?m)\d+[\s]+([\S ]+)', data)
    class_names.insert(0,'__background__')
    return tuple(class_names)

# Get list of validation IDs from ImageSets/val.txt                                               
def get_val_ids():
    val_file = os.path.join(data_dir, 'ImageSets', 'val.txt')
    with open(val_file, 'r') as f:
        val_ids = [line.rstrip() for line in f]
    return val_ids

# extract N elements randomly from list                                                           
def rand_pick_list(check_list, n_check=100):
    checked_indices = []
    check_values = []
    while len(check_values) < n_check:
        rnd_ind = randint(0, len(check_list) - 1)
        if rnd_ind not in checked_indices:
            check_values.append( check_list[rnd_ind] )
            checked_indices.append(rnd_ind)
    return check_values, checked_indices



CLASSES = get_classes()

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
#    print('{} Images above threshold'.format(len(inds)))
    if len(inds) == 0:
        return

#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(np.flipud(im), aspect='equal', origin='lower')
#    ax.imshow(im, aspect='equal')
    max_i = 0
    max_score = 0
    for i in inds:
        score = dets[i, -1]
        if score > max_score:
            max_i = i
            max_score = score
    print('Max score {} at ind {}'.format(max_score, max_i))
        
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        
        print('ind {}, bbox {:6.1f} {:6.1f} {:6.1f} {:6.1f} {}'.format(i, bbox[0], bbox[1], bbox[2], bbox[3], score))

#        if i == max_i:
        if True:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}'.format(class_name, score, bbox[0], bbox[1], bbox[2], bbox[3]),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    #plt.savefig(str(im) + '.png', bbox_inches='tight')

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo2', image_name)
    im_file = os.path.join(data_dir, 'Images', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])



    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(np.flipud(im), aspect='equal', origin='lower')
    ax.imshow(im, aspect='equal')

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
#        keep = nms(dets, NMS_THRESH, force_cpu=True)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])
#    prototxt = '/home/ubuntu/thomas/py-faster-rcnn/models/uecfood256/VGG16/faster_rcnn_end2end/test.prototxt'
    #caffemodel = '/home/ubuntu/thomas/py-faster-rcnn/output/faster_rcnn_end2end_food/uecfood256_2014_train/vgg16_faster_rcnn_food_iter_70000.caffemodel'
#    caffemodel = '/home/ubuntu/thomas/py-faster-rcnn/output/faster_rcnn_end2end_food_2hr/uecfood256_2014_train/vgg16_faster_rcnn_food_iter_75000.caffemodel'

    prototxt = '/home/ubuntu/thomas/py-faster-rcnn/models/uecfood256/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = '/home/ubuntu/thomas/py-faster-rcnn/output/faster_rcnn_alt_opt_food/uecfood256_2014_train/VGG16_faster_rcnn_final.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
        print 'Running in CPU mode'
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    val_ids = get_val_ids()
    check_imgs, checked_indices = rand_pick_list(val_ids, 20)
    im_names = []
    for img in check_imgs:
        im_names.append(str(img) + '.jpg')
                
    for im_name in im_names:
#        chk_im = os.path.join(data_dir, 'Images', im_name)
#        with Image.open(chk_im) as tmp_im:
#            im_width = tmp_im.size[0]
        #print('im_width = {}'.format(im_width))
#        if im_width < 500:
#            continue
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)
        plt.savefig(os.path.splitext(im_name)[0]+'.png', bbox_inches='tight')

    #plt.show()
    # plt.savefig('../../tmp/yurakucho_002_detection.png', bbox_inches='tight')
