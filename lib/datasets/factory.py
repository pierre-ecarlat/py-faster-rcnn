# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.imagenet import imagenet
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.uecfood256 import uecfood256
from datasets.foodinc import foodinc
from datasets.foodinc_sample import foodinc_sample
from datasets.foodinc_reduced import foodinc_reduced
import numpy as np

# Set up imagenet
#imagenet_devkit_path = 'path/to/imagenet'
#for year in ['2012']
#for split in ['train', 'test']:
#    name = 'imagenet_{}_{}'.format(year, split)
#    __sets[name] = (lamda split=split, year=year:, imagenet(split, year, imagenet_devkit_path))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up UEC Food-256, 2014
for year in ['2014']:
    for split in ['train', 'val', 'test', 'try_train', 'try_val', 'try_test', 'try_val_mini']:
        name = 'uecfood256_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: uecfood256(split, year))

# Set up Foodinc
for year in ['2017']:
    for split in ['train', 'val', 'trainval', 'test']:
        # Basic ; sample and reduced (18 categories)
        name = 'foodinc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: foodinc(split, year))
        name = 'foodinc_sample_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: foodinc_sample(split, year))
        name = 'foodinc_reduced_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: foodinc_reduced(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
