#!/bin/bash
# Usage:
# ./experiments/scripts/fast_rcnn.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/fast_rcnn.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=40000
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=280000
    ;;
  foodinc)
    TRAIN_IMDB="foodinc_2017_trainval"
    TEST_IMDB="foodinc_2017_test"
    PT_DIR="foodinc"
    ITERS=70000
    ;;
  foodinc_sample)
    TRAIN_IMDB="foodinc_sample_2017_trainval"
    TEST_IMDB="foodinc_sample_2017_test"
    PT_DIR="foodinc_sample"
    ITERS=100
    ;;
  foodinc_reduced)
    TRAIN_IMDB="foodinc_reduced_2017_trainval"
    TEST_IMDB="foodinc_reduced_2017_test"
    PT_DIR="foodinc_reduced"
    ITERS=70000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/fast_rcnn_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/fast_rcnn/solver.prototxt \
  --weights /mnt2/givenModels/imagenet/${NET}.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/fast_rcnn/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  ${EXTRA_ARGS}
