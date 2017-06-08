#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
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
    echo "Not implemented: use experiments/scripts/faster_rcnn_end2end.sh for coco"
    exit
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

LOG="experiments/logs/faster_rcnn_alt_opt_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_faster_rcnn_alt_opt.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights /mnt2/givenModels/imagenet/${NET}.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}
