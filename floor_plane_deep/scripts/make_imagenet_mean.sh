#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=.
DATA=.
TOOLS=/home/gpu_user/caffe/build/tools
#TOOLS=/cs-share/pradalier/caffe/build/tools

$TOOLS/compute_image_mean floorplane_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
