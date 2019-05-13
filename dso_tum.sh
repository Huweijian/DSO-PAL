#!/bin/bash

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/TUM/sequence_19/images.zip \
# calib=/home/hwj23/Dataset/TUM/sequence_19/camera.txt \
# gamma=/home/hwj23/Dataset/TUM/sequence_19/pcalib.txt \
# vignette=/home/hwj23/Dataset/TUM/sequence_19/vignette.png \
# preset=0 \
# mode=0 \
# start=0

./build/bin/dso_dataset \
files=/home/hwj23/Dataset/TUM/sequence_19/images_720 \
calib=/home/hwj23/Dataset/TUM/720sml_calib_results_tum_fisheye.txt \
gamma=/home/hwj23/Dataset/TUM/sequence_19/pcalib.txt \
vignette=/home/hwj23/Dataset/TUM/sequence_19/720sml_vignette.png \
preset=0 \
mode=0 \
start=0

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/PAL/simu/now/color/ \
# calib=/home/hwj23/Dataset/TUM/720sml_calib_results_tum_fisheye.txt \
# preset=0 \
# mode=2