#!/bin/bash

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/TUM/sequence_19/images \
# calib=/home/hwj23/Dataset/TUM/sequence_19/camera.txt \
# gamma=/home/hwj23/Dataset/TUM/sequence_19/pcalib.txt \
# vignette=/home/hwj23/Dataset/TUM/sequence_19/vignette.png \
# preset=0 \
# mode=0 \
# start=0

./build/bin/dso_dataset \
files=/home/hwj23/Dataset/TUM/sequence_19/images.zip \
calib=/home/hwj23/Dataset/TUM/calib_results_tum_fisheye.txt \
gamma=/home/hwj23/Dataset/TUM/sequence_19/pcalib.txt \
vignette=/home/hwj23/Dataset/TUM/sequence_19/vignette.png \
preset=0 \
mode=0 \
start=650

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/TUM/calib_wide_checkerboard1/images_half \
# calib=/home/hwj23/Dataset/TUM/sml_calib_results_tum_fisheye.txt \
# gamma=/home/hwj23/Dataset/TUM/sequence_19/pcalib.txt \
# vignette=/home/hwj23/Dataset/TUM/sequence_19/sml_vignette.png \
# preset=0 \
# mode=0 \
# start=0

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/TUM/sequence_02/images_720 \
# calib=/home/hwj23/Dataset/TUM/720sml_calib_results_tum_fisheye.txt \
# gamma=/home/hwj23/Dataset/TUM/sequence_02/pcalib.txt \
# vignette=/home/hwj23/Dataset/TUM/sequence_19/720sml_vignette.png \
# preset=0 \
# mode=0 \
# start=0
