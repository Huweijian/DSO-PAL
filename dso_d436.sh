#!/bin/bash

# ./build/bin/dso_dataset_rel \
# files=/home/hwj23/Dataset/simu/movexz/color \
# calib=/home/hwj23/Dataset/simu/camera.cfg \
# preset=0 \
# mode=2

./build/bin/dso_dataset_rel \
files=/home/hwj23/Dataset/D435/s1/images \
calib=/home/hwj23/Dataset/D435/camera.txt \
gamma=/home/hwj23/Dataset/D435/pcalib.txt \
vignette=/home/hwj23/Dataset/D435/vignetteSmoothed.png \
preset=0 \
mode=0