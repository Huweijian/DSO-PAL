#!/bin/bash
./build/bin/dso_dataset \
files=/home/hwj23/Dataset/D435/s12/images \
calib=/home/hwj23/Dataset/D435/camera.txt \
gamma=/home/hwj23/Dataset/D435/pcalib.txt \
vignette=/home/hwj23/Dataset/D435/vignetteSmoothed.png \
preset=0 \
sampleoutput=0 \
mode=0 \
trajectory=/home/hwj23/Dataset/D435/trajectory_304_306.txt

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/PAL/simu/hao/path3/d435/ \
# calib=/home/hwj23/Dataset/D435/camera_color.txt \
# preset=0 \
# mode=2 \
# sampleoutput=1