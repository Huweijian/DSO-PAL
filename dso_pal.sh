#!/bin/bash

# my simu dataset
./build/bin/dso_dataset \
files=/home/hwj23/Dataset/PAL/simu/s16/color/ \
calib=/home/hwj23/Dataset/PAL/calib_results_simu.txt \
preset=0 \
mode=2 \
sampleoutput=1

# hao simu dataset
# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/PAL/simu/hao/path2/bigpal/ \
# calib=/home/hwj23/Dataset/PAL/calib_results_hao.txt \
# preset=0 \
# mode=2 \
# sampleoutput=1

# # real dataset
# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/PAL/real/s42/images \
# calib=/home/hwj23/Dataset/PAL/calib_results_real.txt \
# gamma=/home/hwj23/Dataset/PAL/pcalib.txt \
# vignette=/home/hwj23/Dataset/PAL/vignette.png \
# preset=0 \
# mode=0 \
# start=0 \
# trajectory=/home/hwj23/Dataset/PAL/trajectory_223_231.txt \
# sampleoutput=1 \
