#!/bin/bash

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/PAL/simu/s2/color/ \
# calib=/home/hwj23/Dataset/PAL/calib_results_simu.txt \
# preset=0 \
# mode=2

./build/bin/dso_dataset \
files=/home/hwj23/Dataset/PAL/real/s15/images \
calib=/home/hwj23/Dataset/PAL/calib_results_real.txt \
gamma=/home/hwj23/Dataset/PAL/pcalib.txt \
vignette=/home/hwj23/Dataset/PAL/vignette.png \
preset=0 \
mode=0 \
start=800 \
# sampleoutput=1 \