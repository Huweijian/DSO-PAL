#!/bin/bash

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/PAL/simu/hs1/color/ \
# calib=/home/hwj23/Dataset/PAL/calib_results_simu.txt \
# preset=0 \
# mode=2

./build/bin/dso_dataset \
files=/home/hwj23/Dataset/PAL/real/s4/images \
calib=/home/hwj23/Dataset/PAL/calib_results_real.txt \
gamma=/home/hwj23/Dataset/PAL/pcalib.txt \
preset=0 \
mode=1
