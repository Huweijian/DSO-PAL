#!/bin/bash
./build/bin/dso_dataset \
files=/home/hwj23/Dataset/PAL/simu/s2/color/ \
calib=/home/hwj23/Dataset/PAL/calib_results.txt \
preset=0 \
mode=2

# ./build/bin/dso_dataset \
# files=/home/hwj23/Dataset/PAL/real/s1/images \
# calib=/home/hwj23/Dataset/PAL/calib_results.txt \
# gamma=/home/hwj23/Dataset/PAL/pcalib.txt \
# preset=0 \
# mode=1