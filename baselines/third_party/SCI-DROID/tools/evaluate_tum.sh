#!/bin/bash


TUM_PATH=/mnt/data2/retinex_dba/datasets/TUM-RGBD/

evalset=(
    rgbd_dataset_freiburg1_360
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_tum_dark.py --datapath=$TUM_PATH/$seq --weights=droid.pth --disable_vis $@
done

