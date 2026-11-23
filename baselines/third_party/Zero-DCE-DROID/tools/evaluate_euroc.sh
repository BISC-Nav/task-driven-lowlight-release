#!/bin/bash


EUROC_PATH=/mnt/data2/retinex_dba/datasets/EuRoC

evalset=(
    V1_03_difficult
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --disable_vis --weights=droid.pth $@

done

