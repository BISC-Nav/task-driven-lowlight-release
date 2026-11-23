#!/bin/bash


TUM_PATH=/mnt/data2/retinex_dba/datasets/ROVER

evalset=(
    campus_large
    campus_small
    garden_large
    garden_small
    park
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_rover.py --datapath=$TUM_PATH/$seq --weights=droid.pth --disable_vis $@
done

