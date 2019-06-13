#!/usr/bin/zsh

. ./env.sh activate
run_sequence() {
    sequence=${1##*/}
    python fbms/ochs_pami2014/densify.py \
        --sparse-dir /ssd1/achald/ytvos/keuper-iccv15/results/strict/${sequence}/MulticutResults/ldof0.5000004/ \
        --images-dir /ssd1/achald/ytvos/keuper-iccv15/results/strict/${sequence} \
        --output-dir /ssd1/achald/ytvos/keuper-iccv15/results/strict-densified/${sequence}
}

# prll -c 8 run_sequence /ssd1/achald/ytvos/keuper-iccv15/results/strict/*
for dir in /ssd1/achald/ytvos/keuper-iccv15/results/strict/* ; do
    echo ${dir}
    run_sequence ${dir}
done
