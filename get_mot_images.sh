#!/bin/bash

if [[ "$#" != 2 ]] ; then
    echo "Usage: $0 <MOT17_root> <output_images_root>"
    exit 1
fi

root=$1
out=$2

for split in train test ; do
    for name in $root/${split}/*-DPM ; do
        seq_name=${name##*/}
        seq_name_no_dpm="${seq_name%-DPM}"
        seq_out="${out}/${split}/${seq_name_no_dpm}"
        mkdir -p ${seq_out}
        cp -r ${name}/img1 "${seq_out}"
        echo "Copied ${seq_name_no_dpm} images"
    done
done
