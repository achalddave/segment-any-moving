#!/bin/bash
# Create a directory structure like that of MOT17, but with no detector name in
# the sequences. This lets us evaluate various detectors with the MOT17
# evaluation code.

if [[ "$#" != 2 ]] ; then
    echo "Usage:"
    echo "$0 <mot17-dir> <output>"
    exit 1
fi

mot=$1
output=$2
mkdir -p $output

for i in ${mot}/*-DPM ; do
    dirname=${i##*/}
    sequence=${dirname%-*}
    mkdir -p ${output}/${sequence}
    for j in $i/* ; do
        ln -s $(readlink -e $j) ${output}/${sequence}/
    done
done
