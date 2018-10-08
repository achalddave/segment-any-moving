#!/bin/bash

DETECTIONS_DIR=/data/achald/track/detectron-pytorch/e2e_mask_rcnn_R-50-FPN_1x_two_stream/ytvos-rgb+flow-all-moving-strict-sub-train-8-21-18_from-objectness-and-davis-ft3d_lr-0.002Sep27-12-30-16/step1999/infer-simple_fbms-test/outputs/
IMAGES_DIR=/data/all/FBMS/TestSet/
OUTPUT_DIR=/data/achald/track/detectron-pytorch/e2e_mask_rcnn_R-50-FPN_1x_two_stream/ytvos-rgb+flow-all-moving-strict-sub-train-8-21-18_from-objectness-and-davis-ft3d_lr-0.002Sep27-12-30-16/step1999/infer-simple_fbms-test/tracker/first-attempt/

mkdir -p ${OUTPUT_DIR}

SHOTS_TXT=${OUTPUT_DIR}/all_shots.txt
TRACKS_TXT=${OUTPUT_DIR}/all_tracks.txt
rm ${TRACKS_TXT} ; touch ${TRACKS_TXT}
rm ${SHOTS_TXT} ; touch ${SHOTS_TXT}

num_sequences=$(/bin/ls ${IMAGES_DIR} | wc -l)
echo "${num_sequences}" >> ${SHOTS_TXT}

for seq_dir in ${IMAGES_DIR}/* ; do
    seq=${seq_dir##*/}
    python track.py \
        --detectron-dir ${DETECTIONS_DIR}/${seq} \
        --images-dir ${seq_dir} \
        --output-video ${OUTPUT_DIR}/${seq}.mp4 \
        --output-video-fps 10 \
        --output-fbms-file ${OUTPUT_DIR}/${seq}.dat \
        --fbms-groundtruth ${seq_dir}/GroundTruth \
        --extension '.jpg' \
        --filename-format fbms
    echo "$(readlink -e ${OUTPUT_DIR}/${seq}.dat)" >> ${TRACKS_TXT}
    echo "${seq_dir}/GroundTruth/${seq}Def.dat" >> ${SHOTS_TXT}
done
