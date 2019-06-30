#!/bin/bash

python ytvos/split_annotations_coco_nococo.py \
--annotations-dir /data/achald/track/ytvos/moving-only/labels-8-21-18/all-moving/strict/Annotations/val \
--output-dir /data/achald/track/ytvos/moving-only/labels-8-21-18/all-moving/strict/coco_nococo_split/val/

python ytvos/split_annotations_coco_nococo.py \
--annotations-dir /data/achald/track/ytvos/moving-only/labels-8-21-18/all-moving/strict/Annotations/train \
--output-dir /data/achald/track/ytvos/moving-only/labels-8-21-18/all-moving/strict/coco_nococo_split/train