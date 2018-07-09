"""Visualize nearest neighbors for a few detected objects."""

import argparse
import logging
import os
import pickle
import random
from queue import PriorityQueue

import cv2
import numpy as np
import PIL
import pycocotools.mask as mask_util
from tqdm import tqdm

import utils.vis as vis
from utils.distance import histogram_distance, NeighborsQueue
from track import Detection


def detections_from_detectron_data(detectron_frame_data, image, timestamp):
    boxes, masks, _, labels = vis.convert_from_cls_format(
        detectron_frame_data['boxes'], detectron_frame_data['segmentations'],
        detectron_frame_data['keypoints'])

    masks = mask_util.decode(masks)  # Shape (height, width, num_masks)
    masks = [masks[:, :, i] for i in range(masks.shape[-1])]
    assert len(boxes) == len(masks) == len(labels)
    detections = [
        Detection(box[:4], box[4], label, timestamp, image, mask)
        for box, mask, label in zip(boxes, masks, labels)
    ]
    return detections


def save_image(image_np, output_path):
    PIL.Image.fromarray(image_np).save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detectron-pickle', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-neighbors-dir', required=True)
    parser.add_argument('--extension', default='.png')
    parser.add_argument('--dataset', default='coco', choices=['coco'])
    parser.add_argument('--num-queries', default=5, type=int,
                        help='How many queries to visualize neighbors for')
    parser.add_argument('--num-neighbors', default=100, type=int,
                        help='How many neighbors to visualize')
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_neighbors_dir, exist_ok=True)

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')

    with open(args.detectron_pickle, 'rb') as f:
        data = pickle.load(f)

    frames = sorted(data.keys(), key=lambda x: int(x))[:10]
    sampled_frames = random.sample(frames, args.num_queries)
    logging.info('Loading images')
    images = {}
    for name in tqdm(frames):
        image = cv2.imread(
            os.path.join(args.images_dir, name + args.extension))
        image = image[:, :, ::-1]  # BGR -> RGB
        images[name] = image

    logging.info('Finding neighbors')
    mask_color = [205, 168, 255]
    for query_name in tqdm(sampled_frames):
        query_timestamp = int(query_name)
        query_image = images[query_name]
        query_data = data[query_name]

        query_detections = detections_from_detectron_data(
            query_data, query_image, query_timestamp)
        query_detections = [x for x in query_detections if x.label == 1]
        query_detection = random.choice(query_detections)

        # Save query image
        query_image_mask = vis.vis_mask(
            query_detection.image,
            query_detection.mask,
            color=mask_color,
            alpha=0.0,
            border_thick=5)
        save_image(
            query_image_mask,
            os.path.join(args.output_neighbors_dir,
                         'query-%s.png' % query_name))

        neighbors = NeighborsQueue(maxsize=args.num_neighbors)
        for timestamp, name in enumerate(tqdm(frames)):
            image = images[name]
            frame_data = data[name]
            detections = detections_from_detectron_data(
                frame_data, image, timestamp)
            for detection in detections:
                distance = histogram_distance(query_detection, detection)
                neighbors.put(detection, distance)

        # List of (detection, distance) tuples.
        neighbors_list = [neighbors.get() for _ in range(args.num_neighbors)]
        # neighbors are returned in furthest-to-closest order; reverse that.
        neighbors_list = neighbors_list[::-1]

        for neighbor_index, neighbor in enumerate(neighbors_list):
            neighbor_detection, distance = neighbor
            neighbor_mask = vis.vis_mask(
                neighbor_detection.image,
                neighbor_detection.mask,
                color=mask_color,
                alpha=0.0,
                border_thick=5)
            save_image(
                neighbor_mask,
                os.path.join(
                    args.output_neighbors_dir,
                    'query-%s-neighbor-%s-frame-%s-distance-%.4f.png' %
                    (query_name, neighbor_index, neighbor_detection.timestamp,
                     100 * distance)))


if __name__ == "__main__":
    main()