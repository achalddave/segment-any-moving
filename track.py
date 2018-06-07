"""Playground for tracking objects."""

import argparse
import io
import os
import pickle

import cv2
import numpy as np
import PIL
from matplotlib import pyplot as plt
from tqdm import tqdm

import utils.vis as vis
from utils.colors import colormap
from utils.datasets import get_classes


def visualize_detections(image,
                         boxes,
                         classes,
                         dataset,
                         threshold=0.7,
                         box_alpha=0.7,
                         dpi=200):
    class_list = get_classes(dataset)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    colors = colormap()

    image = image.astype(dtype=np.uint8)
    color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        color = [int(x) for x in colors[color_id, 0:3]]
        if score < threshold:
            continue

        x0, y0, x1, y1 = [int(x) for x in bbox]
        image = vis.vis_bbox(image, (x0, y0, x1 - x0, y1 - y0), color, thick=3)

        class_str = (
            class_list[classes[i]] + ' {:0.2f}'.format(score).lstrip('0'))
        image = vis.vis_class(image, (x0, y0 - 2), class_str)

        color_id = (color_id + 1) % len(colors)

    return image


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detectron-pickle', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--extension', default='.png')
    parser.add_argument('--dataset', default='coco', choices=['coco'])

    args = parser.parse_args()

    with open(args.detectron_pickle, 'rb') as f:
        data = pickle.load(f)

    frames = sorted(data.keys(), key=lambda x: int(x))

    for image_name in tqdm(frames):
        image = cv2.imread(
            os.path.join(args.images_dir, image_name + args.extension))
        image = image[:, :, ::-1]  # BGR -> RGB
        image_data = data[image_name]
        boxes, _, _, classes = vis.convert_from_cls_format(
            image_data['boxes'], image_data['segmentations'],
            image_data['keypoints'])
        new_image = visualize_detections(
            image, boxes, classes, dataset=args.dataset)
        new_image = PIL.Image.fromarray(new_image)
        new_image.save(
            os.path.join(args.output_dir, image_name + '.png'))


if __name__ == "__main__":
    main()
