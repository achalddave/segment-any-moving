"""Extract masks from rendered images.

The first author of [1] provided masks for the output of their method
rendered on the original images. He no longer had access to the original
masks, and I was unable to get their code working. This script attempts to
extract the raw masks from the rendered images through image subtraction +
extracting connected components. This is clearly less than ideal, but is the
closest I can get to evaluating [1] with our proposed metric (primarily for
a reviewer who asked for this comparison).

[1]: Khan, Naeemullah, et al. "Coarse-to-fine segmentation with
shape-tailored continuum scale spaces." Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition. 2017.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from script_utils.common import common_setup
from tqdm import tqdm

from utils.fbms.utils import get_framenumber


def quantize_colors(image, compactness_threshold=10):
    """From
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html"""
    # image = image.copy()
    # image[~np.all(image == [0, 0, 0])] = 255
    # return image, image
    # convert to np.float32
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    n = Z.shape[0]

    for i in range(2, 9):
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,
                    1.0)
        compactness, label, center = cv2.kmeans(Z, i, None, criteria, 10,
                                                cv2.KMEANS_RANDOM_CENTERS)
        compactness /= n
        compactness = np.sqrt(compactness)
        if compactness < compactness_threshold:
            print(f'Using {i}, compactness {compactness}')
            break
        else:
            print(f'NOT using {i}, compactness {compactness}')

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2, label.reshape(image.shape[:-1])


def compute_diff(raw_image, rendered_image):
    raw_image = cv2.imread(str(raw_image))
    rendered_image = cv2.imread(str(rendered_image))
    h, w = rendered_image.shape[:2]
    orig_h, orig_w = raw_image.shape[:2]
    raw_image = cv2.resize(raw_image, (w, h))
    raw_diff = np.abs(rendered_image.astype(float) - raw_image.astype(float))
    threshold = 20
    diff = raw_diff.copy()
    diff[diff < threshold] = 0
    near_gray = (diff.max(axis=2) - diff.min(axis=2)) < 5
    diff[near_gray] = 0
    new_raw_image = raw_image.astype(float)
    new_raw_image[diff > threshold] = raw_image[diff > threshold] * 0.7
    diff = np.abs(rendered_image.astype(float) - new_raw_image)
    diff = cv2.medianBlur(diff.astype(np.uint8), 5)
    return diff


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--rendered-masks',
        type=Path,
        default=Path('/data/achald/track/FBMS/khan-continuum_cvpr17/outputs/'))
    parser.add_argument(
        '--fbms-root',
        help='Should contain TrainingSet and TestSet dirs.',
        type=Path,
        default=Path('/data/all/FBMS'))
    parser.add_argument('--output-dir', type=Path, required=True)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    test_dir = args.fbms_root / 'TestSet'
    train_dir = args.fbms_root / 'TrainingSet'

    output_vis = args.output_dir / 'vis'
    output_mask = args.output_dir / 'masks'
    output_vis.mkdir(exist_ok=True)
    output_mask.mkdir(exist_ok=True)
    for split_dir in [train_dir, test_dir]:
        for subdir in tqdm(list(split_dir.iterdir())):
            if not subdir.is_dir():
                continue
            output_seq_vis = output_vis / subdir.stem
            output_seq_vis.mkdir(exist_ok=True)
            sequence = subdir.name
            mask_dir = args.rendered_masks / sequence
            frames = sorted(subdir.glob('*.jpg'),
                            key=lambda x: get_framenumber(x))
            output_numpy = output_mask / (subdir.stem + '.npz')
            # if output_numpy.exists():
            #     logging.info(f'{output_numpy} already exists, skipping')
            #     continue
            segmentation = []
            image_h, image_w = cv2.imread(str(frames[0])).shape[:2]
            diffs = []
            for i, frame in enumerate(frames):
                if i == 0:
                    continue
                frame_no = get_framenumber(frame) - 1
                segmentation_path = (
                    mask_dir / f'segmentation{frame_no:04d}.png')
                if not segmentation_path.exists():
                    diffs.append(None)
                    logging.error(
                        f'Missing segmentation at {segmentation_path}')
                    continue
                diff = compute_diff(frame, segmentation_path)
                cv2.imwrite(str(output_seq_vis / f'diff{i:04d}.png'),
                            diff)
                diffs.append(diff)

            try:
                valid_diff = next(x for x in diffs if x is not None)
            except StopIteration:
                logging.error(f'No valid segmentation found in {subdir}')
                continue

            for i, x in enumerate(diffs):
                if x is None:
                    diffs[i] = np.zeros_like(valid_diff)

            quantized, segmentation = quantize_colors(np.stack(diffs))
            resized_segmentation = []
            for i in range(quantized.shape[0]):
                resized_segmentation.append(
                    cv2.resize(segmentation[i], (image_w, image_h),
                               interpolation=cv2.INTER_NEAREST))
                cv2.imwrite(str(output_seq_vis / f'mask{i+1:04d}.png'),
                            quantized[i])
            np.savez_compressed(output_numpy, np.stack(resized_segmentation))


if __name__ == "__main__":
    main()