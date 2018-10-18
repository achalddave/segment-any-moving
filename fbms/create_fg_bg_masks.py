"""Create foreground/background motion masks from detections."""

import argparse
import logging
import pickle
import pprint
from pathlib import Path

import numpy as np
from PIL import Image

import pycocotools.mask as mask_util

from utils.fbms import utils as fbms_utils
from utils.log import add_time_to_path, setup_logging


def create_masks_sequence(groundtruth_dir, predictions_dir, output_dir,
                          threshold):
    groundtruth = fbms_utils.FbmsGroundtruth(groundtruth_dir / 'GroundTruth')
    mask_shape = None
    for frame_number, frame_path in groundtruth.frame_label_paths.items():
        filename = frame_path.stem
        filename = filename.replace('_gt', '')
        pickle_file = predictions_dir / (filename + '.pickle')
        output_path = output_dir / (filename + '.png')
        if output_path.exists():
            continue

        if not pickle_file.exists():
            logging.warn("Couldn't find detections for "
                         f"{pickle_file.relative_to(predictions_dir.parent)}")
            continue

        if mask_shape is None:
            image_size = Image.open(frame_path).size
            mask_shape = (image_size[1], image_size[0])

        with open(pickle_file, 'rb') as f:
            frame_data = pickle.load(f)

        if frame_data['segmentations'] is None:
            frame_data['segmentations'] = [
                [] for _ in range(len(frame_data['boxes']))
            ]

        segmentations = []
        scores = []
        # Merge all classes into one.
        for c in range(1, len(frame_data['segmentations'])):
            scores.extend(frame_data['boxes'][c][:, 4])
            segmentations.extend(frame_data['segmentations'][c])

        final_mask = np.zeros(mask_shape, dtype=np.uint8)
        for score, segmentation in zip(scores, segmentations):
            if score <= threshold:
                continue
            mask = mask_util.decode(segmentation)
            final_mask[mask == 1] = 255

        Image.fromarray(final_mask).save(output_path)


def create_masks_split(groundtruth_dir, predictions_dir, output_dir,
                       threshold):
    """
    Args:
        groundtruth_dir (Path)
        predictions_dir (Path)
        output_dir (Path)
    """
    for sequence_groundtruth in groundtruth_dir.iterdir():
        if not sequence_groundtruth.is_dir():
            continue
        sequence_predictions = predictions_dir / sequence_groundtruth.name
        sequence_output = output_dir / sequence_groundtruth.name
        assert sequence_predictions.exists(), (
            f"Couldn't find sequence predictions at {sequence_predictions}")
        sequence_output.mkdir(exist_ok=True, parents=True)
        create_masks_sequence(sequence_groundtruth, sequence_predictions,
                              sequence_output, threshold)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detections-root', type=Path, required=True)
    parser.add_argument('--fbms-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.7)

    args = parser.parse_args()

    fbms_root = args.fbms_root
    detections_root = args.detections_root
    output_dir = args.output_dir

    # assert not output_dir.exists()
    assert detections_root.exists()
    assert fbms_root.exists()

    output_dir.mkdir(exist_ok=True, parents=True)

    setup_logging(
        add_time_to_path(output_dir / (Path(__file__).name + '.log')))
    logging.info('Args: %s\n', pprint.pformat(vars(args)))

    train_split = 'TrainingSet'
    train_fbms = fbms_root / train_split
    if train_fbms.exists():
        train_detections = detections_root / train_split
        train_output = output_dir / train_split
        assert train_detections.exists(), (
            f'No detections found for TrainingSet at {train_detections}')
        create_masks_split(train_fbms, train_detections, train_output,
                           args.threshold)

    test_split = 'TestSet'
    test_fbms = fbms_root / test_split
    if test_fbms.exists():
        test_detections = detections_root / test_split
        test_output = output_dir / test_split
        assert test_detections.exists(), (
            f'No detections found for TestSet at {test_detections}')
        create_masks_split(test_fbms, test_detections, test_output,
                           args.threshold)

    if not (train_fbms.exists() or test_fbms.exists()):
        # Assume that --fbms-root and --detections-root refer to a specific
        # split.
        create_masks_split(fbms_root, detections_root, output_dir,
                           args.threshold)


if __name__ == "__main__":
    main()
