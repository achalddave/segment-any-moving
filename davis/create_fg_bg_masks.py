"""Create foreground/background motion masks from detections."""

import argparse
import logging
import pickle
import pprint
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import pycocotools.mask as mask_util

from utils.fbms import utils as fbms_utils
from utils.log import add_time_to_path, setup_logging


def create_masks_sequence(predictions_dir, output_dir, threshold, mask_shape,
                          duplicate_last_frame):
    pickle_files = sorted(
        predictions_dir.glob('*.pickle'), key=lambda x: int(x.stem))
    output_dir.mkdir(exist_ok=True, parents=True)
    for frame_number, pickle_file in enumerate(pickle_files):
        filename = pickle_file.stem
        output_path = output_dir / (filename + '.png')
        if output_path.exists():
            continue

        if not pickle_file.exists():
            logging.warn("Couldn't find detections for "
                         f"{pickle_file.relative_to(predictions_dir.parent)}")
            continue

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

    if duplicate_last_frame:
        original_last = output_dir / ('%05d.png' % (len(pickle_files) - 1))
        new_last = output_dir / ('%05d.png' % len(pickle_files))
        if not new_last.exists():
            import shutil
            shutil.copy(str(original_last), str(new_last))


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--detections-root',
        type=Path,
        help=('Contains subdirectory for each sequence, containing pickle '
              'files of detectron outputs for each frame.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--output-width', default=854, type=int)
    parser.add_argument('--output-height', default=480, type=int)
    parser.add_argument(
        '--duplicate-last-frame',
        action='store_true',
        help=('Whether to duplicate the last frame. This is useful if we '
              'only have predictions for n-1 frames (since flow is only '
              'computed on the first n-1 frames).'))

    args = parser.parse_args()

    assert args.detections_root.exists()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    setup_logging(
        add_time_to_path(args.output_dir / (Path(__file__).name + '.log')))
    logging.info('Args: %s\n', pprint.pformat(vars(args)))

    all_sequence_predictions = [
        x for x in args.detections_root.iterdir() if x.is_dir()
    ]
    for sequence_predictions in tqdm(all_sequence_predictions):
        create_masks_sequence(
            sequence_predictions,
            args.output_dir / sequence_predictions.name,
            args.threshold,
            mask_shape=(args.output_height, args.output_width),
            duplicate_last_frame=args.duplicate_last_frame)


if __name__ == "__main__":
    main()
