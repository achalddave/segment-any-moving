"""Create foreground/background motion masks from detections."""

import argparse
import collections
import logging
import pickle
import pprint
from pathlib import Path

import numpy as np
from natsort import natsorted, ns
from PIL import Image
from tqdm import tqdm

import pycocotools.mask as mask_util

from utils.log import add_time_to_path, setup_logging


def create_masks_sequence(dir_or_pickles, images_dir, output_dir,
                          threshold, extension, duplicate_last_frame):
    if isinstance(dir_or_pickles, list):
        pickle_files = dir_or_pickles
        predictions_dir = pickle_files[0].parent
    else:
        predictions_dir = dir_or_pickles
        pickle_files = natsorted(predictions_dir.glob('*.pickle'), alg=ns.PATH)
        if not pickle_files:
            logging.warn("Found no pickle files in %s; ignoring.",
                         predictions_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for frame_number, pickle_file in enumerate(pickle_files):
        filename = pickle_file.stem
        output_path = output_dir / (filename + '.png')
        if output_path.exists():
            continue
        w, h = Image.open(images_dir / (filename + extension)).size
        mask_shape = (h, w)

        if not pickle_file.exists():
            logging.warn(f"Couldn't find detections for {pickle_file}")
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
        try:
            int(pickle_files[0].stem)
        except ValueError as e:
            logging.warn('Could not parse pickle file as integer, not '
                         'duplicating last frame.')

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
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument(
        '--extension',
        default='.png',
        help='Extension for images in --images-dir')
    parser.add_argument(
        '--duplicate-last-frame',
        action='store_true',
        help=('Whether to duplicate the last frame. This is useful if we '
              'only have predictions for n-1 frames (since flow is only '
              'computed on the first n-1 frames). NOTE: This only works if '
              'the pickle files are of the form "<frame_id>.pickle".'))

    args = parser.parse_args()

    assert args.detections_root.exists()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    setup_logging(
        add_time_to_path(args.output_dir / (Path(__file__).name + '.log')))
    logging.info('Args: %s\n', pprint.pformat(vars(args)))

    if args.recursive:
        all_pickles = args.detections_root.rglob('*.pickle')
        all_predictions = collections.defaultdict(list)
        for x in all_pickles:
            all_predictions[x.parent].append(x)
    else:
        all_predictions = {
            args.detections_root: list(args.detections_root.glob('*.pickle'))
        }
        if not all_predictions[args.detections_root]:
            raise ValueError("Found no .pickle files in --detections-root. "
                             "Did you mean to specify --recursive?")
    all_predictions = {
        k: natsorted(v, alg=ns.PATH)
        for k, v in all_predictions.items()
    }

    # The DAVIS 2016 evaluation code really doesn't like any other files /
    # directories in the input directory, so we put the masks in a subdirectory
    # without the log file.
    masks_output_dir = args.output_dir / 'masks'
    for sequence_dir, sequence_predictions in tqdm(all_predictions.items()):
        relative_dir = sequence_dir.relative_to(args.detections_root)
        create_masks_sequence(
            sequence_predictions,
            args.images_dir / relative_dir,
            masks_output_dir / relative_dir,
            args.threshold,
            args.extension,
            duplicate_last_frame=args.duplicate_last_frame)


if __name__ == "__main__":
    main()
