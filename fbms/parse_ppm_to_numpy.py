"""Parse dense ppm files output by fbms/ochs_pami2014/densify.py to numpy."""

import argparse
import logging
import numpy as np
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils.fbms.utils import load_ppm_segmentation
from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dense-dir',
        type=Path,
        help=('Contains `frame_dense.ppm` file for each sequence, unless '
              '--ppm-files-in-subdirs is specified, in which case it should '
              'contain a subdirectory for each sequence, which in turn has a '
              'ppm files.'))
    parser.add_argument('--output', required=True)

    parser.add_argument(
        '--ppm-files-in-subdirs',
        action='store_true',
        help='See --dense-dir help for information.')
    parser.add_argument('--ignore-missing-dat', action='store_true')

    args = parser.parse_args()

    args.output = Path(args.output.format(dense_dir=args.dense_dir))

    args.output.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output / (Path(__file__).name + '.log'))

    logging.info('Args:\n%s' % args)

    sequences = []
    inputs = []
    if args.ppm_files_in_subdirs:
        sequence_dirs = [x for x in args.dense_dir.iterdir() if x.is_dir()]
    else:
        sequence_dirs = [args.dense_dir]

    for sequence_dir in sequence_dirs:
        if sequence_dir.is_dir():
            # Although we don't need .dat files to parse ppms, a missing
            # .dat file likely means that the sequence was not properly
            # computed, and we don't want to evaluate on those sequences.
            if not args.ignore_missing_dat:
                dat_file = list(sequence_dir.glob('*.dat'))
                if len(dat_file) != 1:
                    logging.error(
                        'Found %s (!= 1) dat files in %s, skipping.' %
                        (len(dat_file), sequence_dir))
                    continue
            ppm_files = sorted(
                sequence_dir.glob('*_dense.ppm'),
                key=lambda x: int(x.stem.split('_dense')[0]))
            inputs.append(ppm_files)
            sequences.append(sequence_dir.name)
    outputs = [args.output / (x + '.npz') for x in sequences]

    single_input = len(inputs) == 1
    for input_ppms, output_path in zip(
            tqdm(inputs, disable=single_input), outputs):
        if output_path.exists():
            logging.info('%s already exists, skipping.', output_path)
        segmentation = np.stack([load_ppm_segmentation(x) for x in input_ppms])
        unique_object_ids = np.unique(segmentation)
        segmentation_indexed = np.zeros_like(segmentation)
        for i, object_id in enumerate(unique_object_ids):
            segmentation_indexed[segmentation == object_id] = i
        np.savez_compressed(output_path, segmentation=segmentation_indexed)


if __name__ == "__main__":
    main()
