import argparse
import logging
import numpy as np
from pathlib import Path

from utils.fbms.utils import FbmsGroundtruth, parse_tracks
from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dat-file', required=True)
    parser.add_argument('--output-numpy', required=True,)
    parser.add_argument(
        '--groundtruth-dir',
        type=Path,
        required=True,
        help='Groundtruth dir for this sequence, containing a .dat file.')
    parser.add_argument(
        '--label-size-space-separated',
        action='store_true',
        help=('The FBMS code requires that there is a newline between <track '
              'label> and <track size>, but one of the outputs I downloaded '
              'from a method has the two fields separated by a space. This '
              'boolean allows parsing such files.'))

    args = parser.parse_args()
    setup_logging(args.output_numpy + '.log')
    logging.info('Args:\n%s' % args)

    groundtruth = FbmsGroundtruth(args.groundtruth_dir)
    with open(args.dat_file, 'r') as f:
        tracks_txt = f.read()

    segmentation = parse_tracks(
        tracks_txt,
        image_shape=(groundtruth.image_height, groundtruth.image_width),
        track_label_size_space_separated=args.label_size_space_separated)
    np.save(args.output_numpy, segmentation)


if __name__ == "__main__":
    main()
