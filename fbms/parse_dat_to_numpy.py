import argparse
import logging
import numpy as np

from utils.fbms.utils import parse_tracks
from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dat_file')
    parser.add_argument('output_numpy')
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

    with open(args.dat_file, 'r') as f:
        tracks, num_frames = parse_tracks(
            f.read(),
            track_label_size_space_separated=args.label_size_space_separated)

    width = max(p[0] for points in tracks.values() for p in points) + 1
    height = max(p[1] for points in tracks.values() for p in points) + 1
    logging.info('Inferred resolution: %sx%s' % (width, height))

    segmentation = np.zeros((num_frames, height, width)) - 1
    assert -1 not in tracks.keys()

    for label, points in tracks.items():
        for x, y, f in points:
            segmentation[f, y, x] = label

    np.save(args.output_numpy, segmentation)


if __name__ == "__main__":
    main()
