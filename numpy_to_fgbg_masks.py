"""Convert numpy predictions to foreground background masks."""

import argparse
import logging
import pprint
from pathlib import Path

import numpy as np
import scipy.misc
from tqdm import tqdm

from utils.log import add_time_to_path, setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('numpy_dir', type=Path)
    parser.add_argument('output_dir', type=Path)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / 'masks'
    output.mkdir()

    setup_logging(
        add_time_to_path(args.output_dir / (Path(__file__).name + '.log')))
    logging.info('Args: %s\n', pprint.pformat(vars(args)))

    for path in tqdm(list(args.numpy_dir.rglob('*.npy'))):
        segmentation = np.load(path)
        sequence_output = output / path.stem
        sequence_output.mkdir()
        for frame, frame_segmentation in enumerate(segmentation):
            scipy.misc.imsave(sequence_output / f'{frame:05d}.png',
                              (frame_segmentation != 0) * 255)


if __name__ == "__main__":
    main()
