"""Split training set of YTVOS into sub-train/sub-val subsets.

Since annotations for the test set are not available, we will use the
validation set of YTVOS as the test set, and create a train/validation subset
of the YTVOS train set."""

import argparse
import logging
import math
import random
from pathlib import Path

from utils.log import setup_logging


def main():
    with open(__file__, 'r') as f:
        _source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ytvos-train-root', required=True)
    parser.add_argument('--output-sub-train', required=True)
    parser.add_argument('--output-sub-val', required=True)
    parser.add_argument('--train-portion', default=0.8, type=float)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    logging_path = args.output_sub_train + '.log'
    setup_logging(logging_path)

    logging.info('Source file: %s' % Path(__file__).resolve())
    logging.info('Args:\n%s', vars(args))

    ytvos_train_root = Path(args.ytvos_train_root)
    images_root = ytvos_train_root / 'JPEGImages'
    if not images_root.exists():
        raise ValueError('Path to images does not exist: %s' % images_root)

    sequences = [x.stem for x in images_root.iterdir()]
    random.seed(args.seed)
    random.shuffle(sequences)

    num_train = math.floor(len(sequences) * args.train_portion)
    train_sequences = sequences[:num_train]
    val_sequences = sequences[num_train:]

    with open(args.output_sub_train, 'w') as f:
        for sequence in train_sequences:
            f.write(sequence + '\n')

    with open(args.output_sub_val, 'w') as f:
        for sequence in val_sequences:
            f.write(sequence + '\n')

    file_logger = logging.getLogger(logging_path)
    file_logger.info('Source')
    file_logger.info('======')
    file_logger.info(_source)


if __name__ == "__main__":
    main()
