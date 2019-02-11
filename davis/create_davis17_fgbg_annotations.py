"""Convert DAVIS 17 annotations to foreground/background annotations."""

import argparse
import logging
import pprint
import subprocess
from pathlib import Path

import numpy as np
import scipy.misc
from PIL import Image
from tqdm import tqdm

from utils import log as log_utils


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('davis_annotations', type=Path)
    parser.add_argument('output_dir', type=Path)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    output_log_file = log_utils.add_time_to_path(
        args.output_dir / Path(__file__).name)
    log_utils.setup_logging(output_log_file)
    logging.info('Args: %s', pprint.pformat(vars(args)))

    subprocess.call([
        './git-state/save_git_state.sh',
        output_log_file.with_suffix('.git-state')
    ])

    for sequence_dir in tqdm(list(args.davis_annotations.iterdir())):
        if not sequence_dir.is_dir():
            continue
        sequence = sequence_dir.stem
        output_sequence_dir = args.output_dir / sequence
        output_sequence_dir.mkdir(exist_ok=True)
        for image_path in sequence_dir.glob('*.png'):
            image = np.array(Image.open(image_path))
            if image.ndim != 2:
                __import__('ipdb').set_trace()
            image = (image != 0) * 256
            scipy.misc.imsave(args.output_dir / sequence / image_path.name,
                              image)


if __name__ == "__main__":
    main()
