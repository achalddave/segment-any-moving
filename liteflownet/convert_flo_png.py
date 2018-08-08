"""Convert .flo files to .png files with angle/magnitude channels."""

import argparse
import logging
from pathlib import Path

import cv2.optflow
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from utils.log import setup_logging


def convert_flo(input_flo_path, output_image, output_metadata):
    flow = cv2.optflow.readOpticalFlow(str(input_flo_path))
    flow_x, flow_y = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x)

    # Normalize magnitude per frame
    min_magnitude, max_magnitude = magnitude.min(), magnitude.max()
    magnitude = 255 * (magnitude - min_magnitude) / (
        max_magnitude - min_magnitude)

    # Normalize angles generically; map output of arctan2 (in range
    # [-pi, pi]) to image range ([0, 255]).
    angle = 255 * (angle + np.pi) / (2 * np.pi)

    assert angle.min() >= 0, 'angle.min() (%s) < 0' % angle.min()
    assert angle.max() <= 255, 'angle.max() (%s) > 255' % angle.min()
    assert magnitude.min() >= 0, (
        'magnitude.min() (%s) < 0' % magnitude.min())
    assert magnitude.max() >= 0, (
        'magnitude.max() (%s) < 0' % magnitude.max())

    output_image = output_dir / (flo_path.stem + '.png')
    flow = np.zeros(flow.shape, dtype=np.uint8)
    flow[:, :, 0] = angle
    flow[:, :, 1] = magnitude
    image = Image.fromarray(flow)
    image.save(output_image)

    with open(output_metadata, 'w') as f:
        f.write('%s %s' % (min_magnitude, max_magnitude))


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-dir',
        help='Directory containing .flo files.',
        required=True)
    parser.add_argument(
        '--output-dir',
        help='Directory to output angle/magnitude png files to.',
        required=True)
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Look recursively in --input-dir for flo files.')
    parser.add_argument('--num-workers', default=8, type=int)

    args = parser.parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    assert input_root.exists()
    output_root.mkdir(exist_ok=True, parents=True)

    setup_logging(str(output_root / (Path(__file__).stem + '.log')))
    logging.info('Args: %s', vars(args))

    if args.recursive:
        flo_paths = list(input_root.rglob('*.flo'))
    else:
        flo_paths = list(input_root.glob('*.flo'))

    image_outputs = []
    metadata_outputs = []
    for flo_path in flo_paths:
        output_dir = output_root / flo_path.parent.relative_to(input_root)
        output_dir.mkdir(exist_ok=True, parents=True)
        image_outputs.append(output_dir / (flo_path.stem + '.png'))
        metadata_outputs.append(
            output_dir / (flo_path.stem + '_magnitude_minmax.txt'))


    tasks = zip(tqdm(flo_paths), image_outputs, metadata_outputs)
    Parallel(n_jobs=8)(delayed(convert_flo)(*task) for task in tasks)


if __name__ == "__main__":
    main()
