"""Convert .flo files to .png files with angle/magnitude channels."""

import argparse
import collections
import logging
import subprocess
from pathlib import Path

import cv2.optflow
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from utils.log import setup_logging


def convert_flo_pavel_matlab(flo_dir, output_dir):
    pavel_code_dir = Path(__file__).parent / 'pavel_flow'
    command = [
        'matlab', '-r',
        "decodeFlowNet('%s', '%s'); quit" % (flo_dir.resolve(),
                                             output_dir.resolve())
    ]
    subprocess.check_output(command, cwd=pavel_code_dir)


def convert_flo(input_flo_path,
                output_image=None,
                metadata_suffix='_magnitude_minmax.txt',
                overwrite_existing=False):
    if not isinstance(input_flo_path, Path):
        input_flo_path = Path(input_flo_path)

    if output_image is None:
        output_image = input_flo_path.with_suffix('.png')
    output_metadata = output_image.with_name(output_image.stem +
                                             metadata_suffix)
    if not overwrite_existing and (output_image.exists()
                                   and output_metadata.exists()):
        return

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

    # Note that we have to create a 3 channel image, as otherwise PIL
    # saves the image in "LA" format, which OpenCV fails to read properly:
    # https://github.com/opencv/opencv/issues/12185
    flow = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
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
        '--pavel-matlab-conversion',
        action='store_true',
        help="Use Pavel Tokmakov's MATLAB code for converting .flo to .png")
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

    flo_paths_by_dir = collections.defaultdict(list)
    for flo_path in flo_paths:
        flo_paths_by_dir[flo_path.parent].append(flo_path)

    flo_output_dirs = {}
    for flo_dir in flo_paths_by_dir:
        output_dir = output_root / flo_dir.relative_to(input_root)
        flo_output_dirs[flo_dir] = output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

    if not args.pavel_matlab_conversion:
        image_outputs = []
        for flo_path in flo_paths:
            output_image = (
                flo_output_dirs[flo_path.parent] / (flo_path.stem + '.png'))
            image_outputs.append(output_image)

        tasks = zip(tqdm(flo_paths), image_outputs)
        Parallel(n_jobs=8)(
            delayed(convert_flo)(*task) for task in tasks)
    else:
        for flo_dir in tqdm(flo_paths_by_dir):
            convert_flo_pavel_matlab(flo_dir, flo_output_dirs[flo_dir])


if __name__ == "__main__":
    main()
