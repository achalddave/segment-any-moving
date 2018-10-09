"""Concatenate angle/magnitude flows with predicted boundaries.

Given a pair of images, one containing flow angle/magnitude in the R and G
channels, and one a grayscale image containing predicted image boundaries,
output an image where the first two channels (R and G) contain flow
angle/magnitude, and the last channel (B) contains boundary detections."""


import argparse
import logging
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.log import setup_logging


def concat_flows(flow_path, boundary_path, output_path, copy_magnitude):
    if output_path.exists():
        return
    flow = np.array(Image.open(flow_path))
    boundary = np.array(Image.open(boundary_path))
    if boundary.dtype == object:
        __import__('ipdb').set_trace()
        raise ValueError(
            'Boundary png (%s) could not be loaded properly.' % boundary_path)
    flow[:, :, 2] = boundary
    flow = Image.fromarray(flow)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    flow.save(output_path)
    if copy_magnitude:
        magnitude_path = flow_path.with_name(flow_path.stem +
                                             '_magnitude_minmax.txt')
        output_magnitude_path = output_path.with_name(output_path.stem + '_magnitude_minmax.txt')
        shutil.copy(magnitude_path, output_magnitude_path)


def concat_flows_star(args):
    return concat_flows(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flow-root', required=True)
    parser.add_argument('--boundary-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument(
        '--copy-magnitude-txt',
        action='store_true',
        help='Copy magnitude_minmax.txt files in the flow root to output.')
    parser.add_argument('--num-workers', default=4, type=int)

    args = parser.parse_args()

    flow_root = Path(args.flow_root)
    boundary_root = Path(args.boundary_root)
    assert flow_root.exists()
    assert boundary_root.exists()
    output_root = Path(args.output_root)

    output_root.mkdir(exist_ok=True)
    logging_path = str(output_root / (Path(__file__).stem + '.py.log'))
    setup_logging(logging_path)
    logging.info('Script path: %s', Path(__file__).resolve())
    logging.info('Args:\n%s', vars(args))
    file_logger = logging.getLogger(logging_path)

    if args.recursive:
        input_flows = list(flow_root.rglob('*.png'))
        input_boundaries = list(boundary_root.rglob('*.png'))
    else:
        input_flows = list(flow_root.glob('*.png'))
        input_boundaries = list(boundary_root.glob('*.png'))

    input_flow_relative = [x.relative_to(flow_root) for x in input_flows]
    input_boundary_relative = [
        x.relative_to(boundary_root) for x in input_boundaries
    ]

    flow_set = set(input_flow_relative)
    boundary_set = set(input_boundary_relative)

    # Boundary files should always exist, but we can ignore missing flow files
    # since the last frame for each sequence does not have flow.
    missing_boundary = flow_set - boundary_set
    missing_flow = boundary_set - flow_set
    if missing_boundary:
        raise ValueError('%s files not found in boundary root:\n%s' %
                         (len(missing_boundary),
                          '\n'.join([str(x) for x in missing_boundary])))

    if missing_flow:
        logging.fatal('%s files not found in flow root.' % len(missing_flow))
        file_logger.info(
            'Missing files: %s' % ('\n'.join([str(x) for x in missing_flow])))

    tasks = []
    for flow_path in tqdm(input_flows):
        output_path = output_root / (flow_path.relative_to(flow_root))
        boundary_path = boundary_root / (flow_path.relative_to(flow_root))
        tasks.append((flow_path, boundary_path, output_path,
                      args.copy_magnitude_txt))

    pool = Pool(args.num_workers)
    output_generator = pool.imap_unordered(concat_flows_star, tasks)

    with tqdm(total=len(tasks)) as progress_bar:
        for i, _ in enumerate(output_generator):
            progress_bar.update(1)
            if i % 1000 == 0:
                file_logger.info('Processed %s images' % i)


if __name__ == "__main__":
    main()
