"""Visualize flow from png files containing angle/magnitude."""


import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.flow import compute_flow_color, load_flow_png
from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--recursive', action='store_true')

    args = parser.parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    assert input_root.exists()

    output_root.mkdir(exist_ok=True, parents=True)
    setup_logging(str(output_root / (Path(__file__).stem + '.log')))

    if args.recursive:
        flow_paths = input_root.rglob('*.png')
    else:
        flow_paths = input_root.glob('*.png')

    for flow_path in tqdm(flow_paths):
        flow = load_flow_png(str(flow_path), rgb=True)
        angle = (flow[:, :, 0] / 255.0 - 0.5) * 2   # move to range [-1, 1]
        magnitude = flow[:, :, 1]

        image = compute_flow_color(angle, magnitude)
        image = Image.fromarray(image)

        output_path = output_root / flow_path.relative_to(input_root)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        image.save(output_path)


if __name__ == "__main__":
    main()
