"""Visualize flow from png files containing angle/magnitude."""

import argparse
import collections
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from utils.flow import compute_flow_color, load_flow_png
from utils.log import setup_logging


def visualize_flow(flow_or_path,
                   output_visualization,
                   magnitude_scale=None,
                   visualize_angle_only=False,
                   clip_at_magnitude_scale=False,
                   min_normalize_percent=0):
    if isinstance(flow_or_path, str) or isinstance(flow_or_path, Path):
        flow = load_flow_png(str(flow_or_path))
        diagonal = (flow.shape[0]**2 + flow.shape[1]**2)**0.5
        if clip_at_magnitude_scale:
            clip_value = magnitude_scale
        else:
            clip_value = diagonal
        flow[:, :, 1] = flow[:, :, 1].clip(max=clip_value)
    elif isinstance(flow_or_path, np.ndarray):
        flow = flow_or_path
        diagonal = (flow.shape[0]**2 + flow.shape[1]**2)**0.5
    else:
        raise ValueError('Unknown type %s for argument flow_or_path' %
                         type(flow_or_path))

    if visualize_angle_only:
        magnitude = np.ones_like(flow[:, :, 1])
    else:
        magnitude = flow[:, :, 1]

        if magnitude_scale is None:
            magnitude_scale = magnitude.max()
            if min_normalize_percent > 0:
                magnitude_scale = max(magnitude_scale,
                                      diagonal * min_normalize_percent)

        magnitude /= magnitude_scale

    # move to range [-pi, pi]
    angle = (flow[:, :, 0] / 255.0 - 0.5) * 2 * np.pi

    image = compute_flow_color(angle, magnitude)
    image = Image.fromarray(image)

    image.save(output_visualization)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument(
        '--sequence-normalize',
        action='store_true',
        help=("Normalize all images together. If --recursive is True, "
              "normalize all images with the same parent directory together."))
    parser.add_argument(
        '--diagonal-normalize',
        action='store_true',
        help='Normalize magnitude proportional to the image diagonal length.')
    parser.add_argument(
        '--min-normalize-perc',
        default=0,
        type=float,
        help=('Minimum magnitude normalization, as percent of image diagonal.'
              'If <= 0, no minimum is used.'))
    parser.add_argument(
        '--visualize-angle-only',
        action='store_true')

    args = parser.parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    assert input_root.exists()

    output_root.mkdir(exist_ok=True, parents=True)
    setup_logging(str(output_root / (Path(__file__).stem + '.log')))

    logging.info('Args:\n%s', vars(args))
    logging.info('Collecting paths')
    if args.recursive:
        flow_paths = list(tqdm(input_root.rglob('*.png')))
    else:
        flow_paths = list(tqdm(input_root.glob('*.png')))

    output_paths = {
        flow_path: output_root / flow_path.relative_to(input_root)
        for flow_path in flow_paths
    }
    for output_path in output_paths.values():
        output_path.parent.mkdir(exist_ok=True, parents=True)

    assert not (args.diagonal_normalize and args.sequence_normalize)
    if args.min_normalize_perc > 0:
        assert not args.diagonal_normalize
        assert not args.sequence_normalize

    if not args.sequence_normalize:
        tasks = list(output_paths.items())
        if args.diagonal_normalize:
            flow = load_flow_png(str(tasks[0][0]))
            diagonal = (flow.shape[0]**2 + flow.shape[1]**2)**0.5
            magnitude_scale = diagonal / 12
        else:
            magnitude_scale = None
        Parallel(n_jobs=args.num_workers)(
            delayed(visualize_flow)(
                flow_path,
                output_path,
                magnitude_scale,
                args.visualize_angle_only,
                clip_at_magnitude_scale=args.diagonal_normalize,
                min_normalize_percent=args.min_normalize_perc)
            for flow_path, output_path in tqdm(tasks))
    else:
        # Collect flows for each parent directory
        flows_by_parent_dir = collections.defaultdict(list)
        for flow_path in flow_paths:
            flows_by_parent_dir[flow_path.parent].append(flow_path)

        with Parallel(n_jobs=args.num_workers) as parallel:
            for dir_flow_paths in tqdm(flows_by_parent_dir.values()):
                flows = parallel(
                    delayed(load_flow_png)(str(path))
                    for path in dir_flow_paths)
                diagonal = (flows[0].shape[0]**2 + flows[0].shape[1]**2)**0.5
                for flow in flows:
                    flow[:, :, 1] = flow[:, :, 1].clip(max=diagonal)
                magnitude_scale = max(flow[:, :, 1].max() for flow in flows)

                tasks = [(flow_path, output_paths[flow_path])
                         for flow_path in dir_flow_paths]

                parallel(
                    delayed(visualize_flow)
                    (flow, output_path, magnitude_scale,
                     args.visualize_angle_only) for flow, output_path in tasks)


if __name__ == "__main__":
    main()
