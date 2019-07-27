import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import msg, subprocess_call
from release.ytvos.compute_flow import link_splits


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))
    parser.add_argument(
        '--without-ytvos-train',
        action='store_true',
        help=("By default, we infer with model that was trained on YTVOS. "
              "For fair evaluation of generalization, we use a model without "
              "YTVOS training in our manuscript. Set this to True to use "
              "the model without YTVOS training."))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = Path(config['ytvos']['output_dir']) / 'tracks'
    if args.without_ytvos_train:
        output_dir = output_dir / 'without_ytvos'
    else:
        output_dir = output_dir / 'with_ytvos'
    output_dir.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, output_dir, args)

    split_dirs = link_splits(config)
    detections_dir = (
        Path(config['ytvos']['output_dir']) / 'detections' /
        ('with_ytvos' if args.with_ytvos_train else 'without_ytvos'))
    for split in config['ytvos']['splits']:
        image_dir = split_dirs[split][0]
        init_detections = detections_dir / split
        output_split = Path(output_dir) / split
        args = [
            '--images-dir', image_dir,
            '--init-detections-dir', init_detections,
            '--output-dir', output_split,
            '--save-numpy', True,
            '--save-numpy-every-kth-frame', 5,
            '--save-images', False,
            '--bidirectional',
            '--score-init-min', 0.9,
            '--remove-continue-overlap', 0.1,
            '--fps', 30,
            '--filename-format', 'frame',
            '--save-video', config['tracker']['visualize'],
            '--recursive'
        ]
        cmd = ['python', 'tracker/two_detector_track.py'] + args
        msg(f'Running tracker on YTVOS {split}')
        subprocess_call(cmd)


if __name__ == "__main__":
    main()
