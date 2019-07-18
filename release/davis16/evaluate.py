import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.davis16.compute_flow import link_splits
from release.helpers.misc import msg, subprocess_call


def check_tracks(track_output, splits):
    for split in splits:
        np_dir = track_output / split
        if not np_dir.exists():
            raise ValueError(f'Did not find tracks in {np_dir}; '
                             f'did you run release/davis17/track.py?')


def evaluate_proposed(config, output_stage):
    if output_stage == 'detection':
        input_dir = (Path(config['davis16']['output_dir']) / 'detections')
    elif output_stage == 'tracking':
        input_dir = (Path(config['davis16']['output_dir']) / 'tracks')
    else:
        raise ValueError(f'Unknown output stage: {output_stage}')

    for split in config['davis16']['splits']:
        masks_dir = input_dir / split / 'masks' / 'masks'
        cmd = [
            'python', 'davis/eval_fgbg.py',
            '--masks-dir', masks_dir
        ]
        msg(f'Evaluating {split}')
        subprocess_call(cmd)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_stage', choices=['detection', 'tracking'])
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    evaluate_proposed(config, args.output_stage)


if __name__ == "__main__":
    main()
