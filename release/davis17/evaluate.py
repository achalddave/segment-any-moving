import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.davis17.compute_flow import link_splits
from release.helpers.misc import msg, subprocess_call


def check_tracks(track_output, splits):
    for split in splits:
        np_dir = track_output / split
        if not np_dir.exists():
            raise ValueError(f'Did not find tracks in {np_dir}; '
                             f'did you run release/davis17/track.py?')


def evaluate_proposed(config):
    track_output = Path(config['davis17']['output_dir']) / 'tracks'
    check_tracks(track_output, config['davis17']['splits'])

    split_dirs = link_splits(config)
    for split in config['davis17']['splits']:
        np_dir = track_output / split
        annotations_dir = split_dirs[split][1]

        cmd = [
            'python', 'fbms/eval_custom.py',
            '--npy-extension', '.npz',
            '--eval-type', 'davis',
            '--background-id', 0,
            '--groundtruth-dir', annotations_dir,
            '--predictions-dir', np_dir
        ]
        msg(f'Evaluating {split}')
        subprocess_call(cmd)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = Path(config['davis17']['output_dir']) / 'evaluation'
    output_dir.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, output_dir, args)

    logging.info('Evaluating using proposed metric.')
    evaluate_proposed(config)


if __name__ == "__main__":
    main()
