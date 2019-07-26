import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.ytvos.compute_flow import link_splits
from release.helpers.misc import msg, subprocess_call


def check_tracks(track_output, splits):
    for split in splits:
        np_dir = track_output / split
        if not np_dir.exists():
            raise ValueError(f'Did not find tracks in {np_dir}; '
                             f'did you run release/ytvos/track.py?')


def evaluate_proposed(config, with_ytvos_train):
    track_output = Path(config['ytvos']['output_dir']) / 'tracks' / (
        'with_ytvos' if with_ytvos_train else 'without_ytvos')
    check_tracks(track_output, config['ytvos']['splits'])

    split_dirs = link_splits(config)
    for split in config['ytvos']['splits']:
        np_dir = track_output / split
        annotations_dir = split_dirs[split][1]

        cmd = [
            'python', 'fbms/eval_custom.py',
            '--npy-extension', '.npz',
            '--eval-type', 'ytvos',
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
    parser.add_argument(
        '--with-ytvos-train',
        action='store_true',
        help=("By default, we evaluate tracks from a model that was "
              "not trained on YTVOS for fair evaluation of generalization. If "
              "--with-ytvos-train is specified, infer with model that uses "
              "YTVOS for training."))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = Path(config['ytvos']['output_dir']) / 'evaluation' / (
        'with_ytvos' if args.with_ytvos_train else 'without_ytvos')
    output_dir.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, output_dir, args)

    logging.info('Evaluating using proposed metric.')
    evaluate_proposed(config, args.with_ytvos_train)


if __name__ == "__main__":
    main()
