import argparse
import logging
import pprint
import subprocess
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.flow import compute_flow_helper
from release.helpers.misc import msg, subprocess_call


def link_splits(config):
    valid_splits = {
        'moving_train', 'moving_val', 'train', 'val', 'test'
    }
    assert all(x in valid_splits for x in config['ytvos']['splits'])
    output_root = Path(config['ytvos']['output_dir'])
    ytvos_root = Path(config['ytvos']['root'])

    split_dirs = {}
    for split in config['ytvos']['splits']:
        if split in {'moving_train', 'moving_val'}:
            moving_path = Path(__file__).parent / f'{split}_sequences.txt'
            with open(moving_path, 'r') as f:
                sequences = [x.strip() for x in f]

            image_dir = output_root / 'split-links' / 'JPEGImages' / split
            annotation_dir = (
                output_root / 'split-links' / 'Annotations' / split)
            image_dir.mkdir(exist_ok=True, parents=True)
            annotation_dir.mkdir(exist_ok=True, parents=True)

            for sequence in sequences:
                sequence_image = image_dir / sequence
                if not sequence_image.exists():
                    sequence_image.symlink_to(
                        ytvos_root / f'train_all_frames/JPEGImages/{sequence}')
                sequence_annotation = annotation_dir / sequence
                if not sequence_annotation.exists():
                    sequence_annotation.symlink_to(
                        ytvos_root / f'train/Annotations/{sequence}')
        else:
            image_dir = ytvos_root / split / 'JPEGImages'
            annotation_dir = ytvos_root / split / 'Annotations'
        split_dirs[split] = (image_dir, annotation_dir)

    return split_dirs


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_root = Path(config['ytvos']['output_dir'])
    flow_output = output_root / 'flow'
    flow_output.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, flow_output, args)
    logging.debug('Config:\n%s', pprint.pformat(config))

    split_dirs = link_splits(config)
    for split in config['ytvos']['splits']:
        input_dir = split_dirs[split][0]
        print(input_dir)
        output_split = flow_output / split
        msg("Computing flow on YTVOS %s set.")
        compute_flow_helper(config, input_dir, output_split, extension='.jpg')


if __name__ == "__main__":
    main()