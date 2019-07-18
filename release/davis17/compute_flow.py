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
    assert all(x in {'val_moving', 'test', 'test-dev', 'val', 'train'}
               for x in config['davis17']['splits'])
    output_root = Path(config['davis17']['output_dir'])
    davis_root = Path(config['davis17']['root'])

    split_dirs = {}
    for split in config['davis17']['splits']:
        if split == 'val_moving':
            moving_path = Path(__file__).parent / 'val_moving_sequences.txt'
            with open(moving_path, 'r') as f:
                sequences = [x.strip() for x in f]
        else:
            with open(davis_root / f'ImageSets/2017/{split}.txt', 'r') as f:
                sequences = [x.strip() for x in f]

        image_dir = output_root / 'split-links' / 'JPEGImages' / split
        annotation_dir = output_root / 'split-links' / 'Annotations' / split
        image_dir.mkdir(exist_ok=True, parents=True)
        annotation_dir.mkdir(exist_ok=True, parents=True)
        split_dirs[split] = (image_dir, annotation_dir)

        for sequence in sequences:
            sequence_image = image_dir / sequence
            if not sequence_image.exists():
                sequence_image.symlink_to(davis_root /
                                          f'JPEGImages/480p/{sequence}')
            sequence_annotation = annotation_dir / sequence
            if not sequence_annotation.exists():
                sequence_annotation.symlink_to(davis_root /
                                               f'Annotations/480p/{sequence}')
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

    output_root = Path(config['davis17']['output_dir'])
    flow_output = output_root / 'flow'
    flow_output.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, flow_output, args)
    logging.debug('Config:\n%s', pprint.pformat(config))

    split_dirs = link_splits(config)
    for split in config['davis17']['splits']:
        input_dir = split_dirs[split][0]
        output_split = flow_output / split
        msg("Computing flow on DAVIS 2017 %s set.")
        compute_flow_helper(config, input_dir, output_split, extension='.jpg')


if __name__ == "__main__":
    main()