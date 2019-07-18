import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import msg, subprocess_call
from release.davis16.compute_flow import link_splits


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = Path(config['davis16']['output_dir']) / 'tracks'
    output_dir.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, output_dir, args)

    split_dirs = link_splits(config)
    for split in config['davis16']['splits']:
        image_dir = split_dirs[split][0]
        init_detections = (
            Path(config['davis16']['output_dir']) / 'detections' / split)
        output_split = Path(output_dir) / split
        args = [
            '--images-dir', image_dir,
            '--init-detections-dir', init_detections,
            '--output-dir', output_split,
            '--save-numpy', True,
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
        msg(f'Running tracker on DAVIS 2016 {split}')
        subprocess_call(cmd)

        # Convert to foreground-background masks.
        cmd = [
            'python', 'numpy_to_fgbg_masks.py', output_split,
            output_split / 'masks', '--npy-extension', '.npz'
        ]
        subprocess_call(cmd)


if __name__ == "__main__":
    main()
