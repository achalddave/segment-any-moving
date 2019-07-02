import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import subprocess_call


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = Path(config['fbms']['track_output'])
    output_dir.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, output_dir, args)

    for split in ['TrainingSet', 'TestSet']:
        init_detections = Path(config['fbms']['joint_output']) / split
        continue_detections = Path(config['fbms']['appearance_output']) / split
        fbms_split_root = Path(config['fbms']['root']) / split
        output_split = Path(output_dir) / split
        args = [
            '--images-dir', fbms_split_root,
            '--init-detections-dir', init_detections,
            '--continue-detections-dir', continue_detections,
            '--output-dir', output_split,
            '--save-numpy', True,
            '--save-images', False,
            '--bidirectional',
            '--score-init-min', 0.9,
            '--remove-continue-overlap', 0.1,
            '--fps', 30,
            '--filename-format', 'fbms',
            '--save-video', config['tracker']['visualize'],
            '--recursive'
        ]
        cmd = ['python', 'tracker/two_detector_track.py'] + args
        logging.info('\n\n###\n'
                     'Running tracker on FBMS %s\n'
                     '###\n\n', split)
        subprocess_call(cmd)


if __name__ == "__main__":
    main()
