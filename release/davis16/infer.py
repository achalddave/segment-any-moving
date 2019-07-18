import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import subprocess_call
from release.davis16.compute_flow import link_splits


def get_config_ckpt(model_dir, step):
    config = Path(model_dir) / 'config_and_args.pkl'
    model = Path(model_dir) / 'ckpt' / f'model_step{step}.pth'
    return config, model


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args = parser.parse_args()
    output_dir = Path(config['davis16']['output_dir']) / 'detections'
    output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, output_dir, args)

    detectron_dir = (
        Path(__file__).resolve().parent.parent.parent / 'detectron_pytorch')

    split_dirs = link_splits(config)
    for split in config['davis16']['splits']:
        image_dir = split_dirs[split][0]
        flow_dir = Path(config['davis16']['output_dir']) / 'flow' / split

        joint_config, joint_model = get_config_ckpt(
            config['model']['joint']['dir'], config['model']['joint']['step'])
        objectness_config, objectness_model = get_config_ckpt(
            config['model']['objectness']['dir'],
            config['model']['objectness']['step'])

        cmd = ['python', 'tools/infer_simple.py']
        args = [
            '--cfg', joint_config,
            '--num_classes', 2,
            '--load_ckpt', joint_model,
            '--load_appearance_ckpt', objectness_model,
            '--set', 'MODEL.MERGE_WITH_APPEARANCE.ENABLED', 'True',
            '--image_dirs', image_dir, flow_dir,
            '--input_type', 'rgb', 'flow',
            '--save_images', False,
            '--output_dir', output_dir / split,
            '--recursive'
        ]
        subprocess_call(cmd + args, cwd=str(detectron_dir))

        cmd = ['python', 'detectron_to_fgbg_masks.py']
        args = [
            '--detections-root', output_dir / split,
            '--images-dir', image_dir,
            '--output-dir', output_dir / split,
            '--recursive',
            '--extension', '.jpg',
            '--threshold', '0.7'
        ]
        subprocess_call(cmd + args)


if __name__ == "__main__":
    main()