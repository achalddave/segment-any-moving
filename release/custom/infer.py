import argparse
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import subprocess_call


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
    parser.add_argument('--frames-dir',
                        type=Path,
                        help='Required unless --model set to "motion"')
    parser.add_argument('--flow-dir',
                        type=Path,
                        help='Required unless --model set to "appearance".')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument(
        '--model', choices=['joint', 'appearance', 'motion'], default='joint')

    args = parser.parse_args()

    if args.model != 'appearance':
        assert args.flow_dir and args.flow_dir.exists(), (
            f'--flow-dir must be specified for --model {args.model}'
        )

    if args.model != 'motion':
        assert args.frames_dir and args.frames_dir.exists(), (
            f'--frames-dir must be specified for --model {args.model}'
        )

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    detectron_dir = (
        Path(__file__).resolve().parent.parent.parent / 'detectron_pytorch')

    model_config, model = get_config_ckpt(
        config['model'][args.model]['dir'],
        config['model'][args.model]['step'])

    maybe_combine_appearance = []
    if args.model == 'joint':
        appearance_config, appearance_model = get_config_ckpt(
            config['model']['appearance']['dir'],
            config['model']['appearance']['step'])
        maybe_combine_appearance = [
            '--load_appearance_ckpt', appearance_model, '--set',
            'MODEL.MERGE_WITH_APPEARANCE.ENABLED', 'True'
        ]
        image_dirs = ['--image_dirs', args.frames_dir, args.flow_dir]
        input_type = ['--input_type', 'rgb', 'flow']
    elif args.model == 'appearance':
        maybe_combine_appearance = []
        image_dirs = ['--image_dirs', args.frames_dir]
        input_type = ['--input_type', 'rgb']
    elif args.model == 'motion':
        maybe_combine_appearance = []
        image_dirs = ['--image_dirs', args.flow_dir]
        input_type = ['--input_type', 'flow']

    cmd = ['python', 'tools/infer_simple.py']
    args = ([
        '--cfg', model_config,
        '--num_classes', 2,
        '--load_ckpt', model]
        + maybe_combine_appearance
        + image_dirs
        + input_type + [
        '--save_images', args.visualize,
        '--output_dir', args.output_dir,
        '--recursive'])
    subprocess_call(cmd + args, cwd=str(detectron_dir))


if __name__ == "__main__":
    main()