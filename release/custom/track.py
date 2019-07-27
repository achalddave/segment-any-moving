import argparse
import logging
import pprint
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import msg, subprocess_call
from utils.misc import parse_bool


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))
    parser.add_argument('--frames-dir', required=True, type=Path)
    parser.add_argument('--detections-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)

    parser.add_argument('--save-numpy', type=parse_bool, default=True)
    parser.add_argument('--save-images', type=parse_bool, default=False)
    parser.add_argument('--save-video', type=parse_bool, default=True)
    parser.add_argument('--fps', default=30, type=float)
    parser.add_argument(
        '--model', choices=['joint', 'appearance', 'motion'], default='joint')
    parser.add_argument(
        '--filename-format',
        choices=['frame', 'sequence_frame', 'sequence-frame', 'fbms'],
        required=True,
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"sequence_frame": frame number is separated by an underscore, '
              '"sequence-frame": frame number is separated by a dash, '
              '"fbms": assume fbms style frame numbers'))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args.output_dir.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, args.output_dir)
    logging.debug('Args:\n%s', pprint.pformat(vars(args)))

    if args.model == 'joint':
        detections_args = ['--init-detections-dir', args.detections_dir]
        script = 'tracker/two_detector_track.py'
        extra_args = ['--remove-continue-overlap', 0.1]
    else:
        detections_args = ['--detections-dir', args.detections_dir]
        script = 'tracker/track_multiple.py'
        extra_args = []

    args = [
        '--images-dir', args.frames_dir
        ] + detections_args + [
        '--output-dir', args.output_dir,
        '--save-numpy', args.save_numpy,
        '--save-images', args.save_images,
        '--save-video', args.save_video,
        '--bidirectional',
        '--score-init-min', 0.9,
        '--fps', args.fps,
        '--filename-format', args.filename_format,
        '--quiet'
        ] + extra_args
    cmd = ['python', script] + args
    msg(f'Running tracker')
    subprocess_call(cmd)


if __name__ == "__main__":
    main()
