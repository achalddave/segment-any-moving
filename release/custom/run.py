import argparse
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import msg, subprocess_call


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model',
                        default='joint',
                        choices=['joint', 'appearance', 'motion'])
    parser.add_argument(
        '--filename-format',
        choices=[
            'frame', 'frameN', 'sequence_frame', 'sequence-frame', 'fbms'
        ],
        required=True,
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"frameN": format "frame<number>", '
              '"sequence_frame": frame number is separated by an underscore, '
              '"sequence-frame": frame number is separated by a dash, '
              '"fbms": assume fbms style frame numbers'))
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    common_setup(__file__, args.output_dir, args)

    flow_dir = args.output_dir / 'flow'
    if args.model != 'appearance':
        subprocess_call([
            'python', 'release/custom/compute_flow.py',
            '--frames-dir', args.frames_dir,
            '--config', args.config,
            '--output-dir', flow_dir
        ])

    output_dir = args.output_dir / args.model
    detections_dir = output_dir / 'detections'
    subprocess_call([
        'python', 'release/custom/infer.py',
        '--frames-dir', args.frames_dir,
        '--flow-dir', flow_dir,
        '--model', args.model,
        '--config', args.config,
        '--output-dir', detections_dir
    ])

    tracks_dir = output_dir / 'tracks'
    subprocess_call([
        'python', 'release/custom/track.py',
        '--frames-dir', args.frames_dir,
        '--detections-dir', detections_dir,
        '--filename-format', args.filename_format,
        '--config', args.config,
        '--model', args.model,
        '--output-dir', tracks_dir
    ])

    msg(f'Output results to: {tracks_dir}')

if __name__ == "__main__":
    main()
