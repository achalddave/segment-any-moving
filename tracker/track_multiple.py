import argparse
import logging
import pprint
import subprocess
from pathlib import Path

from tqdm import tqdm

import utils.log as log_utils
from tracker.track import (
    create_tracking_parser, load_detectron_pickles, track_and_visualize)
from utils.misc import glob_ext, parse_bool, IMG_EXTENSIONS


def main():
    tracking_parser = create_tracking_parser()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        parents=[tracking_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--detections-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--save-images', type=parse_bool, default=False)
    parser.add_argument('--save-numpy', type=parse_bool, default=False)
    parser.add_argument(
        '--save-numpy-every-kth-frame',
        help='Save only every kth frame in numpy output.',
        type=int,
        default=1)
    parser.add_argument('--save-video', type=parse_bool, default=True)
    parser.add_argument('--fps', default=30, type=float)
    parser.add_argument('--extensions', nargs='*', default=IMG_EXTENSIONS)
    parser.add_argument(
        '--dataset', default='coco', choices=['coco', 'objectness'])
    parser.add_argument(
        '--filename-format',
        choices=['frame', 'sequence_frame', 'sequence-frame', 'fbms'],
        default='frame',
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"sequence_frame": frame number is separated by an underscore, '
              '"sequence-frame": frame number is separated by a dash, '
              '"fbms": assume fbms style frame numbers'))
    parser.add_argument('--quiet', action='store_true')

    tracking_params, remaining_argv = tracking_parser.parse_known_args()
    args = parser.parse_args(remaining_argv)

    tracking_params = vars(tracking_params)

    assert (args.save_images or args.save_video or args.save_numpy), (
        'One of --save-image, --save-video, or --save-numpy must be specified')

    output_log_file = log_utils.add_time_to_path(
        args.output_dir / 'tracker.log')
    output_log_file.parent.mkdir(exist_ok=True, parents=True)
    log_utils.setup_logging(output_log_file)
    if args.quiet:
        logging.root.setLevel(logging.WARN)

    logging.info('Args: %s', pprint.pformat(vars(args)))
    logging.info('Tracking params: %s', pprint.pformat(tracking_params))
    subprocess.call([
        './git-state/save_git_state.sh',
        output_log_file.with_suffix('.git-state')
    ])

    detectron_input = args.detections_dir
    if not detectron_input.is_dir():
        raise ValueError(
            '--detectron-dir %s is not a directory!' % args.detections_dir)

    if args.filename_format == 'fbms':
        from utils.fbms.utils import get_framenumber
    elif args.filename_format == 'sequence-frame':
        def get_framenumber(x):
            return int(x.split('-')[-1])
    elif args.filename_format == 'sequence_frame':
        def get_framenumber(x):
            return int(x.split('_')[-1])
    elif args.filename_format == 'frame':
        get_framenumber = int
    else:
        raise ValueError(
            'Unknown --filename-format: %s' % args.filename_format)

    args.extensions = [x if x[0] == '.' else '.' + x for x in args.extensions]

    images = glob_ext(args.images_dir, args.extensions, recursive=True)

    image_subdirs = sorted(
        set(x.resolve().parent.relative_to(args.images_dir) for x in images))
    for subdir in tqdm(image_subdirs):
        output_numpy = None
        output_images_dir = None
        output_video = None

        if args.save_numpy:
            if subdir == Path('.'):
                output_numpy = args.output_dir / 'tracked.npz'
            else:
                output_numpy = args.output_dir / subdir.with_suffix('.npz')
        if args.save_images:
            output_images_dir = args.output_dir / subdir / 'images'
            output_images_dir.mkdir(exist_ok=True, parents=True)
        if args.save_video:
            if subdir == Path('.'):
                output_video = args.output_dir / 'tracked.mp4'
            else:
                output_video = args.output_dir / subdir.with_suffix('.mp4')

        if all(x is None or x.exists()
               for x in (output_numpy, output_images_dir, output_video)):
            logging.info('%s already processed, skipping', subdir)
            continue
        detections_dir = args.detections_dir / subdir
        if not detections_dir.exists():
            logging.warn('Skipping sequence %s: detections not found at %s',
                         subdir, detections_dir)
            continue

        detection_results = load_detectron_pickles(
            detections_dir, frame_parser=get_framenumber)
        track_and_visualize(
            detection_results,
            args.images_dir / subdir,
            tracking_params,
            get_framenumber,
            args.extensions,
            vis_dataset=args.dataset,
            output_images_dir=output_images_dir,
            output_video=output_video,
            output_video_fps=args.fps,
            output_numpy=output_numpy,
            output_numpy_every_kth=args.save_numpy_every_kth_frame,
            output_track_file=None)


if __name__ == "__main__":
    main()
