"""Wrapper script to run Keuper et al ICCV 2015 on a video directory."""

import argparse
import logging
import subprocess
from pathlib import Path

from PIL import Image

import utils.log as log_utils


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--code-dir',
        type=Path,
        default=Path(
            '/home/achald/research/misc/motion-segmentation-methods/moseg-multicut/'
        ))
    parser.add_argument(
        '--sampling',
        default=4,
        type=int,
        help=("From Keuper et al code: 'sampling' specifies the subsampling "
              "parameter. If you specify 8 (a good default value), only every "
              "8th pixel in x and y direction is taken into account. If you "
              "specify 1, the sampling will be dense (be careful, memory "
              "consumption and computation time will be very large in this "
              "setting)."))
    parser.add_argument(
        '--prior',
        default=0.5,
        type=float,
        help=("From Keuper et al code: prior specifies the prior cut "
              "probability. The higher this value is chosen, the more "
              "segments will be generated. For good performance, choose 0.5."))
    parser.add_argument('--extension', default='.png')

    args = parser.parse_args()

    frames = list(args.frames_dir.glob('*' + args.extension))
    if not frames:
        raise ValueError('Found no images with extension %s.' % args.extension)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    log_utils.setup_logging(
        log_utils.add_time_to_path(
            args.output_dir / (Path(__file__).name + '.log')))
    logging.info('Args:\n%s', vars(args))

    ppm_frames = []
    for frame in frames:
        ppm_frame = args.output_dir / (frame.stem + '.ppm')
        if not ppm_frame.exists():
            Image.open(frame).save(ppm_frame)
        ppm_frames.append(ppm_frame)

    video_name = args.frames_dir.parent.name
    bmf_path = args.output_dir / (video_name + '.bmf')
    with open(bmf_path, 'w') as f:
        f.write('%s %s\n' % (len(frames), 1))
        for ppm_frame in ppm_frames:
            f.write(ppm_frame.name + '\n')

    # Command:
    # ./motionseg_release <bmf_path> 0 <num_frames> <sampling> <prior>
    command = [
        './motionseg_release', bmf_path, 0,
        len(frames), args.sampling, args.prior
    ]
    command = [str(x) for x in command]
    logging.info('Running command:\n%s', ' '.join(command))
    subprocess.check_output(command, cwd=args.code_dir)


if __name__ == "__main__":
    main()
