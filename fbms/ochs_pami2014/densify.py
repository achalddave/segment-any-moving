import argparse
import logging
import os
import subprocess
from pathlib import Path

import utils.log as log_utils

DEFAULT_CODE_DIR = Path(
    '/home/achald/research/misc/motion-segmentation-methods/ochs-malik-brox_pami14/moseg/'
)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--sparse-dir',
        type=Path,
        required=True,
        help='Multicut results directory')
    parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        help='Path to original ppm images.')
    parser.add_argument(
        '--ochs-code-dir',
        type=Path,
        default=DEFAULT_CODE_DIR,
        help=('Ochs et al. PAMI 2014 code directory. Should contain '
              '"./dens100gpu".'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--cuda55-dir',
        help=('Directory containing libcudart.so.5.5. By default, this is '
              'assumed to be the code dir.'),
        default='{code_dir}')
    parser.add_argument('--lambda', dest='lmbda', type=float, default=200.0)
    parser.add_argument('--max-iter', type=int, default=2000)

    args = parser.parse_args()

    if not args.ochs_code_dir.exists():
        raise ValueError(
            'Could not find code directory at %s' % args.ochs_code_dir)

    args.cuda55_dir = Path(args.cuda55_dir.format(code_dir=args.ochs_code_dir))
    if not args.cuda55_dir.exists():
        raise ValueError(
            'Could not find cuda5.5 directory at %s' % args.cuda55_dir)

    dat_path = list(args.sparse_dir.glob('Track*.dat'))
    if len(dat_path) != 1:
        raise ValueError(
            'Expected exactly 1 .dat file in --sparse-dir, found %s' %
            len(dat_path))
    dat_path = dat_path[0]

    args.output_dir.mkdir(parents=True)
    log_path = log_utils.add_time_to_path(
        args.output_dir / (Path(__file__).name + '.log'))
    log_utils.setup_logging(log_path)
    logging.info('Args:\n%s', vars(args))

    filestructure_lines = [
        '',
        's dataDir /',
        's tracksDir /',
        'f lambda %.1f' % args.lmbda,
        'i maxiter %d' % args.max_iter
    ]
    filestructure_path = args.output_dir / 'filestructureDensify.cfg'
    with open(filestructure_path, 'w') as f:
        f.write('\n'.join(filestructure_lines))

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(args.cuda55_dir) + ':' + env['PYTHONPATH']
    dense_output = args.output_dir
    cmd = [
        './dens100gpu',
        str(filestructure_path),
        str(args.images_dir / 'image.ppm'),
        str(dat_path), '-1',
        str(dense_output)
    ]
    logging.info('Running command:\n%s', ' '.join(cmd))
    # bufsize=0: Print output immediately (don't buffer)
    subprocess.call(cmd, bufsize=0, cwd=args.ochs_code_dir, env=env)
    logging.info('Output results to %s' % dense_output)


if __name__ == "__main__":
    main()
