import argparse
import logging
import numpy as np
from pathlib import Path

from tqdm import tqdm

from utils.fbms.utils import FbmsGroundtruth, parse_tracks
from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dat-dir',
        type=Path,
        help=('Contains a .dat file for each sequence, unless '
              '--dat-files-in-subdirs is specified, in which case it should '
              'contain a subdirectory for each sequence, which in turn has a '
              'single dat file.'))
    parser.add_argument('--dat-file', type=Path)
    parser.add_argument(
        '--output',
        help=('Output directory. If --dat-file is specified, this refers to '
              'the output file. Default: {{dat_file_without_extension}}.npy '
              'if --dat-file is specified, else '
              '{{dat_dir}}/numpy-predictions if --dat-dir is specified.'))
    parser.add_argument(
        '--groundtruth-dir',
        type=Path,
        required=True,
        help=("FBMS groundtruth split directory, containing a subdirectory "
              "for each sequence, which in turn contain a 'GroundTruth' "
              "sequence. If --dat-file is specified, this should point to a "
              "specific sequence's 'GroundTruth' directory."))

    parser.add_argument(
        '--label-size-space-separated',
        action='store_true',
        help=('The FBMS code requires that there is a newline between <track '
              'label> and <track size>, but one of the outputs I downloaded '
              'from a method has the two fields separated by a space. This '
              'boolean allows parsing such files.'))
    parser.add_argument(
        '--dat-files-in-subdirs', action='store_true',
        help=('See --dat-dir help for information. Ignored if --dat-file is '
              'specified.'))

    args = parser.parse_args()
    assert (args.dat_file is None) != (args.dat_dir is None), (
        'Exactly one of --dat-file or --dat-dir must be specified.')

    if args.dat_file:
        if not args.output:
            args.output = '{dat_file_without_extension}.npy'
        args.output = Path(
            args.output.format(
                dat_file_without_extension=args.dat_file.with_suffix('')))
    else:
        if not args.output:
            args.output = '{dat_dir}/numpy-predictions'
        args.output = Path(args.output.format(dat_dir=args.dat_dir))

    assert not args.output.exists()
    if args.dat_dir is not None:
        args.output.mkdir(parents=True)
        setup_logging(args.output / (Path(__file__).name + '.log'))
    else:
        setup_logging(args.output + '.log')

    logging.info('Args:\n%s' % args)

    if args.dat_dir is not None:
        sequences = []
        inputs = []
        if args.dat_files_in_subdirs:
            for sequence_dir in args.dat_dir.iterdir():
                if sequence_dir.is_dir():
                    dat_file = list(sequence_dir.glob('*.dat'))
                    if len(dat_file) != 1:
                        raise ValueError('Found %s (!= 1) dat files in %s' %
                                         (len(dat_file), sequence_dir))
                    inputs.append(dat_file[0])
                    sequences.append(sequence_dir.name)
        else:
            for sequence_dat in args.dat_dir.glob('*.dat'):
                sequences.append(sequence_dat.stem)
                inputs.append(sequence_dat)
        groundtruths = [
            args.groundtruth_dir / x / 'GroundTruth'
            for x in sequences
        ]
        outputs = [args.output / (x + '.npy') for x in sequences]
    else:
        inputs = [args.dat_file]
        groundtruths = [args.groundtruth_dir]
        outputs = [args.output]

    single_input = len(inputs) == 1
    for input_path, groundtruth_path, output_path in zip(
            tqdm(inputs, disable=single_input), groundtruths, outputs):
        groundtruth = FbmsGroundtruth(groundtruth_path)

        with open(input_path, 'r') as f:
            tracks_txt = f.read()

        segmentation = parse_tracks(
            tracks_txt,
            image_shape=(groundtruth.image_height, groundtruth.image_width),
            track_label_size_space_separated=args.label_size_space_separated,
            progress=not single_input)
        np.save(output_path, segmentation)


if __name__ == "__main__":
    main()
