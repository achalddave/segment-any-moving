"""Split outputs into TrainingSet and TestSet, and rename sequences.

The zip file shared with me had all the sequence outputs from both Training and
Test sets in one directory, and any sequence that had a left-padded 0 had its 0
removed from the name (e.g. bear01 -> bear1). This script fixes both these
issues."""

import argparse
from pathlib import Path


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--mat-dir',
        type=Path,
        required=True,
        help=('Contains mat files of the form <seq>_obj.mat and '
              '<seq>_obj_reverse.mat'))
    parser.add_argument('--fbms-dir', type=Path, required=True)

    args = parser.parse_args()

    for split in ['TrainingSet', 'TestSet']:
        split_dir = args.mat_dir / split
        split_dir.mkdir()
        sequences = list((args.fbms_dir / split).iterdir())
        for sequence_dir in sequences:
            sequence = sequence_dir.name
            if sequence[-2] == '0':
                mat_glob = sequence[:-2] + sequence[-1] + '_*.mat'
            else:
                mat_glob = sequence + '_*.mat'
            mat_files = list(args.mat_dir.glob(mat_glob))
            assert len(mat_files) == 4, (
                'Expected 4 .mat files for sequence %s, found %s' %
                (sequence, len(mat_files)))
            for mat_path in mat_files:
                # Original name: <seq_name>_<suffix>
                # Replace original <seq_name> with real <seq_name> from
                # groundtruth.
                suffix = mat_path.name.split('_', 1)[1]
                new_name = sequence + '_' + suffix
                mat_path.rename(split_dir / new_name)


if __name__ == "__main__":
    main()
