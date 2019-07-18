import argparse
import logging
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.misc import msg, subprocess_call


def check_tracks(track_output, splits):
    for split in splits:
        np_dir = track_output / split
        if not np_dir.exists():
            raise ValueError(f'Did not find tracks in {np_dir}; '
                            f'did you run release/fbms/track.py?')


def check_dat(track_output, split):
    assert split in ['TrainingSet', 'TestSet']
    dat_dir = track_output / split / 'dat'
    expected_files = 29 if split == 'TrainingSet' else 30
    print(dat_dir)
    return (dat_dir.exists()
            and len(list(dat_dir.glob('*.dat'))) == expected_files)


def evaluate_proposed(config):
    track_output = Path(config['fbms']['output_dir']) / 'tracks'
    check_tracks(track_output, config['fbms']['splits'])

    for split in config['fbms']['splits']:
        np_dir = track_output / split

        cmd = [
            'python', 'fbms/eval_custom.py',
            '--npy-extension', '.npz',
            '--eval-type', 'fbms',
            '--background-id', 0,
            '--groundtruth-dir', Path(config['fbms']['root']) / split,
            '--predictions-dir', np_dir
        ]
        msg(f'Evaluating {split}')
        subprocess_call(cmd)


def evaluate_official(config):
    track_output = Path(config['fbms']['output_dir'] / 'tracks')
    check_tracks(track_output)

    for split in config['fbms']['splits']:
        np_dir = track_output / split
        dat_dir = np_dir / 'dat'
        if not check_dat(track_output, split):
            cmd = [
                'python', 'fbms/numpy_to_dat.py',
                '--numpy-dir', np_dir,
                '--output-dir', dat_dir,
                '--fbms-groundtruth', Path(config['fbms']['root']) / split
            ]
            msg(f'Converting {split} numpy predictions to .dat')
            logging.info('This may take a few minutes...')
            subprocess_call(cmd)
            assert check_dat(track_output, split)

        eval_binary = Path(config['fbms']['eval_dir']) / 'MoSegEvalAllPR'
        cmd = [
            'python', 'fbms/eval_official.py',
            '--eval-binary', eval_binary,
            '--predictions-dir', dat_dir,
            '--split', 'all'
        ]
        msg(f'Evaluating {split}')
        subprocess_call(cmd)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation', default='proposed', choices=['official', 'proposed'])
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = Path(config['fbms']['output_dir']) / 'evaluation'
    output_dir.mkdir(exist_ok=True, parents=True)

    common_setup(__file__, output_dir, args)

    if args.evaluation == 'proposed':
        logging.info('Evaluating using proposed metric.')
        evaluate_proposed(config)
    elif args.evaluation == 'official':
        logging.info('Evaluating using official metric.')
        evaluate_official(config)
    else:
        raise ValueError(f'Unknown evaluation: {args.evaluation}')


if __name__ == "__main__":
    main()
