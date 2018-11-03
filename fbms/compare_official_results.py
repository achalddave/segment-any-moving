"""Compare FBMS official metric results for two different outputs."""

import argparse
import logging
import pprint
from pathlib import Path

import utils.log as log_utils

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results1_dir', type=Path)
    parser.add_argument('results2_dir', type=Path)
    parser.add_argument('--log', required=True, type=Path)

    args = parser.parse_args()

    log_utils.setup_logging(args.log)

    logging.info('Args:\n%s', pprint.pformat(vars(args)))

    def parse_results(path):
        with open(path, 'r') as f:
            results_raw = f.readlines()
        for i, line in enumerate(results_raw):
            if line.startswith('Average Precision, Recall, F-measure:'):
                results_split = results_raw[i + 1].split(' ')
                return {
                    'precision': float(results_split[0]),
                    'recall': float(results_split[1]),
                    'f-measure': float(results_split[2])
                }
        raise ValueError(f'Strange file {path}')


    result_files1 = [
        x for x in args.results1_dir.iterdir()
        if (x.name.endswith('Numbers.txt')
            and not x.name.startswith('all_tracks'))
    ]
    all_results1 = [parse_results(result_path) for result_path in result_files1]
    all_results2 = [
        parse_results(args.results2_dir / result_path.name)
        for result_path in result_files1
    ]
    f_changes = [r2['f-measure'] - r1['f-measure'] for r1, r2 in zip(all_results1, all_results2)]
    f_change_indices_sorted = sorted(
        range(len(f_changes)), key=lambda i: f_changes[i])

    logging.info('F-measure changes from --results1 to --results2')
    logging.info('Positive change means --results2 outperforms --results1')
    for index in f_change_indices_sorted:
        logging.info(
            f'Sequence: {result_files1[index].name},'
            f' F-change: {f_changes[index]*100:.02f}'
        )


if __name__ == "__main__":
    main()
