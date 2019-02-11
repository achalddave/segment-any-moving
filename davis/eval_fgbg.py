import argparse
import logging
import os
import subprocess
import warnings
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

import utils.log as log_utils
from third_party.davis import add_davis16_to_sys_path

add_davis16_to_sys_path()
from davis import cfg
from davis.dataset import db_eval, db_save_eval



def db_statistics(per_frame_values):
    """Compute mean, recall and decay from per-frame evaluation.

    Modified from DAVIS 16 evaluation code.

    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values[1:-1] > 0.5)

    # Compute decay as implemented in Matlab
    per_frame_values = per_frame_values[1:-1]  # Remove first frame

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) +
                   1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


def aggregate_frame_eval(h5py_path):
    """Aggregate frame-level evaluation to sequence level.

    Args:
        h5py_path (path-like): Path to .h5 file.

    Returns:
        db_eval_dict[measure][sequence] (dict): evaluation results.
    """
    measures = ['J', 'F', 'T']
    db = {}
    db_h5 = h5py.File(h5py_path, 'r')
    for m in measures:
        db[m] = OrderedDict()
        for s in sorted(db_h5[m].keys()):
            db[m][s] = db_statistics(db_h5[m][s][...])
    return db


def db_eval_view(db_eval_dict, summary=True):
    """Modified from DAVIS 2016 evaluation code."""
    from prettytable import PrettyTable as ptable
    table = ptable(["Sequence"] +
                   ['J(M)', 'J(O)', 'J(D)', 'F(M)', 'F(O)', 'F(D)', 'T(M)'])

    X = []
    sequences = []
    for key, values in db_eval_dict.items():
        key_sequences, key_results = zip(*db_eval_dict[key].items())
        sequences.extend(key_sequences)
        X.append(list(key_results))
    X = np.hstack(X)[:, :7]

    for s, row in zip(sequences, X):
        table.add_row([s] + ["{: .3f}".format(n) for n in row])

    table.add_row(['' for _ in range(1 + X.shape[1])])
    table.add_row(
        ['Average'] +
        ["{: .3f}".format(n) for n in np.nanmean(X, axis=0)])

    return str(table)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--masks-dir', type=Path, required=True)
    parser.add_argument(
        '--output-dir', default='{masks_parent}/davis16-evaluation')
    parser.add_argument(
        '--annotations-dir',
        type=Path,
        help='Specify alternative annotations, e.g. DAVIS 2017 fg/bg.')
    parser.add_argument(
        '--sequences-dir',
        type=Path,
        help='Specify alternative sequences dir, e.g. DAVIS 2017.')

    args = parser.parse_args()

    args.output_dir = Path(
        args.output_dir.format(masks_parent=args.masks_dir.parent))
    args.output_dir.mkdir(exist_ok=True)

    log_path = log_utils.add_time_to_path(
        args.output_dir / (Path(__file__).name + '.log'))
    log_utils.setup_logging(log_path)

    if args.annotations_dir is not None:
        cfg.PATH.ANNOTATION_DIR = str(args.annotations_dir)
        cfg.N_JOBS = 1  # Config changes don't propagate if this is > 1

    if args.sequences_dir is not None:
        cfg.PATH.SEQUENCES_DIR = str(args.sequences_dir)
        cfg.N_JOBS = 1  # Config changes don't propagate if this is > 1

    db_eval_dict = db_eval(
        args.masks_dir.name, [x.name for x in args.masks_dir.iterdir()],
        str(args.masks_dir.parent),
        metrics=['J', 'F', 'T'])

    output_h5 = args.output_dir / (args.masks_dir.name + '.h5')
    logging.info("Saving results in: %s", output_h5)
    db_save_eval(db_eval_dict, outputdir=args.output_dir)

    davis_data = aggregate_frame_eval(output_h5)
    output = db_eval_view(davis_data)
    logging.info('Results:\n%s\n', output)


if __name__ == "__main__":
    main()
