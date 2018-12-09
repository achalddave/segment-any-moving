import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from utils.fbms.utils import get_frameoffset
from utils.log import setup_logging
from utils.misc import simple_table


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        required=True,
        help='Contains .npy file for each sequence.')
    parser.add_argument(
        '--groundtruth-dir',
        type=Path,
        required=True,
        help='3D motion groundtruth')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--eval-code-dir',
        default='/home/achald/research/misc/datasets/fbms/fbms-3d-eval/')

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(args.output_dir / 'eval_3dmotion_numpy.log')
    logging.info('Script path: %s', Path(__file__).resolve())
    logging.info('Args:\n%s', vars(args))

    # N0.75: Number of objects detected at F-measure > 0.75
    # N-predicted: # of predicted objects
    # N-groundtruth: # of gt objects
    all_metrics = {
        'F': [],
        'P': [],
        'R': [],
        'N0.75': [],
        'N-predicted': [],
        'N-groundtruth': []
    }
    sequence_dirs = [x for x in args.groundtruth_dir.iterdir() if x.is_dir()]
    for sequence_dir in tqdm(sequence_dirs):
        sequence = sequence_dir.name
        # Contains predictions for every frame in the video
        all_predictions = np.load(args.predictions_dir / (sequence + '.npy'))

        sequence_groundtruth = sequence_dir / 'GroundTruth'
        groundtruth_pngs = sequence_groundtruth.glob('*.png')
        groundtruth_png_indices = [(x, int(x.stem.split('_gt')[0]))
                                   for x in groundtruth_pngs]
        groundtruth_png_indices = sorted(groundtruth_png_indices,
                                         key=lambda x: x[1])
        predictions = np.zeros(
            (len(groundtruth_png_indices), all_predictions.shape[1],
             all_predictions.shape[2]))
        for i, (groundtruth_png,
                raw_frame) in enumerate(groundtruth_png_indices):
            frame = get_frameoffset(sequence, raw_frame)
            predictions[i] = all_predictions[frame]
        predictions_mat = args.output_dir / (sequence + '.mat')
        savemat(predictions_mat, {'predictions': predictions})
        cmd = [
            'matlab', '-nodesktop', '-nosplash', '-r',
            "evaluate('{gt}', '{pred}', '{code}'); quit".format(
                gt=sequence_groundtruth,
                pred=predictions_mat,
                code=args.eval_code_dir)
        ]
        output = subprocess.check_output(cmd, cwd=Path(__file__).parent)
        output = output.decode('utf-8')
        metrics = {k: None for k in all_metrics}
        for line in output.split('\n'):
            for metric in metrics:
                if line.startswith(metric + ':'):
                    metrics[metric] = float(line.split(': ')[-1])
        for metric, value in metrics.items():
            if value is None:
                raise ValueError("Couldn't find metric (%s) in output:\n%s" %
                                 (metric, output))
            all_metrics[metric].append(value)

    # N diff: | # of predicted objs - # of groundtruth objs|
    rows = [('Sequence', 'Precision', 'Recall', 'F-measure', 'N >0.75',
             'N pred', 'N gt', 'N diff')]
    for i, sequence_dir in enumerate(sequence_dirs):
        N_diff = abs(all_metrics['N-predicted'][i] -
                     all_metrics['N-groundtruth'][i])
        rows.append(
            (sequence_dir.name,
             '%.2f' % all_metrics['P'][i],
             '%.2f' % all_metrics['R'][i],
             '%.2f' % all_metrics['F'][i],
             '%d' % all_metrics['N0.75'][i],
             '%d' % all_metrics['N-predicted'][i],
             '%d' % all_metrics['N-groundtruth'][i],
             '%d' % N_diff))

    logging.info('Results:\n%s', simple_table(rows))
    for metric, values in all_metrics.items():
        assert len(values) == len(sequence_dirs)
    for metric in ['F', 'P', 'R']:
        logging.info('{}: {:.2f}'.format(
            metric,
            sum(all_metrics[metric]) / len(sequence_dirs)))
    logging.info('N0.75: {:.0f}'.format(sum(all_metrics['N0.75'])))
    logging.info('N diff: {:.2f}'.format(
        sum([
            abs(x - y) for x, y in zip(all_metrics['N-predicted'],
                                       all_metrics['N-groundtruth'])
        ]) / len(sequence_dirs)))


if __name__ == "__main__":
    main()
