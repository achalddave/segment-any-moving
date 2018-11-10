"""Evaluate FBMS outputs using a metric that penalizes false positives.

The metric is computed as follows:

For each video, let P = {p_1, ..., p_n} be the predicted tracks and G = {g_1,
.., g_m} be the groundtruth tracks.

Let intersection(a, b) for track a and track b be the intersection between
tracks a and b: i.e. the total number of (x, y, t) "pixels" that belong to both
tracks.

Let g(p_i) be the groundtruth that is matched to prediction p_i using some
algorithm that will be defined later. If p_i is matched ot no groundtruth,
let g(p_i) be an empty track, so that intersection(p_i, g(p_i)) = 0.

Metric:
    sum(intersection(p, g(p)) for p in predictions)
    -----------------------------------------------
           sum(area(p) for p in predictions)

Groundtruth-prediction assignment:
    1. Compute similarities between each p_i and g_i using 3D IoU.
    2. Assign a predicted track to each groundtruth track with hungarian
       matching. If there are more predicted track than groundtruth
       track, create dummy predictions and assign them to each groundtruth
       track.
"""

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import scipy.stats
import scipy.optimize
from tqdm import tqdm

import utils.fbms.utils as fbms_utils
import utils.log as log_utils


def compute_f_measure(precision, recall):
    return 2 * precision * recall / (max(precision + recall, 1e-10))


def simple_table(rows):
    lengths = [
        max(len(row[i]) for row in rows) + 1 for i in range(len(rows[0]))
    ]
    row_format = ' '.join(('{:<%s}' % length) for length in lengths[:-1])
    row_format += ' {}'  # The last column can maintain its length.

    output = ''
    for i, row in enumerate(rows):
        if i > 0:
            output += '\n'
        output += row_format.format(*row)
    return output


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--groundtruth-dir',
        type=Path,
        required=True)
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        help='Contains numpy file of predictions for each sequence.',
        required=True)
    parser.add_argument('--npy-extension', default='.npy')
    parser.add_argument(
        '--background-id',
        required=True,
        help=('ID of background track in predictions. Can be an integer or '
              '"infer", in which case the background id is assumed to be the '
              'id of the track with the most pixels.'))
    parser.add_argument(
        '--include-unknown-labels',
        action='store_true',
        help=('Whether to include labels in ppm files that are not in '
              'groundtruth .dat file when evaluating. See '
              'FbmsGroundtruth.frame_labels method for more details.'))
    args = parser.parse_args()

    log_path = args.predictions_dir / (Path(__file__).name + '.log')
    if args.include_unknown_labels:
        log_path = args.predictions_dir / (
            Path(__file__).name + '_with-unknown.log')
    else:
        log_path = args.predictions_dir / (Path(__file__).name + '.log')
    log_path = log_utils.add_time_to_path(log_path)
    log_utils.setup_logging(log_path)
    file_logger = logging.getLogger(str(log_path))

    subprocess.call([
        './git-state/save_git_state.sh',
        str(log_path.with_suffix('.git-state'))
    ])

    logging.info('Args:\n%s', vars(args))

    if args.background_id != 'infer':
        background_prediction_id = int(args.background_id)
    else:
        background_prediction_id = None

    prediction_paths = sorted(x for x in args.predictions_dir.iterdir()
                              if x.name.endswith(args.npy_extension))
    if not prediction_paths:
        raise ValueError(
            'Found no numpy files (ending in "%s") in --predictions-dir.' %
            args.npy_extension)

    groundtruth_paths = [
        args.groundtruth_dir / x.stem / 'GroundTruth'
        for x in prediction_paths
    ]

    # Maps track_id to list of (x, y, t) tuples.
    sequence_metrics = []  # List of (sequence, precision, recall, f-measure)
    for groundtruth_path, prediction_path in zip(
            tqdm(groundtruth_paths), prediction_paths):
        groundtruth_info = fbms_utils.FbmsGroundtruth(groundtruth_path)

        # (num_frames, height, width)
        prediction_all_frames = np.load(prediction_path)

        # Maps frame index to (height, width)
        groundtruth_dict = groundtruth_info.frame_labels(
            args.include_unknown_labels)

        num_labeled_frames = len(groundtruth_dict)
        groundtruth = np.zeros((
            num_labeled_frames, groundtruth_info.image_height,
            groundtruth_info.image_width))
        prediction = np.zeros_like(groundtruth)

        for f, frame in enumerate(sorted(groundtruth_dict.keys())):
            groundtruth[f] = groundtruth_dict[frame]
            prediction[f] = prediction_all_frames[frame]

        groundtruth_track_ids = set(np.unique(groundtruth))
        predicted_track_ids = set(np.unique(prediction))

        predictions_by_id = {
            p: (prediction == p) for p in predicted_track_ids
        }
        groundtruth_by_id = {
            g: (groundtruth == g) for g in groundtruth_track_ids
        }

        num_predicted = {
            p: id_prediction.sum()
            for p, id_prediction in predictions_by_id.items()
        }
        num_groundtruth = {
            g: id_groundtruth.sum()
            for g, id_groundtruth in groundtruth_by_id.items()
        }

        if background_prediction_id is None:  # Infer background id
            background_prediction_id = max(
                num_predicted.items(), key=lambda x: x[1])[0]

        groundtruth_track_ids.remove(0)
        predicted_track_ids.remove(background_prediction_id)

        groundtruth_track_ids = sorted(groundtruth_track_ids)
        predicted_track_ids = sorted(predicted_track_ids)

        f_measures = np.zeros((len(groundtruth_track_ids),
                               len(predicted_track_ids)))
        intersections = {}
        for g, groundtruth_id in enumerate(groundtruth_track_ids):
            track_groundtruth = groundtruth_by_id[groundtruth_id]
            for p, predicted_id in enumerate(predicted_track_ids):
                track_prediction = predictions_by_id[predicted_id]
                intersection = (track_groundtruth & track_prediction).sum()
                intersections[(groundtruth_id, predicted_id)] = intersection
                precision = intersection / num_predicted[predicted_id]
                recall = intersection / num_groundtruth[groundtruth_id]
                f_measures[g, p] = compute_f_measure(precision, recall)
        # Tuple of (groundtruth_indices, predicted_indices)
        assignment = scipy.optimize.linear_sum_assignment(-f_measures)
        # List of (groundtruth_track_id, predicted_track_id) tuples
        assignment = [(groundtruth_track_ids[assignment[0][i]],
                       predicted_track_ids[assignment[1][i]])
                      for i in range(len(assignment[0]))]

        num_predicted = (prediction != background_prediction_id).sum()
        num_groundtruth = (groundtruth != 0).sum()
        num_correct = sum(intersections[(g, p)] for g, p in assignment)

        precision = 100 * num_correct / num_predicted
        recall = 100 * num_correct / num_groundtruth
        f_measure = compute_f_measure(precision, recall)
        sequence_metrics.append((groundtruth_path.parent.stem, precision,
                                 recall, f_measure))

    file_logger.info('Per sequence metrics:')
    formatted_metrics = [
        [metrics[0]] + ['{:.2f}'.format(m) for m in metrics[1:]]
        for metrics in sequence_metrics
    ]
    file_logger.info(
        '\n%s' % simple_table([('Sequence', 'Precision', 'Recall',
                                'F-measure')] + formatted_metrics))

    avg_precision = np.mean([m[1] for m in sequence_metrics])
    avg_recall = np.mean([m[2] for m in sequence_metrics])
    logging.info('Average precision: %.2f', avg_precision)
    logging.info('Average recall: %.2f', avg_recall)
    logging.info('Average f-measure: %.2f',
                 np.mean([m[3] for m in sequence_metrics]))

    logging.info('F-measure of average prec/rec: %.2f',
                 2 * avg_precision * avg_recall / (avg_precision + avg_recall))


if __name__ == "__main__":
    main()
