"""Per-frame version of proposed evaluation for video instance segmentation.

See fbms/eval_custom.py for a video-level evaluation that also works with
DAVIS."""

import argparse
import collections
import logging
import pickle
from pathlib import Path

import numpy as np
import scipy.optimize
import pycocotools.mask as mask_util
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image

import utils.log as log_utils


def compute_f_measure(precision, recall):
    return 2 * precision * recall / (max(precision + recall, 1e-10))


def get_unique_objects(groundtruth):
    """Get unique object ids from segmentation mask

    Adapted from DAVIS evaluation code.
    """
    ids = sorted(np.unique(groundtruth))
    if ids[-1] == 255:  # Remove unknown-label
        ids = ids[:-1]
    if ids[0] == 0:  # Remove background
        ids = ids[1:]
    return ids


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detections-pickle', type=Path, required=True)
    parser.add_argument('--annotations-json', type=Path, required=True)
    parser.add_argument('--davis-root', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.7)

    args = parser.parse_args()

    output_log = log_utils.add_time_to_path(
        args.detections_pickle.parent / (Path(__file__).name + '.log'))
    log_utils.setup_logging(output_log)
    logging.info('Args:\n%s', vars(args))

    groundtruth = COCO(str(args.annotations_json))
    image_ids = sorted(groundtruth.getImgIds())
    # Map <sequence_name>/<frame>.png to list of segmentations, sorted in
    # ascending order of scores.
    results = {}
    with open(args.detections_pickle, 'rb') as f:
        data = pickle.load(f)
        boxes = data['all_boxes']
        masks = data['all_segms']
        num_classes = len(boxes)
        for c in range(num_classes):
            assert len(boxes[c]) == len(image_ids), (
                f'Expected {len(image_ids)} boxes for class {c}, got '
                f'{len(boxes[c])}')
        for i, image_id in enumerate(image_ids):
            scores = []
            segmentations = []
            # Merge all classes into one.
            for c in range(1, num_classes):
                scores.extend(boxes[c][i][:, 4])
                segmentations.extend(masks[c][i])
            segmentation_scores = sorted(
                zip(segmentations, scores), key=lambda x: x[1])
            results[groundtruth.imgs[image_id]['file_name']] = [
                segmentation for segmentation, score in segmentation_scores
                if score > args.threshold
            ]

    sequence_frames = collections.defaultdict(list)
    for x in results.keys():
        x = Path(x)
        sequence_frames[x.parent.name].append(x)
    annotations_dir = args.davis_root / 'Annotations' / '480p'

    metrics = []  # List of (frame name, precision, recall, f-measure) tuples
    for sequence, frames in tqdm(sequence_frames.items()):
        frames = sorted(frames, key=lambda x: int(x.stem))
        davis_sequence = annotations_dir / sequence
        davis_frames = sorted(
            davis_sequence.glob('*.png'), key=lambda x: int(x.stem))
        assert (
            len(davis_frames) == len(frames)
            or len(davis_frames) == (len(frames) + 1)
        ), 'Unexpected number of frames. Expected: %s or %s, saw %s' % (
            len(frames), len(frames) + 1, len(davis_frames))
        for i, frame_path in enumerate(davis_frames):
            frame_name = str(frame_path.relative_to(annotations_dir))
            groundtruth = np.array(Image.open(frame_path))
            # Some frames in DAVIS 16 have an extra channel, but this code
            # should only be used with DAVIS 17.
            assert groundtruth.ndim == 2, (
                'Groundtruth has multiple channels. This may be because you '
                'are passing DAVIS 2016 annotations, which is not supported.')
            unique_objects = get_unique_objects(groundtruth)
            groundtruth_masks = [
                groundtruth == i for i in unique_objects
            ]
            if i == (len(davis_frames) - 1) and frame_name not in results:
                previous_frame_name = '%s/%05d.png' % (sequence, i - 1)
                results[frame_name] = results[previous_frame_name]

            prediction = np.full(groundtruth.shape, fill_value=-1)
            for p, predicted_mask in enumerate(results[frame_name]):
                prediction[mask_util.decode(predicted_mask) != 0] = p
            predicted_masks = [
                (prediction == p) for p in np.unique(prediction)
                if p != -1
            ]

            num_predicted = [m.sum() for m in predicted_masks]
            num_groundtruth = [x.sum() for x in groundtruth_masks]
            f_measures = np.zeros((len(groundtruth_masks),
                                   len(predicted_masks)))
            intersections = {}
            for g, groundtruth_mask in enumerate(groundtruth_masks):
                for p, predicted_mask in enumerate(predicted_masks):
                    intersection = (groundtruth_mask & predicted_mask).sum()
                    intersections[g, p] = intersection
                    precision = intersection / num_predicted[p]
                    recall = intersection / num_groundtruth[g]
                    f_measures[g, p] = compute_f_measure(precision, recall)

            # Tuple of (groundtruth_indices, predicted_indices)
            assignment = scipy.optimize.linear_sum_assignment(-f_measures)
            assignment = zip(assignment[0].tolist(), assignment[1].tolist())

            num_predicted = (prediction != -1).sum()
            num_groundtruth = sum(groundtruth_mask.sum()
                                  for groundtruth_mask in groundtruth_masks)
            num_correct = sum(intersections[(g, p)] for g, p in assignment)

            precision = 100 * num_correct / max(num_predicted, 1e-10)
            recall = 100 * num_correct / num_groundtruth
            f_measure = compute_f_measure(precision, recall)
            metrics.append((frame_name, precision, recall, f_measure))

    logging.info('Average precision: %.2f', np.mean([m[1] for m in metrics]))
    logging.info('Average recall: %.2f', np.mean([m[2] for m in metrics]))
    logging.info('Average f-measure: %.2f', np.mean([m[3] for m in metrics]))


if __name__ == "__main__":
    main()
