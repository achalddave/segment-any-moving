"""Evaluate FBMS detectron outputs using COCO evaluation."""

import argparse
import io
import json
import logging
import pickle
from contextlib import redirect_stdout
from pathlib import Path
from pprint import pformat

import numpy as np
from PIL import Image
from tqdm import tqdm

import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.log import setup_logging


def translate_range(value, old_range, new_range):
    """
    Translate value in one range to another. Useful for scaling scores.

    >>> translate_range(0.5, (0, 1), (0, 2))
    1.0
    >>> translate_range(1, (0, 1), (0, 2))
    2.0
    >>> translate_range(3, (2, 4), (5, 6))
    5.5
    >>> translate_range(0.5, (0, 1), (0, 2))
    1.0
    >>> translate_range(np.array([2, 2.5, 3, 3.5, 4]), (2, 4), (5, 7)).tolist()
    [5.0, 5.5, 6.0, 6.5, 7.0]
    >>> translate_range(np.array([2, 2.5, 3, 3.5, 4]), (2, 4), (5, 5)).tolist()
    [5.0, 5.0, 5.0, 5.0, 5.0]
    """
    value = np.asarray(value)
    old_min, old_max = old_range

    if np.any(value < old_min):
        raise ValueError('Value(s) (%s) < min(old_range) (%s)' %
                         (value[value < old_min], old_min))

    if np.any(value > old_max):
        raise ValueError('Value(s) (%s) > max(old_range) (%s)' %
                         (value[value > old_max], old_max))

    if (old_max - old_min) < 1e-10:
        return old_max

    new_min, new_max = new_range
    scale = (new_max - new_min) / (old_max - old_min)
    return (value - old_min) * scale + new_min


def load_motion_masks(motion_root):
    """Load motion masks from disk.

    Args:
        motion_root (Path): Points to a directory which contains a
            subdirectory for each sequence, which in turn contains a .png for
            each frame in the sequence.

    Returns:
        motion_masks (dict): Map sequence to dict mapping frame name to motion
            mask path.
    """
    motion_mask_paths = {}
    for sequence_path in motion_root.iterdir():
        if not sequence_path.is_dir():
            continue

        sequence = sequence_path.stem
        motion_mask_paths[sequence] = {}
        for motion_path in sequence_path.glob('*.png'.format(sequence)):
            # Pavel's ICCV 2017 method outputs an extra set of soft masks that
            # start with 'raw_' or 'input_'; ignore them by starting the glob
            # with the sequence name.
            if (motion_path.stem.startswith('raw_')
                    or motion_path.stem.startswith('input_')):
                continue
            motion_mask_paths[sequence][motion_path.stem] = motion_path
    return motion_mask_paths


def load_detectron_predictions(detectron_root):
    """Load detectron predictions from root directort.

    Args:
        detectron_root (Path): Points to a directory which contains a
            subdirectory for each sequence, which in turn contains a .pickle
            file for each frame in the sequence.

    Returns:
        predictions (dict): Map sequence to dict mapping from frame name to
            a dictionary containining keys 'boxes', and 'segmentations'.
    """
    predictions = {}
    for sequence_path in detectron_root.iterdir():
        if not sequence_path.is_dir():
            continue

        sequence = sequence_path.stem
        predictions[sequence] = {}
        for detectron_path in sequence_path.glob('*.pickle'):
            with open(detectron_path, 'rb') as f:
                frame_data = pickle.load(f)

            if frame_data['segmentations'] is None:
                frame_data['segmentations'] = [
                    [] for _ in range(len(frame_data['boxes']))
                ]

            frame_name = detectron_path.stem
            predictions[sequence][frame_name] = {
                'boxes': [],
                'segmentations': []
            }
            for c in range(len(frame_data['boxes'])):
                predictions[sequence][frame_name]['boxes'].extend(
                    frame_data['boxes'][c])
                predictions[sequence][frame_name]['segmentations'].extend(
                    frame_data['segmentations'][c])
    return predictions


def main():
    with open(__file__, 'r') as f:
        _file_source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--fbms-annotation-json',
        help='FBMS JSON annotations',
        required=True)
    parser.add_argument(
        '--motion-masks-root',
        required=True,
        help='Directory containing estimated PNG motion masks for each frame.')
    parser.add_argument(
        '--detectron-root',
        help='Directory containing outputs of detectron on FBMS.',
        required=True)
    parser.add_argument(
        '--save-pickle',
        action='store_true')
    parser.add_argument('--moving-threshold', default=0.5, type=float)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--filename-format', choices=['frame', 'sequence_frame', 'fbms'],
        default='fbms',
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"sequence_frame": the frame number is separated by an '
                                'underscore, '  # noqa: E127
              '"fbms": assume fbms style frame numbers'))
    args = parser.parse_args()

    detectron_root = Path(args.detectron_root)
    motion_root = Path(args.motion_masks_root)

    dataset = COCO(args.fbms_annotation_json)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True)

    logging_path = str(output_root / (Path(__file__).stem + '.log'))
    setup_logging(logging_path)

    file_logger = logging.getLogger(logging_path)
    file_logger.info('Source:\n%s' % _file_source)

    logging.info('Args:\n %s', pformat(vars(args)))

    # Map (sequence, frame_name) to frame_id.
    frame_key_to_id = {}
    for annotation in dataset.imgs.values():
        # Path ends in 'sequence/frame_name'
        path = Path(annotation['file_name'])
        frame_key_to_id[(path.parent.stem, path.stem)] = annotation['id']

    logging.info('Loading motion paths')
    # Map sequence to dict mapping frame index to motion mask path
    motion_mask_paths = load_motion_masks(motion_root)

    logging.info('Loading detectron paths')
    predictions = load_detectron_predictions(detectron_root)

    logging.info('Outputting moving detections')
    detection_results = []
    segmentation_results = []

    if args.filename_format == 'fbms':
        from utils.fbms.utils import get_framenumber
    elif args.filename_format == 'sequence_frame':
        def get_framenumber(x):
            return int(x.split('_')[-1])
    elif args.filename_format == 'frame':
        get_framenumber = int
    else:
        raise ValueError(
            'Unknown --filename-format: %s' % args.filename_format)

    # The last frame won't have a motion mask, so we use the second to last
    # frame's mask as the last frame's mask.
    for sequence in predictions.keys():
        frame_index_names = sorted(
            predictions[sequence].keys(), key=lambda x: get_framenumber(x))
        second_last_frame, last_frame = frame_index_names[-2:]
        if last_frame not in motion_mask_paths:
            motion_mask_paths[sequence][last_frame] = (
                motion_mask_paths[sequence][second_last_frame])

    tasks = [(sequence, frame_name) for sequence in predictions.keys()
             for frame_name in predictions[sequence]]
    for sequence, frame_name in tasks:
        frame_key = (sequence, frame_name)
        # If --save-pickle is true, process every frame. Otherwise, only
        # process frames that are in --fbms-annotations-json.
        if not args.save_pickle and frame_key not in frame_key_to_id:
            continue

        boxes = predictions[sequence][frame_name]['boxes']
        segmentations = predictions[sequence][frame_name]['segmentations']

        motion_mask = np.array(
            Image.open(motion_mask_paths[sequence][frame_name])) != 0

        if args.save_pickle:
            updated_boxes = []
            updated_segmentations = []
        for i, (box, segmentation) in enumerate(zip(boxes, segmentations)):
            mask = mask_util.decode(segmentation)
            x1, y1, x2, y2, score = box.tolist()
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if mask.sum() < 1e-10:
                moving_portion = 0
            else:
                moving_portion = (mask & motion_mask).sum() / mask.sum()

            if moving_portion < args.moving_threshold:
                score = translate_range(score, (0, 1), (0, 0.5))
            else:
                score = translate_range(score, (0, 1), (0.5, 1))

            if frame_key in frame_key_to_id:
                frame_id = frame_key_to_id[frame_key]
                detection_results.append({
                    'image_id': frame_id,
                    'category_id': 1,
                    'bbox': [x1, y1, w, h],
                    'score': score
                })
                segmentation_results.append({
                    'image_id': frame_id,
                    'category_id': 1,
                    'segmentation': segmentation,
                    'score': score
                })

            if args.save_pickle:
                updated_boxes.append([x1, y1, x2, y2, score])
                updated_segmentations.append(segmentation)

        if args.save_pickle:
            output_path = (output_root / 'pickle' / sequence /
                           frame_name).with_suffix('.pickle')
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'wb') as f:
                # TODO(achald): Make this work for multiple classes.
                updated_boxes = [[], updated_boxes]
                if len(updated_segmentations):
                    updated_segmentations = [[], updated_segmentations]
                else:
                    updated_segmentations = None
                pickle.dump({
                    'boxes': updated_boxes,
                    'segmentations': updated_segmentations,
                    'keypoints': [[], []]
                }, f)

    box_output = output_root / 'bbox_fbms_results.json'
    logging.info('Writing box results to %s' % box_output)
    with open(box_output, 'w') as f:
        json.dump(detection_results, f)

    segmentation_output = output_root / 'segmentation_fbms_results.json'
    logging.info('Writing segmentation results to %s' % segmentation_output)
    with open(segmentation_output, 'w') as f:
        json.dump(segmentation_results, f)

    for eval_type, results in (('bbox', detection_results),
                               ('segm', segmentation_results)):
        predictions_dataset = dataset.loadRes(results)
        coco_eval = COCOeval(dataset, predictions_dataset, eval_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        summary_f = io.StringIO()
        with redirect_stdout(summary_f):
            coco_eval.summarize()
        summary = summary_f.getvalue()
        logging.info('COCO evaluation:')
        logging.info('\n%s', summary)


if __name__ == "__main__":
    main()
