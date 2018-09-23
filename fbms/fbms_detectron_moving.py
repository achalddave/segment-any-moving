"""Evaluate FBMS detectron outputs using COCO evaluation."""

import argparse
import io
import json
import logging
import pickle
from contextlib import redirect_stdout
from pathlib import Path, PurePath
from pprint import pformat

import numpy as np
from PIL import Image
from tqdm import tqdm

import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import utils.fbms.utils as fbms_utils
from utils.log import setup_logging
from utils import vis


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
        path = Path(annotation['file_name'])
        sequence = path.parent.stem
        frame_key_to_id[(sequence, path.stem)] = annotation['id']

    # Map image paths to dict containing 'boxes', 'segmentations'
    logging.info('Loading motion paths')
    # Map sequence to dict mapping frame index to motion mask path
    motion_mask_paths = {}
    for sequence_path in tqdm(list(motion_root.iterdir())):
        if not sequence_path.is_dir():
            continue

        sequence = sequence_path.stem
        motion_mask_paths[sequence] = {}
        for motion_path in sequence_path.glob('{}*.png'.format(sequence)):
            frame_index = fbms_utils.get_framenumber(motion_path.stem)
            motion_mask_paths[sequence][frame_index] = motion_path
        # The last frame doesn't have a motion segmentation mask, so we use the
        # second to last frame's motion mask as the last frame's motion mask.
        last_frame = max(motion_mask_paths[sequence].keys()) + 1
        motion_mask_paths[sequence][last_frame] = (
            motion_mask_paths[sequence][last_frame - 1])

    logging.info('Loading detectron paths')
    predictions = {}  # Map (sequence, frame_name) to predictions
    for sequence_path in tqdm(list(detectron_root.iterdir())):
        if not sequence_path.is_dir():
            continue

        for detectron_path in sequence_path.glob('*.pickle'):
            with open(detectron_path, 'rb') as f:
                frame_data = pickle.load(f)

            if frame_data['segmentations'] is None:
                frame_data['segmentations'] = [
                    [] for _ in range(len(frame_data['boxes']))
                ]
            frame_key = (sequence_path.stem, detectron_path.stem)
            predictions[frame_key] = {'boxes': [], 'segmentations': []}
            if frame_data['segmentations'] is None:
                continue
            for c in range(len(frame_data['boxes'])):
                predictions[frame_key]['boxes'].extend(frame_data['boxes'][c])
                predictions[frame_key]['segmentations'].extend(
                    frame_data['segmentations'][c])

    logging.info('Outputting moving detections')
    detection_results = []
    segmentation_results = []

    for sequence, frame_name in tqdm(predictions.keys()):
        frame_key = (sequence, frame_name)

        # If --save-pickle is true, process every frame. Otherwise, only
        # process frames that are in --fbms-annotations-json.
        if not args.save_pickle and frame_key not in frame_key_to_id:
            continue

        boxes = predictions[frame_key]['boxes']
        segmentations = predictions[frame_key]['segmentations']

        frame_index = fbms_utils.get_framenumber(frame_name)
        motion_mask = np.array(
            Image.open(motion_mask_paths[sequence][frame_index])) != 0

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
