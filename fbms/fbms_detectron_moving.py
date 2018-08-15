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
        '--visualize',
        action='store_true')
    parser.add_argument('--output-dir', required=True)

    args = parser.parse_args()
    detectron_root = Path(args.detectron_root)
    motion_root = Path(args.motion_masks_root)
    dataset = COCO(args.fbms_annotation_json)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True)
    logging_path = str(output_root / (Path(__file__).stem + '.log'))
    setup_logging(logging_path)

    with open(output_root / (Path(__name__).stem + '.py'), 'w') as f:
        f.write(_file_source)

    logging.info('Args:\n %s', pformat(vars(args)))

    frame_id_paths = {
        x['id']: PurePath(x['file_name'])
        for x in dataset.imgs.values()
    }
    # Map image paths to dict containing 'boxes', 'segmentations'
    predictions = {}
    logging.info('Loading motion paths')
    # Map sequence to dict mapping frame index to motion mask path
    motion_mask_paths = {}
    for sequence_path in tqdm(list(motion_root.iterdir())):
        if not sequence_path.is_dir():
            continue

        sequence = sequence_path.stem
        motion_mask_paths[sequence] = {}
        for motion_path in sequence_path.glob('*.png'):
            frame_index = fbms_utils.get_framenumber(motion_path.stem)
            motion_mask_paths[sequence][frame_index] = motion_path
        # The last frame doesn't have a motion segmentation mask, so we use the
        # second to last frame's motion mask as the last frame's motion mask.
        last_frame = max(motion_mask_paths[sequence].keys()) + 1
        motion_mask_paths[sequence][last_frame] = (
            motion_mask_paths[sequence][last_frame - 1])

    logging.info('Loading detectron paths')
    for frame_id, frame_path in tqdm(frame_id_paths.items()):
        # frame_path is of the form <split>/<sequence>/<frame_name>
        _, sequence, frame_name = frame_path.parts
        detectron_path = (
            detectron_root / sequence / frame_name).with_suffix('.pickle')
        if not detectron_path.exists():
            raise ValueError(
                'Could not find detectron path at %s' % detectron_path)
        with open(detectron_path, 'rb') as f:
            frame_data = pickle.load(f)

        predictions[frame_id] = {'boxes': [], 'segmentations': []}
        for c in range(len(frame_data['boxes'])):
            predictions[frame_id]['boxes'].extend(frame_data['boxes'][c])
            predictions[frame_id]['segmentations'].extend(
                frame_data['segmentations'][c])

    logging.info('Outputting moving detections')
    detection_results = []
    segmentation_results = []
    if args.visualize:
        output_images_dir = output_root / 'vis'
    for frame_id, frame_path in tqdm(frame_id_paths.items()):
        boxes = predictions[frame_id]['boxes']
        segmentations = predictions[frame_id]['segmentations']

        _, sequence, frame_name = frame_path.parts
        frame_index = fbms_utils.get_framenumber(frame_name)
        motion_mask = np.array(
            Image.open(motion_mask_paths[sequence][frame_index])) != 0

        is_moving = []
        for box, segmentation in zip(boxes, segmentations):
            mask = mask_util.decode(segmentation)
            moving_portion = (mask & motion_mask).sum() / mask.sum()
            is_moving.append(True)

        for i, (box, segmentation) in enumerate(zip(boxes, segmentations)):
            if not is_moving[i]:
                continue
            x1, y1, x2, y2, score = box.tolist()
            w = x2 - x1 + 1
            h = y2 - y1 + 1
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

        if args.visualize:
            output_image = np.zeros((motion_mask.shape[0],
                                     motion_mask.shape[1], 3), dtype=np.uint8)
            output_image = vis.vis_one_image_opencv(
                output_image,
                np.array([b for i, b in enumerate(boxes) if is_moving[i]]),
                [s for i, s in enumerate(segmentations) if is_moving[i]])
            output_path = output_images_dir / frame_path
            output_path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(output_image).save(output_images_dir / frame_path)

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
