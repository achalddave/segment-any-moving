"""Evaluate detectron outputs using COCO evaluation."""

import argparse
import io
import json
import logging
import pickle
from contextlib import redirect_stdout
from pathlib import Path
from pprint import pformat

from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.log import setup_logging


def main():
    with open(__file__, 'r') as f:
        _file_source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--annotation-json',
        help='JSON annotations',
        required=True)
    parser.add_argument(
        '--detectron-root',
        help='Directory containing outputs of detectron.',
        required=True)
    parser.add_argument('--output-dir', required=True)

    args = parser.parse_args()
    detectron_root = Path(args.detectron_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True)
    logging_path = str(output_root / (Path(__file__).stem + '.log'))
    setup_logging(logging_path)
    dataset = COCO(args.annotation_json)

    with open(output_root / (Path(__name__).stem + '.py'), 'w') as f:
        f.write(_file_source)

    logging.info('Args:\n %s', pformat(vars(args)))

    frame_id_paths = {x['id']: x['file_name'] for x in dataset.imgs.values()}
    # Map image paths to annotations
    predictions = {}
    for frame_id, frame_path in tqdm(frame_id_paths.items()):
        detectron_path = (detectron_root / frame_path).with_suffix('.pickle')
        if not detectron_path.exists():
            raise ValueError(
                'Could not find detectron path at %s' % detectron_path)
        with open(detectron_path, 'rb') as f:
            predictions[frame_id] = pickle.load(f)

    detection_results = []
    segmentation_results = []
    for frame_id in tqdm(frame_id_paths.keys()):
        boxes = []
        segmentations = []
        for c in range(len(predictions[frame_id]['boxes'])):
            boxes.extend(predictions[frame_id]['boxes'][c])
            segmentations.extend(predictions[frame_id]['segmentations'][c])
        for box, segmentation in zip(boxes, segmentations):
            x1, y1, x2, y2, score = box.tolist()
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            detection_results.append({
                'image_id': frame_id,
                'category_id': 1,
                'bbox': [x1, y1, w, h],
                'score': score,
            })

            segmentation_results.append({
                'image_id': frame_id,
                'category_id': 1,
                'segmentation': segmentation,
                'score': score
            })

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
        logging.info('COCO evaluation:')
        logging.info('\n%s', summary_f.getvalue())


if __name__ == "__main__":
    main()
