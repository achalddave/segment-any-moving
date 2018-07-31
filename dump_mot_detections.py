"""Convert MaskRCNN detections to MOT format text file."""

import argparse
import logging
import pickle
from pathlib import Path
from pprint import pformat

from tqdm import tqdm

from track import Detection
from utils.datasets import get_classes
from utils.log import setup_logging
from utils.vis import convert_from_cls_format


# The last three fields are 'x', 'y', and 'z', and are only used for
# 3D object detection.
DETECTION_FORMAT = (
    '{frame},{track_id},{left},{top},{width},{height},{conf},-1,-1,-1'
    '\n')


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detectron-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="""Search recursively in detectron-dir for pickle files. Any
                subdirectory containing a pickle file is considered to be
                a sequence.""")
    parser.add_argument(
        '--detectron-dataset', default='coco', choices=['coco'])

    args = parser.parse_args()
    detectron_dir = Path(args.detectron_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(str(output_dir / (Path(__file__).stem + '.log')))

    input_sequences = set(x.parent for x in detectron_dir.rglob('*.pickle'))
    logging.info('Input sequences: %s' % pformat(map(str, input_sequences)))

    label_list = get_classes(args.detectron_dataset)
    for sequence_path in tqdm(input_sequences):
        output_path = output_dir / (
            sequence_path.relative_to(detectron_dir)).with_suffix('.txt')

        detections = {}
        for pickle_path in sequence_path.glob('*.pickle'):
            timestamp = int(pickle_path.stem)
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            boxes, _, _, labels = convert_from_cls_format(
                data['boxes'], data['segmentations'], data['keypoints'])
            detections[timestamp] = [
                Detection(box[:4], box[4], label, timestamp)
                for box, label in zip(boxes, labels)
                if label_list[label] == 'person'
            ]

        output_str = ''
        for frame, frame_detections in sorted(
                detections.items(), key=lambda x: x[0]):
            for detection in frame_detections:
                x0, y0, x1, y1 = detection.box
                width = x1 - x0
                height = y1 - y0
                output_str += DETECTION_FORMAT.format(
                    frame=frame,
                    track_id=-1,
                    left=x0,
                    top=y0,
                    width=width,
                    height=height,
                    conf=detection.score,
                    x=-1,
                    y=-1,
                    z=-1)
        with open(output_path, 'w') as f:
            f.write(output_str)



if __name__ == "__main__":
    main()
