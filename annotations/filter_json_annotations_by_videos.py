"""Select subset of JSON annotations belonging to the specified sequences.

Assumes that the input COCO-style JSON annotations have images whose
'file_name' field of the format '<sequence_name>/<frame>.<extension>'."""

import argparse
import json
import logging
from pathlib import Path, PurePath
from pprint import pformat

from utils.log import setup_logging


def main():
    with open(__file__, 'r') as f:
        _source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-split-videos', required=True)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', required=True)

    args = parser.parse_args()
    output_json = Path(args.output_json)
    logging_path = args.output_json + '.log'
    setup_logging(logging_path)
    file_logger = logging.getLogger(logging_path)

    logging.info('Args: %s', pformat(vars(args)))
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    if args.input_split_videos.endswith('.csv'):
        import csv
        with open(args.input_split_videos, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            valid_sequences = [row['video'] for row in reader]
    else:
        with open(args.input_split_videos, 'r') as f:
            valid_sequences = [x.strip() for x in f.readlines()]
            logging.info('Valid sequences: %s' % valid_sequences)

    images = []
    image_ids = set()
    present_sequences = set()
    for image in data['images']:
        sequence = str(PurePath(image['file_name']).parent)
        present_sequences.add(sequence)
        if sequence in valid_sequences:
            images.append(image)
            image_ids.add(image['id'])

    new_annotations = [
        x for x in data['annotations'] if x['image_id'] in image_ids
    ]
    new_images = images

    logging.info(
        'Kept %s/%s sequences' % (len(present_sequences & valid_sequences),
                                  len(present_sequences)))
    logging.info('Kept %s/%s annotations, %s/%s images' %
                 (len(new_annotations), len(data['annotations']),
                  len(new_images), len(data['images'])))
    data['annotations'] = new_annotations
    data['images'] = new_images
    with open(args.output_json, 'w') as f:
        json.dump(data, f)


    file_logger.info('Source:')
    file_logger.info('=======')
    file_logger.info(_source)
    file_logger.info('=======')


if __name__ == "__main__":
    main()
