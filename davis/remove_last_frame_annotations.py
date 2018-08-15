import argparse
import collections
import json
import logging
from pathlib import Path, PurePath

from utils.log import setup_logging


def main():
    with open(__file__, 'r') as f:
        _source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', required=True)

    args = parser.parse_args()

    logging_path = str(PurePath(args.output_json).with_suffix('.log'))
    setup_logging(logging_path)
    file_logger = logging.getLogger(logging_path)
    logging.info('Source path: %s', Path(__file__).resolve())
    logging.info('Args:\n%s' % vars(args))

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    sequence_images = collections.defaultdict(list)
    for image in data['images']:
        sequence = PurePath(image['file_name']).parent
        sequence_images[sequence].append(image)

    valid_images = []
    valid_image_ids = set()
    removed_images = []
    for sequence in sequence_images:
        sorted_images = sorted(
            sequence_images[sequence],
            key=lambda x: int(PurePath(x['file_name']).stem))
        valid_images.extend(sorted_images[:-1])
        removed_images.append(sorted_images[-1]['file_name'])

    logging.info(
        'Removing %s images: %s' % (len(removed_images), removed_images))

    for image in valid_images:
        valid_image_ids.add(image['id'])

    annotations = [
        x for x in data['annotations'] if x['image_id'] in valid_image_ids
    ]
    logging.info('Kept %s/%s images, %s/%s annotations' %
                 (len(valid_images), len(data['images']), len(annotations),
                  len(data['annotations'])))
    data['images'] = valid_images
    data['annotations'] = annotations
    with open(args.output_json, 'w') as f:
        json.dump(data, f)

    file_logger.info('Source:')
    file_logger.info('=======')
    file_logger.info(_source)
    file_logger.info('=======')


if __name__ == "__main__":
    main()
