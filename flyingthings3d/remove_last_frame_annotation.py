"""Remove annotations for the last frame of each video.

We cannot compute forward optical flow for the last frame, so it doesn't make
sense to train on it. At test time, we may want to evaluate on the last frame
(if we care about it), but we may want to remove it as well to get a fair
comparison.
"""

import argparse
import json
import logging
from pathlib import Path
from pprint import pformat

from tqdm import tqdm

from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', required=True)

    args = parser.parse_args()

    with open(__file__, 'r') as f:
        _source = f.read()

    input_path = Path(args.input_json)
    output_json = Path(args.output_json)

    assert input_path.exists()
    assert not output_json.exists()

    log_path = args.output_json + '.log'
    setup_logging(log_path)
    logging.info('Args: %s' % pformat(vars(args)))

    logging.info('Loading json from %s' % input_path)
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    logging.info('Loaded json')

    valid_images = []
    valid_image_ids = set()
    for image in tqdm(input_data['images']):
        image_path = Path(image['file_name'])
        if image_path.stem != '0015':
            valid_images.append(image)
            valid_image_ids.add(image['id'])
        else:
            logging.info(
                'Removing image id: %s, path: %s' % (image['id'], image_path))

    valid_annotations = []
    for annotation in tqdm(input_data['annotations']):
        if annotation['image_id'] in valid_image_ids:
            valid_annotations.append(annotation)

    logging.info('Removed %s images, %s annotations' %
                 (len(input_data['images']) - len(valid_images),
                  len(input_data['annotations']) - len(valid_annotations)))
    logging.info('Kept %s images, %s annotations' %
                 (len(valid_images), len(valid_annotations)))
    input_data['images'] = valid_images
    input_data['annotations'] = valid_annotations

    with open(output_json, 'w') as f:
        json.dump(input_data, f)

    file_logger = logging.getLogger(log_path)
    file_logger.info('Source:')
    file_logger.info('=======\n%s', _source)
    file_logger.info('=======')


if __name__ == "__main__":
    main()
