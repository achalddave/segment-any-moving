"""Remove annotations for the last frame from YTVOS JSON annotations.

We cannot compute forward optical flow for the last frame, so it doesn't make
sense to train on it. At test time, we may want to evaluate on the last frame
(if we care about it), but we may want to remove it as well to get a fair
comparison.

Note that we cannot use the same script as for DAVIS. In DAVIS, every frame is
annotated, so we can simply remove the last frame for every sequence. For
YTVOS, frames are annotated at 6FPS, so we have to check if the last frame for
a sequence in the annotations is actually the last frame in the video."""

import argparse
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
    parser.add_argument('--images-root', required=True)

    args = parser.parse_args()

    images_root = Path(args.images_root)
    if not images_root.exists():
        raise ValueError('Could not find --images-root: %s' % images_root)

    logging_path = str(args.output_json + '.log')
    setup_logging(logging_path)
    file_logger = logging.getLogger(logging_path)
    logging.info('Source path: %s', Path(__file__).resolve())
    logging.info('Args:\n%s' % vars(args))

    if (images_root / 'JPEGImages').exists():
        logging.info('Found JPEGImages in %s, updating --images-root to %s' %
                     (images_root, images_root / 'JPEGImages'))
        images_root = images_root / 'JPEGImages'

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    # Set of (sequence, frame_filename) tuples.
    to_remove_images = set()
    for sequence in images_root.iterdir():
        images = sorted(sequence.iterdir(), key=lambda x: int(x.stem))
        to_remove_images.add((sequence.stem, images[-1].stem))
    logging.info('%s images marked for removal.' % len(to_remove_images))

    valid_images = []
    removed_images = []
    annotated_sequences = set()
    for image in data['images']:
        path = PurePath(image['file_name'])
        sequence = path.parent.stem
        annotated_sequences.add(sequence)
        image_name = path.stem
        if (sequence, image_name) not in to_remove_images:
            valid_images.append(image)
        else:
            removed_images.append(image['file_name'])

    logging.info('Removing %s images from %s sequences.' %
                 (len(removed_images), len(annotated_sequences)))

    valid_image_ids = set(x['id'] for x in valid_images)
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
