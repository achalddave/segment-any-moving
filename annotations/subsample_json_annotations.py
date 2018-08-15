import argparse
import json
import logging
import random
from pathlib import Path

from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--keep-num-images', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    log_path = args.output_json + '.log'
    setup_logging(log_path)
    logging.info('Args:\n%s' % vars(args))

    assert input_path.exists()
    assert not output_path.exists()

    with open(input_path, 'r') as f:
        data = json.load(f)

    image_ids = [x['id'] for x in data['images']]
    import collections
    ids_count = collections.Counter(image_ids)
    repeated = {x: y for x, y in ids_count.items() if y > 1}
    random.shuffle(image_ids)
    kept_image_ids = set(image_ids[:args.keep_num_images])
    __import__('ipdb').set_trace()

    subsampled_images = [
        x for x in data['images'] if x['id'] in kept_image_ids
    ]
    subsampled_annotations = [
        x for x in data['annotations'] if x['image_id'] in kept_image_ids
    ]

    logging.info(
        'Kept %s/%s images' % (len(subsampled_images), len(data['images'])))
    logging.info('Kept %s/%s annotations' % (len(subsampled_annotations),
                                             len(data['annotations'])))

    data['images'] = subsampled_images
    data['annotations'] = subsampled_annotations

    with open(output_path, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
