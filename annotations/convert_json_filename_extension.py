import argparse
import json
import logging
from pathlib import Path, PurePath

from tqdm import tqdm

from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--new-extension', required=True)

    args = parser.parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    if not args.new_extension.startswith('.'):
        args.new_extension = '.' + args.new_extension

    log_path = args.output_json + '.log'
    setup_logging(log_path)
    logging.info('Args:\n%s' % vars(args))

    assert input_path.exists()
    assert not output_path.exists()

    with open(input_path, 'r') as f:
        data = json.load(f)

    for image in tqdm(data['images']):
        image['file_name'] = str(
            PurePath(image['file_name']).with_suffix(args.new_extension))

    with open(output_path, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
