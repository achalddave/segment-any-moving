import argparse
import json
import logging
from pathlib import Path

from get_videos import _set_logging


def main():
    with open(__file__, 'r') as f:
        _source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-json', nargs='+')
    parser.add_argument('--output-json', required=True)

    args = parser.parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(exist_ok=True, parents=True)
    logging_path = str(output_json.with_suffix('.log'))
    _set_logging(logging_path)
    file_logger = logging.getLogger(logging_path)
    logging.info('Args: %s' % vars(args))

    datasets = []
    for input_json in args.input_json:
        with open(input_json, 'r') as f:
            datasets.append(json.load(f))

    merged_dataset = datasets[0]
    for d in datasets[1:]:
        merged_dataset['images'].extend(d['images'])
        merged_dataset['annotations'].extend(d['annotations'])

    with open(args.output_json, 'w') as f:
        json.dump(merged_dataset, f)

    file_logger.info('Source:')
    file_logger.info('=======')
    file_logger.info(_source)
    file_logger.info('=======')



if __name__ == "__main__":
    main()
