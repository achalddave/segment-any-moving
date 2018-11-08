"""Print sequences from a JSON annotations file."""

import argparse
import json
import os


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('annotations_json')

    args = parser.parse_args()
    with open(args.annotations_json, 'r') as f:
        data = json.load(f)
        sequences = set(os.path.split(x['file_name'])[0] for x in data['images'])
    print('\n'.join(sorted(sequences)))

if __name__ == "__main__":
    main()
