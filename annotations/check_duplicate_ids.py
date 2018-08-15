import argparse
import collections
import json


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_json')

    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    image_id_counts = collections.Counter([x['id'] for x in data['images']])
    annotation_id_counts = collections.Counter(
        [x['id'] for x in data['annotations']])

    image_id_duplicates = {x: y for x, y in image_id_counts.items() if y > 1}
    annotation_id_duplicates = {
        x: y
        for x, y in annotation_id_counts.items() if y > 1
    }

    if image_id_duplicates:
        print('ERROR: Image id duplicates found: %s' % image_id_duplicates)

    if annotation_id_duplicates:
        print('ERROR: Annotation id duplicates found: %s' %
              annotation_id_duplicates)

    if not image_id_duplicates and not annotation_id_duplicates:
        print('SUCCESS: No image id nor annotation id duplicates found.')


if __name__ == "__main__":
    main()
