import argparse
import json
import logging
import pprint
from pathlib import Path

from script_utils.common import common_setup


COCO_CATEGORIES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}


def levenshtein(s1, s2):
    """https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one
            # character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--meta-json',
                        default=Path('/data/all/youtube-vos/train/meta.json'),
                        type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    with open(args.meta_json, 'r') as f:
        data = json.load(f)
    ytvos_categories = set([
        y['category'] for x in data['videos'].values()
        for y in x['objects'].values()
    ])

    logging.info('Total YTVOS categories: %s', len(ytvos_categories))
    common_categories = ytvos_categories & COCO_CATEGORIES
    ytvos_only = ytvos_categories - COCO_CATEGORIES
    logging.info('Common: %s', ', '.join(list(common_categories)))
    logging.info('YTVOS only: %s', ', '.join(list(ytvos_only)))

    with open(args.output_dir / 'ytvos_not_in_coco_exact.txt', 'w') as f:
        f.write('\n'.join(sorted(ytvos_only)))

    with open(args.output_dir / 'coco_labels.txt', 'w') as f:
        f.write('\n'.join(sorted(COCO_CATEGORIES)))

    with open(args.output_dir / 'exact_match_labels.json', 'w') as f:
        json.dump({x: x for x in common_categories}, f, indent=True)

    near_matches = {}
    for category in ytvos_only:
        near_matches[category] = sorted(
            COCO_CATEGORIES, key=lambda l: levenshtein(category, l))[:5]
    with open(args.output_dir / 'closest_matches.json', 'w') as f:
        json.dump(near_matches, f, indent=True)


if __name__ == "__main__":
    main()