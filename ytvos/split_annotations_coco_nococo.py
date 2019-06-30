import argparse
import json
import logging
from pathlib import Path

from script_utils.common import common_setup


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ytvos-matched',
        help='JSON mapping some YTVOS labels to COCO labels.',
        default=Path(
            '/data/achald/track/ytvos/compare-labels-with-coco/all_matched_labels.json'
        ),
        type=Path)
    parser.add_argument(
        '--annotations-dir',
        required=True,
        type=Path,
        help='Contains directory with annotations for each video')
    parser.add_argument('--meta-json',
                        help='YTVOS train meta.json',
                        default=Path('/data/all/youtube-vos/train/meta.json'),
                        type=Path)
    parser.add_argument('--output-dir',
                        type=Path,
                        required=True)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    with open(args.meta_json, 'r') as f:
        all_labels = json.load(f)

    labels = {
        video: {x['category'] for x in info['objects'].values()}
        for video, info in all_labels['videos'].items()
    }

    with open(args.ytvos_matched, 'r') as f:
        known_labels = set(json.load(f).keys())

    unknown_only = []
    known_only = []
    mixed = []
    for video_dir in args.annotations_dir.iterdir():
        if not video_dir.is_dir():
            continue
        video = video_dir.name
        video_labels = labels[video]
        if video_labels.isdisjoint(known_labels):
            unknown_only.append(video)
        elif video_labels.issubset(known_labels):
            known_only.append(video)
        else:
            mixed.append(video)

    logging.info('New category videos: %s', len(unknown_only))
    logging.info('Known category videos: %s', len(known_only))
    logging.info('Mixed new/known category videos: %s', len(mixed))

    unknown_dir = args.output_dir / 'unknown'
    unknown_dir.mkdir(exist_ok=True)
    for video in unknown_only:
        destination = unknown_dir / video
        source = args.annotations_dir / video
        destination.symlink_to(source.resolve())

    known_dir = args.output_dir / 'known'
    known_dir.mkdir(exist_ok=True)
    for video in known_only:
        destination = known_dir / video
        source = args.annotations_dir / video
        destination.symlink_to(source.resolve())

    mixed_dir = args.output_dir / 'mixed'
    mixed_dir.mkdir(exist_ok=True)
    for video in mixed:
        destination = mixed_dir / video
        source = args.annotations_dir / video
        destination.symlink_to(source.resolve())


if __name__ == "__main__":
    main()