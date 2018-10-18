import argparse
import collections
import logging
import pprint
from pathlib import Path

import numpy as np
from tqdm import tqdm

import tracker.track as tracker
import utils.log as log_utils
from utils.fbms import utils as fbms_utils


def output_fbms_tracks(tracks, groundtruth_dir, output_file, progress=True):
    # Map frame number to list of Detections
    detections_by_frame = collections.defaultdict(list)
    for track in tracks:
        for detection in track.detections:
            detections_by_frame[detection.timestamp].append(detection)
    # Disable defaultdict functionality so missing keys raise errors.
    detections_by_frame.default_factory = None
    assert len(detections_by_frame) > 0

    groundtruth = fbms_utils.FbmsGroundtruth(Path(groundtruth_dir))
    segmentations = {}
    image_size = tracks[0].detections[0].image.shape[:2]

    for frame_offset, frame_path in groundtruth.frame_label_paths.items():
        segmentation = np.zeros(image_size)
        if frame_offset not in detections_by_frame:
            segmentations[frame_offset] = segmentation
            continue
        # Sort by ascending score; this is the order we will paint
        # segmentations in.
        detections = sorted(
            detections_by_frame[frame_offset], key=lambda x: x.score)
        for i, detection in enumerate(detections):
            mask = detection.decoded_mask()
            label = detection.track.id + 1
            segmentation[mask == 1] = label
        segmentations[frame_offset] = segmentation

    fbms_tracks = fbms_utils.masks_to_tracks(segmentations)
    fbms_tracks_str = fbms_utils.get_tracks_text(
        fbms_tracks, groundtruth.num_frames, verbose=progress)
    with open(output_file, 'w') as f:
        f.write(fbms_tracks_str)


def main():
    tracking_parser = tracker.create_tracking_parser()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        parents=[tracking_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--detections-dir',
        type=Path,
        help=('Contains subdirectories for each FBMS sequence, each of which '
              'contain pickle files of detections for each frame.'),
        required=True)
    parser.add_argument(
        '--fbms-split-root',
        type=Path,
        required=True,
        help=('Directory containing subdirectories for each sequence, each of '
              'which contains a ".dat" file of groundtruth. E.g. '
              '<FBMS_ROOT>/TestSet'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--extension', default='.jpg')
    parser.add_argument(
        '--vis-dataset',
        default='objectness',
        choices=['coco', 'objectness'],
        help='Dataset to use for mapping label indices to names.')

    tracking_params, remaining_argv = tracking_parser.parse_known_args()
    args = parser.parse_args(remaining_argv)

    tracking_params = vars(tracking_params)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_log_file = log_utils.add_time_to_path(
        args.output_dir / 'tracker.log')
    log_utils.setup_logging(output_log_file)
    logging.info('Args: %s', pprint.pformat(vars(args)))
    logging.info('Tracking params: %s', pprint.pformat(tracking_params))

    detectron_input = args.detections_root
    if not detectron_input.is_dir():
        raise ValueError(
            '--detectron-dir %s is not a directory!' % args.detections_root)

    for split in ['TestSet', 'TrainingSet']:
        if (detectron_input / split).exists():
            raise ValueError(
                f"--detectron-dir contains a '{split}' subdirectory; it "
                "should just contain a subdirectory for each sequence.")

    sequences = [s for s in detectron_input.iterdir() if s.is_dir()]
    all_shot_paths = []  # Paths for all_shots.txt
    all_track_paths = []  # Paths for all_tracks.txt

    for sequence_dir in tqdm(sequences):
        detection_results = tracker.load_detectron_pickles(
            sequence_dir, fbms_utils.get_framenumber)
        frames = sorted(
            detection_results.keys(), key=fbms_utils.get_framenumber)
        frame_paths = [
            args.fbms_split_root / sequence_dir.name / (frame + args.extension)
            for frame in frames
        ]
        groundtruth_dir = (
            args.fbms_split_root / sequence_dir.name / 'GroundTruth')

        all_tracks = tracker.track(
            frame_paths, [detection_results[frame] for frame in frames],
            tracking_params, progress=False)

        output_track = args.output_dir / (sequence_dir.name + '.dat')
        output_fbms_tracks(
            all_tracks, groundtruth_dir, output_track, progress=False)

        if args.save_video:
            tracker.visualize_tracks(
                all_tracks,
                frame_paths,
                args.vis_dataset,
                tracking_params,
                output_video=output_track.with_suffix('.mp4'),
                output_video_fps=args.fps,
                progress=False)

        all_shot_paths.append(
            groundtruth_dir / (sequence_dir.name + 'Def.dat'))
        all_track_paths.append(output_track)

    with open(args.output_dir / 'all_shots.txt', 'w') as f:
        f.write(str(len(all_shot_paths)) + '\n')
        f.write('\n'.join(str(x.resolve()) for x in all_shot_paths))

    with open(args.output_dir / 'all_tracks.txt', 'w') as f:
        f.write('\n'.join(str(x.resolve()) for x in all_track_paths))


if __name__ == "__main__":
    main()
